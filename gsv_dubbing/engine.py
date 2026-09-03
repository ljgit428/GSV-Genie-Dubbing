"""配音引擎：串联 字幕解析 → 说话人路由 → GPT-SoVITS 合成 → 时间轴导出。

核心策略：
- 逐句合成（cut0 不切分），保证每句独立、可控
- 语速自适应：第一次按 speed=1 合成；若音频比字幕窗口长，按比例提速重试（上限 max_speed）
- 断点续跑：done 的句子直接复用缓存
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .audio_utils import build_timeline, normalize_peak, resample, save_wav
from .gsv_client import GSVClient, GSVError
from .session_state import SessionState
from .speaker_router import SpeakerConfig, SpeakerRouter
from .subtitle_parser import SubtitleBlock, filter_blocks, load_subtitles

ProgressCallback = Callable[..., None]


@dataclass
class DubConfig:
    api_url: str = "http://127.0.0.1:9880"
    ref_audio: str = ""
    prompt_text: str = ""
    text_lang: str = "zh"
    prompt_lang: str = "zh"
    gpt_weights: str = ""
    sovits_weights: str = ""
    speed: float = 1.0
    max_speed: float = 1.4
    min_speed: float = 0.6
    fit_timeline: bool = True          # 超窗自动提速
    seed: int = -1
    top_k: int = 5
    top_p: float = 1.0
    temperature: float = 1.0
    retry: int = 3
    retry_wait: float = 2.0
    sample_steps: int = 32
    repetition_penalty: float = 1.35
    strip_brackets: bool = True
    only_cjk: bool = False
    out_sr: int = 32000
    normalize: bool = True
    speaker_profile: str = ""         # JSON 路径；空则用上面的全局默认
    max_workers: int = 1              # >1 时按相同说话人分组并行
    # 进度回调
    on_progress: Optional[ProgressCallback] = None   # (done, total, block, msg)
    on_line_done: Optional[Callable] = None          # (block, path, speed)


@dataclass
class DubResult:
    out_wav: str
    total_lines: int
    voiced_lines: int
    failed_lines: int
    clips_dir: str
    session_file: str


class DubbingEngine:
    def __init__(
        self,
        subtitle_path: str,
        output_dir: str,
        config: DubConfig,
        session_dir: Optional[str] = None,
    ):
        self.subtitle_path = Path(subtitle_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = config
        self.session_dir = Path(session_dir) if session_dir else (
            self.output_dir / self.subtitle_path.stem / "session"
        )
        self.clips_dir = self.session_dir / "clips"
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.state = SessionState(self.session_dir)
        self.client = GSVClient(config.api_url)
        self.router = (
            SpeakerRouter.from_file(config.speaker_profile)
            if config.speaker_profile
            else None
        )
        self._stop = threading.Event()
        self._paused = threading.Event()

    # ------------------------------------------------------------ 控制
    def stop(self):
        self._stop.set()

    def pause(self):
        self._paused.set()

    def resume(self):
        self._paused.clear()

    def _notify(self, done: int, total: int, block, msg: str):
        if self.cfg.on_progress:
            try:
                self.cfg.on_progress(done, total, block, msg)
            except Exception:
                pass

    # ------------------------------------------------------------ 主流程
    def run(self) -> DubResult:
        blocks_raw = load_subtitles(self.subtitle_path)
        blocks = filter_blocks(
            blocks_raw,
            strip_brackets=self.cfg.strip_brackets,
            only_cjk=self.cfg.only_cjk,
        )
        if not blocks:
            raise RuntimeError("字幕中未找到可朗读的句子")

        self.state.init_lines(
            blocks,
            meta={
                "subtitle": str(self.subtitle_path),
                "api": self.cfg.api_url,
                "ref_audio": self.cfg.ref_audio,
                "gpt_weights": self.cfg.gpt_weights,
                "sovits_weights": self.cfg.sovits_weights,
                "text_lang": self.cfg.text_lang,
            },
        )

        todo = [b for b in blocks if not self.state.is_done(b.index)]
        total = len(blocks)
        done = total - len(todo)
        self._notify(done, total, None, f"共 {total} 句，待合成 {len(todo)} 句（已完成 {done} 句复用缓存）")

        # 确保默认权重已加载
        if self.cfg.gpt_weights or self.cfg.sovits_weights:
            self.client.ensure_weights(self.cfg.gpt_weights, self.cfg.sovits_weights)

        for b in todo:
            if self._stop.is_set():
                break
            while self._paused.is_set():
                time.sleep(0.3)
                if self._stop.is_set():
                    break

            spk = self._route(b)
            try:
                self._dub_line(b, spk)
                done += 1
                self._notify(done, total, b, "OK")
            except GSVError as e:
                self.state.mark(b.index, "failed")
                self.state.save()
                done += 1
                self._notify(done, total, b, f"FAIL: {e}")
            self.state.save()

        # -------------------------------------------------- 合并导出
        if self._stop.is_set():
            raise InterruptedError("用户中止")
        return self._merge(blocks)

    # ------------------------------------------------------------ 单句
    def _route(self, b: SubtitleBlock) -> SpeakerConfig:
        if self.router:
            return self.router.route(b)
        return SpeakerConfig(
            name="__default__",
            ref_audio=self.cfg.ref_audio,
            prompt_text=self.cfg.prompt_text,
            prompt_lang=self.cfg.prompt_lang,
            gpt_weights=self.cfg.gpt_weights,
            sovits_weights=self.cfg.sovits_weights,
            speed=self.cfg.speed,
        )

    def _clean_for_tts(self, text: str, spk: SpeakerConfig) -> str:
        if self.router:
            return self.router.strip_speaker_prefix(text, spk)
        return SpeakerRouter.PREFIX_RE.sub("", text, count=1).strip()

    def _dub_line(self, b: SubtitleBlock, spk: SpeakerConfig) -> str:
        """合成一句并落盘缓存。返回相对文件名。"""
        rel = f"{b.index:05d}.wav"
        out = self.clips_dir / rel
        text = self._clean_for_tts(b.text, spk)
        if not text.strip():
            self.state.mark(b.index, "done", file=rel)
            return rel

        if spk.gpt_weights or spk.sovits_weights:
            self.client.ensure_weights(spk.gpt_weights or None, spk.sovits_weights or None)

        base_speed = spk.speed or self.cfg.speed
        speed = self.state.speed_history.get(b.index, base_speed)
        window = b.duration if self.cfg.fit_timeline else None

        last_err: Optional[Exception] = None
        for attempt in range(1, self.cfg.retry + 1):
            if self._stop.is_set():
                raise InterruptedError("用户中止")
            try:
                audio, sr = self.client.tts(
                    text=text,
                    text_lang=self.cfg.text_lang,
                    ref_audio_path=spk.ref_audio or self.cfg.ref_audio,
                    prompt_text=spk.prompt_text or self.cfg.prompt_text,
                    prompt_lang=spk.prompt_lang or self.cfg.prompt_lang,
                    top_k=self.cfg.top_k,
                    top_p=self.cfg.top_p,
                    temperature=self.cfg.temperature,
                    text_split_method="cut0",
                    speed_factor=speed,
                    seed=self.cfg.seed,
                    sample_steps=self.cfg.sample_steps,
                    repetition_penalty=self.cfg.repetition_penalty,
                )
                dur = audio.size / sr
                # 超窗则按比例提速重试
                if window and dur > window + 0.05 and speed < self.cfg.max_speed:
                    need = dur / max(window, 0.2)
                    new_speed = min(speed * need, self.cfg.max_speed)
                    if new_speed > speed * 1.02:
                        speed = new_speed
                        continue
                save_wav(out, audio, sr)
                self.state.mark(b.index, "done", file=rel, speed=speed)
                self.state.save()
                if self.cfg.on_line_done:
                    self.cfg.on_line_done(b, out, speed)
                return rel
            except (GSVError, InterruptedError) as e:
                last_err = e
                if isinstance(e, InterruptedError):
                    raise
                time.sleep(self.cfg.retry_wait)
        raise GSVError(f"重试 {self.cfg.retry} 次仍失败: {last_err}")

    # ------------------------------------------------------------ 合并
    def _merge(self, blocks: List[SubtitleBlock]) -> DubResult:
        import wave as _wave

        clips = []
        voiced = failed = 0
        for b in blocks:
            st = self.state.lines.get(b.index)
            if not st or st.status != "done" or not st.file:
                failed += 1
                continue
            path = self.clips_dir / st.file
            if not path.exists():
                failed += 1
                continue
            with _wave.open(str(path), "rb") as w:
                sr = w.getframerate()
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
            audio = pcm.astype(np.float32) / 32768.0
            clips.append((b.start, b.end, audio, sr))
            voiced += 1

        if not clips:
            raise RuntimeError("没有任何合成成功的句子，无法导出")

        total = max((e for _, e, _, _ in clips), default=0.0) + 0.5
        mixed = build_timeline(clips, total_duration=total, sr=self.cfg.out_sr)
        if self.cfg.normalize:
            mixed = normalize_peak(mixed)
        out_wav = self.output_dir / f"{self.subtitle_path.stem}_dubbed.wav"
        save_wav(out_wav, mixed, self.cfg.out_sr)

        return DubResult(
            out_wav=str(out_wav),
            total_lines=len(blocks),
            voiced_lines=voiced,
            failed_lines=failed,
            clips_dir=str(self.clips_dir),
            session_file=str(self.state.file),
        )
