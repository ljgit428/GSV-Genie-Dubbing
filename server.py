"""GSV-Genie-Dubbing Web 服务端。

启动: python server.py --port 8765
浏览器打开 http://127.0.0.1:8765

REST API:
  GET  /api/ping                      探测 GPT-SoVITS API
  GET  /api/defaults                  返回默认参数 + 上次会话
  POST /api/subtitle                  上传/加载字幕 → 返回逐句列表(可编辑)
  POST /api/subtitle/file             用服务器本地路径加载字幕
  POST /api/start                     开始批量配音 {session_id, overrides}
  POST /api/stop                      停止
  GET  /api/progress/{session_id}     轮询进度 (含完成的句子)
  POST /api/regen/{session_id}        重新生成单句 {index, text?, speed?}
  GET  /api/clip/{session_id}/{index} 播放/下载单句 WAV
  GET  /api/download/{session_id}    下载整轨 WAV
  POST /api/merge/{session_id}       重新按时间轴合并导出
"""

from __future__ import annotations

import argparse
import threading
import time
import traceback
import uuid
import wave
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from gsv_dubbing import DubConfig, DubbingEngine
from gsv_dubbing.audio_utils import build_timeline, normalize_peak, save_wav
from gsv_dubbing.gsv_client import GSVClient, GSVError
from gsv_dubbing.session_state import SessionState
from gsv_dubbing.subtitle_parser import SubtitleBlock, filter_blocks, load_subtitles

ROOT = Path(__file__).parent.resolve()
WORKSPACE = ROOT / "workspace"
WORKSPACE.mkdir(exist_ok=True)

app = FastAPI(title="GSV-Genie-Dubbing")


# ================================================================ 数据模型

class TTSParams(BaseModel):
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
    fit_timeline: bool = True
    seed: int = -1
    top_k: int = 5
    top_p: float = 1.0
    temperature: float = 1.0
    retry: int = 3
    sample_steps: int = 32
    repetition_penalty: float = 1.35
    strip_brackets: bool = True
    only_cjk: bool = False
    out_sr: int = 32000
    normalize: bool = True
    speaker_profile: str = ""


class SubtitleFileReq(BaseModel):
    path: str
    params: TTSParams


class StartReq(BaseModel):
    session_id: str
    # 逐句编辑覆盖 {index: {text, speed, skip}}
    overrides: Dict[int, dict] = {}
    params: TTSParams


class RegenReq(BaseModel):
    index: int
    text: Optional[str] = None
    speed: Optional[float] = None
    params: TTSParams


# ================================================================ 会话管理

class Session:
    def __init__(self, sid: str, subtitle_path: Path, params: TTSParams):
        self.id = sid
        self.subtitle_path = subtitle_path
        self.params = params
        self.blocks: List[SubtitleBlock] = []
        self.out_dir = WORKSPACE / sid / "output"
        self.clips_dir = WORKSPACE / sid / "clips"
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.engine: Optional[DubbingEngine] = None
        self.state = SessionState(WORKSPACE / sid / "session")
        self.lock = threading.RLock()
        # 进度
        self.total = 0
        self.done = 0
        self.running = False
        self.error: Optional[str] = None
        self.finished_at: Optional[float] = None
        self.merged_wav: Optional[Path] = None
        self.stop_flag = threading.Event()
        self.thread: Optional[threading.Thread] = None
        # index → 当前编辑文本 / 语速 / 跳过
        self.edits: Dict[int, dict] = {}

    def load_blocks(self):
        raw = load_subtitles(self.subtitle_path)
        self.blocks = filter_blocks(
            raw, strip_brackets=self.params.strip_brackets, only_cjk=self.params.only_cjk
        )
        # 用会话参数初始化状态文件（保留已完成）
        self.state.init_lines(
            [b for b in self.blocks],
            meta={"subtitle": str(self.subtitle_path), "api": self.params.api_url},
        )
        self.total = len(self.blocks)
        return self.blocks

    def block_by_index(self, index: int) -> Optional[SubtitleBlock]:
        for b in self.blocks:
            if b.index == index:
                return b
        return None

    def clip_path(self, index: int) -> Path:
        return self.clips_dir / f"{index:05d}.wav"


SESSIONS: Dict[str, Session] = {}
_sessions_lock = threading.Lock()


def get_session(sid: str) -> Session:
    with _sessions_lock:
        s = SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, f"session {sid} 不存在")
    return s


def _cfg_from(params: TTSParams, on_progress=None) -> DubConfig:
    return DubConfig(
        api_url=params.api_url,
        ref_audio=params.ref_audio,
        prompt_text=params.prompt_text,
        text_lang=params.text_lang,
        prompt_lang=params.prompt_lang,
        gpt_weights=params.gpt_weights,
        sovits_weights=params.sovits_weights,
        speaker_profile=params.speaker_profile or "",
        speed=params.speed,
        max_speed=params.max_speed,
        fit_timeline=params.fit_timeline,
        seed=params.seed,
        top_k=params.top_k,
        top_p=params.top_p,
        temperature=params.temperature,
        retry=params.retry,
        sample_steps=params.sample_steps,
        repetition_penalty=params.repetition_penalty,
        strip_brackets=params.strip_brackets,
        only_cjk=params.only_cjk,
        out_sr=params.out_sr,
        normalize=params.normalize,
        on_progress=on_progress,
    )


def _ts(sec: float) -> str:
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ================================================================ API

@app.get("/api/ping")
def api_ping():
    """探测 GPT-SoVITS API。"""
    return {"ok": GSVClient("http://127.0.0.1:9880").ping()}


@app.post("/api/ping")
def api_ping_url(params: TTSParams):
    return {"ok": GSVClient(params.api_url, timeout=6).ping()}


@app.post("/api/subtitle")
async def api_upload_subtitle(file: bytes, name: str = ""):
    """上传字幕文件内容（前端 File API 读成 bytes）。"""
    if not name:
        name = f"upload_{uuid.uuid4().hex[:6]}.srt"
    suffix = Path(name).suffix or ".srt"
    if suffix.lower() not in (".srt", ".ass", ".ssa", ".vtt", ".txt"):
        raise HTTPException(400, f"不支持的字幕格式: {suffix}")
    sid = uuid.uuid4().hex[:10]
    p = WORKSPACE / sid / "subtitle" / f"sub{suffix}"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(file)
    return {"session_id": sid, "path": str(p)}


@app.post("/api/subtitle/file")
def api_load_subtitle_file(req: SubtitleFileReq):
    p = Path(req.path)
    if not p.exists():
        raise HTTPException(400, f"文件不存在: {p}")
    sid = uuid.uuid4().hex[:10]
    s = Session(sid, p, req.params)
    try:
        s.load_blocks()
    except Exception as e:
        raise HTTPException(400, f"字幕解析失败: {e}")
    with _sessions_lock:
        SESSIONS[sid] = s
    return {
        "session_id": sid,
        "total": s.total,
        "done": s.state.done_count(),
        "lines": [
            {
                "index": b.index,
                "start": round(b.start, 3),
                "end": round(b.end, 3),
                "time": f"{_ts(b.start)} → {_ts(b.end)}",
                "duration": round(b.duration, 3),
                "speaker": b.speaker,
                "text": b.text,
                "status": "done" if s.state.is_done(b.index) else "pending",
            }
            for b in s.blocks
        ],
    }


@app.post("/api/start")
def api_start(req: StartReq):
    s = get_session(req.session_id)
    with s.lock:
        if s.running:
            raise HTTPException(409, "已在运行中")
        # 应用逐句编辑
        for idx_str, ov in req.overrides.items():
            idx = int(idx_str)
            b = s.block_by_index(idx)
            if b and ov.get("text") is not None:
                b.text = str(ov["text"])
                st = s.state.lines.get(idx)
                if st and st.text != b.text:      # 文本变了 → 缓存失效
                    st.status = "pending"
                    st.file = None
                    s.state.save(force=True)
            if ov.get("skip"):
                s.edits[idx] = {"skip": True}
            if ov.get("speed"):
                s.edits.setdefault(idx, {})["speed"] = float(ov["speed"])
        s.stop_flag.clear()
        s.error = None
        s.running = True

        def on_progress(done, total, block, msg):
            s.done = done
            s.total = total

        cfg = _cfg_from(req.params, on_progress=on_progress)

        def worker():
            try:
                todo = [b for b in s.blocks
                        if not s.state.is_done(b.index)
                        and not s.edits.get(b.index, {}).get("skip")]
                from gsv_dubbing.speaker_router import SpeakerRouter
                router = SpeakerRouter.from_file(req.params.speaker_profile) if req.params.speaker_profile else None
                client = GSVClient(req.params.api_url)
                if req.params.gpt_weights or req.params.sovits_weights:
                    client.ensure_weights(req.params.gpt_weights, req.params.sovits_weights)
                total = len(s.blocks)
                done = total - len(todo)
                s.total, s.done = total, done
                for b in todo:
                    if s.stop_flag.is_set():
                        break
                    spk_cfg = (
                        router.route(b) if router else None
                    )
                    text = b.text
                    if router:
                        text = router.strip_speaker_prefix(text, spk_cfg)
                    else:
                        text = SpeakerRouter.PREFIX_RE.sub("", text, count=1).strip()
                    speed = (
                        s.edits.get(b.index, {}).get("speed")
                        or (spk_cfg.speed if spk_cfg else None)
                        or req.params.speed
                    )
                    eff = _effective_cfg(cfg, spk_cfg, req.params)
                    audio, sr = _tts_one(client, b, text, eff, speed)
                    save_wav(s.clip_path(b.index), audio, sr)
                    s.state.mark(b.index, "done", file=f"{b.index:05d}.wav", speed=speed)
                    s.state.save()
                    done += 1
                    s.done = done
                s.merged_wav = None
                _merge_session(s)
            except Exception as e:
                s.error = f"{e}\n{traceback.format_exc(limit=3)}"
            finally:
                s.running = False
                s.finished_at = time.time()

        s.thread = threading.Thread(target=worker, daemon=True)
        s.thread.start()
    return {
        "ok": True,
        "total": s.total,
        "resume": len([
            b for b in s.blocks
            if not s.state.is_done(b.index) and not s.edits.get(b.index, {}).get("skip")
        ]),
    }


def _effective_cfg(cfg: DubConfig, spk_cfg, params: TTSParams) -> DubConfig:
    if spk_cfg is None:
        return cfg
    import copy
    c = copy.copy(cfg)
    if spk_cfg.ref_audio:
        c.ref_audio = spk_cfg.ref_audio
    if spk_cfg.prompt_text:
        c.prompt_text = spk_cfg.prompt_text
    c.prompt_lang = spk_cfg.prompt_lang or params.prompt_lang
    return c


def _tts_one(client: GSVClient, b: SubtitleBlock, text: str, cfg: DubConfig, speed: float):
    """合成一句，超窗自动提速。"""
    last_err = None
    window = b.duration if cfg.fit_timeline else None
    for _ in range(max(1, cfg.retry)):
        audio, sr = client.tts(
            text=text,
            text_lang=cfg.text_lang,
            ref_audio_path=cfg.ref_audio,
            prompt_text=cfg.prompt_text,
            prompt_lang=cfg.prompt_lang,
            top_k=cfg.top_k, top_p=cfg.top_p, temperature=cfg.temperature,
            text_split_method="cut0",
            speed_factor=speed, seed=cfg.seed,
            sample_steps=cfg.sample_steps,
            repetition_penalty=cfg.repetition_penalty,
        )
        dur = audio.size / sr
        if window and dur > window + 0.05 and speed < cfg.max_speed:
            need = dur / max(window, 0.2)
            new_speed = min(speed * need, cfg.max_speed)
            if new_speed > speed * 1.02:
                speed = new_speed
                continue
        return audio, sr
    return audio, sr


def _merge_session(s: Session) -> Path:
    clips = []
    for b in s.blocks:
        if s.edits.get(b.index, {}).get("skip"):
            continue
        st = s.state.lines.get(b.index)
        if not st or st.status != "done":
            continue
        f = s.clip_path(b.index)
        if not f.exists():
            continue
        with wave.open(str(f), "rb") as w:
            sr = w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        clips.append((b.start, b.end, pcm, sr))
    if not clips:
        raise RuntimeError("没有可合并的句子")
    total = max(e for _, e, _, _ in clips) + 0.5
    mixed = build_timeline(clips, total_duration=total, sr=s.params.out_sr)
    if s.params.normalize:
        mixed = normalize_peak(mixed)
    out = s.out_dir / "dubbed.wav"
    save_wav(out, mixed, s.params.out_sr)
    s.merged_wav = out
    return out


@app.post("/api/stop/{sid}")
def api_stop(sid: str):
    s = get_session(sid)
    s.stop_flag.set()
    return {"ok": True}


@app.get("/api/progress/{sid}")
def api_progress(sid: str):
    s = get_session(sid)
    done_lines = [
        {"index": st.index, "status": st.status, "speed": st.speed}
        for st in s.state.lines.values() if st.status in ("done", "failed")
    ]
    return {
        "running": s.running,
        "done": s.done,
        "total": s.total,
        "error": s.error,
        "finished": s.finished_at is not None and not s.running,
        "merged_available": s.merged_wav.exists() if s.merged_wav else False,
        "lines": done_lines,
    }


@app.post("/api/regen/{sid}")
def api_regen(sid: str, req: RegenReq):
    """重新生成单句（同步，前端 await 即拿到结果）。"""
    s = get_session(sid)
    if s.running and not s.stop_flag.is_set():
        # 批量进行中：等一小会儿也行，这里直接拒绝避免并发冲突
        raise HTTPException(409, "批量合成进行中，请先停止")
    b = s.block_by_index(req.index)
    if not b:
        raise HTTPException(404, f"句子 {req.index} 不存在")
    if req.text is not None and req.text.strip():
        b.text = req.text.strip()
    speed = req.speed or s.edits.get(req.index, {}).get("speed") or s.params.speed
    s.edits[req.index] = {**s.edits.get(req.index, {}), "speed": speed}
    try:
        client = GSVClient(req.params.api_url)
        if req.params.gpt_weights or req.params.sovits_weights:
            client.ensure_weights(req.params.gpt_weights, req.params.sovits_weights)
        cfg = _cfg_from(req.params)
        text = SpeakerRouter_prefix_strip(b.text) if not req.params.speaker_profile else None
        if req.params.speaker_profile:
            from gsv_dubbing.speaker_router import SpeakerRouter
            router = SpeakerRouter.from_file(req.params.speaker_profile)
            spk = router.route(b)
            if spk.ref_audio:
                cfg.ref_audio = spk.ref_audio
            if spk.prompt_text:
                cfg.prompt_text = spk.prompt_text
            text = router.strip_speaker_prefix(b.text, spk)
        audio, sr = _tts_one(client, b, text, cfg, speed)
        save_wav(s.clip_path(req.index), audio, sr)
        s.state.mark(req.index, "done", file=f"{req.index:05d}.wav", speed=speed)
        s.state.save()
        s.merged_wav = None
        return {"ok": True, "index": req.index, "duration": round(audio.size / sr, 2), "speed": speed}
    except GSVError as e:
        raise HTTPException(502, str(e))
    except Exception as e:
        raise HTTPException(500, f"{e}\n{traceback.format_exc(limit=2)}")


def SpeakerRouter_prefix_strip(text: str) -> str:
    from gsv_dubbing.speaker_router import SpeakerRouter
    return SpeakerRouter.PREFIX_RE.sub("", text, count=1).strip()


@app.get("/api/clip/{sid}/{index}")
def api_clip(sid: str, index: int):
    s = get_session(sid)
    f = s.clip_path(index)
    if not f.exists():
        raise HTTPException(404, "该句还没有音频，请先生成")
    return FileResponse(f, media_type="audio/wav", filename=f"{index:05d}.wav")


@app.get("/api/download/{sid}")
def api_download(sid: str):
    s = get_session(sid)
    if s.merged_wav is None or not s.merged_wav.exists():
        try:
            _merge_session(s)
        except Exception as e:
            raise HTTPException(400, f"合并失败: {e}")
    return FileResponse(s.merged_wav, media_type="audio/wav",
                        filename=f"{Path(s.subtitle_path).stem}_dubbed.wav")


@app.post("/api/merge/{sid}")
def api_merge(sid: str):
    s = get_session(sid)
    try:
        _merge_session(s)
    except Exception as e:
        raise HTTPException(400, f"合并失败: {e}")
    return {"ok": True, "path": str(s.merged_wav)}


# ================================================================ 前端静态页

@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html", media_type="text/html")


@app.get("/app.js")
def app_js():
    return FileResponse(ROOT / "web" / "app.js", media_type="application/javascript")


@app.get("/style.css")
def style_css():
    return FileResponse(ROOT / "web" / "style.css", media_type="text/css")


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
