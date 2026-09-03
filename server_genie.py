"""GSV-Genie-Dubbing Web 服务端 —— Genie (genie-tts) 后端。

与 server.py 同一套前端，推理层从 GPT-SoVITS api_v2 换成 Genie：
- Genie = GPT-SoVITS 的 ONNX 轻量推理引擎（pip install genie-tts）
- 角色 = ONNX 模型目录 + 参考音频；三步：load_character → set_reference_audio → tts

启动:
  python server_genie.py --port 8766                # 进程内直调（本机已装 genie-tts）
  python server_genie.py --port 8766 --engine http # 对接已运行的 Genie HTTP 服务

REST API（与前端配套）:
  GET  /api/ping                      探测 Genie（local=导入测试 / http=连通性）
  POST /api/genie/load                {character_name, onnx_dir, language}  加载角色
  POST /api/genie/ref                 {character_name, audio_path, audio_text, language}
  POST /api/subtitle/file             加载字幕 → 逐句列表
  POST /api/start                     批量配音 {session_id, overrides, params}
  POST /api/stop/{sid}
  GET  /api/progress/{sid}
  POST /api/regen/{sid}               单句重生成 {index, text?, speed?}
  GET  /api/clip/{sid}/{index}        播放/下载单句
  GET  /api/download/{sid}            下载整轨
  POST /api/merge/{sid}               重新合并
"""

from __future__ import annotations

import argparse
import threading
import time
import traceback
import uuid
import wave
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from gsv_dubbing.audio_utils import build_timeline, normalize_peak, save_wav
from gsv_dubbing.genie_client import (
    GenieError,
    make_client,
    normalize_language,
    parse_wav_bytes,
)
from gsv_dubbing.speaker_router import SpeakerRouter
from gsv_dubbing.session_state import SessionState
from gsv_dubbing.subtitle_parser import SubtitleBlock, filter_blocks, load_subtitles

ROOT = Path(__file__).parent.resolve()
WORKSPACE = ROOT / "workspace"
WORKSPACE.mkdir(exist_ok=True)

app = FastAPI(title="GSV-Genie-Dubbing (Genie backend)")

# ---------------------------------------------------------------- 全局

ENGINE = {"mode": "local", "base_url": "http://127.0.0.1:8000"}
_client = None
_client_lock = threading.Lock()


def client():
    global _client
    with _client_lock:
        if _client is None:
            _client = make_client(ENGINE["mode"], ENGINE["base_url"])
        return _client


# ---------------------------------------------------------------- 数据模型

class GenieLoadReq(BaseModel):
    character_name: str
    onnx_dir: str
    language: str = "zh"


class GenieRefReq(BaseModel):
    character_name: str
    audio_path: str
    audio_text: str
    language: str = "zh"


class TTSParams(BaseModel):
    # Genie 核心
    character_name: str = "default"
    onnx_dir: str = ""                 # ONNX 模型目录
    language: str = "zh"               # zh/ja/en（Genie 映射为 Chinese/Japanese/English）
    ref_audio: str = ""                # 参考音频路径（本机可访问）
    ref_text: str = ""                 # 参考音频文本
    # 合成控制
    speed: float = 1.0
    max_speed: float = 1.4
    fit_timeline: bool = True
    retry: int = 2
    # 字幕清洗
    strip_brackets: bool = True
    only_cjk: bool = False
    # 导出
    out_sr: int = 32000
    normalize: bool = True
    # 多说话人（Genie 版：按说话人切 character/参考音频）
    speaker_profile: str = ""


class SubtitleFileReq(BaseModel):
    path: str
    params: TTSParams


class StartReq(BaseModel):
    session_id: str
    overrides: Dict[int, dict] = {}
    params: TTSParams


class RegenReq(BaseModel):
    index: int
    text: Optional[str] = None
    speed: Optional[float] = None
    params: TTSParams


# ---------------------------------------------------------------- 会话

class Session:
    def __init__(self, sid: str, subtitle_path: Path, params: TTSParams):
        self.id = sid
        self.subtitle_path = subtitle_path
        self.params = params
        self.blocks: List[SubtitleBlock] = []
        self.out_dir = WORKSPACE / sid / "output"
        self.clips_dir = WORKSPACE / sid / "clips"
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.state = SessionState(WORKSPACE / sid / "session")
        self.lock = threading.RLock()
        self.total = 0
        self.done = 0
        self.running = False
        self.error: Optional[str] = None
        self.finished_at: Optional[float] = None
        self.merged_wav: Optional[Path] = None
        self.stop_flag = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.edits: Dict[int, dict] = {}

    def load_blocks(self):
        raw = load_subtitles(self.subtitle_path)
        self.blocks = filter_blocks(
            raw, strip_brackets=self.params.strip_brackets, only_cjk=self.params.only_cjk
        )
        self.state.init_lines(
            self.blocks,
            meta={"subtitle": str(self.subtitle_path), "engine": ENGINE["mode"]},
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


def _ts(sec: float) -> str:
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------- Genie 角色准备

def ensure_character_ready(params: TTSParams, speaker_cfg=None):
    """加载角色 + 设置参考音频。speaker_cfg（多说话人）可覆盖 character/ref。"""
    name = getattr(speaker_cfg, "genie_character", "") or params.character_name or "default"
    onnx_dir = getattr(speaker_cfg, "genie_onnx_dir", "") or params.onnx_dir
    ref_audio = getattr(speaker_cfg, "genie_ref_audio", "") or params.ref_audio
    ref_text = getattr(speaker_cfg, "genie_ref_text", "") or params.ref_text
    lang = getattr(speaker_cfg, "prompt_lang", "") or params.language

    c = client()
    if not onnx_dir:
        raise GenieError("未设置 ONNX 模型目录（角色 = ONNX 目录 + 参考音频）")
    if not ref_audio or not ref_text:
        raise GenieError("未设置参考音频或参考文本（Genie 音色克隆必需）")

    c.load_character(name, onnx_dir, normalize_language(lang))
    c.set_reference_audio(name, ref_audio, ref_text, lang)
    return name


def gen_voice(client_obj, name: str, text: str, retry: int):
    """调用一次 Genie TTS，返回 (audio, sr)。"""
    last = None
    for i in range(max(1, retry)):
        try:
            return client_obj.tts(name, text, split_sentence=False)
        except GenieError as e:
            last = e
            time.sleep(1.0)
    raise GenieError(f"Genie 合成失败（重试 {retry} 次）: {last}")


def fit_speed(audio: np.ndarray, sr: int, window: Optional[float],
              speed: float, max_speed: float, client_obj, name: str, text: str, retry: int):
    """Genie 无原生语速参数 —— 超窗时用重采样时长压缩模拟提速。"""
    dur = audio.size / sr
    if window and dur > window + 0.05:
        need = dur / max(window, 0.2)
        if need <= max_speed:
            # 线性插值压缩时长（等价于语速 need 倍）
            import numpy as _np
            n_out = max(int(audio.size / need), 1)
            t_in = _np.arange(audio.size, dtype=_np.float64) / sr
            t_out = _np.arange(n_out, dtype=_np.float64) * (t_in[-1] / max(n_out - 1, 1))
            audio = _np.interp(t_out, t_in, audio.astype(_np.float64)).astype(_np.float32)
    return audio


# ---------------------------------------------------------------- API

@app.get("/api/ping")
def api_ping():
    try:
        ok = client().ping()
    except Exception:
        ok = False
    return {"ok": ok, "engine": ENGINE["mode"]}


@app.post("/api/genie/load")
def api_genie_load(req: GenieLoadReq):
    try:
        client().load_character(req.character_name, req.onnx_dir, req.language)
        return {"ok": True}
    except GenieError as e:
        raise HTTPException(400, str(e))


@app.post("/api/genie/ref")
def api_genie_ref(req: GenieRefReq):
    try:
        client().set_reference_audio(req.character_name, req.audio_path,
                                     req.audio_text, req.language)
        return {"ok": True}
    except GenieError as e:
        raise HTTPException(400, str(e))


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
        for idx_str, ov in req.overrides.items():
            idx = int(idx_str)
            b = s.block_by_index(idx)
            if b and ov.get("text") is not None:
                b.text = str(ov["text"])
                st = s.state.lines.get(idx)
                if st and st.text != b.text:
                    st.status = "pending"
                    st.file = None
                    s.state.save(force=True)
            if ov.get("skip"):
                s.edits[idx] = {**(s.edits.get(idx) or {}), "skip": True}
        s.stop_flag.clear()
        s.error = None
        s.running = True
        s.params = req.params

        def worker():
            try:
                todo = [b for b in s.blocks
                        if not s.state.is_done(b.index)
                        and not s.edits.get(b.index, {}).get("skip")]
                total = len(s.blocks)
                done = total - len(todo)
                s.total, s.done = total, done
                router = (
                    SpeakerRouter.from_file(req.params.speaker_profile)
                    if req.params.speaker_profile else None
                )
                c = client()
                ensure_character_ready(req.params)
                for b in todo:
                    if s.stop_flag.is_set():
                        break
                    spk = router.route(b) if router else None
                    # 多说话人：切换角色/参考音频后合成
                    if router and spk and (spk.genie_character or spk.genie_ref_audio):
                        name = ensure_character_ready(req.params, spk)
                    else:
                        name = req.params.character_name or "default"
                    text = (
                        router.strip_speaker_prefix(b.text, spk) if router
                        else SpeakerRouter.PREFIX_RE.sub("", b.text, count=1).strip()
                    )
                    audio, sr = gen_voice(c, name, text, req.params.retry)
                    audio = fit_speed(audio, sr, b.duration if req.params.fit_timeline else None,
                                      req.params.speed, req.params.max_speed, c, name, text,
                                      req.params.retry)
                    save_wav(s.clip_path(b.index), audio, sr)
                    s.state.mark(b.index, "done", file=f"{b.index:05d}.wav", speed=1.0)
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


@app.post("/api/stop/{sid}")
def api_stop(sid: str):
    s = get_session(sid)
    s.stop_flag.set()
    return {"ok": True}


@app.get("/api/progress/{sid}")
def api_progress(sid: str):
    s = get_session(sid)
    return {
        "running": s.running,
        "done": s.done,
        "total": s.total,
        "error": s.error,
        "finished": s.finished_at is not None and not s.running,
        "merged_available": s.merged_wav.exists() if s.merged_wav else False,
        "lines": [
            {"index": st.index, "status": st.status, "speed": st.speed}
            for st in s.state.lines.values() if st.status in ("done", "failed")
        ],
    }


@app.post("/api/regen/{sid}")
def api_regen(sid: str, req: RegenReq):
    s = get_session(sid)
    if s.running:
        raise HTTPException(409, "批量合成进行中，请先停止")
    b = s.block_by_index(req.index)
    if not b:
        raise HTTPException(404, f"句子 {req.index} 不存在")
    if req.text is not None and req.text.strip():
        b.text = req.text.strip()
    try:
        c = client()
        params = req.params
        router = SpeakerRouter.from_file(params.speaker_profile) if params.speaker_profile else None
        spk = router.route(b) if router else None
        if router and spk and (spk.genie_character or spk.genie_ref_audio):
            name = ensure_character_ready(params, spk)
        else:
            name = ensure_character_ready(params)
        text = (
            router.strip_speaker_prefix(b.text, spk) if router
            else SpeakerRouter.PREFIX_RE.sub("", b.text, count=1).strip()
        )
        audio, sr = gen_voice(c, name, text, params.retry)
        audio = fit_speed(audio, sr, b.duration if params.fit_timeline else None,
                          params.speed, params.max_speed, c, name, text, params.retry)
        save_wav(s.clip_path(req.index), audio, sr)
        s.state.mark(req.index, "done", file=f"{req.index:05d}.wav", speed=1.0)
        s.state.save()
        s.merged_wav = None
        return {"ok": True, "index": req.index, "duration": round(audio.size / sr, 2)}
    except GenieError as e:
        raise HTTPException(502, str(e))
    except Exception as e:
        raise HTTPException(500, f"{e}\n{traceback.format_exc(limit=2)}")


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


# ---------------------------------------------------------------- 静态页（Genie 版参数）

@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index_genie.html", media_type="text/html")


@app.get("/app_genie.js")
def app_genie_js():
    return FileResponse(ROOT / "web" / "app_genie.js", media_type="application/javascript")


@app.get("/style.css")
def style_css():
    return FileResponse(ROOT / "web" / "style.css", media_type="text/css")


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser(description="GSV-Genie-Dubbing server (Genie backend)")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--engine", choices=["local", "http"], default="local",
                    help="local=进程内 genie_tts；http=对接已运行的 Genie HTTP 服务")
    ap.add_argument("--genie-url", default="http://127.0.0.1:8000",
                    help="engine=http 时的 Genie 服务地址")
    args = ap.parse_args()
    ENGINE["mode"] = args.engine
    ENGINE["base_url"] = args.genie_url
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
