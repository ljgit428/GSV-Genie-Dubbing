"""模拟 Genie (genie-tts) HTTP 服务，用于无模型环境下全链路测试。

复刻 genie_tts.Server 的接口契约:
  POST /load_character  {character_name, onnx_model_dir, language}
  POST /set_reference_audio {character_name, audio_path, audio_text, language}
  POST /tts {character_name, text, split_sentence, save_path} → 流式 WAV

运行: python tests/mock_genie_api.py --port 8000
"""

import argparse
import io
import json
import wave
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI()

_loaded = set()
_refs = {}


class CharacterPayload(BaseModel):
    character_name: str
    onnx_model_dir: str
    language: str


class ReferenceAudioPayload(BaseModel):
    character_name: str
    audio_path: str
    audio_text: str
    language: str


class TTSPayload(BaseModel):
    character_name: str
    text: str
    split_sentence: bool = False
    save_path: str | None = None


@app.post("/load_character")
def load_character(p: CharacterPayload):
    d = Path(p.onnx_model_dir)
    if not d.exists():
        raise HTTPException(500, f"The model directory '{d}' does not exist. (mock)")
    _loaded.add(p.character_name)
    return {"status": "success", "message": f"Character '{p.character_name}' loaded."}


@app.post("/set_reference_audio")
def set_ref(p: ReferenceAudioPayload):
    if not Path(p.audio_path).exists():
        raise HTTPException(500, f"Reference audio not found: {p.audio_path} (mock)")
    _refs[p.character_name] = {"text": p.audio_text, "lang": p.language}
    return {"status": "success"}


@app.post("/tts")
def tts(p: TTSPayload):
    if p.character_name not in _refs:
        raise HTTPException(404, "Character not found or reference audio not set.")
    # 生成与文本长度成比例的"语音"
    dur = max(len(p.text) * 0.1, 0.2)
    sr = 32000
    t = np.arange(int(dur * sr)) / sr
    f0 = 180 + (hash(p.text) % 60)
    audio = 0.4 * np.sin(2 * np.pi * f0 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((audio * 32767).astype("<i2").tobytes())
    data = buf.getvalue()

    async def gen():
        # 分块吐出，模拟流式
        for i in range(0, len(data), 8192):
            yield data[i: i + 8192]

    return StreamingResponse(gen(), media_type="audio/wav")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    print(f"mock Genie HTTP API on :{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
