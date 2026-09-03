"""模拟 GPT-SoVITS api_v2 的最小服务器，用于本地全链路测试。

运行: python tests/mock_gsv_api.py --port 9880
"""

import argparse
import io
import sys
import wave
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
import uvicorn

app = FastAPI()


def make_wav(text: str, speed: float = 1.0) -> bytes:
    """生成一段与文本长度成比例的正弦+噪声，模拟语音时长。"""
    dur = max(len(text) * 0.12 / max(speed, 0.1), 0.2)
    sr = 32000
    t = np.arange(int(dur * sr)) / sr
    f0 = 200 + (hash(text) % 100)
    audio = 0.3 * np.sin(2 * np.pi * f0 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((audio * 32767).astype("<i2").tobytes())
    return buf.getvalue()


@app.get("/tts")
def tts(text: str = "", text_lang: str = "zh", ref_audio_path: str = "",
        prompt_text: str = "", prompt_lang: str = "zh",
        speed_factor: float = 1.0, text_split_method: str = "cut0",
        seed: int = -1, media_type: str = "wav",
        top_k: int = 5, top_p: float = 1.0, temperature: float = 1.0,
        batch_size: int = 1, fragment_interval: float = 0.3,
        sample_steps: int = 32, repetition_penalty: float = 1.35,
        parallel_infer: bool = True, streaming_mode: bool = False):
    if not ref_audio_path:
        return JSONResponse(status_code=400, content={"message": "ref_audio_path is required"})
    if not text:
        return JSONResponse(status_code=400, content={"message": "text is required"})
    return Response(make_wav(text, speed_factor), media_type="audio/wav")


@app.get("/set_gpt_weights")
def set_gpt(weights_path: str = ""):
    return {"message": "success"}


@app.get("/set_sovits_weights")
def set_sovits(weights_path: str = ""):
    return {"message": "success"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9880)
    args = ap.parse_args()
    print(f"mock GPT-SoVITS API on :{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
