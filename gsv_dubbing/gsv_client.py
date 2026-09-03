"""GPT-SoVITS api_v2 HTTP 客户端（线程安全）。

用法：
    client = GSVClient("http://127.0.0.1:9880")
    wav_bytes, sr = client.tts("你好世界", ref_audio=..., prompt_text=..., prompt_lang="zh")
"""

from __future__ import annotations

import io
import json
import re
import threading
import urllib.parse
import urllib.request
import wave
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class GSVError(RuntimeError):
    pass


class GSVClient:
    def __init__(self, base_url: str = "http://127.0.0.1:9880", timeout: float = 300):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self._lock = threading.Lock()   # 权重切换全局互斥
        self._params_lock = threading.Lock()
        self._current_gpt: Optional[str] = None
        self._current_sovits: Optional[str] = None

    # ------------------------------------------------------------ 基础请求
    def _get(self, path: str, params: dict, timeout: Optional[float] = None) -> Tuple[int, bytes]:
        qs = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{self.base}{path}"
        if qs:
            url += "?" + qs
        req = urllib.request.Request(url, headers={"User-Agent": "GSV-Genie-Dubbing/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except urllib.error.URLError as e:
            raise GSVError(f"无法连接 GPT-SoVITS API ({self.base}): {e.reason}") from e

    def _get_json(self, path: str, params: dict) -> dict:
        status, body = self._get(path, params, timeout=30)
        try:
            data = json.loads(body.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            raise GSVError(f"{path} 返回非 JSON: {body[:200]!r}")
        if status != 200:
            msg = data.get("message", data) if isinstance(data, dict) else data
            raise GSVError(f"{path} 失败 (HTTP {status}): {msg}")
        return data

    # ------------------------------------------------------------ 探测/控制
    def ping(self) -> bool:
        """探测服务是否可达（/tts 空 text 会返回 400，能返回说明活着）。"""
        try:
            self._get("/tts", {"text": "", "text_lang": "zh"}, timeout=5)
            return True
        except GSVError:
            return True   # 400 也是服务在线
        except Exception:
            return False

    def set_gpt_weights(self, path: str) -> None:
        with self._lock:
            self._get_json("/set_gpt_weights", {"weights_path": path})

    def set_sovits_weights(self, path: str) -> None:
        with self._lock:
            self._get_json("/set_sovits_weights", {"weights_path": path})

    def ensure_weights(self, gpt: Optional[str], sovits: Optional[str]) -> None:
        """只在权重与当前不符时切换，避免重复加载。"""
        with self._params_lock:
            if gpt and gpt != self._current_gpt:
                self.set_gpt_weights(gpt)
                self._current_gpt = gpt
            if sovits and sovits != self._current_sovits:
                self.set_sovits_weights(sovits)
                self._current_sovits = sovits

    # ------------------------------------------------------------ TTS
    def tts(
        self,
        text: str,
        text_lang: str = "zh",
        ref_audio_path: str = "",
        prompt_text: str = "",
        prompt_lang: str = "zh",
        aux_ref_audio_paths: Optional[list] = None,
        top_k: int = 5,
        top_p: float = 1.0,
        temperature: float = 1.0,
        text_split_method: str = "cut0",
        speed_factor: float = 1.0,
        fragment_interval: float = 0.3,
        seed: int = -1,
        batch_size: int = 1,
        sample_steps: int = 32,
        repetition_penalty: float = 1.35,
    ) -> Tuple[np.ndarray, int]:
        """合成一段文本，返回 (float32 PCM[-1,1], sample_rate)。"""
        params = {
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": ref_audio_path,
            "aux_ref_audio_paths": aux_ref_audio_paths,
            "prompt_text": prompt_text,
            "prompt_lang": prompt_lang,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "text_split_method": text_split_method,
            "batch_size": batch_size,
            "speed_factor": speed_factor,
            "fragment_interval": fragment_interval,
            "seed": seed,
            "media_type": "wav",
            "streaming_mode": False,
            "parallel_infer": True,
            "repetition_penalty": repetition_penalty,
            "sample_steps": sample_steps,
        }
        status, body = self._get("/tts", params)
        if status != 200:
            try:
                msg = json.loads(body).get("message", body)
            except Exception:
                msg = body[:300]
            raise GSVError(f"TTS 失败 (HTTP {status}): {msg}")
        return self._parse_wav(body)

    @staticmethod
    def _parse_wav(data: bytes) -> Tuple[np.ndarray, int]:
        try:
            with wave.open(io.BytesIO(data), "rb") as w:
                nch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
                raw = w.readframes(n)
            if sw == 2:
                audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            elif sw == 4:
                audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
            else:
                audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
            if nch > 1:
                audio = audio.reshape(-1, nch).mean(axis=1)
            return audio.copy(), sr
        except Exception as e:
            raise GSVError(f"返回的音频无法解析: {e}") from e
