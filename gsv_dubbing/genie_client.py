"""Genie (genie-tts) 客户端：GPT-SoVITS 的轻量级 ONNX 推理引擎。

两种后端，同一接口：
- GenieHTTPClient: 对接 `genie_tts.start_server()` 起的 HTTP 服务
  （POST /load_character → /set_reference_audio → /tts 返回流式 WAV）
- GenieLocalClient: 进程内直调 `import genie_tts`（免部署，server.py 直接用）

流程（两种后端一致）：
  load_character(name, onnx_dir, language)   # 加载 ONNX 角色
  set_reference_audio(name, wav, text, lang) # 设置参考音频
  tts(name, text) -> (np.ndarray, sr)        # 合成
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class GenieError(RuntimeError):
    pass


# Genie 支持的语言（normalize_language 后的值）
LANG_MAP = {
    "zh": "Chinese", "chinese": "Chinese", "中文": "Chinese",
    "ja": "Japanese", "jp": "Japanese", "japanese": "Japanese", "日语": "Japanese",
    "en": "English", "english": "English", "英语": "English",
}


def normalize_language(lang: str) -> str:
    """zh/ja/en → Chinese/Japanese/English；已是标准值直接过。"""
    key = (lang or "").strip().lower()
    if key in LANG_MAP:
        return LANG_MAP[key]
    raise GenieError(f"Genie 只支持中文/日语/英语，无法识别语言: {lang!r}")


def parse_wav_bytes(data: bytes) -> Tuple[np.ndarray, int]:
    """解析 WAV 字节为 (float32 PCM[-1,1], sample_rate)。"""
    import io
    import wave

    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            nch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(n)
        if sw == 2:
            audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif sw == 4:
            audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        elif sw == 1:
            audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
        else:
            raise GenieError(f"不支持的位宽: {sw}")
        if nch > 1:
            audio = audio.reshape(-1, nch).mean(axis=1)
        return audio.copy(), sr
    except GenieError:
        raise
    except Exception as e:
        raise GenieError(f"WAV 解析失败: {e}") from e


# ================================================================ HTTP 后端

class GenieHTTPClient:
    """对接 `python -m genie_tts` / `genie_tts.start_server()` 起的服务。"""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 300):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self._lock = threading.RLock()          # 串行化合成请求
        self._loaded: set = set()               # 已 load 的角色

    # ------------------------------------------------ 基础请求
    def _post_json(self, path: str, payload: dict, timeout: Optional[float] = None) -> dict:
        req = urllib.request.Request(
            f"{self.base}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read().decode("utf-8")).get("detail", e.code)
            except Exception:
                detail = e.code
            raise GenieError(f"{path} 失败 (HTTP {e.code}): {detail}") from e
        except urllib.error.URLError as e:
            raise GenieError(f"无法连接 Genie 服务 ({self.base}): {e.reason}") from e

    # ------------------------------------------------ 接口
    def ping(self) -> bool:
        """探测服务（GET / 是 FastAPI 默认 404，能响应即在线）。"""
        try:
            urllib.request.urlopen(f"{self.base}/openapi.json", timeout=5)
            return True
        except urllib.error.HTTPError:
            return True
        except Exception:
            return False

    def load_character(self, name: str, onnx_model_dir: str, language: str) -> None:
        lang = normalize_language(language)
        with self._lock:
            self._post_json("/load_character", {
                "character_name": name,
                "onnx_model_dir": str(onnx_model_dir),
                "language": lang,
            })
            self._loaded.add(name)

    def set_reference_audio(self, name: str, audio_path: str,
                            audio_text: str, language: Optional[str] = None) -> None:
        with self._lock:
            self._post_json("/set_reference_audio", {
                "character_name": name,
                "audio_path": str(audio_path),
                "audio_text": audio_text,
                "language": normalize_language(language) if language else "Chinese",
            })

    def tts(self, name: str, text: str,
            split_sentence: bool = False) -> Tuple[np.ndarray, int]:
        """同步合成：POST /tts 返回流式 WAV，读完整响应后解析。"""
        import io
        req = urllib.request.Request(
            f"{self.base}/tts",
            data=json.dumps({
                "character_name": name,
                "text": text,
                "split_sentence": split_sentence,
                "save_path": None,          # 直接收流，不落盘到服务器
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._lock:
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = resp.read()
            except urllib.error.HTTPError as e:
                try:
                    detail = json.loads(e.read().decode("utf-8")).get("detail", e.code)
                except Exception:
                    detail = e.code
                raise GenieError(f"Genie /tts 失败 (HTTP {e.code}): {detail}") from e
            except urllib.error.URLError as e:
                raise GenieError(f"无法连接 Genie 服务: {e.reason}") from e
        if len(data) < 44 or data[:4] != b"RIFF":
            raise GenieError(f"Genie 返回的不是有效 WAV ({len(data)}B)")
        return parse_wav_bytes(data)


# ================================================================ 进程内后端

class GenieLocalClient:
    """进程内直调 genie_tts（本机安装了 genie-tts 时可用，免起服务）。"""

    _import_lock = threading.RLock()

    def __init__(self):
        self._genie = None
        self._loaded: set = set()
        self._lock = threading.RLock()

    def _mod(self):
        with GenieLocalClient._import_lock:
            if self._genie is None:
                try:
                    import genie_tts
                except Exception as e:
                    raise GenieError(
                        f"导入 genie_tts 失败: {e}。请 pip install genie-tts，"
                        "或改用 HTTP 模式对接已运行的 Genie 服务。"
                    ) from e
                self._genie = genie_tts
        return self._genie

    def ping(self) -> bool:
        try:
            self._mod()
            return True
        except GenieError:
            return False

    def load_character(self, name: str, onnx_model_dir: str, language: str) -> None:
        lang = normalize_language(language)
        g = self._mod()
        with self._lock:
            g.load_character(
                character_name=name,
                onnx_model_dir=str(onnx_model_dir),
                language=lang,
            )
            self._loaded.add(name)

    def set_reference_audio(self, name: str, audio_path: str,
                            audio_text: str, language: Optional[str] = None) -> None:
        g = self._mod()
        with self._lock:
            g.set_reference_audio(
                character_name=name,
                audio_path=str(audio_path),
                audio_text=audio_text,
                language=normalize_language(language) if language else "Chinese",
            )

    def tts(self, name: str, text: str,
            split_sentence: bool = False) -> Tuple[np.ndarray, int]:
        """genie.tts() 写入临时 wav 再读回。"""
        import tempfile

        g = self._mod()
        with self._lock:
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            import os
            os.close(fd)
            try:
                g.tts(
                    character_name=name,
                    text=text,
                    play=False,
                    split_sentence=split_sentence,
                    save_path=tmp,
                )
                return parse_wav_bytes(Path(tmp).read_bytes())
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass


# ================================================================ 统一入口

def make_client(mode: str = "local", base_url: str = "http://127.0.0.1:8000",
                timeout: float = 300):
    """mode: 'local'（进程内）| 'http'（对接 Genie 服务）。"""
    if mode == "local":
        return GenieLocalClient()
    if mode == "http":
        return GenieHTTPClient(base_url, timeout)
    raise GenieError(f"未知后端模式: {mode}")
