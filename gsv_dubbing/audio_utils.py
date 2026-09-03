"""音频工具：重采样、时间轴合成导出。"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """线性插值重采样（够用且无依赖；GPT-SoVITS 输出一般已是 32k）。"""
    if sr_in == sr_out or audio.size == 0:
        return audio
    n_out = int(round(audio.size * sr_out / sr_in))
    t_in = np.arange(audio.size, dtype=np.float64) / sr_in
    t_out = np.arange(n_out, dtype=np.float64) / sr_out
    return np.interp(t_out, t_in, audio.astype(np.float64)).astype(np.float32)


def normalize_peak(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(audio))) if audio.size else 0.0
    if m > 0:
        return (audio / m * peak).astype(np.float32)
    return audio


def build_timeline(
    clips: List[Tuple[float, float, np.ndarray, int]],
    total_duration: Optional[float] = None,
    sr: int = 32000,
) -> np.ndarray:
    """按 (start, end, audio, clip_sr) 列表铺设整条时间轴，返回 sr 采样率的音轨。

    - clip 音频重采样到 sr 后放在 start 位置
    - 与已铺设内容重叠时做 20ms 交叉淡化，避免叠加爆音
    - 超出总长度的部分丢弃
    """
    if total_duration is None:
        total_duration = max((e for _, e, _, _ in clips), default=0.0)
    total_n = int(np.ceil(total_duration * sr))
    if total_n <= 0:
        return np.zeros(0, dtype=np.float32)
    canvas = np.zeros(total_n, dtype=np.float32)

    # 按 start 排序，逐个铺设
    for start, _end, audio, clip_sr in sorted(clips, key=lambda c: c[0]):
        if audio.size == 0:
            continue
        a = resample(audio, clip_sr, sr)
        so = int(start * sr)
        if so >= total_n:
            continue
        a = a[: total_n - so]  # 超出总长截断
        n = a.size
        if n == 0:
            continue
        seg = canvas[so : so + n]
        occupied = np.abs(seg) > 1e-9
        if occupied.any():
            # 与已有内容重叠：整个重叠区做交叉淡化（旧淡出、新淡入），再相加
            ov_start = int(np.argmax(occupied))
            ov_end = n - 1 - int(np.argmax(occupied[::-1]))
            ov_len = max(ov_end - ov_start + 1, 1)
            w = np.ones(n, dtype=np.float32)
            # 新音频在重叠区从 0 → 1 渐进
            w[ov_start : ov_end + 1] = np.linspace(0, 1, ov_len, dtype=np.float32)
            g_old = np.ones(n, dtype=np.float32)
            g_old[ov_start : ov_end + 1] = 1 - w[ov_start : ov_end + 1]
            canvas[so : so + n] = seg * g_old + a * w
        else:
            canvas[so : so + n] = a
    return canvas[:total_n]


def save_wav(path: str | Path, audio: np.ndarray, sr: int) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    data = (pcm * 32767).astype("<i2")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())
