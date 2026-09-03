"""Genie 后端端到端测试：对运行中的 server_genie.py (--engine http) 跑全链路。

前置: python server_genie.py --port 8766 --engine http
      python tests/mock_genie_api.py --port 8000
"""

import io
import json
import sys
import tempfile
import time
import urllib.request
import wave
from pathlib import Path

BASE = "http://127.0.0.1:8766"
PARAMS = {
    "character_name": "yukikaze",
    "onnx_dir": "D:/mock_onnx_dir",       # mock 检查目录存在即可，先建真目录
    "language": "zh",
    "ref_audio": "D:/mock_ref.wav",       # 同上
    "ref_text": "参考音频文本",
}


def call(path, data=None, method=None):
    if data is not None:
        req = urllib.request.Request(
            BASE + path, data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
    else:
        req = urllib.request.Request(BASE + path)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read()
            ct = r.headers.get("content-type", "")
            return r.status, (json.loads(body) if "json" in ct else body)
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, body


def main():
    # mock 需要 onnx_dir 与 ref_audio 真实存在 —— 用临时目录/文件
    tmp = Path(tempfile.mkdtemp(prefix="genie_mock_"))
    onnx_dir = tmp / "onnx"
    onnx_dir.mkdir()
    (onnx_dir / "t2s_encoder_fp32.onnx").write_bytes(b"mock")
    ref_wav = tmp / "ref.wav"
    import numpy as np
    with wave.open(str(ref_wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(32000)
        w.writeframes((np.zeros(3200) + 1000).astype("<i2").tobytes())
    PARAMS["onnx_dir"] = str(onnx_dir)
    PARAMS["ref_audio"] = str(ref_wav)

    # 1. ping（engine=http 探测 mock）
    st, r = call("/api/ping")
    assert st == 200 and r["ok"], (st, r)
    print(f"[1] Genie ping OK (engine={r['engine']})")

    # 2. 加载字幕
    small = tmp / "sub.srt"
    small.write_text("""1
00:00:01,000 --> 00:00:03,000
第一句测试。

2
00:00:04,000 --> 00:00:06,000
第二句测试来了。

3
00:00:07,000 --> 00:00:09,000
♪ 音乐行应被跳过 ♪

4
00:00:10,000 --> 00:00:12,000
[纯旁白]

5
00:00:13,000 --> 00:00:15,000
第三句结束。
""", encoding="utf-8")
    st, r = call("/api/subtitle/file", {"path": str(small), "params": PARAMS})
    assert st == 200, (st, r)
    sid = r["session_id"]
    idxs = [l["index"] for l in r["lines"]]
    assert r["total"] == 3, f"音乐/旁白应被过滤，实际 {r['total']} 句"
    print(f"[2] 字幕加载: session={sid}, {r['total']} 句 {idxs}")

    # 3. 单句重生成（带文本编辑）
    st, r = call(f"/api/regen/{sid}", {"index": idxs[0], "text": "第一句测试！（编辑后）", "params": PARAMS})
    assert st == 200, (st, r)
    print(f"[3] 单句重生成: index={idxs[0]}, duration={r['duration']}s")

    # 4. 单句下载
    with urllib.request.urlopen(f"{BASE}/api/clip/{sid}/{idxs[0]}", timeout=30) as resp:
        audio = resp.read()
    assert audio[:4] == b"RIFF", "非 WAV"
    print(f"[4] 单句下载: {len(audio)}B WAV OK")

    # 5. 批量合成
    st, r = call("/api/start", {"session_id": sid, "overrides": {}, "params": PARAMS})
    assert st == 200, (st, r)
    assert r["resume"] == 2, f"应有 2 句待合成（第 1 句已单句生成），实际 {r['resume']}"
    print(f"[5] 批量启动: total={r['total']}, 本次需生成={r['resume']}")
    t0 = time.time()
    while time.time() - t0 < 60:
        st, pr = call(f"/api/progress/{sid}")
        if not pr["running"]:
            break
        time.sleep(0.5)
    assert pr["done"] == pr["total"], pr
    print(f"[6] 批量完成: {pr['done']}/{pr['total']}, merged={pr['merged_available']}")

    # 6. 整轨下载
    with urllib.request.urlopen(f"{BASE}/api/download/{sid}", timeout=60) as resp:
        wav = resp.read()
    with wave.open(io.BytesIO(wav)) as w:
        dur = w.getnframes() / w.getframerate()
    assert 14.0 < dur < 16.5, f"整轨时长异常: {dur}"
    print(f"[7] 整轨下载: {len(wav)}B, 时长 {dur:.1f}s")

    # 7. 断点续跑
    st, r = call("/api/start", {"session_id": sid, "overrides": {}, "params": PARAMS})
    assert st == 200 and r["resume"] == 0, r
    print(f"[8] 断点续跑: 本次需生成 {r['resume']} 句 ✓")

    print("\nGenie 后端全部端到端测试通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
