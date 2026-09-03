"""端到端 API 测试脚本：对运行中的 server.py 跑完整链路。"""

import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8765"
PARAMS = {
    "api_url": "http://127.0.0.1:9880",
    "ref_audio": "D:/mock_ref.wav",
    "prompt_text": "参考文本",
}


def call(path, data=None):
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
        return e.code, json.loads(body) if body[:1] == b"{" else body


def main():
    # 1. 加载字幕
    st, r = call("/api/subtitle/file", {"path": "C:/Users/yukik/Desktop/nanoda.srt", "params": PARAMS})
    assert st == 200, (st, r)
    sid = r["session_id"]
    print(f"[1] 加载字幕: session={sid}, {r['total']} 句")

    # 2. 单句重生成（编辑文本）
    st, r = call(f"/api/regen/{sid}", {"index": 2, "text": "赤城起火了！！（编辑测试）", "params": PARAMS})
    assert st == 200, (st, r)
    print(f"[2] 单句重生成: index=2, duration={r['duration']}s, speed={r['speed']}")

    # 3. 下载该句音频
    req = urllib.request.Request(f"{BASE}/api/clip/{sid}/2")
    with urllib.request.urlopen(req, timeout=30) as resp:
        audio = resp.read()
        assert resp.status == 200 and audio[:4] == b"RIFF", "非 WAV"
    print(f"[3] 单句音频下载: {len(audio)}B WAV OK")

    # 4. 覆盖式批量：只跑前 3 句（用 overrides skip 掉其余，只测管线）
    overrides = {}
    lines_keep = {2, 4, 6}
    first_lines = None
    st, r = call(f"/api/progress/{sid}")
    # 拿前几句 index
    st2, r2 = call("/api/subtitle/file", {"path": "C:/Users/yukik/Desktop/nanoda.srt",
                                          "params": PARAMS})
    # 会重新开 session…… 所以直接在原 session 上用 start，但 1274 句太多。
    # 改用小字幕做批量测试
    small_srt = """1
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
"""
    import tempfile, pathlib
    f = pathlib.Path(tempfile.gettempdir()) / "e2e_small.srt"
    f.write_text(small_srt, encoding="utf-8")
    st, r = call("/api/subtitle/file", {"path": str(f), "params": PARAMS})
    assert st == 200, (st, r)
    sid2 = r["session_id"]
    idxs = [l["index"] for l in r["lines"]]
    print(f"[4] 小字幕 session={sid2}: {r['total']} 句 {idxs} (音乐/旁白应已过滤)")

    # 5. 批量合成
    st, r = call("/api/start", {"session_id": sid2, "overrides": {}, "params": PARAMS})
    assert st == 200, (st, r)
    print(f"[5] 批量启动: total={r['total']}, 本次需生成={r['resume']}")
    t0 = time.time()
    while time.time() - t0 < 60:
        st, pr = call(f"/api/progress/{sid2}")
        if not pr["running"]:
            break
        time.sleep(0.5)
    assert pr["done"] == pr["total"], pr
    print(f"[6] 批量完成: {pr['done']}/{pr['total']}, merged={pr['merged_available']}")

    # 6. 下载整轨
    req = urllib.request.Request(f"{BASE}/api/download/{sid2}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        wav = resp.read()
    import io, wave
    with wave.open(io.BytesIO(wav)) as w:
        dur = w.getnframes() / w.getframerate()
    print(f"[7] 整轨下载: {len(wav)}B, 时长 {dur:.1f}s (应 ≈15s)")

    # 7. 断点续跑：再次 start 应全部复用
    st, r = call("/api/start", {"session_id": sid2, "overrides": {}, "params": PARAMS})
    print(f"[8] 断点续跑: 本次需生成 {r['resume']} 句 (应为 0)")
    assert r["resume"] == 0
    while True:
        st, pr = call(f"/api/progress/{sid2}")
        if not pr["running"]:
            break
        time.sleep(0.3)

    print("\n全部端到端测试通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
