#!/usr/bin/env python3
"""GSV-Genie-Dubbing 命令行界面。

示例：
  # 最简：全靠默认（API 已启动且已带默认参考音频时）
  python dubbing_cli.py sub.srt -o out/

  # 指定参考音频与权重
  python dubbing_cli.py sub.srt -o out/ \
      --api http://127.0.0.1:9880 \
      --ref "D:/参考音频/雪风.wav" --prompt "雪风的台词" \
      --gpt "GPT_weights_v2ProPlus/Yukikaze-e15.ckpt" \
      --sovits "SoVITS_weights_v2ProPlus/Yukikaze_e8_s704.pth"

  # 多说话人（profile.json 里配置每个角色的参考音频）
  python dubbing_cli.py sub.ass -o out/ --profile profile.json
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gsv_dubbing import DubConfig, DubbingEngine
from gsv_dubbing.gsv_client import GSVClient


def fmt_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="输入字幕文件，调用 GPT-SoVITS API 逐句配音并导出整轨 WAV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("subtitle", help="字幕文件 (.srt/.ass/.vtt)")
    p.add_argument("-o", "--output", default="output", help="输出目录")
    p.add_argument("--api", default="http://127.0.0.1:9880", help="GPT-SoVITS api_v2 地址")
    p.add_argument("--ref", default="", help="参考音频路径（服务器可访问的路径）")
    p.add_argument("--prompt", default="", help="参考音频的文本")
    p.add_argument("--text-lang", default="zh")
    p.add_argument("--prompt-lang", default="zh")
    p.add_argument("--gpt", default="", help="GPT 权重路径（服务器可访问）")
    p.add_argument("--sovits", default="", help="SoVITS 权重路径（服务器可访问）")
    p.add_argument("--profile", default="", help="多说话人配置 JSON（见 profile.example.json）")
    p.add_argument("--speed", type=float, default=1.0, help="基础语速")
    p.add_argument("--max-speed", type=float, default=1.4, help="为对齐时间轴自动提速的上限")
    p.add_argument("--no-fit", action="store_true", help="不做时间轴语速自适应")
    p.add_argument("--no-brackets", action="store_true", help="保留【旁白】等方括号内容")
    p.add_argument("--only-cjk", action="store_true", help="只配中文行")
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--retry", type=int, default=3)
    p.add_argument("--sr", type=int, default=32000, help="导出采样率")
    p.add_argument("--dry-run", action="store_true", help="只解析字幕并打印计划，不合成")
    p.add_argument("--list", action="store_true", help="解析后列出全部句子")
    return p


def main(argv=None) -> int:
    args = make_parser().parse_args(argv)

    if not Path(args.subtitle).exists():
        print(f"[错误] 字幕文件不存在: {args.subtitle}")
        return 1

    cfg = DubConfig(
        api_url=args.api,
        ref_audio=args.ref,
        prompt_text=args.prompt,
        text_lang=args.text_lang,
        prompt_lang=args.prompt_lang,
        gpt_weights=args.gpt,
        sovits_weights=args.sovits,
        speaker_profile=args.profile,
        speed=args.speed,
        max_speed=args.max_speed,
        fit_timeline=not args.no_fit,
        strip_brackets=not args.no_brackets,
        only_cjk=args.only_cjk,
        seed=args.seed,
        retry=args.retry,
        out_sr=args.sr,
    )

    # 解析预览
    from gsv_dubbing import filter_blocks, load_subtitles

    blocks = filter_blocks(
        load_subtitles(args.subtitle),
        strip_brackets=cfg.strip_brackets,
        only_cjk=cfg.only_cjk,
    )
    print(f"[字幕] {args.subtitle}: 共 {len(blocks)} 句可朗读")
    if args.list:
        for b in blocks:
            spk = b.speaker or "-"
            print(f"  #{b.index:>4} [{fmt_time(b.start)} → {fmt_time(b.end)}] ({spk}) {b.text}")
        return 0
    if args.dry_run:
        total = sum(b.duration for b in blocks)
        print(f"[计划] 预计配音 {len(blocks)} 句，字幕总时长 {fmt_time(total)}")
        return 0

    # API 探测
    client = GSVClient(args.api)
    if not client.ping():
        print(f"[错误] GPT-SoVITS API 不可达: {args.api}")
        print("       请先在 GPT-SoVITS 目录运行: python api_v2.py -a 127.0.0.1 -p 9880")
        return 1
    print(f"[API] {args.api} 在线")

    lock = threading.Lock()
    t0 = time.time()

    def on_progress(done, total, block, msg):
        with lock:
            if block is None:
                print(f"[进度] {msg}")
                return
            frac = done / max(total, 1)
            bar = "█" * int(frac * 30) + "·" * (30 - int(frac * 30))
            eta = (time.time() - t0) / max(done, 1) * (total - done)
            print(
                f"\r[{bar}] {done}/{total} ({frac:.0%}) ETA {fmt_time(eta)} | "
                f"#{block.index} {msg[:40]:<40}",
                end="",
            )
            if done == total:
                print()

    cfg.on_progress = on_progress

    engine = DubbingEngine(args.subtitle, args.output, cfg)
    try:
        result = engine.run()
    except KeyboardInterrupt:
        print("\n[中断] 已停止，进度已保存，重跑同一命令即可续接")
        return 130
    except InterruptedError:
        print("\n[中断] 已停止，进度已保存，重跑同一命令即可续接")
        return 130
    except Exception as e:
        print(f"\n[失败] {e}")
        return 1

    print(f"\n[完成] 导出: {result.out_wav}")
    print(f"       句子: {result.voiced_lines}/{result.total_lines} 成功"
          + (f"，{result.failed_lines} 句失败" if result.failed_lines else ""))
    print(f"       单句缓存: {result.clips_dir}（可单独试听/替换后重新导出）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
