#!/usr/bin/env python3
"""GSV-Genie-Dubbing 图形界面（Tkinter，零第三方依赖）。

双击运行或: python dubbing_gui.py
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
import traceback
import wave
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from gsv_dubbing import DubConfig, DubbingEngine
from gsv_dubbing.gsv_client import GSVClient
from gsv_dubbing.subtitle_parser import filter_blocks, load_subtitles

try:
    import numpy as np
except ImportError:  # numpy 应该有；兜底只影响播放按钮
    np = None


APP_TITLE = "GSV-Genie-Dubbing 字幕配音"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("880x720")
        self.minsize(760, 620)
        self._jobs: queue.Queue = queue.Queue()
        self.engine: Optional[DubbingEngine] = None
        self.worker: Optional[threading.Thread] = None
        self._build_ui()
        self._load_settings()
        self.after(100, self._poll_jobs)

    # ------------------------------------------------------------ UI
    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}
        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        # ---- API 连接区
        api = ttk.LabelFrame(frm, text="GPT-SoVITS API", padding=8)
        api.pack(fill="x", **pad)
        ttk.Label(api, text="地址:").grid(row=0, column=0, sticky="w")
        self.var_api = tk.StringVar(value="http://127.0.0.1:9880")
        ttk.Entry(api, textvariable=self.var_api, width=44).grid(row=0, column=1, sticky="we", padx=6)
        self.btn_ping = ttk.Button(api, text="测试连接", command=self.on_ping)
        self.btn_ping.grid(row=0, column=2, padx=4)
        self.lbl_ping = ttk.Label(api, text="未连接", foreground="#888")
        self.lbl_ping.grid(row=0, column=3, sticky="w", padx=6)
        api.columnconfigure(1, weight=1)

        ttk.Separator(frm).pack(fill="x", **pad)

        # ---- 输入区
        inp = ttk.LabelFrame(frm, text="输入", padding=8)
        inp.pack(fill="x", **pad)
        ttk.Label(inp, text="字幕文件:").grid(row=0, column=0, sticky="w")
        self.var_sub = tk.StringVar()
        ttk.Entry(inp, textvariable=self.var_sub).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(inp, text="浏览…", command=self.browse_sub).grid(row=0, column=2)
        ttk.Button(inp, text="预览句子", command=self.preview).grid(row=0, column=3, padx=4)

        ttk.Label(inp, text="输出目录:").grid(row=1, column=0, sticky="w")
        self.var_out = tk.StringVar(value="output")
        ttk.Entry(inp, textvariable=self.var_out).grid(row=1, column=1, sticky="we", padx=6)
        ttk.Button(inp, text="浏览…", command=self.browse_out).grid(row=1, column=2)
        inp.columnconfigure(1, weight=1)

        # ---- 音色区
        voice = ttk.LabelFrame(frm, text="音色（参考音频与权重 — 填 GPT-SoVITS 服务器本机可访问的路径）", padding=8)
        voice.pack(fill="x", **pad)
        ttk.Label(voice, text="参考音频:").grid(row=0, column=0, sticky="w")
        self.var_ref = tk.StringVar()
        ttk.Entry(voice, textvariable=self.var_ref).grid(row=0, column=1, columnspan=3, sticky="we", padx=6)
        ttk.Label(voice, text="参考文本:").grid(row=1, column=0, sticky="w")
        self.var_prompt = tk.StringVar()
        ttk.Entry(voice, textvariable=self.var_prompt).grid(row=1, column=1, columnspan=3, sticky="we", padx=6)
        ttk.Label(voice, text="GPT权重:").grid(row=2, column=0, sticky="w")
        self.var_gpt = tk.StringVar()
        ttk.Entry(voice, textvariable=self.var_gpt).grid(row=2, column=1, columnspan=3, sticky="we", padx=6)
        ttk.Label(voice, text="SoVITS权重:").grid(row=3, column=0, sticky="w")
        self.var_sovits = tk.StringVar()
        ttk.Entry(voice, textvariable=self.var_sovits).grid(row=3, column=1, columnspan=3, sticky="we", padx=6)
        ttk.Label(voice, text="多说话人:").grid(row=4, column=0, sticky="w")
        self.var_profile = tk.StringVar()
        ttk.Entry(voice, textvariable=self.var_profile).grid(row=4, column=1, columnspan=3, sticky="we", padx=6)
        ttk.Button(voice, text="编辑示例…", command=self.edit_profile).grid(row=4, column=4)
        for c in range(5):
            voice.columnconfigure(c, weight=0)
        voice.columnconfigure(4, weight=0)

        # ---- 参数区
        opt = ttk.LabelFrame(frm, text="参数", padding=8)
        opt.pack(fill="x", **pad)
        ttk.Label(opt, text="文本语言:").grid(row=0, column=0, sticky="w")
        self.var_lang = tk.StringVar(value="zh")
        ttk.Combobox(opt, textvariable=self.var_lang, values=["zh", "ja", "en", "ko", "yue"], width=6, state="readonly").grid(row=0, column=1, padx=6)
        ttk.Label(opt, text="基础语速:").grid(row=0, column=2, sticky="w")
        self.var_speed = tk.DoubleVar(value=1.0)
        ttk.Spinbox(opt, from_=0.5, to=2.0, increment=0.05, textvariable=self.var_speed, width=6).grid(row=0, column=3, padx=6)
        ttk.Label(opt, text="限速上限:").grid(row=0, column=4, sticky="w")
        self.var_maxspeed = tk.DoubleVar(value=1.4)
        ttk.Spinbox(opt, from_=1.0, to=2.5, increment=0.05, textvariable=self.var_maxspeed, width=6).grid(row=0, column=5, padx=6)
        ttk.Label(opt, text="重试:").grid(row=0, column=6, sticky="w")
        self.var_retry = tk.IntVar(value=3)
        ttk.Spinbox(opt, from_=1, to=10, textvariable=self.var_retry, width=4).grid(row=0, column=7, padx=6)
        self.var_fit = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="时间轴自适应（超窗自动提速）", variable=self.var_fit).grid(row=1, column=0, columnspan=3, sticky="w")
        self.var_brackets = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="清洗【旁白】/标签", variable=self.var_brackets).grid(row=1, column=3, columnspan=2, sticky="w")
        self.var_only_cjk = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt, text="仅配中文行", variable=self.var_only_cjk).grid(row=1, column=5, columnspan=3, sticky="w")

        # ---- 句子列表
        lst = ttk.LabelFrame(frm, text="句子（双击可预听该句缓存）", padding=4)
        lst.pack(fill="both", expand=True, **pad)
        cols = ("idx", "time", "status", "text")
        self.tree = ttk.Treeview(lst, columns=cols, show="headings", selectmode="browse")
        for c, w, a in (("idx", 50, "center"), ("time", 130, "center"), ("status", 70, "center"), ("text", 480, "w")):
            self.tree.heading(c, text={"idx": "#", "time": "时间", "status": "状态", "text": "文本"}[c])
            self.tree.column(c, width=w, anchor=a, stretch=(c == "text"))
        ysb = ttk.Scrollbar(lst, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=ysb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")
        self.tree.tag_configure("done", foreground="#0a7")
        self.tree.tag_configure("failed", foreground="#c33")
        self.tree.tag_configure("current", background="#eef4ff")
        self.tree.bind("<Double-1>", self.on_play_clip)

        # ---- 进度与按钮
        bot = ttk.Frame(frm)
        bot.pack(fill="x", **pad)
        self.progress = ttk.Progressbar(bot, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.btn_run = ttk.Button(bot, text="开始配音", command=self.on_run)
        self.btn_run.pack(side="left")
        self.btn_stop = ttk.Button(bot, text="停止", command=self.on_stop, state="disabled")
        self.btn_stop.pack(side="left", padx=6)
        self.btn_reexport = ttk.Button(bot, text="仅重新合并", command=self.on_reexport)
        self.btn_reexport.pack(side="left")

        self.log = tk.Text(frm, height=8, state="disabled", font=("Consolas", 9))
        self.log.pack(fill="x", **pad)

    # ------------------------------------------------------------ 工具
    def logline(self, msg: str):
        self._jobs.put(("log", msg))

    def _poll_jobs(self):
        try:
            while True:
                kind, payload = self._jobs.get_nowait()
                if kind == "log":
                    self.log.configure(state="normal")
                    self.log.insert("end", f"{time.strftime('%H:%M:%S')}  {payload}\n")
                    self.log.see("end")
                    self.log.configure(state="disabled")
                elif kind == "progress":
                    done, total, msg = payload
                    self.progress.configure(maximum=max(total, 1), value=done)
                    self.statusbar_set(f"{done}/{total}  {msg}")
                elif kind == "row":
                    i, status = payload
                    for item in self.tree.get_children():
                        if self.tree.set(item, "idx") == str(i):
                            self.tree.set(item, "status", status)
                            tags = list(self.tree.item(item, "tags") or [])
                            for t in ("done", "failed", "current"):
                                if t in tags:
                                    tags.remove(t)
                            tags.append(status if status in ("done", "failed") else "current")
                            self.tree.item(item, tags=tags)
                            break
                elif kind == "done":
                    result = payload
                    self.progress.configure(value=self.progress.cget("maximum"))
                    self.run_finished(result)
        except queue.Empty:
            pass
        self.after(120, self._poll_jobs)

    def statusbar_set(self, s: str):
        self.title(f"{APP_TITLE} — {s}" if s else APP_TITLE)

    # ------------------------------------------------------------ 设置持久化
    SETTINGS = Path.home() / ".gsv_genie_dubbing.json"

    def _save_settings(self):
        data = {
            "api": self.var_api.get(), "sub": self.var_sub.get(), "out": self.var_out.get(),
            "ref": self.var_ref.get(), "prompt": self.var_prompt.get(),
            "gpt": self.var_gpt.get(), "sovits": self.var_sovits.get(),
            "profile": self.var_profile.get(), "lang": self.var_lang.get(),
            "speed": self.var_speed.get(), "maxspeed": self.var_maxspeed.get(),
            "fit": self.var_fit.get(), "brackets": self.var_brackets.get(),
            "only_cjk": self.var_only_cjk.get(), "retry": self.var_retry.get(),
        }
        try:
            self.SETTINGS.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:
            pass

    def _load_settings(self):
        try:
            if self.SETTINGS.exists():
                d = json.loads(self.SETTINGS.read_text(encoding="utf-8"))
                self.var_api.set(d.get("api", self.var_api.get()))
                self.var_sub.set(d.get("sub", ""))
                self.var_out.set(d.get("out", "output"))
                self.var_ref.set(d.get("ref", ""))
                self.var_prompt.set(d.get("prompt", ""))
                self.var_gpt.set(d.get("gpt", ""))
                self.var_sovits.set(d.get("sovits", ""))
                self.var_profile.set(d.get("profile", ""))
                self.var_lang.set(d.get("lang", "zh"))
                self.var_speed.set(float(d.get("speed", 1.0)))
                self.var_maxspeed.set(float(d.get("maxspeed", 1.4)))
                self.var_fit.set(bool(d.get("fit", True)))
                self.var_brackets.set(bool(d.get("brackets", True)))
                self.var_only_cjk.set(bool(d.get("only_cjk", False)))
                self.var_retry.set(int(d.get("retry", 3)))
        except Exception:
            pass

    # ------------------------------------------------------------ 事件
    def browse_sub(self):
        f = filedialog.askopenfilename(filetypes=[
            ("字幕文件", "*.srt *.ass *.ssa *.vtt *.txt"), ("所有文件", "*.*")])
        if f:
            self.var_sub.set(f)
            self.preview()

    def browse_out(self):
        d = filedialog.askdirectory()
        if d:
            self.var_out.set(d)

    def edit_profile(self):
        f = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if f:
            self.var_profile.set(f)

    def on_ping(self):
        url = self.var_api.get().strip()
        self.lbl_ping.config(text="连接中…", foreground="#888")
        self.btn_ping.config(state="disabled")

        def work():
            ok = GSVClient(url, timeout=6).ping()
            self._jobs.put(("log", f"API 探测 {url}: {'在线' if ok else '不可达'}"))

            def finish():
                self.lbl_ping.config(
                    text="● 在线" if ok else "● 不可达",
                    foreground="#0a7" if ok else "#c33",
                )
                self.btn_ping.config(state="normal")
            self.after(0, finish)
        threading.Thread(target=work, daemon=True).start()

    def preview(self):
        path = self.var_sub.get().strip()
        if not path or not Path(path).exists():
            messagebox.showwarning(APP_TITLE, "请先选择字幕文件")
            return
        blocks = filter_blocks(
            load_subtitles(path),
            strip_brackets=self.var_brackets.get(),
            only_cjk=self.var_only_cjk.get(),
        )
        self.tree.delete(*self.tree.get_children())
        for b in blocks:
            def ts(x: float) -> str:
                h, rem = divmod(int(x), 3600)
                m, s = divmod(rem, 60)
                return f"{h:02d}:{m:02d}:{s:02d}"
            self.tree.insert("", "end", values=(
                b.index, f"{ts(b.start)} → {ts(b.end)}", "待合成", b.text))
        self.logline(f"解析 {Path(path).name}: {len(blocks)} 句可朗读，总时长 "
                     f"{sum(b.duration for b in blocks):.1f}s")

    def on_run(self):
        sub = self.var_sub.get().strip()
        if not sub or not Path(sub).exists():
            messagebox.showwarning(APP_TITLE, "请先选择有效的字幕文件")
            return
        ref = self.var_ref.get().strip()
        prof = self.var_profile.get().strip()
        if not ref and not prof:
            if not messagebox.askyesno(
                APP_TITLE,
                "未设置参考音频（也未用多说话人 profile）。\n"
                "GPT-SoVITS API 必须有 ref_audio_path 才能合成。\n仍要继续吗？",
            ):
                return

        cfg = DubConfig(
            api_url=self.var_api.get().strip(),
            ref_audio=ref,
            prompt_text=self.var_prompt.get().strip(),
            text_lang=self.var_lang.get(),
            gpt_weights=self.var_gpt.get().strip(),
            sovits_weights=self.var_sovits.get().strip(),
            speaker_profile=prof or None,
            speed=float(self.var_speed.get()),
            max_speed=float(self.var_maxspeed.get()),
            fit_timeline=self.var_fit.get(),
            strip_brackets=self.var_brackets.get(),
            only_cjk=self.var_only_cjk.get(),
            retry=int(self.var_retry.get()),
        )
        self._save_settings()
        self.preview()

        self.btn_run.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.logline("开始配音…（关闭窗口或点停止可中断，进度自动保存）")

        def on_progress(done, total, block, msg):
            self._jobs.put(("progress", (done, total, msg)))
            if block is not None:
                self._jobs.put(("row", (block.index, "done" if msg == "OK" else "failed")))

        cfg.on_progress = on_progress

        def work():
            try:
                engine = DubbingEngine(sub, self.var_out.get().strip(), cfg)
                self.engine = engine
                result = engine.run()
                self._jobs.put(("done", result))
            except InterruptedError:
                self._jobs.put(("log", "已停止。再次点击「开始配音」可续跑（已完成句子自动跳过）"))
                self._jobs.put(("done", None))
            except Exception as e:
                self._jobs.put(("log", f"失败: {e}\n{traceback.format_exc(limit=2)}"))
                self._jobs.put(("done", None))
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def run_finished(self, result):
        self.btn_run.config(state="normal")
        self.btn_stop.config(state="disabled")
        if result:
            self.logline(f"完成！导出 {result.out_wav}（{result.voiced_lines}/{result.total_lines} 句成功）")
            if messagebox.askyesno(APP_TITLE, f"配音完成：\n{result.out_wav}\n\n立即打开所在文件夹？"):
                try:
                    import os
                    os.startfile(str(Path(result.out_wav).parent))  # type: ignore
                except Exception:
                    pass

    def on_stop(self):
        if self.engine:
            self.engine.stop()
            self.logline("正在停止…（等当前句合成完）")

    def on_reexport(self):
        sub = self.var_sub.get().strip()
        if not sub or not Path(sub).exists():
            messagebox.showwarning(APP_TITLE, "请先选择字幕文件")
            return
        cfg = DubConfig(api_url=self.var_api.get().strip())
        try:
            engine = DubbingEngine(sub, self.var_out.get().strip(), cfg)
            result = engine._merge(filter_blocks(load_subtitles(sub)))
            self.logline(f"重新合并完成: {result.out_wav}")
        except Exception as e:
            messagebox.showerror(APP_TITLE, str(e))

    def on_play_clip(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(self.tree.set(sel[0], "idx"))
        sub = self.var_sub.get().strip()
        clips = Path(self.var_out.get() or "output") / Path(sub).stem / "session" / "clips"
        f = clips / f"{idx:05d}.wav"
        if not f.exists():
            messagebox.showinfo(APP_TITLE, f"该句还没有缓存音频:\n{f}")
            return
        try:
            import winsound  # Windows 专用；其他平台静默跳过
            winsound.PlaySound(str(f), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception:
            pass


if __name__ == "__main__":
    App().mainloop()
