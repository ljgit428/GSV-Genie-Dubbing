"""断点续跑状态：记录每句字幕的合成状态与音频文件名。"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

STATUS_SCHEMA = 1


@dataclass
class LineStatus:
    index: int
    text: str
    start: float
    end: float
    speaker: Optional[str]
    file: Optional[str] = None        # 成功后的相对文件名
    speed: float = 1.0
    attempts: int = 0
    status: str = "pending"          # pending / done / failed


class SessionState:
    """session.json：断点续跑 + 语速历史（同一句重跑时用上次的 speed 起步）。"""

    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.file = self.session_dir / "session.json"
        self.lines: Dict[int, LineStatus] = {}
        self.meta: dict = {}
        self.speed_history: Dict[int, float] = {}
        self._dirty = False
        if self.file.exists():
            try:
                self._load()
            except Exception:
                pass

    def _load(self):
        data = json.loads(self.file.read_text(encoding="utf-8"))
        if data.get("schema") != STATUS_SCHEMA:
            return
        self.meta = data.get("meta", {})
        self.speed_history = {int(k): v for k, v in data.get("speed_history", {}).items()}
        for l in data.get("lines", []):
            st = LineStatus(**{k: l.get(k) for k in LineStatus.__dataclass_fields__})
            self.lines[st.index] = st

    def init_lines(self, blocks, meta: dict):
        """以字幕块初始化（保留已有同 index 且文本相同的状态）。"""
        self.meta = meta
        keep = {}
        for b in blocks:
            old = self.lines.get(b.index)
            if old and old.text == b.text and old.status == "done":
                keep[b.index] = old
            else:
                keep[b.index] = LineStatus(
                    index=b.index, text=b.text, start=b.start, end=b.end,
                    speaker=b.speaker, status="pending",
                )
        self.lines = keep
        self.save(force=True)

    def save(self, force: bool = False):
        if not self._dirty and not force:
            return
        data = {
            "schema": STATUS_SCHEMA,
            "meta": self.meta,
            "lines": [asdict(l) for l in sorted(self.lines.values(), key=lambda x: x.index)],
            "speed_history": self.speed_history,
        }
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.file)
        self._dirty = False

    def mark(self, index: int, status: str, file: Optional[str] = None, speed: float = 1.0):
        l = self.lines.get(index)
        if l:
            l.status = status
            l.file = file
            l.speed = speed
            l.attempts += 1
            if status == "done":
                self.speed_history[index] = speed
            self._dirty = True

    def is_done(self, index: int) -> bool:
        l = self.lines.get(index)
        return bool(l and l.status == "done" and l.file)

    def done_count(self) -> int:
        return sum(1 for l in self.lines.values() if l.status == "done")
