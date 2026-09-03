"""字幕解析模块：支持 SRT / ASS / VTT，统一输出 SubtitleBlock 列表。

清洗规则（默认可关）：
- 去掉对话标签、旁白方括号 [xxx]、注释大括号 {xxx}、ASS 覆盖标签 {\\...}
- 合并多行文本
- 过滤纯符号 / 无有效文字的行
- 可选：跳过非目标语言行（用中文字符占比判断）
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

ASS_OVERRIDE_TAG = re.compile(r"\{\\.*?\}")          # ASS 覆盖标签 {\i1} 等
COMMENT_TAG = re.compile(r"\{[^\\{}]*?\}")           # 注释 {xxx}
SQUARE_BRACKET = re.compile(r"^[【\[][^】\]]*[】\]]")  # 行首旁白 【1942年…】 / [xxx]
DIALOGUE_MARK = re.compile(r"^[「『“\"'-]+|[\」』”\"']+$")
HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
ANY_LANGUAGE_TEXT = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9fa-zA-Z]{2,}")
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
MUSIC_NOTE = re.compile(r"[♪♫♬♩]")


@dataclass
class SubtitleBlock:
    index: int              # 原字幕序号（1 起）
    start: float            # 秒
    end: float              # 秒
    text: str               # 清洗后的文本
    speaker: Optional[str] = None   # ASS 说话人 / VTT voice 标签
    raw: str = field(default="", repr=False)

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)

    def has_cjk(self) -> bool:
        return bool(CJK_RE.search(self.text))


def _to_seconds(ts: str) -> float:
    ts = ts.strip().replace(",", ".")
    # 支持 h:mm:ss.ms 与 mm:ss.ms
    parts = ts.split(":")
    parts = [float(p) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts[-3], parts[-2], parts[-1]
    return h * 3600 + m * 60 + s


def clean_text(text: str, strip_brackets: bool = True) -> str:
    """清洗单条字幕文本。"""
    lines = text.replace("\r", "\n").split("\n")
    cleaned_lines = []
    for ln in lines:
        ln = ASS_OVERRIDE_TAG.sub("", ln)
        ln = COMMENT_TAG.sub("", ln)
        ln = HTML_TAG.sub("", ln)
        if strip_brackets:
            prev = None
            while prev != ln:  # 行首可能有多个连续标签 [xx][yy]
                prev = ln
                ln = SQUARE_BRACKET.sub("", ln, count=1)
        ln = ln.strip()
        if ln:
            cleaned_lines.append(ln)
    merged = " ".join(cleaned_lines).strip()
    merged = DIALOGUE_MARK.sub("", merged).strip()
    return merged


def is_speakable(text: str) -> bool:
    """判断清洗后文本是否含可读文字（过滤空/纯符号/音乐标记行）。"""
    if MUSIC_NOTE.search(text):
        return False
    return bool(ANY_LANGUAGE_TEXT.search(text))


# ---------------------------------------------------------------- SRT / VTT

def parse_srt(content: str) -> List[SubtitleBlock]:
    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n").strip())
    out = []
    idx = 0
    for blk in blocks:
        blk = blk.strip()
        if not blk:
            continue
        lines = [l for l in blk.split("\n") if l.strip()]
        if not lines:
            continue
        # 跳过可能的序号行
        if re.match(r"^\d+$", lines[0].strip()):
            lines = lines[1:]
        if not lines:
            continue
        m = re.search(
            r"(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}|\d{1,2}:\d{2}[.,]\d{1,3})"
            r"\s*-->\s*"
            r"(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}|\d{1,2}:\d{2}[.,]\d{1,3})",
            lines[0],
        )
        if not m:
            continue
        start, end = _to_seconds(m.group(1)), _to_seconds(m.group(2))
        text = "\n".join(lines[1:])
        idx += 1
        out.append(SubtitleBlock(index=idx, start=start, end=end, text=text, raw=text))
    return out


def parse_vtt(content: str) -> List[SubtitleBlock]:
    content = re.sub(r"^WEBVTT.*?\n", "", content, flags=re.S)
    content = re.sub(r"^(NOTE|STYLE|REGION)[^\n]*\n", "", content, flags=re.M)
    return parse_srt(content)


# ---------------------------------------------------------------- ASS

def parse_ass(content: str) -> List[SubtitleBlock]:
    out = []
    idx = 0
    fmt_fields: List[str] = []
    in_events = False
    for line in content.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if line.startswith("["):
            in_events = line.lower().startswith("[events]")
            continue
        if not in_events or not line:
            continue
        if line.startswith("Format:"):
            fmt_fields = [f.strip().lower() for f in line.split(":", 1)[1].split(",")]
            continue
        if not (line.startswith("Dialogue:") or line.startswith("Comment:")):
            continue
        if line.startswith("Comment:"):
            continue
        body = line.split(":", 1)[1]
        parts = [p.strip() for p in body.split(",")]
        # Text 字段可能含逗号：取到 Text 位置为止，剩余全部并入 Text
        if "text" in fmt_fields:
            ti = fmt_fields.index("text")
            if len(parts) > len(fmt_fields):
                parts = parts[:ti] + [",".join(parts[ti:])]
        try:
            d = dict(zip(fmt_fields, parts))
            start, end = _to_seconds(d["start"]), _to_seconds(d["end"])
        except (KeyError, ValueError):
            continue
        idx += 1
        out.append(
            SubtitleBlock(
                index=idx,
                start=start,
                end=end,
                text=d.get("text", ""),
                speaker=d.get("name") or None,
                raw=d.get("text", ""),
            )
        )
    return out


# ---------------------------------------------------------------- 统一入口

def load_subtitles(path: str | Path) -> List[SubtitleBlock]:
    p = Path(path)
    content = p.read_text(encoding="utf-8-sig", errors="replace")
    suffix = p.suffix.lower()
    if suffix in (".ass", ".ssa"):
        return parse_ass(content)
    if suffix in (".vtt",):
        return parse_vtt(content)
    if suffix in (".srt", ".txt") or "-->" in content:
        return parse_srt(content)
    return parse_srt(content)


def filter_blocks(
    blocks: List[SubtitleBlock],
    strip_brackets: bool = True,
    only_cjk: bool = False,
    skip_empty: bool = True,
) -> List[SubtitleBlock]:
    """文本清洗 + 过滤，返回新列表（index 保留原值）。"""
    out = []
    for b in blocks:
        text = clean_text(b.text, strip_brackets=strip_brackets)
        b2 = SubtitleBlock(b.index, b.start, b.end, text, b.speaker, b.raw)
        if skip_empty and not is_speakable(text):
            continue
        if only_cjk and not b2.has_cjk():
            continue
        out.append(b2)
    return out
