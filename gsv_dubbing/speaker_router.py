"""旁白/说话人规则：为不同说话人分配不同参考音频与权重。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .subtitle_parser import SubtitleBlock


@dataclass
class SpeakerConfig:
    name: str
    ref_audio: str = ""
    prompt_text: str = ""
    prompt_lang: str = "zh"
    gpt_weights: str = ""
    sovits_weights: str = ""
    speed: float = 1.0


DEFAULT_PROFILE = {
    "text_lang": "zh",
    "prompt_lang": "zh",
    "default_speaker": {
        "ref_audio": "",
        "prompt_text": "",
        "gpt_weights": "",
        "sovits_weights": "",
    },
    "speakers": {},
    # 说话人识别：ASS Name 字段 或 文本前缀「雪风：」
}


class SpeakerRouter:
    """决定每条字幕用哪个参考音频/权重。

    匹配优先级：ASS/VTT 说话人字段 > 文本前缀「名字：/名字:」 > default_speaker
    speakers 配置键支持通配（fnmatch 风格）。
    """

    PREFIX_RE = re.compile(r"^([^\s「『\\【\[]{1,12})[：:]\s*")

    def __init__(self, profile: dict):
        self.profile = profile
        self.default = SpeakerConfig(
            name="__default__",
            **{k: v for k, v in profile.get("default_speaker", {}).items()
               if k in SpeakerConfig.__dataclass_fields__},
        )
        self.speakers: Dict[str, SpeakerConfig] = {}
        for name, cfg in profile.get("speakers", {}).items():
            self.speakers[name] = SpeakerConfig(
                name=name,
                **{k: v for k, v in cfg.items() if k in SpeakerConfig.__dataclass_fields__},
            )

    @classmethod
    def from_file(cls, path: str | Path) -> "SpeakerRouter":
        p = Path(path)
        if p.exists():
            profile = json.loads(p.read_text(encoding="utf-8"))
        else:
            profile = json.loads(json.dumps(DEFAULT_PROFILE, ensure_ascii=False))
            profile["_loaded_from"] = str(p)
        return cls(profile)

    def save_template(self, path: str | Path):
        Path(path).write_text(
            json.dumps(self.profile, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    def route(self, block: SubtitleBlock) -> SpeakerConfig:
        name = block.speaker
        if not name:
            m = self.PREFIX_RE.match(block.text)
            if m:
                name = m.group(1)
        if name:
            if name in self.speakers:
                return self.speakers[name]
            import fnmatch
            for pat, cfg in self.speakers.items():
                if fnmatch.fnmatch(name, pat):
                    return cfg
        return self.default

    def strip_speaker_prefix(self, text: str, speaker: SpeakerConfig) -> str:
        """去掉文本中的说话人前缀「名字：」；名字对不上时原样返回。"""
        m = self.PREFIX_RE.match(text)
        if not m:
            return text
        prefix_name = m.group(1)
        if speaker.name == "__default__" or prefix_name == speaker.name:
            return text[m.end():].strip()
        return text
