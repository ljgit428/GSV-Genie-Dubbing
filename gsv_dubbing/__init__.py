"""GSV-Genie-Dubbing：基于 GPT-SoVITS api_v2 的字幕逐句自动配音工具。"""

__version__ = "0.1.0"

from .engine import DubConfig, DubbingEngine, DubResult  # noqa: F401
from .subtitle_parser import SubtitleBlock, load_subtitles, filter_blocks  # noqa: F401
