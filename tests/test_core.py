"""测试：字幕解析（用桌面真实 nanoda.srt 风格样本）+ 音频时间轴管线。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from gsv_dubbing.audio_utils import build_timeline, normalize_peak, resample, save_wav
from gsv_dubbing.speaker_router import SpeakerRouter
from gsv_dubbing.subtitle_parser import (
    SubtitleBlock,
    clean_text,
    filter_blocks,
    is_speakable,
    parse_ass,
    parse_srt,
)


SRT_SAMPLE = """1
00:01:03,000 --> 00:01:04,380
[1942年6月——中途岛近海]

2
00:01:04,380 --> 00:01:05,590
[1942年6月——中途岛近海]
赤城起火了！

3
00:01:05,590 --> 00:01:05,710
♪ BGM ♪

4
00:01:07,800 --> 00:01:09,760
雪风：取消作战！所有舰艇驶出敌方射程！

5
00:01:10,000 --> 00:01:12,500
{\\i1}警告：敌方空袭{\\i0}

"""

ASS_SAMPLE = """[Script Info]
Title: test

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, Bold, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:01:03.00,0:01:04.38,Default,雪风,0,0,0,,{\\i1}起火了{\\i0}，快撤退！
Dialogue: 0,0:01:05.00,0:01:06.00,Default,旁白,0,0,0,,【1942年6月】中途岛近海
Comment: 0,0:01:07.00,0:01:08.00,Default,,0,0,0,,被注释的行不应出现
"""


class TestSRTParser(unittest.TestCase):
    def test_parse_basic(self):
        blocks = parse_srt(SRT_SAMPLE)
        self.assertEqual(len(blocks), 5)
        self.assertAlmostEqual(blocks[0].start, 63.0)
        self.assertAlmostEqual(blocks[0].end, 64.38)
        self.assertIn("赤城", blocks[1].text)

    def test_clean_brackets(self):
        blocks = parse_srt(SRT_SAMPLE)
        filtered = filter_blocks(blocks)
        # 1 号是纯旁白 → 清洗后空；3 号是 ♪ BGM ♪ → 无文字
        idx = [b.index for b in filtered]
        self.assertNotIn(1, idx)
        self.assertNotIn(3, idx)
        self.assertIn(2, idx)
        # 第 4 句去掉方括号后保留正文
        b4 = [b for b in filtered if b.index == 4][0]
        self.assertEqual(b4.text, "雪风：取消作战！所有舰艇驶出敌方射程！")
        # 第 5 句 ASS 覆盖标签被剥掉
        b5 = [b for b in filtered if b.index == 5][0]
        self.assertEqual(b5.text, "警告：敌方空袭")

    def test_is_speakable(self):
        self.assertTrue(is_speakable("赤城起火了！"))
        self.assertFalse(is_speakable("♪ ♪"))
        self.assertFalse(is_speakable(""))


class TestASSParser(unittest.TestCase):
    def test_parse_ass(self):
        # parse 阶段保留原文；标签/旁白在 filter_blocks 清洗阶段剥掉
        blocks = parse_ass(ASS_SAMPLE)
        self.assertEqual(len(blocks), 2)  # Comment 行被跳过
        self.assertEqual(blocks[0].speaker, "雪风")
        self.assertIn("快撤退", blocks[0].text)
        self.assertNotIn("{\\i1}", clean_text(blocks[0].text))
        b2 = filter_blocks(blocks)[1]
        # 【1942年6月】旁白被剥掉
        self.assertEqual(b2.text, "中途岛近海")


class TestSpeakerRouter(unittest.TestCase):
    def test_prefix_routing(self):
        router = SpeakerRouter({
            "default_speaker": {"ref_audio": "default.wav", "prompt_text": "默认"},
            "speakers": {"雪风": {"ref_audio": "yukikaze.wav", "prompt_text": "雪风台词"}},
        })
        b = SubtitleBlock(1, 0, 1, "雪风：全舰撤退！")
        spk = router.route(b)
        self.assertEqual(spk.name, "雪风")
        text = router.strip_speaker_prefix(b.text, spk)
        self.assertEqual(text, "全舰撤退！")

    def test_ass_field_routing(self):
        router = SpeakerRouter({
            "default_speaker": {},
            "speakers": {"旁白": {"speed": 0.9}},
        })
        b = SubtitleBlock(2, 0, 1, "中途岛近海", speaker="旁白")
        spk = router.route(b)
        self.assertEqual(spk.name, "旁白")
        self.assertEqual(spk.speed, 0.9)


class TestAudioPipeline(unittest.TestCase):
    def test_resample(self):
        a = np.ones(32000, dtype=np.float32)
        out = resample(a, 32000, 16000)
        self.assertEqual(out.size, 16000)

    def test_timeline_placement(self):
        sr = 1000
        clips = [
            (1.0, 2.0, np.ones(500, dtype=np.float32), sr),   # 0.5s @1.0s → 1.0~1.5s
            (3.0, 4.0, np.ones(1000, dtype=np.float32), sr), # 1.0s @3.0s → 3.0~4.0s
        ]
        mixed = build_timeline(clips, total_duration=4.5, sr=sr)
        self.assertEqual(mixed.size, 4500)
        self.assertAlmostEqual(mixed[500], 0.0)    # 0.5s 静音
        self.assertAlmostEqual(mixed[1200], 1.0)  # 1.2s 有声（第一段内）
        self.assertAlmostEqual(mixed[1600], 0.0)   # 1.6s 第一段已结束
        self.assertAlmostEqual(mixed[2500], 0.0)   # 2.5s 静音
        self.assertAlmostEqual(mixed[3500], 1.0)   # 3.5s 有声（第二段内）

    def test_timeline_overlap_fade(self):
        sr = 1000
        clips = [
            (1.0, 2.0, np.ones(1500, dtype=np.float32), sr),  # 延伸到 2.5s
            (2.0, 3.0, np.ones(500, dtype=np.float32), sr),   # 2.0s 开始
        ]
        mixed = build_timeline(clips, total_duration=3.5, sr=sr)
        self.assertEqual(mixed.size, 3500)
        # 重叠处应有交叉淡化，无爆音（值不超过 1.0）
        self.assertLessEqual(float(np.max(mixed)), 1.0 + 1e-6)

    def test_save_and_normalize(self):
        import tempfile
        import wave
        a = np.array([0.1, -0.2, 0.5], dtype=np.float32)
        a = normalize_peak(a, peak=0.9)
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "t.wav"
            save_wav(f, a, 16000)
            with wave.open(str(f), "rb") as w:
                self.assertEqual(w.getframerate(), 16000)
                self.assertEqual(w.getnframes(), 3)


class TestRealSrtFile(unittest.TestCase):
    """用桌面上的真实 nanoda.srt（若存在）做冒烟测试。"""

    REAL = Path(r"C:\Users\yukik\Desktop\nanoda.srt")

    def test_real(self):
        if not self.REAL.exists():
            self.skipTest("nanoda.srt 不存在")
        from gsv_dubbing.subtitle_parser import load_subtitles
        blocks = filter_blocks(load_subtitles(self.REAL))
        self.assertGreater(len(blocks), 10)
        # 全部句子时间有效
        for b in blocks:
            self.assertGreaterEqual(b.start, 0)
            self.assertGreater(b.end, b.start)
            self.assertTrue(b.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
