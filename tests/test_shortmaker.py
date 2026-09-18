"""ShortMaker birim testleri: ag / GPU / model gerektirmez."""
from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from shortmaker import silence, smart_crop, subtitle_generator as sg
from shortmaker.analyzer import MediaInfo, choose_fps
from shortmaker.config import SettingsError, load_settings
from shortmaker.downloader import download
from shortmaker.errors import MSG_UNSUPPORTED, ShortMakerError
from shortmaker.pipeline import _short_detail, next_output_dir
from shortmaker.segment_selector import select_segment
from shortmaker.transcriber import Transcript, Word


def words_from(text: str, start: float = 0.0, step: float = 0.3, gap_after: dict | None = None) -> list[Word]:
    out, t = [], start
    for i, w in enumerate(text.split()):
        out.append(Word(round(t, 3), round(t + step * 0.8, 3), w))
        t += step + (gap_after or {}).get(i, 0.0)
    return out


class SubtitleTests(unittest.TestCase):
    def test_groups_are_2_to_7_words(self):
        ws = words_from("bu cok uzun bir cumle ve hic noktalama yok ama yine de kisa "
                        "gruplara bolunmesi gerekiyor cunku ekranda paragraf istemiyoruz")
        cues = sg.group_words(ws)
        self.assertGreater(len(cues), 2)
        for c in cues:
            self.assertLessEqual(len(c.words), 7)
            self.assertGreaterEqual(len(c.words), 2)

    def test_sentence_end_and_pause_split(self):
        ws = words_from("Merhaba arkadaslar. Bugun cok guzel bir konu var", gap_after={1: 0.0})
        cues = sg.group_words(ws)
        self.assertEqual(cues[0].text, "Merhaba arkadaslar.")
        ws = words_from("bir iki uc dort", gap_after={1: 0.6})
        self.assertEqual([c.text for c in sg.group_words(ws)], ["bir iki", "uc dort"])

    def test_cues_do_not_overlap_and_follow_speech(self):
        cues = sg.group_words(words_from("a b c d e f g h i j k l m n o p", step=0.25))
        for a, b in zip(cues, cues[1:]):
            self.assertLessEqual(a.end, b.start + 1e-9)
        self.assertEqual(cues[0].start, 0.0)

    def test_srt_format(self):
        srt = sg.to_srt([sg.Cue(1.5, 3661.25, [Word(1.5, 2, "Hola"), Word(2, 3, "mundo")])])
        self.assertIn("1\n00:00:01,500 --> 01:01:01,250\nHola mundo\n", srt)

    def test_ass_highlight_one_event_per_word_and_keeps_language(self):
        cue = sg.Cue(0.0, 2.0, [Word(0, 0.5, "Çok"), Word(0.5, 1.0, "güzel"), Word(1.0, 1.8, "şey")])
        ass = sg.to_ass([cue], 1080, 1920, highlight=True)
        events = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
        self.assertEqual(len(events), 3)
        self.assertIn("{\\c&H0000FFFF&}güzel{\\c&H00FFFFFF&}", events[1])
        self.assertIn("PlayResY: 1920", ass)
        margin_v = int(re.search(r"Style: Short,.*,(\d+),1$", ass, re.M).group(1))
        self.assertGreater(margin_v, 1920 * 0.2)  # Shorts arayuzunun ustunde
        plain = sg.to_ass([cue], 720, 1280, highlight=False)
        self.assertEqual(len([l for l in plain.splitlines() if l.startswith("Dialogue:")]), 1)

    def test_ass_escapes_braces(self):
        ass = sg.to_ass([sg.Cue(0, 1, [Word(0, 1, "{x}")])], 1080, 1920, highlight=False)
        self.assertIn("(x)", ass)


class SilenceTests(unittest.TestCase):
    def test_keep_intervals_leaves_padding(self):
        keeps = silence.keep_intervals(0.0, 10.0, [(3.0, 5.0)], 0.7, 0.15)
        self.assertEqual(keeps, [(0.0, 3.15), (4.85, 10.0)])

    def test_leading_and_trailing_silence(self):
        keeps = silence.keep_intervals(0.0, 10.0, [(0.0, 1.0), (9.0, 10.0)], 0.7, 0.15)
        self.assertEqual(keeps, [(0.85, 9.15)])

    def test_short_gaps_are_kept(self):
        self.assertEqual(silence.keep_intervals(0, 10, [(3.0, 3.3)], 0.7, 0.15), [(0, 10)])

    def test_remap_words(self):
        keeps = [(0.0, 3.15), (4.85, 10.0)]
        ws = [Word(1.0, 1.5, "a"), Word(5.0, 5.5, "b"), Word(4.0, 4.2, "silinen")]
        out = silence.remap_words(ws, keeps)
        self.assertEqual([w.text for w in out], ["a", "b"])
        self.assertAlmostEqual(out[1].start, 3.15 + 0.15, places=3)
        self.assertAlmostEqual(silence.total(keeps), 8.3, places=3)

    def test_word_gaps(self):
        ws = [Word(0.2, 0.5, "a"), Word(2.0, 2.4, "b")]
        self.assertEqual(silence.word_gaps(ws, 0.0, 3.0, 0.7), [(0.5, 2.0)])


class CropTests(unittest.TestCase):
    def test_crop_size_keeps_aspect(self):
        self.assertEqual(smart_crop.crop_size(1920, 1080, 1080, 1920), (608, 1080))
        cw, ch = smart_crop.crop_size(1080, 1920, 1080, 1920)
        self.assertEqual((cw, ch), (1080, 1920))
        self.assertTrue(smart_crop.needs_horizontal_crop(1920, 1080, 1080, 1920))
        self.assertFalse(smart_crop.needs_horizontal_crop(608, 1080, 1080, 1920))

    def test_path_is_smooth_clamped_and_cuts_on_scene_change(self):
        rng = np.random.default_rng(0)
        samples = []
        for f in range(0, 300, 5):     # 10 sn @30fps, 6 ornek/sn
            if f < 150:
                x = 500 + rng.normal(0, 15)          # titreyen yuz tespiti
            else:
                x = 1500 + rng.normal(0, 15)
            samples.append((f, x, f == 150))         # 150. karede sahne kesimi
        path = smart_crop._build_path(samples, 300, 1920, 608, 30.0)
        self.assertEqual(len(path), 300)
        self.assertTrue(np.all(path >= 304) and np.all(path <= 1920 - 304))
        self.assertLess(np.abs(np.diff(path[10:140])).max(), 3.0)   # titreme yok
        self.assertGreater(abs(path[151] - path[148]), 500)        # kesimde aninda gecis

    def test_missing_detections_hold_position(self):
        samples = [(0, 800.0, False), (5, None, False), (10, None, False), (15, 800.0, False)]
        path = smart_crop._build_path(samples, 20, 1920, 606, 30.0)
        self.assertTrue(np.allclose(path, 800.0))


class SelectionTests(unittest.TestCase):
    def _tr(self, sentences):
        segs, words, t = [], [], 0.0
        for s in sentences:
            ws = words_from(s, start=t, step=0.4)
            words += ws
            segs.append((ws[0].start, ws[-1].end, s))
            t = ws[-1].end + 0.6
        return Transcript("tr", 0.99, words, segs, "cpu")

    def test_short_video_is_not_extended(self):
        sel = select_segment(20.0, 60, self._tr(["kisa bir video."]))
        self.assertEqual((sel.start, sel.end), (0.0, 20.0))

    def test_long_video_respects_target_and_word_boundaries(self):
        sents = [f"Bu {i}. cumle ve burada anlatilan onemli bir fikir var." for i in range(40)]
        tr = self._tr(sents)
        total = tr.words[-1].end + 1
        for target in (30, 45, 60):
            sel = select_segment(total, target, tr)
            self.assertLessEqual(sel.duration, target + 1e-6)
            self.assertGreater(sel.duration, target * 0.4)
            for w in tr.words:  # hicbir kelime ortadan bolunmez
                self.assertFalse(w.start < sel.start < w.end, w)
                self.assertFalse(w.start < sel.end < w.end, w)

    def test_mid_sentence_start_moves_to_next_sentence(self):
        from shortmaker.segment_selector import _align_to_sentences
        ws = words_from("on the planet, Snake Island. Why are we going there? " + "word " * 40 + "end.",
                        step=0.4)
        # bolum "planet," kelimesinden basliyor: onceki kelime cumle sonu degil
        s, e = _align_to_sentences(ws[2].start, ws[-1].end, ws, 8.0)
        self.assertEqual(s, ws[5].start)  # "Why"
        # bitis cumle ortasinda -> son tam cumlenin sonuna
        s, e = _align_to_sentences(ws[5].start, ws[20].end, ws, 1.0)
        self.assertEqual(e, ws[9].end)    # "there?"

    def test_no_transcript_uses_target_from_start(self):
        sel = select_segment(100.0, 45, None)
        self.assertEqual((sel.start, sel.end), (0.0, 45))


class MiscTests(unittest.TestCase):
    def test_choose_fps(self):
        base = dict(duration=10, width=1920, height=1080, has_audio=True, audio_mean_db=-20,
                    video_codec="h264", audio_codec="aac")
        self.assertEqual(choose_fps(MediaInfo(fps=25, variable_fps=False, **base), 60, 30), 25)
        self.assertEqual(choose_fps(MediaInfo(fps=59.94, variable_fps=False, **base), 60, 30), 59.94)
        self.assertEqual(choose_fps(MediaInfo(fps=120, variable_fps=False, **base), 60, 30), 30)
        self.assertEqual(choose_fps(MediaInfo(fps=29.2, variable_fps=True, **base), 60, 30), 30)

    def test_invalid_url_is_unsupported(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ShortMakerError) as ctx:
                download("bu bir url degil", Path(d))
        self.assertEqual(ctx.exception.message, MSG_UNSUPPORTED)

    def test_next_output_dir(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertEqual(next_output_dir(root).name, "video_001")
            (root / "video_001").mkdir()
            (root / "video_007").mkdir()
            (root / "test").mkdir()
            self.assertEqual(next_output_dir(root).name, "video_008")

    def test_short_detail(self):
        self.assertEqual(_short_detail("ERROR: [youtube] abc: This video is unavailable"),
                         "This video is unavailable")

    def test_settings_validation(self):
        st = load_settings(Path("does-not-exist.yaml"), {"video": {"duration": 45}})
        self.assertEqual(st.duration, 45)
        with self.assertRaises(SettingsError):
            load_settings(Path("does-not-exist.yaml"), {"video": {"duration": 90}})


if __name__ == "__main__":
    unittest.main()
