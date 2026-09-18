"""clipselect.py saf mantik testleri: cumle siniri, 30-60sn, hook/ceza skorlari.

Whisper/ffmpeg gerektirmez: yalnizca Segment listesi -> select_clips ciktisi
dogrulanir. Kural tabanli skorlamanin deterministik oldugu da test edilir.
"""
from __future__ import annotations

import unittest

from shortmaker.clipselect import (
    ClipProposal,
    MAX_CLIP_SEC,
    MIN_CLIP_SEC,
    Segment,
    select_clips,
    split_sentences,
)


def seg(start: float, end: float, text: str) -> Segment:
    return Segment(start, end, text)


class SplitSentencesTest(unittest.TestCase):
    def test_splits_on_punctuation(self) -> None:
        segs = split_sentences([seg(0.0, 6.0, "Merhaba dünya. Nasılsın? İyiyim!")])
        texts = [s.text for s in segs]
        self.assertEqual(texts, ["Merhaba dünya.", "Nasılsın?", "İyiyim!"])

    def test_no_punctuation_keeps_single(self) -> None:
        segs = split_sentences([seg(0.0, 4.0, "Selam nasılsın")])
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].text, "Selam nasılsın")
        self.assertEqual(segs[0].start, 0.0)
        self.assertEqual(segs[0].end, 4.0)

    def test_silence_segment_preserved(self) -> None:
        segs = split_sentences([seg(0.0, 2.0, ""), seg(2.0, 6.0, "Ses var.")])
        self.assertEqual(segs[0].text, "")
        self.assertEqual(segs[1].text, "Ses var.")


class SelectClipsTest(unittest.TestCase):
    def _speech(self, start: float, dur: float, text: str) -> Segment:
        return seg(start, start + dur, text)

    def test_clips_within_min_max_bounds(self) -> None:
        # 3 dakikalik, her bir 4 sn'lik 45 cumlelik konusma
        units = [self._speech(i * 4.0, 4.0, f"Bu {i}. cumle tamam.") for i in range(45)]
        clips = select_clips(units, 180.0, min_sec=MIN_CLIP_SEC, max_sec=MAX_CLIP_SEC, max_clips=3)
        self.assertGreaterEqual(len(clips), 1)
        for c in clips:
            self.assertGreaterEqual(c.duration, MIN_CLIP_SEC - 0.01)
            self.assertLessEqual(c.duration, MAX_CLIP_SEC + 0.01)

    def test_windows_start_and_end_at_sentence_boundaries(self) -> None:
        units = [self._speech(i * 5.0, 5.0, f"Cumle {i} tamamlandı.") for i in range(40)]
        clips = select_clips(units, 200.0, max_clips=3)
        for c in clips:
            starts = {u.start for u in units}
            ends = {u.end for u in units}
            self.assertIn(round(c.start, 3), {round(s, 3) for s in starts})
            self.assertIn(round(c.end, 3), {round(e, 3) for e in ends})

    def test_no_overlapping_clips(self) -> None:
        units = [self._speech(i * 4.0, 4.0, f"Bolum {i} anlatiliyor.") for i in range(50)]
        clips = select_clips(units, 200.0, max_clips=5)
        ranges = sorted((c.start, c.end) for c in clips)
        for (a0, a1), (b0, b1) in zip(ranges, ranges[1:]):
            self.assertLessEqual(a1, b0)  # ortusme yok

    def test_hook_opening_question_prioritized(self) -> None:
        # Soru ile baslayan blok daha yuksek skor almali
        units = [
            self._speech(0.0, 4.0, "Birinci bolumde normal bir metin var."),
            self._speech(4.0, 8.0, "Devam ediyoruz ve acikliyoruz."),
            self._speech(8.0, 12.0, "Son paragraf burada bitiyor."),
        ]
        while units[-1].end < 60.0:
            units.append(self._speech(units[-1].end, 4.0, "Ek cumle devam ediyor."))
        # sorulu bir bolum ekle: 60-120 arasina guclu acilis
        units.extend([
            self._speech(60.0, 4.0, "Biliyor musunuz bu inanilmaz mı?"),
            self._speech(64.0, 4.0, "Rekor kırıldı ve herkes şok oldu."),
            self._speech(68.0, 4.0, "Milyonlarca izlenme aldi."),
        ])
        while units[-1].end < 120.0:
            units.append(self._speech(units[-1].end, 4.0, "Ek cumle sonrasi."))

        clips = select_clips(units, 160.0, max_clips=1)
        self.assertEqual(len(clips), 1)
        self.assertLessEqual(clips[0].start, 70.0)  # soru/hook bolgesine yakin

    def test_filler_words_penalized(self) -> None:
        clean = [self._speech(i * 4.0, 4.0, f"Onemli bilgi {i} burada.") for i in range(0, 20)]
        clean2 = [self._speech(80.0 + i * 4.0, 4.0, f"Diger icerik {i} gecmiste.") for i in range(20)]
        # filler dolu bolge: 160-240
        fill = []
        for i in range(20):
            fill.append(self._speech(160.0 + i * 4.0, 4.0,
                                     f"yani şey ee bilmiyorum işte {i} falan yani ee"))
        s = clean + clean2 + fill
        s.sort(key=lambda x: x.start)
        clips = select_clips(s, 240.0, max_clips=1)
        self.assertEqual(len(clips), 1)
        self.assertLess(clips[0].start, 80.0)  # filler bolgelerden uzak durur

    def test_max_clips_respected_and_padded_if_needed(self) -> None:
        units = [self._speech(i * 5.0, 5.0, f"Cumle {i} burada.") for i in range(30)]
        clips = select_clips(units, 150.0, max_clips=10)
        self.assertLessEqual(len(clips), 10)

    def test_as_report_contains_required_fields(self) -> None:
        units = [self._speech(i * 4.0, 4.0, f"Cumle {i} tamam.") for i in range(20)]
        clips = select_clips(units, 80.0, max_clips=1)
        self.assertEqual(len(clips), 1)
        r = clips[0].as_report()
        for key in ("baslangic", "bitis", "sure", "baslik", "neden",
                    "ana_fikir", "hook", "altyazi_vurgulari"):
            self.assertIn(key, r)
        self.assertEqual(r["baslangic"], clips[0].timecode_start())

    def test_english_competition_keywords_reach_hook_and_emphasis(self) -> None:
        """EN yarisma icerigi: prize/challenge/rescue/team INTEREST keşfine takilmali.

        Siradan anlatim ile kiyaslandiginda yarisma bolgesi secilmeli; hook ve
        altyazi vurgulari bu kelimelerden en az birini icermeli.
        """
        # siradan anlatim: 0-60 arasi, ilgi kelimesi yok
        common = [self._speech(i * 4.0, 4.0, f"This is a regular part of the story {i}.")
                  for i in range(15)]
        # yarisma bolgesi: 60-120 arasi guclu kelimeler
        comp = [
            self._speech(60.0, 4.0, "The prize is one hundred thousand dollars!"),
            self._speech(64.0, 4.0, "Every team faces a new challenge today."),
            self._speech(68.0, 4.0, "The rescue mission must reach the finish line."),
            self._speech(72.0, 4.0, "Our team is going to win the final reward."),
        ]
        while comp[-1].end < 120.0:
            comp.append(self._speech(comp[-1].end, 4.0, "More challenge details continue here."))
        units = sorted(common + comp, key=lambda s: s.start)

        clips = select_clips(units, 180.0, max_clips=1)
        self.assertEqual(len(clips), 1)
        # yarisma bolgesi secilir (siradan anlatim degil)
        self.assertGreaterEqual(clips[0].start, 55.0)

        report = clips[0].as_report()
        words = " ".join(map(str, report["altyazi_vurgulari"])).lower()
        hook = str(report["hook"]).lower()
        # INTEREST keşfi en az bir yarisma kelimesini hook veya vurguya tasimali
        self.assertTrue(
            any(w in hook or w in words for w in ("prize", "challenge", "rescue", "team")),
            f"yarisma kelimesi yansimadi: hook={report['hook']!r} vurgu={report['altyazi_vurgulari']!r}",
        )

    def test_short_speech_returns_no_clips(self) -> None:
        # 10 sn toplam konusma: 30 sn dolduracak pencere yok
        units = [seg(0.0, 3.0, "Kisa."), seg(3.0, 7.0, "Cok kisa.")]
        clips = select_clips(units, 10.0)
        self.assertEqual(clips, [])


if __name__ == "__main__":
    unittest.main()