"""En uygun <=N saniyelik bolumu secer.

Transkript varsa cumle sinirli puanlayici (clipselect.py)
kullanilir: bolum cumle basinda baslar, cumle sonunda biter, hook/icerik odullu.
Sinirlar kelime zamanlarina oturtulur ve kucuk bir nefes payi eklenir.
Video hedeften kisaysa tamami kullanilir (uzatma yok).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .clipselect import Segment, select_clips
from .transcriber import Transcript

log = logging.getLogger(__name__)

LEAD_IN = 0.12   # ilk kelimeden once birakilan pay
TAIL = 0.35      # son kelimeden sonra (cumle sonu nefesi)
OPENER_TOLERANCE = 0.15  # acilis bolumu en iyi puandan en fazla %15 dusukse tercih edilir


@dataclass
class Selection:
    start: float
    end: float
    reason: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def select_segment(duration: float, target: float, transcript: Transcript | None) -> Selection:
    if duration <= target + 0.05:
        return Selection(0.0, duration, "video hedef sureden kisa: tamami kullanildi")
    if not transcript or not transcript.segments:
        return Selection(0.0, target, "konusma yok: videonun basindan hedef sure")

    segs = [Segment(s, e, t) for s, e, t in transcript.segments]
    proposals = []
    try:
        proposals = select_clips(segs, duration, min_sec=max(8.0, target * 0.5),
                                 max_sec=target, max_clips=3)
    except Exception as exc:  # puanlayici hata verirse basit yonteme dus
        log.warning("Klip puanlayici basarisiz: %s", exc)

    if proposals:
        best = proposals[0]
        start, end, why = best.start, best.end, "; ".join(best.why[:3]) or "en yuksek puanli bolum"
        # Videonun acilis cumlesi genelde yaraticinin kurdugu kancadir; puani yakinsa
        # bastan baslayan bolumu sec (baglam kaybolmaz, anlamsiz yerde baslamaz).
        opener = _best_opener(segs, duration, max(8.0, target * 0.5), target)
        if opener and abs(opener[0] - best.start) > 0.5 \
                and opener[2] >= best.score - abs(best.score) * OPENER_TOLERANCE:
            start, end = opener[0], opener[1]
            why = "videonun acilis kancasindan basliyor; cumle sinirinda bitiyor"
    else:
        start, end = _greedy_from_start(transcript.segments, target)
        why = "cumle sinirinda bastan kesildi"

    start, end = _snap_to_words(start, end, transcript, duration, target)
    return Selection(start, end, why)


def select_segments(duration: float, target: float, transcript: Transcript | None,
                    count: int = 1) -> list[Selection]:
    """Birbiriyle ortusmeyen en iyi `count` bolum (1. = en guclu). Kisa videoda tek bolum."""
    first = select_segment(duration, target, transcript)
    if count <= 1 or duration <= target + 0.05 or not transcript or not transcript.segments:
        return [first]

    chosen = [first]
    segs = [Segment(s, e, t) for s, e, t in transcript.segments]
    try:
        proposals = select_clips(segs, duration, min_sec=max(8.0, target * 0.5),
                                 max_sec=target, max_clips=count + 4)
    except Exception as exc:
        log.warning("Klip puanlayici basarisiz: %s", exc)
        proposals = []
    for p in proposals:  # puana gore sirali
        if len(chosen) >= count:
            break
        s, e = _snap_to_words(p.start, p.end, transcript, duration, target)
        if any(s < c.end and e > c.start for c in chosen):  # hic ortusme yok
            continue
        chosen.append(Selection(s, e, "; ".join(p.why[:3]) or "yuksek puanli bolum"))

    # puanlayici yeterli aday bulamadiysa bos kalan yerlerden cumle sinirli dilim ekle
    while len(chosen) < count:
        extra = _next_free_window(transcript, duration, target, chosen)
        if extra is None:
            break
        chosen.append(extra)
    return chosen


def _next_free_window(tr: Transcript, duration: float, target: float,
                      chosen: list[Selection]) -> Selection | None:
    for s0, _e0, _t in tr.segments:
        if any(c.start - 0.05 <= s0 < c.end for c in chosen):
            continue
        rest = [seg for seg in tr.segments if seg[0] >= s0]
        s, e = _greedy_from_start(rest, target)
        s, e = _snap_to_words(s, e, tr, duration, target)
        if e - s >= max(8.0, target * 0.5) and not any(s < c.end and e > c.start for c in chosen):
            return Selection(s, e, "cumle sinirinda kalan bolumden secildi")
    return None


def _best_opener(segs: list[Segment], duration: float, min_sec: float,
                 max_sec: float) -> tuple[float, float, float] | None:
    """Ilk cumleden baslayan en yuksek puanli pencere: (bas, son, puan)."""
    try:
        from .clipselect import _build_windows, _window_score, split_sentences
        units = split_sentences(segs)
        first = next((k for k, u in enumerate(units) if u.text.strip()), None)
        if first is None:
            return None
        best = None
        for i, j, _dur in _build_windows(units, duration, min_sec, max_sec):
            if i != first:
                continue
            score, _ = _window_score(units, i, j, [])
            if best is None or score > best[2]:
                best = (units[i].start, units[j - 1].end, score)
        return best
    except Exception as exc:  # puanlayici ic API degisirse sadece bu iyilestirme devre disi
        log.debug("Acilis penceresi puanlanamadi: %s", exc)
        return None


def _greedy_from_start(segments: list[tuple[float, float, str]], target: float) -> tuple[float, float]:
    start = segments[0][0]
    end = start
    for s, e, _t in segments:
        if e - start > target:
            break
        end = e
    if end <= start:  # tek cumle bile hedeften uzun
        end = start + target
    return start, end


SENTENCE_END = (".", "!", "?", "…", "。", "！", "？")
SENTENCE_GAP = 0.6  # bu kadar duraklama da cumle siniri sayilir


def _ends_sentence(text: str) -> bool:
    return text.rstrip("\"'”’)»").endswith(SENTENCE_END)


def _align_to_sentences(start: float, end: float, words: list, min_len: float) -> tuple[float, float]:
    """Whisper parcalari cumle ortasinda baslayabilir; sinirlari gercek cumle sinirina tasir.

    Baslangic cumle ortasindaysa sonraki cumle basina ilerler, bitis cumle ortasindaysa
    son tam cumlenin sonuna geri cekilir. Bolum min_len'in altina dusecekse dokunulmaz.
    """
    idx = [i for i, w in enumerate(words) if w.end > start + 0.01 and w.start < end - 0.01]
    if not idx:
        return start, end
    i0, i1 = idx[0], idx[-1]

    def is_start(i: int) -> bool:
        return i == 0 or _ends_sentence(words[i - 1].text) or words[i].start - words[i - 1].end >= SENTENCE_GAP

    def is_end(i: int) -> bool:
        return (i == len(words) - 1 or _ends_sentence(words[i].text)
                or words[i + 1].start - words[i].end >= SENTENCE_GAP)

    s_i = next((i for i in range(i0, i1 + 1) if is_start(i)), i0)
    e_i = next((i for i in range(i1, s_i - 1, -1) if is_end(i)), i1)
    if words[e_i].end - words[s_i].start < min_len:
        return start, end
    if s_i != i0 or e_i != i1:
        log.info("Bolum cumle sinirina hizalandi: %.2f-%.2f -> %.2f-%.2f",
                 start, end, words[s_i].start, words[e_i].end)
    return (words[s_i].start if s_i != i0 else start), (words[e_i].end if e_i != i1 else end)


def _snap_to_words(start: float, end: float, tr: Transcript, duration: float,
                   target: float) -> tuple[float, float]:
    """Sinirlari kelime aralarina oturtur, kelimeyi ortadan bolmez, hedefi asmaz."""
    words = tr.words
    start, end = _align_to_sentences(start, end, words, max(8.0, target * 0.4))
    inside = [w for w in words if w.end > start + 0.01 and w.start < end - 0.01]
    if inside:
        start = min(start, inside[0].start)
        end = max(end, inside[-1].end)
    prev_end = max((w.end for w in words if w.end <= start + 0.01), default=0.0)
    next_start = min((w.start for w in words if w.start >= end - 0.01), default=duration)
    s = max(0.0, prev_end, start - LEAD_IN)
    e = min(duration, next_start, end + TAIL)
    if e - s > target:
        # once payi kirp; yine sigmiyorsa hedef icindeki son kelimede bitir
        e = min(e, s + target)
        fitting = [w for w in words if s <= w.start and w.end <= e]
        if fitting:
            e = min(e, fitting[-1].end + 0.2)
    return round(s, 3), round(max(e, s + 0.5), 3)
