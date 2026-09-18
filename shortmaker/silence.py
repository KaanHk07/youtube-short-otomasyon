"""Uzun sessizlikleri bulup kisaltir; kelime zamanlarini yeni zaman cizelgesine esler.

Bosluk kaynagi: ffmpeg silencedetect (sessiz ortam) + kelime aralari (Whisper).
Ikisi de "konusma yok" dediginde kesilir; boylece muzik/efekt olan anlar
korunur. Her kesimin iki yaninda dogal pay birakilir (agresif jump-cut yok).
"""
from __future__ import annotations

import re
from pathlib import Path

from .ffmpeg_utils import FFmpegError, run_ffmpeg
from .transcriber import Word

Interval = tuple[float, float]


def detect_silences(path: Path, start: float, end: float, min_len: float,
                    noise_db: float = -35.0) -> list[Interval]:
    """[start,end] araligindaki sessiz bolgeler (kaynak zamaninda)."""
    try:
        err = run_ffmpeg(["-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(path),
                          "-map", "0:a:0", "-af", f"silencedetect=noise={noise_db}dB:d={min_len}",
                          "-vn", "-f", "null", "-"], timeout=900)
    except FFmpegError:
        return []
    out: list[Interval] = []
    cur: float | None = None
    for line in err.splitlines():
        m = re.search(r"silence_start:\s*(-?[\d.]+)", line)
        if m:
            cur = float(m.group(1))
            continue
        m = re.search(r"silence_end:\s*(-?[\d.]+)", line)
        if m and cur is not None:
            out.append((start + max(0.0, cur), start + float(m.group(1))))
            cur = None
    if cur is not None:
        out.append((start + cur, end))
    return out


def word_gaps(words: list[Word], start: float, end: float, min_len: float) -> list[Interval]:
    """Kelimeler arasindaki (ve bolum basi/sonundaki) konusmasiz araliklar."""
    ws = [w for w in words if w.end > start and w.start < end]
    if not ws:
        return []
    gaps: list[Interval] = []
    edges = [(start, ws[0].start)]
    edges += [(a.end, b.start) for a, b in zip(ws, ws[1:])]
    edges.append((ws[-1].end, end))
    for a, b in edges:
        if b - a >= min_len:
            gaps.append((a, b))
    return gaps


def _intersect(a: list[Interval], b: list[Interval]) -> list[Interval]:
    out = []
    for s1, e1 in a:
        for s2, e2 in b:
            s, e = max(s1, s2), min(e1, e2)
            if e > s:
                out.append((s, e))
    return sorted(out)


def keep_intervals(start: float, end: float, silences: list[Interval],
                   min_len: float, pad: float) -> list[Interval]:
    """Sessizlikleri (paylari birakarak) cikarip tutulacak araliklari dondurur."""
    cuts = []
    for s, e in sorted(silences):
        cs, ce = max(start, s + pad), min(end, e - pad)
        if s <= start + 0.01:
            cs = start          # bastaki sessizligin tamami atilabilir (pay sadece sonda)
            ce = min(end, e - pad)
        if e >= end - 0.01:
            ce = end
            cs = max(start, s + pad)
        if ce - cs >= max(0.2, min_len - 2 * pad):
            cuts.append((cs, ce))
    keeps: list[Interval] = []
    cur = start
    for cs, ce in cuts:
        if cs > cur + 0.05:
            keeps.append((cur, cs))
        cur = max(cur, ce)
    if end > cur + 0.05:
        keeps.append((cur, end))
    return keeps or [(start, end)]


def plan_cuts(path: Path, words: list[Word], start: float, end: float,
              min_len: float, pad: float, use_audio: bool = True) -> list[Interval]:
    gaps = word_gaps(words, start, end, min_len)
    if not gaps:
        return [(start, end)]
    if use_audio:
        audio_sil = detect_silences(path, start, end, min_len * 0.6)
        gaps = [g for g in _intersect(gaps, audio_sil) if g[1] - g[0] >= min_len]
    return keep_intervals(start, end, gaps, min_len, pad)


def remap_time(t: float, keeps: list[Interval]) -> float | None:
    """Kaynak zamanini kesilmis zaman cizelgesine cevirir; kesilen alandaysa None."""
    offset = 0.0
    for s, e in keeps:
        if t < s - 1e-6:
            return None
        if t <= e + 1e-6:
            return offset + (min(t, e) - s)
        offset += e - s
    return None


def remap_words(words: list[Word], keeps: list[Interval]) -> list[Word]:
    """Tutulan araliklara dusen kelimeleri yeni zamanlara tasir (sinira kirpar)."""
    out: list[Word] = []
    for w in words:
        for s, e in keeps:
            if w.end > s and w.start < e:
                ns = remap_time(max(w.start, s), keeps)
                ne = remap_time(min(w.end, e), keeps)
                if ns is not None and ne is not None and ne > ns:
                    out.append(Word(round(ns, 3), round(ne, 3), w.text))
                break
    return out


def total(keeps: list[Interval]) -> float:
    return sum(e - s for s, e in keeps)
