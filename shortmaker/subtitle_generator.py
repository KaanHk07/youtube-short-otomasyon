"""Dinamik altyazi: 2-7 kelimelik gruplar -> .srt ve .ass (aktif kelime vurgulu).

Metin kaynak dildedir (Whisper transcribe; ceviri yok). ASS stili Shorts/TikTok
tarzi: kalin beyaz yazi, kalin siyah kontur, orta-alt bolge; YouTube Shorts
arayuzunun (alt baslik/aciklama alani, sag butonlar) ortmeyecegi guvenli alan.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .transcriber import Word

SENTENCE_END = (".", "!", "?", "…", "。", "！", "？")
CLAUSE_END = (",", ";", ":", "،", "、")
PAUSE_SPLIT = 0.35     # bu kadar duraklama yeni grup baslatir
HOLD_AFTER = 0.30      # son kelimeden sonra ekranda kalma
BRIDGE_GAP = 0.60      # gruplar arasi bu kadar kisa boslukta onceki grup uzatilir

YELLOW = "&H0000FFFF&"  # ASS BGR
WHITE = "&H00FFFFFF&"


@dataclass
class Cue:
    start: float
    end: float
    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


def group_words(words: list[Word], min_words: int = 2, max_words: int = 7,
                max_chars: int = 24) -> list[Cue]:
    """Kelimeleri konusma ritmine gore kisa gruplara ayirir (2 satiri gecmez)."""
    groups: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        if cur:
            gap = w.start - cur[-1].end
            last = cur[-1].text
            chars = len(" ".join(x.text for x in cur)) + 1 + len(w.text)
            split = (len(cur) >= max_words or chars > max_chars * 2 or gap >= PAUSE_SPLIT
                     or last.endswith(SENTENCE_END)
                     or (last.endswith(CLAUSE_END) and len(cur) >= min_words + 1))
            if split:
                groups.append(cur)
                cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    groups = _merge_singles(groups, min_words, max_words, max_chars)

    cues = [Cue(g[0].start, g[-1].end, g) for g in groups]
    for a, b in zip(cues, cues[1:]):
        a.end = b.start if b.start - a.end < BRIDGE_GAP else a.end + HOLD_AFTER
    if cues:
        cues[-1].end += HOLD_AFTER
    return cues


def _merge_singles(groups: list[list[Word]], min_words: int, max_words: int,
                   max_chars: int) -> list[list[Word]]:
    """Tek kelimelik gruplari, ritmi bozmuyorsa komsusuna ekler."""
    out: list[list[Word]] = []
    for g in groups:
        if out and (len(g) < min_words or len(out[-1]) < min_words):
            prev = out[-1]
            merged = prev + g
            text = " ".join(w.text for w in merged)
            ends_sentence = prev[-1].text.endswith(SENTENCE_END) and len(prev) >= min_words
            close = g[0].start - prev[-1].end < 0.8 and not ends_sentence
            if close and len(merged) <= max_words and len(text) <= max_chars * 2:
                out[-1] = merged
                continue
            if close and len(g) < min_words and len(prev) > min_words:
                # birlesemiyorsa dengele: onceki gruptan son kelimeyi al (7+1 -> 6+2)
                need = min_words - len(g)
                out[-1], g = prev[:-need], prev[-need:] + g
        out.append(g)
    return out


# ------------------------------------------------------------------ SRT

def _srt_time(t: float) -> str:
    ms = max(0, int(round(t * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(cues: list[Cue]) -> str:
    blocks = []
    for i, c in enumerate(cues, 1):
        blocks.append(f"{i}\n{_srt_time(c.start)} --> {_srt_time(c.end)}\n{c.text}\n")
    return "\n".join(blocks)


# ------------------------------------------------------------------ ASS

def _ass_time(t: float) -> str:
    cs = max(0, int(round(t * 100)))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    return text.replace("\\", "/").replace("{", "(").replace("}", ")")


def _line_break_index(words: list[Word], max_chars: int) -> int | None:
    """Uzun grubu iki dengeli satira boler; kelime indeksini dondurur."""
    text = " ".join(w.text for w in words)
    if len(text) <= max_chars or len(words) < 2:
        return None
    best, best_diff = None, 10**9
    for i in range(1, len(words)):
        a = len(" ".join(w.text for w in words[:i]))
        b = len(" ".join(w.text for w in words[i:]))
        if abs(a - b) < best_diff:
            best, best_diff = i, abs(a - b)
    return best


def _ass_line(words: list[Word], active: int | None, brk: int | None) -> str:
    parts = []
    for i, w in enumerate(words):
        sep = "" if i == 0 else ("\\N" if i == brk else " ")
        t = _ass_escape(w.text)
        if i == active:
            t = f"{{\\c{YELLOW}}}{t}{{\\c{WHITE}}}"
        parts.append(sep + t)
    return "".join(parts)


def to_ass(cues: list[Cue], width: int, height: int, font: str = "Arial Black",
           font_size: int = 86, bottom_margin_ratio: float = 0.28,
           highlight: bool = True, max_chars: int = 24) -> str:
    k = width / 1080.0
    size = round(font_size * k)
    outline = max(2, round(6 * k))
    shadow = max(1, round(2 * k))
    side = round(90 * k)
    margin_v = round(height * bottom_margin_ratio)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Short,{font},{size},&H00FFFFFF,&H0000FFFF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},2,{side},{side},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for c in cues:
        brk = _line_break_index(c.words, max_chars)
        if not highlight:
            lines.append(_dialogue(c.start, c.end, _ass_line(c.words, None, brk)))
            continue
        for i, w in enumerate(c.words):
            s = c.start if i == 0 else w.start
            e = c.words[i + 1].start if i + 1 < len(c.words) else c.end
            if e - s < 0.01:
                continue
            lines.append(_dialogue(s, e, _ass_line(c.words, i, brk)))
    return header + "\n".join(lines) + "\n"


def _dialogue(s: float, e: float, text: str) -> str:
    return f"Dialogue: 0,{_ass_time(s)},{_ass_time(e)},Short,,0,0,0,,{text}"


def write_transcript(path: Path, sentences: list[tuple[float, float, str]], language: str) -> None:
    body = "\n".join(t for _s, _e, t in sentences)
    path.write_text(f"[language: {language}]\n{body}\n", encoding="utf-8")
