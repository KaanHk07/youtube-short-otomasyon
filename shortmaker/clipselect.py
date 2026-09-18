"""Akilli klip secimi: transkript segmentlerinden en degerli 30-60 sn dilimlerini cikarir.

Saf Python mantigi (bagimsiz, test edilebilir): girdi = zaman damgali konusma
segmentleri, cikti = sirali ClipProposal listesi. Dilimler cumle sinirlarinda
baslar/biter (konusma ortasindan kesme yok), sessizlik/dolgu/tekrar cezalandirilir,
dikkat cekici acilis ve ilginc icerik odullendirilir.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MIN_CLIP_SEC = 30.0
MAX_CLIP_SEC = 60.0

# ---------------------------------------------------------------- veri modeli

@dataclass
class Segment:
    """Zaman damgali konusma parcasi (transkript veya sessizlik tabanli)."""
    start: float
    end: float
    text: str = ""


@dataclass
class ClipProposal:
    """Secilmis klip: zaman kodlari + rapor alanlari."""
    start: float
    end: float
    duration: float
    title: str
    why: list[str]
    main_idea: str
    hook: str
    emphasis_words: list[str]
    text: str = ""
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def timecode_start(self) -> str:
        return _fmt_tc(self.start)

    def timecode_end(self) -> str:
        return _fmt_tc(self.end)

    def as_report(self) -> dict[str, object]:
        """Kullanici istenen rapor formati (8 alan)."""
        return {
            "baslangic": self.timecode_start(),
            "bitis": self.timecode_end(),
            "sure": _fmt_dur(self.duration),
            "baslik": self.title,
            "neden": self.why,
            "ana_fikir": self.main_idea,
            "hook": self.hook,
            "altyazi_vurgulari": self.emphasis_words,
        }


# ---------------------------------------------------------------- kelime setleri

# Dolgu/tereddut kelimeleri (TR + EN): klipte olmasi cezalandirilir.
FILLERS = {
    "ee", "e", "ııı", "ıı", "ı", "eee", "eh", "hmm", "mmm", "ah", "uh", "um", "er",
    "yani", "şey", "gibi", "falan", "filan", "aynen", "işte", "bak", "bakın",
    "you know", "like", "actually", "basically", "i mean", "sort of", "kind of",
}

_SN = {"a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "for", "with",
       "at", "by", "from", "is", "are", "was", "were", "be", "been", "it", "this",
       "that", "i", "you", "he", "she", "we", "they", "my", "your", "his", "her",
       "our", "their", "as", "so", "if", "then", "than", "not", "no", "do", "does",
       "did", "have", "has", "had", "will", "would", "can", "could", "should",
       "bir", "bu", "şu", "o", "ve", "ile", "için", "ama", "fakat", "ancak", "çünkü",
       "gibi", "kadar", "sonra", "önce", "ben", "sen", "o", "biz", "siz", "onlar",
       "benim", "senin", "bizim", "sizin", "mi", "mu", "mı", "de", "da", "daha",
       "en", "çok", "az", "her", "hiç", "hep", "bütün", "tüm", "kendi", "olan",
       "oldu", "olur", "olmak", "var", "yok", "dedi", "dedim", "diyor", "geldi",
       "gidiyor", "biliyor", "bakalım", "hadi", "artık"}

# Ilgi/duygu cekici icerik kelimeleri (TR + EN): klipte olmasi odullendirilir.
INTEREST = {
    "rekor", "milyon", "milyar", "asla", "ilk", "son", "tehlikeli", "yasak", "sır",
    "para", "aşk", "ölüm", "korku", "korkunç", "komik", "patlama", "şok", "hile",
    "ücretsiz", "bedava", "kazan", "kaybet", "kaybetti", "ağladı", "ağlıyor",
    "inanılmaz", "imkansız", "gerçek", "yanlış", "önemli", "dikkat", "sakın",
    "en iyi", "en kötü", "tarihi", "bebek", "anne", "baba", "çocuk", "kalp",
    "dünya", "uzay", "robot", "yapay zeka", "vergi", "maaş", "fiyat", "zam",
    "indirim", "fırsat", "yöntem", "iptal", "rezalet", "skandal", "viral", "trend",
    "moda", "sahte", "gizli", "yeni", "rekor", "anı", "hayat", "ölüm", "savaş",
    # yarisma/mucadele/kurtarma/macera turu (TR)
    "ödül", "yarışma", "mücadele", "kurtarma", "takım", "kazanan", "final",
    "hayatta kalma", "ada", "macera", "yarış", "rakip", "helikopter", "oy",
    # yarisma/mucadele/kurtarma/macera turu (EN)
    "record", "million", "billion", "never", "first", "last", "danger", "secret",
    "money", "love", "death", "fear", "funny", "explosion", "shock", "hack", "free",
    "win", "lost", "cried", "crying", "crazy", "amazing", "impossible", "true",
    "false", "important", "attention", "careful", "best", "worst", "history",
    "baby", "mom", "dad", "child", "world", "space", "robot", "ai", "artificial",
    "price", "tax", "salary", "discount", "deal", "fake", "real", "viral", "trend",
    "scandal", "ban", "illegal", "murder", "survive", "new", "hidden", "life",
    "war", "secret",
    "prize", "challenge", "rescue", "team", "winner", "reward", "race", "survival",
    "helicopter", "extreme", "adventure", "island", "ocean", "final", "eliminated",
    "jackpot", "quest", "opponent", "vote", "competition", "compete", "finish",
    "alliance", "immunity", "enemies", "tribe",
}

_SENT_RE = re.compile(r".*?[.!?…]+[\"'”’)]*")  # noktalama dahil cumle yakala


# ---------------------------------------------------------------- yardimcilar

def _fmt_tc(sec: float) -> str:
    """Saniyeyi zaman koduna cevirir: 75.3 -> '1:15.3' (>=1sa: '1:02:03.4')."""
    s = max(0.0, sec)
    m, ss = divmod(int(s * 10), 600)
    h, mm = divmod(m, 60)
    frac = int(round((s * 10) % 10))
    if h:
        return f"{h}:{mm:02d}:{ss // 10:02d}.{frac}"
    return f"{mm}:{ss // 10:02d}.{frac}"


def _fmt_dur(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    return f"{m}:{s:02d}"


def split_sentences(segments: list[Segment]) -> list[Segment]:
    """Segmentleri noktalama sinirlarindan cumlelere boler (zaman korunur).

    Noktalama isaretleri cumle metninde KALIR (".", "?", "!" dahil) —
    altyazi/hook icin gerekli, zaman oranlamasi da dogru olur. Noktalama
    yoksa segment kendi cumlesi sayilir. Bitis zamani bir sonraki segmentin
    baslangicina kadar uzatilir (sessizlik dahil).
    """
    out: list[Segment] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            # sessizlik parcasi: ayri cumle sayilir, secimde engelleyici
            out.append(Segment(seg.start, seg.end, ""))
            continue
        parts = _split_punct(text)
        if not parts:
            out.append(Segment(seg.start, seg.end, text))
            continue
        # her cumlenin yaklasik baslangicini metin oranina gore dagit
        total = len(text)
        span = max(0.0, seg.end - seg.start)
        cursor = seg.start
        for p in parts:
            frac = len(p) / total if total else 0
            s = cursor
            e = s + max(0.0, span * frac)
            out.append(Segment(s, e, p))
            cursor = e
    # bosluklari kapat: cumle bitisini sonraki baslangica uzat
    for i in range(len(out) - 1):
        if out[i].end < out[i + 1].start and not out[i].text.strip():
            pass
    return out


def _split_punct(text: str) -> list[str]:
    """Metni noktalama dahil cumlelere boler; noktalamasiz kuyruk korunur."""
    parts: list[str] = []
    pos = 0
    for m in _SENT_RE.finditer(text):
        parts.append(text[pos:m.end()].strip())
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        if parts:
            parts[-1] = f"{parts[-1]} {tail}"
        else:
            parts.append(tail)
    return [p for p in parts if p]


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9']+", text.lower())]


def _content_words(text: str) -> list[str]:
    return [w for w in _words(text) if w not in _SN and len(w) > 2]


def _interest_hits(words: list[str]) -> list[str]:
    t = " ".join(words)
    return [k for k in INTEREST if k in t]


def _filler_count(text: str) -> int:
    low = " " + text.lower() + " "
    return sum(low.count(f" {f} ") for f in FILLERS)


def _ends_terminal(text: str) -> bool:
    return bool(re.search(r"[.!?…]\s*$", text.strip()))


def _has_number(text: str) -> bool:
    return bool(re.search(r"\d", text))


# ---------------------------------------------------------------- klip adaylari

def _build_windows(units: list[Segment], total: float,
                   min_sec: float = MIN_CLIP_SEC,
                   max_sec: float = MAX_CLIP_SEC) -> list[tuple[int, int, float]]:
    """Cumle sinirlarina hizali, [min_sec, max_sec] araliginda pencere indeksleri.

    (bas, bitis_haric, sure) seklinde; bitis_haric dahil degildir. Her baslangic
    icin uygun TUM bitis sinirlari uretilir: en guclu 30-60sn dilimi, maksimal
    pencerenin icinde kalabilecek zayif icerigi disari birakabilir. 60sn'yi tek
    basina asan uzun cumle atlanir (konusma ortasindan kesme kurali).
    """
    windows: list[tuple[int, int, float]] = []
    n = len(units)
    for i in range(n):
        if not units[i].text.strip():
            # sessizlikten baslama (dikkat cekici acilis icin)
            continue
        # tek cumle 60sn'yi asiyorsa kesmek yasak: bu baslangici atla
        if units[i].end - units[i].start > max_sec:
            continue
        for j in range(i + 1, n + 1):
            end = units[j - 1].end
            wdur = end - units[i].start
            if wdur > max_sec:
                break  # daha uzun pencereler de tasar, bu baslangic icin bitti
            if wdur >= min_sec:
                windows.append((i, j, wdur))
    return windows


def _window_score(units: list[Segment], i: int, j: int, selected: list[ClipProposal]) -> tuple[float, list[str]]:
    """Pencereyi puanlar: hook, icerik, yogunluk, dolgu, tekrar, kapanis."""
    win = units[i:j]
    text = " ".join(s.text for s in win).strip()
    words = _content_words(text)
    total_dur = win[-1].end - win[0].start
    speech = sum(max(0.0, s.end - s.start) for s in win if s.text.strip())
    density = speech / total_dur if total_dur else 0.0

    score = 0.0
    reasons: list[str] = []

    # --- hook (ilk ~3 sn): soru/ünlem/sayi/ilginc kelime ---
    hook_text = " ".join(s.text for s in win[:2] if s.text.strip()).strip()
    if re.search(r"\?\s*$", hook_text) or "?" in hook_text:
        score += 3.0
        reasons.append("Acik soruyla basliyor - merak uyandirir")
    if re.search(r"!\s*$", hook_text) or "!" in hook_text:
        score += 2.0
        reasons.append("Guclu unlem/duygu acilisi")
    if _has_number(hook_text):
        score += 2.0
        reasons.append("Sayi/istatistik iceren acilis")
    hits = _interest_hits(_words(hook_text))
    if hits:
        score += min(3.0, 1.0 * len(hits))
        reasons.append(f"Dikkat cekici acilis: {', '.join(sorted(set(hits))[:3])}")

    # --- icerik ilgincligi ---
    ihits = _interest_hits(words)
    if ihits:
        score += min(4.0, 0.5 * len(ihits))

    # --- yogunluk (sessizlik cezasi) ---
    if density >= 0.7:
        score += 2.0
        reasons.append("Yuksek konusma yogunlugu - gereksiz sessizlik yok")
    elif density >= 0.5:
        score += 1.0
    else:
        score -= 2.0
        reasons.append("Dusuk konusma yogunlugu (uzun sessizlikler)")

    # --- dolgu kelime cezasi ---
    fills = _filler_count(text)
    if fills:
        score -= min(3.0, 0.5 * fills)
        if fills >= 4:
            reasons.append("Cok dolgu kelimesi iceriyor")

    # --- kapanis tamamlanmislik ---
    last_txt = win[-1].text.strip()
    if last_txt and _ends_terminal(last_txt):
        score += 1.0
        reasons.append("Tamamlanmis cumle ile kapaniyor")

    # --- tekrar cezasi (onceki secimlerle metin ortusmesi) ---
    prior = " ".join(c.text for c in selected).lower()
    if prior:
        overlap = sum(1 for w in set(words) if f" {w} " in f" {prior} ")
        if overlap:
            score -= min(2.0, 0.25 * overlap)

    return score, reasons


def select_clips(segments: list[Segment], total_duration: float,
                 min_sec: float = MIN_CLIP_SEC, max_sec: float = MAX_CLIP_SEC,
                 max_clips: int = 5) -> list[ClipProposal]:
    """Transkriptten en degerli, ortusmeyen klipleri secer (acgozlu).

    Kural tabanli: cumle siniri, 30-60sn, hook/icerik odulu, sessizlik/dolgu/
    tekrar cezasi. Ayni konuyu tekrar eden ortusen klipler elenir.
    """
    units = split_sentences(segments)
    windows = _build_windows(units, total_duration, min_sec, max_sec)
    proposals: list[ClipProposal] = []
    used_ranges: list[tuple[float, float]] = []

    while len(proposals) < max_clips and windows:
        best: tuple[float, int, int, list[str]] | None = None
        for (i, j, dur) in windows:
            s, e = units[i].start, units[j - 1].end
            # onceki kliplerle >%40 ortuse seci
            if any(max(0.0, min(e, ur[1]) - max(s, ur[0]))
                   > 0.4 * (e - s) for ur in used_ranges):
                continue
            score, reasons = _window_score(units, i, j, proposals)
            cand = (score, i, j, reasons)
            if best is None or cand[0] > best[0]:
                best = cand
        if best is None:
            break
        _, i, j, reasons = best
        win = units[i:j]
        start, end = win[0].start, win[-1].end
        text = " ".join(s.text for s in win).strip()
        prop = _proposal_from_window(win, start, end, text, reasons)
        prop.score = _
        proposals.append(prop)
        used_ranges.append((start, end))
        # bu pencereyi tekrar secme; kismi ortusenleri de sil
        windows = [w for w in windows
                   if not (w[0] <= i < w[1] or w[0] <= j - 1 < w[1])]

    # degerine gore sirala: en guclu klip once (kullanici kurali: guc onceligi)
    proposals.sort(key=lambda p: p.score, reverse=True)
    return proposals[:max_clips]


def _proposal_from_window(win: list[Segment], start: float, end: float,
                          text: str, reasons: list[str]) -> ClipProposal:
    first = next((s.text.strip() for s in win if s.text.strip()), "")
    hook = _shorten(first, 12)
    main = _shorten(first, 22)
    title = _make_title(win, first)
    why = reasons or ["Guclu, tamamlanmis bir bolum"]
    emphasis = _emphasis_words(win)
    return ClipProposal(
        start=start, end=end, duration=end - start,
        title=title, why=why, main_idea=main, hook=hook,
        emphasis_words=emphasis, text=text, reasons=reasons)


def _shorten(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def _make_title(win: list[Segment], first: str) -> str:
    """Baslik: ilk cumlenin ozu veya en guclu ilginc kelime."""
    strong = _interest_hits(_words(first))
    if strong:
        return strong[0].capitalize()
    t = _shorten(first, 6)
    return t if len(t.split()) >= 2 else ("Ilginc bir bolum" if not t else t)


def _emphasis_words(win: list[Segment]) -> list[str]:
    """Altyazida vurgulanacak kelimeler: pencerede sik gecen icerik kelimeleri."""
    all_words = _content_words(" ".join(s.text for s in win))
    freq: dict[str, int] = {}
    for w in all_words:
        freq[w] = freq.get(w, 0) + 1
    # once ilginc/duygu kelimeleri, sonra siklik
    ranked = sorted(freq.items(), key=lambda kv: (kv[0] in INTEREST, kv[1]), reverse=True)
    out: list[str] = []
    for w, c in ranked:
        if len(out) >= 5:
            break
        if c >= 2 or w in INTEREST:
            out.append(w)
    return out