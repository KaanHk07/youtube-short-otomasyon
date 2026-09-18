"""Konusma dili tespiti (Whisper). Birden cok 30 sn pencereye bakarak karar verir."""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# ISO 639-1 -> okunur ad (info.json ve konsol icin; digerleri kodla gosterilir)
LANGUAGE_NAMES = {
    "tr": "Türkçe", "en": "English", "es": "Español", "de": "Deutsch", "fr": "Français",
    "it": "Italiano", "pt": "Português", "ru": "Русский", "ar": "العربية", "ja": "日本語",
    "ko": "한국어", "zh": "中文", "hi": "हिन्दी", "nl": "Nederlands", "pl": "Polski",
    "az": "Azərbaycanca", "fa": "فارسی", "uk": "Українська", "id": "Bahasa Indonesia",
}


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def detect_language(model: Any, audio: Any) -> tuple[str, float]:
    """(dil_kodu, olasilik). Konusma olan bolumlerden (VAD) en fazla 4 pencereye bakar."""
    lang, prob, _all = model.detect_language(
        audio, vad_filter=True, language_detection_segments=4,
        language_detection_threshold=0.6,
    )
    log.info("Tespit edilen dil: %s (%.0f%%)", lang, prob * 100)
    return lang, float(prob)
