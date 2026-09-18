"""ShortMaker ayarlari: shortmaker.yaml + guvenli varsayilanlar."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict[str, Any] = {
    "output_dir": "output",
    "work_dir": ".temp_files/shortmaker",
    "video": {"duration": 60, "resolution": "1080x1920", "max_fps": 60,
              "fallback_fps": 30, "crf": 20, "preset": "medium"},
    "subtitles": {"enabled": True, "highlight_active_word": True, "min_words": 2,
                  "max_words": 7, "max_chars": 24, "font": "Arial Black",
                  "font_size": 86, "bottom_margin_ratio": 0.28},
    "crop": {"smart": True, "sample_fps": 6},
    "silence": {"remove": True, "min_silence_sec": 0.7, "keep_padding_sec": 0.15},
    "whisper": {"model": "small", "device": "auto"},
    "rights": {"default": "own", "allowed": ["own", "licensed"]},
}

ALLOWED_DURATIONS = (30, 45, 60)
ALLOWED_RESOLUTIONS = ("1080x1920", "720x1280")


class SettingsError(ValueError):
    pass


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Settings:
    data: dict[str, Any]
    root: Path = ROOT

    def section(self, name: str) -> dict[str, Any]:
        return self.data[name]

    @property
    def duration(self) -> int:
        return int(self.data["video"]["duration"])

    @property
    def size(self) -> tuple[int, int]:
        w, h = str(self.data["video"]["resolution"]).lower().split("x")
        return int(w), int(h)

    @property
    def output_dir(self) -> Path:
        return self._abs(self.data["output_dir"])

    @property
    def work_dir(self) -> Path:
        return self._abs(self.data["work_dir"])

    def _abs(self, raw: str) -> Path:
        p = Path(raw)
        return p if p.is_absolute() else self.root / p

    def validate(self) -> None:
        if self.duration not in ALLOWED_DURATIONS:
            raise SettingsError(f"video.duration 30/45/60 olmali (su an {self.duration})")
        if str(self.data["video"]["resolution"]) not in ALLOWED_RESOLUTIONS:
            raise SettingsError("video.resolution 1080x1920 veya 720x1280 olmali")


def load_settings(path: Path | None = None, overrides: dict[str, Any] | None = None) -> Settings:
    path = path or ROOT / "shortmaker.yaml"
    raw: dict[str, Any] = {}
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    data = _merge(DEFAULTS, raw)
    if overrides:
        data = _merge(data, overrides)
    s = Settings(data, path.resolve().parent if path.exists() else ROOT)
    s.validate()
    return s
