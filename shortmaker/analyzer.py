"""Video analizi: sure, cozunurluk, FPS, ses durumu (ffprobe + volumedetect)."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from .errors import MSG_PROCESS, ShortMakerError
from .ffmpeg_utils import FFmpegError, probe, run_ffmpeg

SILENT_DB = -60.0  # bu ortalamanin altindaki ses "yok" sayilir


@dataclass
class MediaInfo:
    duration: float
    width: int           # dondurme (rotation) uygulanmis goruntu genisligi
    height: int
    fps: float
    variable_fps: bool
    has_audio: bool
    audio_mean_db: float | None
    video_codec: str
    audio_codec: str

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["resolution"] = self.resolution
        return d


def _fraction(text: str | None) -> float:
    try:
        f = Fraction(text or "0")
        return float(f) if f.denominator else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def _rotation(stream: dict[str, Any]) -> int:
    rot = stream.get("tags", {}).get("rotate")
    for side in stream.get("side_data_list", []) or []:
        if "rotation" in side:
            rot = side["rotation"]
    try:
        return abs(int(float(rot or 0))) % 180
    except ValueError:
        return 0


def mean_volume_db(path: Path) -> float | None:
    """Ses akisinin ortalama seviyesi (dB). Ses yoksa None."""
    try:
        err = run_ffmpeg(["-i", str(path), "-map", "0:a:0", "-af", "volumedetect",
                          "-vn", "-f", "null", "-"], timeout=900)
    except FFmpegError:
        return None
    m = re.search(r"mean_volume:\s*(-?[\d.]+|-inf) dB", err)
    if not m:
        return None
    return -120.0 if m.group(1) == "-inf" else float(m.group(1))


def analyze(path: Path) -> MediaInfo:
    try:
        data = probe(path)
    except FFmpegError as exc:
        raise ShortMakerError(MSG_PROCESS, f"ffprobe okuyamadi: {exc}") from exc
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    if video is None:
        raise ShortMakerError(MSG_PROCESS, "dosyada goruntu akisi yok")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    w, h = int(video.get("width") or 0), int(video.get("height") or 0)
    if _rotation(video) == 90:
        w, h = h, w
    r_fps = _fraction(video.get("r_frame_rate"))
    avg_fps = _fraction(video.get("avg_frame_rate"))
    fps = avg_fps or r_fps
    vfr = bool(r_fps and avg_fps and abs(r_fps - avg_fps) / r_fps > 0.02)

    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0.0)
    mean_db = mean_volume_db(path) if audio else None
    has_audio = audio is not None and mean_db is not None and mean_db > SILENT_DB

    return MediaInfo(duration=duration, width=w, height=h, fps=round(fps, 3),
                     variable_fps=vfr, has_audio=has_audio, audio_mean_db=mean_db,
                     video_codec=video.get("codec_name", ""),
                     audio_codec=(audio or {}).get("codec_name", ""))


def choose_fps(info: MediaInfo, max_fps: float, fallback: float) -> float:
    """Kaynak FPS'i korur; degisken, bilinmeyen veya cok yuksekse fallback (30)."""
    if info.variable_fps or not info.fps or info.fps > max_fps + 0.01 or info.fps < 15:
        return float(fallback)
    return info.fps
