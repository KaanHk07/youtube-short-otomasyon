"""ffmpeg / ffprobe cagrilari icin ince sarmalayicilar."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


class FFmpegError(RuntimeError):
    pass


def ffmpeg_bin() -> str:
    return shutil.which("ffmpeg") or "ffmpeg"


def ffprobe_bin() -> str:
    return shutil.which("ffprobe") or "ffprobe"


def run_ffmpeg(args: list[str], timeout: float = 3600) -> str:
    """ffmpeg'i calistirir; basarisizlikta stderr sonunu iceren FFmpegError firlatir."""
    cmd = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)
    if proc.returncode != 0:
        raise FFmpegError(proc.stderr[-1500:])
    return proc.stderr


def probe(path: Path) -> dict[str, Any]:
    cmd = [ffprobe_bin(), "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=120)
    if proc.returncode != 0:
        raise FFmpegError(proc.stderr[-800:])
    return json.loads(proc.stdout or "{}")


def ffmpeg_path_arg(path: Path) -> str:
    """Filtre grafigi icindeki Windows yolu (ass=...) icin kacis: C\:/a/b.ass"""
    s = str(path.resolve()).replace("\\", "/")
    return s.replace(":", "\:").replace("'", "\'")
