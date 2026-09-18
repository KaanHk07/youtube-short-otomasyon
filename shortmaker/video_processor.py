"""Render: (1) secilen bolum + sessizlik kesimleri -> ara video, (2) dikey kadraj +
altyazi + ses zinciri -> H.264/AAC MP4.

Ara video kaynak cozunurlukte, sabit FPS'te ve neredeyse kayipsizdir (CRF 12);
ses PCM tutulur, boylece AAC yalnizca bir kez kodlanir.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from .audio import filter_chain
from .errors import MSG_PROCESS, ShortMakerError
from .ffmpeg_utils import FFmpegError, ffmpeg_bin, ffmpeg_path_arg, run_ffmpeg
from .smart_crop import CropPlan

log = logging.getLogger(__name__)

FADE = 0.02  # kesim noktalarinda tik sesini onleyen mikro fade


def render_intermediate(src: Path, keeps: list[tuple[float, float]], fps: float,
                        has_audio: bool, out: Path) -> Path:
    """Tutulacak araliklari birlestirip sabit FPS'li ara dosya uretir."""
    base = max(0.0, keeps[0][0] - 1.0)           # hizli giris aramasi (-ss), sonra hassas trim
    end = keeps[-1][1] + 0.5
    parts, labels = [], []
    for i, (s, e) in enumerate(keeps):
        s0, e0 = s - base, e - base
        parts.append(f"[0:v]trim=start={s0:.3f}:end={e0:.3f},setpts=PTS-STARTPTS,fps={fps:g}[v{i}]")
        if has_audio:
            d = e - s
            fo = max(0.0, d - FADE)
            parts.append(f"[0:a]atrim=start={s0:.3f}:end={e0:.3f},asetpts=PTS-STARTPTS,"
                         f"afade=t=in:d={FADE},afade=t=out:st={fo:.3f}:d={FADE}[a{i}]")
            labels.append(f"[v{i}][a{i}]")
        else:
            labels.append(f"[v{i}]")
    n = len(keeps)
    parts.append(f"{''.join(labels)}concat=n={n}:v=1:a={1 if has_audio else 0}"
                 + ("[v][a]" if has_audio else "[v]"))
    graph = ";".join(parts)
    args = ["-ss", f"{base:.3f}", "-to", f"{end:.3f}", "-i", str(src),
            "-filter_complex", graph, "-map", "[v]"]
    if has_audio:
        args += ["-map", "[a]", "-c:a", "pcm_s16le", "-ar", "48000"]
    args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "12", "-pix_fmt", "yuv420p",
             "-r", f"{fps:g}", "-fps_mode", "cfr", str(out)]
    try:
        run_ffmpeg(args)
    except FFmpegError as exc:
        raise ShortMakerError(MSG_PROCESS, f"kesim/birlestirme: {exc}") from exc
    return out


def _video_filters(out_w: int, out_h: int, ass: Path | None) -> str:
    vf = f"setsar=1,format=yuv420p"
    if ass is not None:
        vf = f"ass='{ffmpeg_path_arg(ass)}'," + vf
    return vf


def _encode_args(cfg_video: dict, fps: float, has_audio: bool, normalize: bool) -> list[str]:
    args = ["-c:v", "libx264", "-preset", str(cfg_video.get("preset", "medium")),
            "-crf", str(cfg_video.get("crf", 20)), "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-r", f"{fps:g}", "-fps_mode", "cfr", "-g", str(int(round(fps * 2)))]
    args += ["-af", filter_chain(normalize) if has_audio else "aresample=48000",
             "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
             "-movflags", "+faststart", "-shortest"]
    return args


def render_final(inter: Path, plan: CropPlan, src_w: int, src_h: int, out_w: int, out_h: int,
                 fps: float, has_audio: bool, ass: Path | None, out: Path,
                 cfg_video: dict, normalize_audio: bool = True) -> Path:
    try:
        if plan.mode == "smart" and plan.xs is not None:
            _render_piped(inter, plan, out_w, out_h, fps, has_audio, ass, out, cfg_video, normalize_audio)
        else:
            cx = (src_w - plan.crop_w) // 2
            cy = (src_h - plan.crop_h) // 2
            vf = (f"crop={plan.crop_w}:{plan.crop_h}:{cx}:{cy},"
                  f"scale={out_w}:{out_h}:flags=lanczos," + _video_filters(out_w, out_h, ass))
            args = ["-i", str(inter)]
            if not has_audio:
                args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
            args += ["-map", "0:v:0", "-map", "0:a:0" if has_audio else "1:a:0", "-vf", vf]
            args += _encode_args(cfg_video, fps, has_audio, normalize_audio)
            run_ffmpeg(args + [str(out)])
    except FFmpegError as exc:
        raise ShortMakerError(MSG_PROCESS, f"son kodlama: {exc}") from exc
    return out


def _render_piped(inter: Path, plan: CropPlan, out_w: int, out_h: int, fps: float,
                  has_audio: bool, ass: Path | None, out: Path, cfg_video: dict,
                  normalize_audio: bool) -> None:
    """Kareleri OpenCV ile kirpip ffmpeg'e boru ile verir (kare basina farkli x)."""
    import cv2

    cmd = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", f"{fps:g}",
           "-i", "-"]
    cmd += ["-i", str(inter)] if has_audio else ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
    cmd += ["-map", "0:v:0", "-map", "1:a:0", "-vf", _video_filters(out_w, out_h, ass)]
    cmd += _encode_args(cfg_video, fps, has_audio, normalize_audio) + [str(out)]

    cap = cv2.VideoCapture(str(inter))
    if not cap.isOpened():
        raise FFmpegError(f"ara video acilamadi: {inter}")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    xs = plan.xs
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            x = int(xs[min(i, len(xs) - 1)])
            crop = frame[plan.y:plan.y + plan.crop_h, x:x + plan.crop_w]
            interp = cv2.INTER_AREA if plan.crop_w >= out_w else cv2.INTER_CUBIC
            proc.stdin.write(cv2.resize(crop, (out_w, out_h), interpolation=interp).tobytes())
            i += 1
    except (BrokenPipeError, OSError):
        pass
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except OSError:
            pass
    err = proc.stderr.read().decode("utf-8", "replace")
    if proc.wait() != 0 or i == 0:
        raise FFmpegError(err[-1500:] or "kare yazilamadi")
