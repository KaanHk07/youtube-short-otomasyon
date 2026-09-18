"""URL -> Shorts orkestrasyonu. Her URL bagimsizdir: biri basarisiz olursa digerleri surer."""
from __future__ import annotations

import json
import logging
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import analyzer, downloader, segment_selector, silence, smart_crop, subtitle_generator
from .config import Settings
from .errors import MSG_NO_AUDIO, MSG_PROCESS, MSG_RIGHTS, MSG_SUBTITLE, ShortMakerError
from .ffmpeg_utils import probe
from .language_detector import language_name
from .transcriber import Transcript, transcribe
from .video_processor import render_final, render_intermediate

log = logging.getLogger(__name__)

Progress = Callable[[str], None]


@dataclass
class Result:
    url: str
    ok: bool
    out_dir: Path | None = None
    message: str = ""
    detail: str = ""
    warnings: list[str] = field(default_factory=list)


def _short_detail(detail: str) -> str:
    """yt-dlp/ffmpeg ayrintisini tek okunur satira indirir."""
    lines = (detail or "").strip().splitlines()
    text = re.sub(r"^ERROR:\s*(\[[^\]]+\]\s*)?", "", lines[-1] if lines else "")
    text = re.sub(r"^[\w-]+:\s+(?=[A-Z])", "", text)   # "sayfa: Unable..." -> "Unable..."
    text = re.sub(r"\s*\(caused by .*\)$", "", text)
    return text[:160]


def next_output_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    used = [int(m.group(1)) for p in root.iterdir()
            if p.is_dir() and (m := re.fullmatch(r"video_(\d+)", p.name))]
    return root / f"video_{(max(used) + 1 if used else 1):03d}"


@dataclass
class Source:
    """Bir URL icin bir kez hazirlanan veri; her klip bunu paylasir."""
    url: str
    rights: str
    path: Path
    title: str
    uploader: str
    info: analyzer.MediaInfo
    fps: float
    transcript: Transcript | None
    warnings: list[str]

    @property
    def lang(self) -> str:  # konusma yoksa dil yok
        return self.transcript.language if self.transcript and self.transcript.words else ""


def process_url(url: str, st: Settings, out_dir: Path | None = None, rights: str | None = None,
                progress: Progress = print, clips: int = 1) -> list[Result]:
    """URL'den `clips` adet Short uretir; her klip icin bir Result dondurur. Istisna firlatmaz.

    out_dir verilirse: tek klipte o klasor, coklu klipte out_dir/clip_1, clip_2 ...
    verilmezse her klip icin output/video_NNN.
    """
    rights = rights or st.section("rights")["default"]
    if rights not in st.section("rights")["allowed"]:
        return [Result(url=url, ok=False, message=MSG_RIGHTS)]
    work = st.work_dir / f"job_{int(time.time() * 1000)}"
    results: list[Result] = []
    try:
        work.mkdir(parents=True, exist_ok=True)
        try:
            src = _prepare(url, st, work, rights, progress)
        except ShortMakerError as exc:
            log.error("%s -> %s", url, exc)
            return [Result(url, False, message=exc.message, detail=_short_detail(exc.detail))]
        except Exception as exc:  # beklenmeyen hata: toplu is durmasin
            log.exception("%s -> beklenmeyen hata: %s", url, exc)
            return [Result(url, False, message=MSG_PROCESS)]

        selections = segment_selector.select_segments(src.info.duration, float(st.duration),
                                                      src.transcript, clips)
        if len(selections) < clips:
            progress(f"   ! Video {clips} ayrı klip için yeterince uzun/konuşmalı değil; "
                     f"{len(selections)} klip üretilecek.")
        for k, sel in enumerate(selections, 1):
            if len(selections) > 1:
                progress(f"-- Klip {k}/{len(selections)}")
            if out_dir is None:
                target_dir, auto = next_output_dir(st.output_dir), True
            else:
                target_dir, auto = (out_dir / f"clip_{k}" if len(selections) > 1 else out_dir), False
            results.append(_render_clip(src, sel, st, work / f"clip_{k}", target_dir, auto,
                                        progress, k, len(selections)))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return results


def _prepare(url: str, st: Settings, work: Path, rights: str, progress: Progress) -> Source:
    vcfg, wcfg = st.section("video"), st.section("whisper")
    warnings: list[str] = []

    progress("Video indiriliyor...")
    dl = downloader.download(url, work)

    progress("Ses analiz ediliyor...")
    info = analyzer.analyze(dl.path)
    fps = analyzer.choose_fps(info, float(vcfg["max_fps"]), float(vcfg["fallback_fps"]))
    progress(f"   süre {info.duration:.1f} sn · {info.resolution} · {info.fps:g} FPS · "
             f"ses: {'var' if info.has_audio else 'yok'}")
    if not info.has_audio:
        warnings.append(MSG_NO_AUDIO)
        progress(f"   ! {MSG_NO_AUDIO} Altyazısız devam ediliyor.")

    transcript: Transcript | None = None
    if info.has_audio:
        progress("Konuşma dili belirleniyor...")
        try:
            transcript = transcribe(dl.path, str(wcfg["model"]), str(wcfg["device"]))
            if transcript.words:
                progress(f"   dil: {language_name(transcript.language)} ({transcript.language}, "
                         f"%{transcript.language_probability * 100:.0f}) · {len(transcript.words)} kelime")
            else:
                progress("   videoda konuşma bulunamadı; altyazı eklenmeyecek")
        except ShortMakerError as exc:
            warnings.append(MSG_SUBTITLE)
            progress(f"   ! {MSG_SUBTITLE} ({exc.detail[:160]})")
    return Source(url, rights, dl.path, dl.title, dl.uploader, info, fps, transcript, warnings)


def _render_clip(src: Source, sel: segment_selector.Selection, st: Settings, work: Path,
                 out_dir: Path, auto_dir: bool, progress: Progress, index: int, total: int) -> Result:
    res = Result(url=src.url, ok=False, out_dir=out_dir, warnings=list(src.warnings))
    try:
        work.mkdir(parents=True, exist_ok=True)
        _build_clip(src, sel, st, work, out_dir, progress, res, index, total)
        res.ok = True
        res.message = f"Hazır: {out_dir / 'short.mp4'}"
    except ShortMakerError as exc:
        res.message = exc.message
        res.detail = _short_detail(exc.detail)
        log.error("%s (klip %d) -> %s", src.url, index, exc)
    except Exception as exc:
        res.message = MSG_PROCESS
        log.exception("%s (klip %d) -> beklenmeyen hata: %s", src.url, index, exc)
    finally:
        if not res.ok and auto_dir and out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)  # yarim klasor birakma
    progress(("   ✔ " if res.ok else "   ✖ ") + res.message)
    return res


def _build_clip(src: Source, sel: segment_selector.Selection, st: Settings, work: Path,
                out_dir: Path, progress: Progress, res: Result, index: int, total: int) -> None:
    vcfg, scfg, ccfg, qcfg = (st.section(k) for k in ("video", "subtitles", "crop", "silence"))
    target = float(st.duration)
    out_w, out_h = st.size
    info, fps, transcript, lang = src.info, src.fps, src.transcript, src.lang
    progress(f"   bölüm: {sel.start:.2f}–{sel.end:.2f} sn ({sel.duration:.1f} sn) · {sel.reason}")

    words = transcript.words if transcript else []
    if qcfg["remove"] and info.has_audio:
        keeps = silence.plan_cuts(src.path, words, sel.start, sel.end,
                                  float(qcfg["min_silence_sec"]), float(qcfg["keep_padding_sec"]))
    else:
        keeps = [(sel.start, sel.end)]
    removed = sel.duration - silence.total(keeps)
    if removed > 0.05:
        progress(f"   sessizlik: {removed:.1f} sn kaldırıldı ({len(keeps) - 1} kesim)")
    clip_words = silence.remap_words(words, keeps)

    out_dir.mkdir(parents=True, exist_ok=True)
    progress("Altyazılar hazırlanıyor...")
    ass_path: Path | None = None
    cues = subtitle_generator.group_words(clip_words, int(scfg["min_words"]), int(scfg["max_words"]),
                                          int(scfg["max_chars"])) if clip_words else []
    (out_dir / "subtitles.srt").write_text(subtitle_generator.to_srt(cues), encoding="utf-8")
    sentences = [s for s in (transcript.segments if transcript else [])
                 if s[1] > sel.start and s[0] < sel.end]
    subtitle_generator.write_transcript(out_dir / "transcript.txt", sentences, lang or "-")
    if cues and scfg["enabled"]:
        ass_path = work / "subs.ass"
        ass_path.write_text(subtitle_generator.to_ass(
            cues, out_w, out_h, font=str(scfg["font"]), font_size=int(scfg["font_size"]),
            bottom_margin_ratio=float(scfg["bottom_margin_ratio"]),
            highlight=bool(scfg["highlight_active_word"]), max_chars=int(scfg["max_chars"])),
            encoding="utf-8")
        progress(f"   {len(cues)} altyazı grubu ({lang})")

    progress("Shorts formatına dönüştürülüyor...")
    inter = render_intermediate(src.path, keeps, fps, info.has_audio, work / "cut.mkv")
    cw, ch = smart_crop.crop_size(info.width, info.height, out_w, out_h)
    plan = smart_crop.CropPlan("center", cw, ch, None)
    if smart_crop.needs_horizontal_crop(info.width, info.height, out_w, out_h):
        if ccfg["smart"]:
            try:
                plan = smart_crop.analyze_path(inter, fps, info.width, info.height, out_w, out_h,
                                               float(ccfg["sample_fps"]))
                progress(f"   akıllı crop: {'yüz takibi' if plan.faces_found else 'hareket takibi'}")
            except Exception as exc:
                log.warning("Akilli crop basarisiz, merkez crop: %s", exc)
                res.warnings.append("Akıllı crop uygulanamadı; merkez crop kullanıldı.")
    else:
        plan.mode = "scale"

    progress("Video kaydediliyor...")
    short = out_dir / "short.mp4"
    render_final(inter, plan, info.width, info.height, out_w, out_h, fps, info.has_audio,
                 ass_path, short, vcfg)
    final = _verify(short, out_w, out_h, target)

    meta = {
        "source_url": src.url,
        "source_title": src.title,
        "source_uploader": src.uploader,
        "rights": src.rights,
        "clip_index": index,
        "clip_count": total,
        "original_duration": round(info.duration, 2),
        "short_duration": round(final, 2),
        "detected_language": lang or None,
        "detected_language_name": language_name(lang) if lang else None,
        "language_probability": round(transcript.language_probability, 3) if lang else None,
        "resolution": f"{out_w}x{out_h}",
        "source_resolution": info.resolution,
        "fps": fps,
        "source_fps": info.fps,
        "video_codec": "h264",
        "audio_codec": "aac",
        "has_audio": info.has_audio,
        "segment": {"start": sel.start, "end": sel.end, "reason": sel.reason},
        "removed_silence_sec": round(max(0.0, removed), 2),
        "cuts": len(keeps) - 1,
        "crop_mode": plan.mode,
        "subtitles_burned": ass_path is not None,
        "subtitle_cues": len(cues),
        "whisper_device": transcript.device if transcript else None,
        "warnings": res.warnings,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "youtube_video_id": None,
    }
    (out_dir / "info.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _verify(path: Path, w: int, h: int, target: float) -> float:
    data = probe(path)
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    a = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})
    dur = float(data.get("format", {}).get("duration") or 0)
    problems = []
    if (v.get("width"), v.get("height")) != (w, h):
        problems.append(f"cozunurluk {v.get('width')}x{v.get('height')}")
    if v.get("codec_name") != "h264" or a.get("codec_name") != "aac":
        problems.append(f"codec {v.get('codec_name')}/{a.get('codec_name')}")
    if dur > target + 0.5 or dur <= 0:
        problems.append(f"sure {dur:.2f}")
    if problems:
        raise ShortMakerError(MSG_PROCESS, "cikti dogrulamasi: " + ", ".join(problems))
    return dur


def process_many(urls: list[str], st: Settings, rights: str | None = None,
                 out_dir: Path | None = None, progress: Progress = print,
                 clips: int = 1) -> list[Result]:
    results = []
    for i, url in enumerate(urls, 1):
        progress(f"\n[{i}/{len(urls)}] {url}")
        target = out_dir if (out_dir and len(urls) == 1) else None
        rs = process_url(url, st, out_dir=target, rights=rights, progress=progress, clips=clips)
        if len(rs) == 1 and not rs[0].ok and rs[0].out_dir is None:
            progress("   ✖ " + rs[0].message)  # hazirlik asamasinda durdu
        results.extend(rs)
    return results
