"""faster-whisper ile kelime zamanli transkripsiyon.

Kural: task="transcribe" — ceviri YAPILMAZ. Dil once tespit edilir, sonra ayni
dil sabitlenerek yazıya dokulur; boylece altyazi kaynak videonun dilinde kalir.
GPU (CUDA) kullanilamazsa sessizce CPU'ya duser.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import MSG_SUBTITLE, ShortMakerError

log = logging.getLogger(__name__)


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    language: str
    language_probability: float
    words: list[Word]
    segments: list[tuple[float, float, str]]  # (start, end, cumle)
    device: str

    @property
    def text(self) -> str:
        return " ".join(s[2] for s in self.segments).strip()


def _add_nvidia_dll_dirs() -> None:
    """pip ile gelen nvidia-cublas/cudnn DLL'lerini Windows arama yoluna ekler."""
    if os.name != "nt":
        return
    for base in map(Path, sys.path):
        nv = base / "nvidia"
        if not nv.is_dir():
            continue
        for bin_dir in nv.glob("*/bin"):
            try:
                os.add_dll_directory(str(bin_dir))
            except OSError:
                pass
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


_models: dict[tuple[str, str], Any] = {}


def load_model(name: str, device: str) -> Any:
    from faster_whisper import WhisperModel
    key = (name, device)
    if key not in _models:
        compute = "float16" if device == "cuda" else "int8"
        log.info("Whisper modeli yukleniyor: %s (%s/%s)", name, device, compute)
        _models[key] = WhisperModel(name, device=device, compute_type=compute)
    return _models[key]


def _devices(pref: str) -> list[str]:
    if pref == "cpu":
        return ["cpu"]
    _add_nvidia_dll_dirs()
    if pref == "cuda" or (pref == "auto" and cuda_available()):
        return ["cuda", "cpu"]
    return ["cpu"]


def transcribe(audio_path: Path, model_name: str = "small", device: str = "auto") -> Transcript:
    """Sesi tespit edilen kendi dilinde, kelime zamanlariyla yaziya doker."""
    try:
        from faster_whisper import decode_audio
    except ImportError as exc:
        raise ShortMakerError(MSG_SUBTITLE, "faster-whisper kurulu degil: install.bat") from exc

    from .language_detector import detect_language

    audio = decode_audio(str(audio_path), sampling_rate=16000)
    last_exc: Exception | None = None
    for dev in _devices(device):
        try:
            model = load_model(model_name, dev)
            lang, prob = detect_language(model, audio)
            seg_iter, _info = model.transcribe(
                audio, language=lang, task="transcribe", word_timestamps=True,
                vad_filter=True, vad_parameters={"min_silence_duration_ms": 400},
                beam_size=5, condition_on_previous_text=False,
            )
            words: list[Word] = []
            segments: list[tuple[float, float, str]] = []
            for seg in seg_iter:  # jenerator: hata burada da cikabilir
                text = seg.text.strip()
                if text:
                    segments.append((float(seg.start), float(seg.end), text))
                for w in seg.words or []:
                    t = w.word.strip()
                    if t:
                        words.append(Word(float(w.start), float(w.end), t))
            return Transcript(lang, prob, words, segments, dev)
        except ShortMakerError:
            raise
        except Exception as exc:
            last_exc = exc
            log.warning("Transkripsiyon %s uzerinde basarisiz: %s", dev, exc)
            _models.pop((model_name, dev), None)
    raise ShortMakerError(MSG_SUBTITLE, f"{type(last_exc).__name__}: {last_exc}")
