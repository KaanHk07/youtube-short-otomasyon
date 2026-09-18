"""Bagimlilik kontrolu: ffmpeg/ffprobe (+libass), Python paketleri, GPU, yuz modeli."""
from __future__ import annotations

import importlib.util
import shutil
import subprocess

from .smart_crop import MODEL_PATH

PACKAGES = {
    "yt_dlp": "yt-dlp",
    "faster_whisper": "faster-whisper",
    "cv2": "opencv-python",
    "numpy": "numpy",
    "yaml": "PyYAML",
}


YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")


def setup_models(whisper_model: str = "small") -> None:
    """Yuz modeli ve Whisper modelini onceden indirir (ilk calistirma beklemesin)."""
    import urllib.request

    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(YUNET_URL, MODEL_PATH)
            print("  · Yüz modeli indirildi.")
        except Exception as exc:
            print(f"  ! Yüz modeli indirilemedi ({exc}); Haar cascade kullanılacak.")
    try:
        from faster_whisper.utils import download_model
        download_model(whisper_model)
        print(f"  · Whisper '{whisper_model}' modeli hazır.")
    except Exception as exc:
        print(f"  ! Whisper modeli indirilemedi ({exc}); ilk çalıştırmada tekrar denenecek.")


def check() -> tuple[list[str], list[str]]:
    """(hatalar, bilgiler). Hata listesi bossa sistem calismaya hazirdir."""
    errors: list[str] = []
    info: list[str] = []

    ff = shutil.which("ffmpeg")
    if not ff or not shutil.which("ffprobe"):
        errors.append("FFmpeg bulunamadı. Kurulum: winget install Gyan.FFmpeg  (sonra terminali yeniden açın)")
    else:
        out = subprocess.run([ff, "-hide_banner", "-filters"], capture_output=True, text=True,
                             encoding="utf-8", errors="replace").stdout
        ver = subprocess.run([ff, "-version"], capture_output=True, text=True,
                             encoding="utf-8", errors="replace").stdout.split("\n")[0]
        info.append(f"FFmpeg: {ver[:60]}")
        if " ass " not in out:
            errors.append("FFmpeg libass desteği yok (altyazı gömülemez). 'full' sürümünü kurun: winget install Gyan.FFmpeg")
        enc = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True,
                             encoding="utf-8", errors="replace").stdout
        if "libx264" not in enc:
            errors.append("FFmpeg libx264 (H.264) kodlayıcısı yok.")

    for mod, pkg in PACKAGES.items():
        if importlib.util.find_spec(mod) is None:
            errors.append(f"Python paketi eksik: {pkg}  (install.bat çalıştırın)")

    try:
        from .transcriber import _add_nvidia_dll_dirs, cuda_available
        _add_nvidia_dll_dirs()
        info.append("GPU (CUDA): " + ("var — Whisper GPU'da çalışacak" if cuda_available()
                                      else "yok — Whisper CPU'da çalışacak (daha yavaş)"))
    except Exception:
        pass
    info.append("Yüz modeli (YuNet): " + ("hazır" if MODEL_PATH.exists() else "yok — Haar cascade kullanılacak"))
    return errors, info
