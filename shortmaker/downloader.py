"""yt-dlp ile indirme. Desteklenmeyen URL ile indirme hatasini ayirir."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .errors import MSG_DOWNLOAD, MSG_UNSUPPORTED, ShortMakerError

log = logging.getLogger(__name__)

_URL_RE = re.compile(r"^https?://[^\s/$.?#].[^\s]*$", re.IGNORECASE)

# En yuksek cozunurluk (<=1080p) oncelikli; ayni cozunurlukte H.264 tercih edilir.
# AV1/VP9 da sorun degil: ffmpeg ara dosyayi zaten H.264'e cevirir.
FORMAT = "bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b"
FORMAT_SORT = ["res:1080", "fps", "vcodec:h264", "acodec:aac"]


class _QuietLogger:
    """yt-dlp ciktisini konsol yerine log dosyasina yonlendirir."""

    def debug(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        log.debug("yt-dlp: %s", msg)

    def error(self, msg: str) -> None:
        log.info("yt-dlp: %s", msg)


@dataclass
class Download:
    path: Path
    title: str
    video_id: str
    uploader: str
    extractor: str


def download(url: str, dest_dir: Path) -> Download:
    url = url.strip().strip('"')
    local = Path(url)
    if not url.lower().startswith(("http://", "https://")) and local.is_file():
        # kendi bilgisayarindaki video: indirme yok, dogrudan islenir
        return Download(path=local.resolve(), title=local.stem, video_id="", uploader="", extractor="local")
    if not _URL_RE.match(url):
        raise ShortMakerError(MSG_UNSUPPORTED, "gecerli bir http(s) adresi degil")
    try:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import DownloadError, UnsupportedError
    except ImportError as exc:
        raise ShortMakerError(MSG_DOWNLOAD, "yt-dlp kurulu degil: install.bat") from exc

    dest_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "outtmpl": str(dest_dir / "source.%(ext)s"),
        "format": FORMAT,
        "format_sort": FORMAT_SORT,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "overwrites": True,
        "logger": _QuietLogger(),  # hatalar ShortMakerError ile tek satir raporlanir
    }
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info))
    except UnsupportedError as exc:
        raise ShortMakerError(MSG_UNSUPPORTED, str(exc)) from exc
    except DownloadError as exc:
        text = str(exc)
        if "Unsupported URL" in text or "is not a valid URL" in text:
            raise ShortMakerError(MSG_UNSUPPORTED, text) from exc
        raise ShortMakerError(MSG_DOWNLOAD, text) from exc
    except Exception as exc:  # yt-dlp cok cesitli hatalar firlatir
        raise ShortMakerError(MSG_DOWNLOAD, f"{type(exc).__name__}: {exc}") from exc

    if not path.exists():  # birlestirme sonrasi uzanti degisebilir
        found = sorted(dest_dir.glob("source.*"))
        if not found:
            raise ShortMakerError(MSG_DOWNLOAD, "indirilen dosya bulunamadi")
        path = found[0]
    log.info("Indirildi: %s -> %s", url, path.name)
    return Download(path=path, title=info.get("title") or "", video_id=info.get("id") or "",
                    uploader=info.get("uploader") or "", extractor=info.get("extractor_key") or "")
