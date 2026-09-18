"""Kullaniciya gosterilen hata mesajlari. Her hata tek bir URL'yi durdurur, toplu isi degil."""
from __future__ import annotations

MSG_DOWNLOAD = "Video indirilemedi."
MSG_UNSUPPORTED = "URL desteklenmiyor."
MSG_NO_AUDIO = "Video içerisinde ses bulunamadı."
MSG_SUBTITLE = "Altyazı oluşturulamadı."
MSG_PROCESS = "Video işlenemedi."
MSG_RIGHTS = "Kullanım hakkı doğrulanmadı (yalnızca own / licensed)."


class ShortMakerError(RuntimeError):
    """message: kullaniciya gosterilecek kisa metin; detail: log icin teknik ayrinti."""

    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def __str__(self) -> str:
        return f"{self.message} ({self.detail})" if self.detail else self.message
