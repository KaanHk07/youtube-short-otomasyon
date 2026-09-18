"""Ses zinciri: kaynak ses korunur, konusma netlestirilir, seviye normalize edilir.

- highpass 70 Hz: ugultu/ruzgar/masaya vurma gibi alcak frekans gurultusu
- hafif compressor: fisilti ile bagirma arasindaki farki daraltir (konusma anlasilir)
- loudnorm -14 LUFS / -1.5 dBTP: YouTube hedef seviyesi, ses patlamasi yok
- alimiter: kalan ani tepeler icin son guvenlik
Muzik eklenmez.
"""
from __future__ import annotations

SPEECH_CHAIN = ",".join([
    "highpass=f=70",
    "acompressor=threshold=0.089:ratio=2.5:attack=15:release=250:makeup=1.5",
    "loudnorm=I=-14:TP=-1.5:LRA=11",
    "alimiter=limit=0.89:attack=5:release=50:level=disabled",
    "aresample=48000",
])


def filter_chain(normalize: bool = True) -> str:
    return SPEECH_CHAIN if normalize else "aresample=48000"
