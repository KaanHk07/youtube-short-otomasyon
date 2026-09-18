"""ShortMaker — URL'lerden altyazili dikey Shorts videolari uretir (output/ klasorune).

Ornekler:
  python app.py "https://youtube.com/shorts/..."            tek URL -> output/video_NNN/
  python app.py --urls urls.txt                             toplu uretim
  python app.py URL --out output/test --duration 45         belirli klasor / sure
  python app.py URL --clips 2                               ayni videodan 2 ayri Short
  python app.py --check                                     bagimlilik kontrolu

Yalnizca sahibi oldugun, kullanma iznin olan veya yeniden kullanim lisansi
bulunan iceriklerle kullan (--rights own | licensed).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # cv2 DNN uyarilari konsolu kirletmesin


def _setup_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _setup_logging(verbose: bool) -> None:
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_dir / "shortmaker.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    # Ilerleme ve hata mesajlari zaten ekrana yaziliyor; teknik ayrinti logs/shortmaker.log'da.
    ch.setLevel(logging.INFO if verbose else logging.CRITICAL)
    logging.basicConfig(level=logging.INFO, handlers=[fh, ch],
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "faster_whisper", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore")


def read_urls(args: argparse.Namespace) -> list[str]:
    urls = list(args.url or [])
    if args.urls:
        p = Path(args.urls)
        if not p.exists():
            raise SystemExit(f"URL dosyası bulunamadı: {p}")
        for line in p.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    seen, unique = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="app.py", description="URL -> altyazili dikey YouTube Shorts")
    p.add_argument("url", nargs="*", help="video URL'leri")
    p.add_argument("--urls", help="her satirda bir URL olan dosya (# ile yorum)")
    p.add_argument("--out", help="cikti klasoru (yalnizca tek URL icin; varsayilan output/video_NNN)")
    p.add_argument("--rights", choices=["own", "licensed"], help="icerik hakki (varsayilan: shortmaker.yaml)")
    p.add_argument("--duration", type=int, choices=[30, 45, 60], help="Shorts suresi")
    p.add_argument("--clips", type=int, default=1, metavar="N",
                   help="her URL'den birbiriyle ortusmeyen N klip uret (varsayilan 1)")
    p.add_argument("--resolution", choices=["1080x1920", "720x1280"])
    p.add_argument("--no-subtitles", action="store_true", help="altyaziyi videoya gomme")
    p.add_argument("--no-highlight", action="store_true", help="aktif kelime vurgusunu kapat")
    p.add_argument("--no-smart-crop", action="store_true", help="merkez crop kullan")
    p.add_argument("--keep-silence", action="store_true", help="sessizlikleri kaldirma")
    p.add_argument("--whisper-model", help="tiny/base/small/medium/large-v3")
    p.add_argument("--check", action="store_true", help="bagimliliklari kontrol et")
    p.add_argument("--setup", action="store_true", help="modelleri indir + kontrol et (install.bat)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _overrides(a: argparse.Namespace) -> dict:
    o: dict = {"video": {}, "subtitles": {}, "crop": {}, "silence": {}, "whisper": {}}
    if a.duration:
        o["video"]["duration"] = a.duration
    if a.resolution:
        o["video"]["resolution"] = a.resolution
    if a.no_subtitles:
        o["subtitles"]["enabled"] = False
    if a.no_highlight:
        o["subtitles"]["highlight_active_word"] = False
    if a.no_smart_crop:
        o["crop"]["smart"] = False
    if a.keep_silence:
        o["silence"]["remove"] = False
    if a.whisper_model:
        o["whisper"]["model"] = a.whisper_model
    return o


def main(argv: list[str] | None = None) -> int:
    _setup_console()
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)

    from shortmaker import deps
    from shortmaker.config import SettingsError, load_settings

    if args.setup:
        st = load_settings(ROOT / "shortmaker.yaml")
        deps.setup_models(str(st.section("whisper")["model"]))
        args.check = True
    errors, info = deps.check()
    if args.check:
        for line in info:
            print("  ·", line)
        for line in errors:
            print("  ✖", line)
        print("Hazır." if not errors else "Eksikler var, yukarıdaki adımları uygulayın.")
        return 1 if errors else 0
    if errors:
        for line in errors:
            print("✖", line)
        return 2

    try:
        st = load_settings(ROOT / "shortmaker.yaml", _overrides(args))
    except SettingsError as exc:
        print("Ayar hatası:", exc)
        return 2

    urls = read_urls(args)
    if not urls:
        print("URL verilmedi. Örnek: python app.py \"https://youtube.com/shorts/...\"  "
              "veya urls.txt dosyasına URL'leri yazıp start.bat'ı çalıştırın.")
        return 1
    print("Not: Yalnızca sahibi olduğunuz / kullanma izniniz olan içerikleri işleyin.")
    print(f"Ayarlar: {st.duration} sn · {st.size[0]}x{st.size[1]} · altyazı "
          f"{'açık' if st.section('subtitles')['enabled'] else 'kapalı'} · vurgu "
          f"{'açık' if st.section('subtitles')['highlight_active_word'] else 'kapalı'} · akıllı crop "
          f"{'açık' if st.section('crop')['smart'] else 'kapalı'} · sessizlik kaldırma "
          f"{'açık' if st.section('silence')['remove'] else 'kapalı'}")

    from shortmaker.pipeline import process_many
    out = Path(args.out) if args.out else None
    if out and not out.is_absolute():
        out = ROOT / out
    results = process_many(urls, st, rights=args.rights, out_dir=out, clips=max(1, args.clips))
    ok_count = sum(r.ok for r in results)
    fail_count = len(results) - ok_count
    print(f"\nBitti: {ok_count} başarılı, {fail_count} başarısız.")
    for r in results:
        if not r.ok:
            print(f"  ✖ {r.url}\n     {r.message}" + (f" — {r.detail}" if r.detail else ""))

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
