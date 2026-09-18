# ShortMaker — URL → altyazılı dikey Shorts

URL listesi ver; her video için **≤60 sn, 1080x1920, H.264/AAC** dikey bir video üretir ve
konuşmayı **kendi dilinde** (çeviri yok) kelime-senkron altyazıyla gömer. Çıktılar
`output/` klasörüne kaydedilir; yükleme yapılmaz.

> Yalnızca **sahibi olduğun, kullanma iznin olan veya yeniden kullanım lisansı
> bulunan** içerikleri işle.

## Kurulum ve kullanım (Windows 10/11)

1. `install.bat` → Python paketleri, (NVIDIA varsa) GPU hızlandırma, yüz ve Whisper
   modelleri, FFmpeg/libass kontrolü. FFmpeg yoksa `winget` ile kurmayı dener.
2. `urls.txt` içine her satıra bir URL yaz (YouTube, Shorts, TikTok, Instagram… yt-dlp'nin
   desteklediği siteler; yerel dosya yolu da olur).
3. `start.bat` → her URL için `output\video_NNN\` klasörü oluşturur.

```bat
.venv\Scripts\python app.py "https://youtube.com/shorts/..."          :: tek URL -> output\video_NNN
.venv\Scripts\python app.py --urls urls.txt --duration 45             :: toplu
.venv\Scripts\python app.py URL --clips 2                             :: aynı videodan 2 ayrı Short
.venv\Scripts\python app.py URL --out output\deneme                   :: belirli klasör
.venv\Scripts\python app.py --check                                   :: bağımlılık kontrolü
```

Ayarlar `shortmaker.yaml` (veya komut satırı): süre 30/45/60 (`--duration`), çözünürlük
1080x1920/720x1280 (`--resolution`), altyazı (`--no-subtitles`), aktif kelime vurgusu
(`--no-highlight`), akıllı crop (`--no-smart-crop`), sessizlik kaldırma (`--keep-silence`),
Whisper modeli (`--whisper-model medium`), klip sayısı (`--clips N`).

## Çıktı

```
output/video_001/
  short.mp4        1080x1920, H.264 High + AAC 48 kHz, -14 LUFS, kaynak FPS (değişken/>60 ise 30)
  subtitles.srt    kaynak dilde, 2–7 kelimelik gruplar
  transcript.txt   seçilen bölümün tam metni
  info.json        source_url, original/short_duration, detected_language, resolution, fps,
                   segment, removed_silence_sec, crop_mode, warnings ...
```

## Nasıl çalışır (`shortmaker/`)

| Adım | Modül |
|---|---|
| İndirme (≤1080p), desteklenmeyen URL / indirme hatası ayrımı | `downloader.py` |
| Süre, çözünürlük, FPS, ses var mı (ffprobe + volumedetect) | `analyzer.py` |
| Dil tespiti + kelime zamanlı transkript (faster-whisper, GPU→CPU yedekli) | `language_detector.py`, `transcriber.py` |
| Cümle sınırlı en iyi ≤N sn bölüm(ler) | `segment_selector.py`, `clipselect.py` |
| Uzun sessizlikleri paylı kısaltma, kelime zamanlarını yeniden eşleme | `silence.py` |
| 2–7 kelimelik dinamik altyazı, SRT + ASS (sarı aktif kelime, güvenli alan) | `subtitle_generator.py` |
| Yatay kaynakta yüz (YuNet) / hareket takipli, titremesiz 9:16 kadraj | `smart_crop.py` |
| Highpass + compressor + loudnorm + limiter | `audio.py` |
| Kesim/birleştirme → kadraj + altyazı gömme → MP4 | `video_processor.py` |
| URL başına akış, hata mesajları, info.json | `pipeline.py` |

Hatalar URL bazındadır ("Video indirilemedi.", "URL desteklenmiyor.", "Video içerisinde ses
bulunamadı.", "Altyazı oluşturulamadı."); biri başarısız olsa da diğerleri işlenir.
Teknik ayrıntı: `logs/shortmaker.log`.

## Testler

```bat
.venv\Scripts\python -m unittest discover -s tests -t . -v
```
