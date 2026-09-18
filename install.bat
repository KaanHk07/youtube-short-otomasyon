@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
echo ============================================
echo   ShortMaker kurulumu
echo ============================================

rem --- Python ---
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (where python >nul 2>nul && set "PY=python")
if not defined PY (
  echo [HATA] Python bulunamadi. https://www.python.org/downloads/ adresinden Python 3.10+ kurun
  echo        ^(kurulumda "Add python.exe to PATH" secenegini isaretleyin^).
  pause & exit /b 1
)
%PY% -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" || (
  echo [HATA] Python 3.10 veya daha yeni bir surum gerekli.
  pause & exit /b 1
)

rem --- FFmpeg ---
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo [BILGI] FFmpeg bulunamadi, winget ile kuruluyor...
  winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
  echo [BILGI] FFmpeg kurulduysa bu pencereyi kapatip install.bat'i yeniden calistirin.
  pause & exit /b 1
)

rem --- Sanal ortam + paketler ---
if not exist ".venv\Scripts\python.exe" (
  echo [1/4] Sanal ortam olusturuluyor...
  %PY% -m venv .venv || (echo [HATA] venv olusturulamadi & pause & exit /b 1)
)
echo [2/4] Python paketleri kuruluyor...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo [HATA] Paket kurulumu basarisiz & pause & exit /b 1)

where nvidia-smi >nul 2>nul
if not errorlevel 1 (
  echo [3/4] NVIDIA ekran karti bulundu, GPU hizlandirma kuruluyor...
  ".venv\Scripts\python.exe" -m pip install -r requirements-gpu.txt || echo [UYARI] GPU paketleri kurulamadi, CPU kullanilacak.
) else (
  echo [3/4] NVIDIA ekran karti yok, Whisper CPU'da calisacak.
)

echo [4/4] Modeller indiriliyor ve sistem kontrol ediliyor...
".venv\Scripts\python.exe" app.py --setup
if errorlevel 1 (echo [HATA] Eksikler var, yukaridaki mesajlara bakin. & pause & exit /b 1)

if not exist urls.txt (echo # Her satira bir URL> urls.txt)
echo.
echo Kurulum tamam. urls.txt dosyasina URL'leri yazip start.bat'i calistirin.
pause
