@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Once install.bat dosyasini calistirin.
  pause & exit /b 1
)
if not exist urls.txt (echo # Her satira bir URL> urls.txt)
findstr /r /v /c:"^#" /c:"^ *$" urls.txt >nul
if errorlevel 1 (
  echo urls.txt bos. URL'leri yazip kaydedin, sonra start.bat'i tekrar calistirin.
  start "" notepad urls.txt
  pause & exit /b 0
)
rem Shorts'lari uretir; her video output\video_NNN klasorune kaydedilir.
rem Ornek ek secenekler: start.bat --clips 2 --duration 45
".venv\Scripts\python.exe" app.py --urls urls.txt %*
echo.
echo Ciktilar: %~dp0output
pause
