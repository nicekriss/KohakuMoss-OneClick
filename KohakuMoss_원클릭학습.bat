@echo off
rem Keep this file ASCII only: cmd misreads UTF-8 lines. The Korean UI lives in ui.html / app.py.
rem Double-click: opens the trainer in the browser. Closing this window stops the trainer.
title KohakuMoss OneClick LoRA
cd /d "%~dp0"
set "KOHAKU=C:\KohakuMossTrainer"
if exist "%~dp0kohaku_path.txt" set /p KOHAKU=<"%~dp0kohaku_path.txt"
if not exist "%KOHAKU%\.venv\Scripts\python.exe" (
  echo.
  echo  KohakuMossTrainer was not found at %KOHAKU%
  echo  Install it first, or write its folder path into kohaku_path.txt next to this file.
  echo.
  pause
  exit /b 1
)
rem launcher.py restarts the trainer after an in-app update; keep this file unchanged from here on.
"%KOHAKU%\.venv\Scripts\python.exe" "%~dp0launcher.py"
echo.
pause
