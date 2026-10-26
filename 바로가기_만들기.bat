@echo off
rem Keep this file ASCII only (cmd misreads UTF-8 lines); the Korean text lives in make_shortcut.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0make_shortcut.ps1"
pause
