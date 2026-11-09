@echo off
title KohakuMoss OneClick LoRA
cd /d "%~dp0"
set "KOHAKU=C:\KohakuMossTrainer"
if exist kohaku_path.txt set /p KOHAKU=<kohaku_path.txt
if exist "%KOHAKU%\.venv\Scripts\python.exe" goto run
for %%R in (H: G: F: E: D: C: "%~dp0..") do if exist "%%~R\KohakuMossTrainer\.venv\Scripts\python.exe" set "KOHAKU=%%~R\KohakuMossTrainer"
if exist "%KOHAKU%\.venv\Scripts\python.exe" goto run
echo KohakuMossTrainer was not found.
set /p KOHAKU=Paste its folder path, then Enter: 
set KOHAKU=%KOHAKU:"=%
if not exist "%KOHAKU%\.venv\Scripts\python.exe" echo Not found: "%KOHAKU%"& pause& exit /b 1
<nul set /p "=%KOHAKU%">kohaku_path.txt
:run
rem Keep the launch line at this byte 
"%KOHAKU%\.venv\Scripts\python.exe" "%~dp0launcher.py"
echo.
pause
