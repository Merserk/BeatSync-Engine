@echo off
chcp 65001 >nul
setlocal

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "PYTHON=%SCRIPT_DIR%bin\python-3.13.17-embed-amd64\python.exe"
set "FFMPEG=%SCRIPT_DIR%bin\ffmpeg"

if not exist "%PYTHON%" (
    echo ERROR: Portable Python was not found.
    echo Run install.bat first.
    pause
    exit /b 1
)

set "PATH=%FFMPEG%;%SCRIPT_DIR%bin\python-3.13.17-embed-amd64;%SCRIPT_DIR%bin\python-3.13.17-embed-amd64\Scripts;%PATH%"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH=%SCRIPT_DIR%src"

"%PYTHON%" -X utf8 src\gui.py

pause
