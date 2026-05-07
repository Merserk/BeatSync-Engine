@echo off
chcp 65001 >nul
setlocal

:: Get script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: Set paths
set "PYTHON=%SCRIPT_DIR%bin\python-3.13.9-embed-amd64\python.exe"

:: Environment
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"

:: Run
echo Starting Music Video Cutter...
"%PYTHON%" -X utf8 gui.py

pause
