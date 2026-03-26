@echo off
chcp 65001 >nul
setlocal

:: Get script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: Set paths
set "PYTHON=%SCRIPT_DIR%bin\python-3.13.9-embed-amd64\python.exe"

if not exist "%PYTHON%" (
  echo [Error] Portable Python not found: %PYTHON%
  pause
  exit /b 1
)

:: Environment
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"

:: Auto-check and repair NumPy/Numba compatibility before launch
"%PYTHON%" -X utf8 dependency_guard.py --check
if %errorlevel% neq 0 (
  echo [Warning] Detected NumPy/Numba compatibility issue. Trying auto-repair...
  "%PYTHON%" -X utf8 dependency_guard.py --fix
  if %errorlevel% neq 0 (
    echo [Error] Auto-repair failed. Please run repair_env.bat manually.
    pause
    exit /b 1
  )
)

:: Run
echo Starting Music Video Cutter...
"%PYTHON%" -X utf8 gui.py

pause
