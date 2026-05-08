@echo off
chcp 65001 >nul
setlocal

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%scripts\install.ps1"
set "INSTALL_EXIT=%ERRORLEVEL%"

if not "%INSTALL_EXIT%"=="0" (
    echo.
    echo Install failed with exit code %INSTALL_EXIT%.
    pause
    exit /b %INSTALL_EXIT%
)

echo.
echo Install completed. Use run.bat to start BeatSync Engine.
pause
