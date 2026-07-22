@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\installer.ps1"
if errorlevel 1 (
  echo.
  echo L'installation a rencontre une erreur.
  pause
  exit /b 1
)
echo.
pause
