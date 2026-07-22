@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\creer_package_partage.ps1"
if errorlevel 1 (
  echo.
  echo La creation du package a rencontre une erreur.
  pause
  exit /b 1
)
echo.
pause
