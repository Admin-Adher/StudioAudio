@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\lancer.ps1"
if errorlevel 1 (
  echo.
  echo L'application s'est arretee avec une erreur.
  pause
  exit /b 1
)
