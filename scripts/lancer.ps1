$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
Set-Location $ProjectRoot

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host "L'application n'est pas encore installee. Lancement de l'installation..." -ForegroundColor Yellow
    & (Join-Path $PSScriptRoot "installer.ps1")
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Installation incomplete : Python local introuvable."
}

$env:PYTHONUTF8 = "1"
$env:GRADIO_ANALYTICS_ENABLED = "False"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:DO_NOT_TRACK = "1"
$env:WESPEAKER_HOME = Join-Path $ProjectRoot "modeles\wespeaker"
$env:OMP_NUM_THREADS = [Math]::Min([Environment]::ProcessorCount, 8).ToString()

Write-Host ""
Write-Host "Studio Audio demarre sur http://127.0.0.1:7860" -ForegroundColor Green
Write-Host "Gardez cette fenetre ouverte. Ctrl+C permet d'arreter l'application."
& $VenvPython (Join-Path $ProjectRoot "app.py")
