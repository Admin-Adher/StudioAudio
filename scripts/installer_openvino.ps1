$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
Set-Location $ProjectRoot

Write-Host ""
Write-Host "Pilote OpenVINO pour Intel Arc" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Ce composant est experimental et reste desactive par defaut."

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host "Installation de l'application principale..." -ForegroundColor Yellow
    & (Join-Path $PSScriptRoot "installer.ps1")
}
if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Installation principale incomplete : Python local introuvable."
}

Write-Host "Installation du runtime OpenVINO stable..."
& $VenvPython -m pip install --requirement (Join-Path $ProjectRoot "requirements-openvino.txt")
if ($LASTEXITCODE -ne 0) { throw "L'installation du runtime OpenVINO a echoue." }

Write-Host "Telechargement du modele Intel Arc (environ 0,8 Go)..."
& $VenvPython (Join-Path $PSScriptRoot "prepare_openvino.py")
if ($LASTEXITCODE -ne 0) { throw "Le telechargement du modele OpenVINO a echoue." }

Write-Host ""
Write-Host "Pilote OpenVINO installe." -ForegroundColor Green
Write-Host "Fermez puis relancez l'application avant de l'activer dans Parametres."
