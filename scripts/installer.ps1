$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
Set-Location $ProjectRoot

Write-Host ""
Write-Host "Installation de Studio Audio" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan

if (-not (Test-Path -LiteralPath $VenvPython)) {
    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    $Python = Get-Command python -ErrorAction SilentlyContinue
    if ($PyLauncher) {
        Write-Host "Creation de l'environnement Python..."
        & $PyLauncher.Source -3 -m venv (Join-Path $ProjectRoot ".venv")
    }
    elseif ($Python) {
        Write-Host "Creation de l'environnement Python..."
        & $Python.Source -m venv (Join-Path $ProjectRoot ".venv")
    }
    else {
        throw "Python 3.10 a 3.12 est requis. Installez Python depuis https://www.python.org/downloads/windows/ puis relancez INSTALLER.bat."
    }
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "La creation de l'environnement Python a echoue."
}

$VersionCheck = & $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$VersionParts = $VersionCheck.Split('.')
$Major = [int]$VersionParts[0]
$Minor = [int]$VersionParts[1]
if ($Major -ne 3 -or $Minor -lt 10 -or $Minor -gt 12) {
    throw "Version Python $VersionCheck detectee. Utilisez Python 3.10, 3.11 ou 3.12."
}

Write-Host "Mise a jour de l'outil d'installation..."
& $VenvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "La mise a jour de pip a echoue." }

Write-Host "Installation des moteurs locaux (cela peut prendre plusieurs minutes)..."
& $VenvPython -m pip install --requirement (Join-Path $ProjectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "L'installation des dependances a echoue." }

New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "modeles") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "resultats") | Out-Null

Write-Host ""
Write-Host "Installation terminee." -ForegroundColor Green
Write-Host "Double-cliquez maintenant sur LANCER.bat."
