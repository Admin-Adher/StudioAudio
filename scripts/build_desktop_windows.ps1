param(
    [string]$Version = "0.2.0",
    [string]$PythonPath = "",
    [switch]$SkipDependencies,
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"

if (-not $IsWindows -and $PSVersionTable.PSEdition -eq "Core") {
    throw "Ce build Windows doit être exécuté sous Windows."
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DesktopVenv = Join-Path $ProjectRoot ".venv-desktop"
$VenvPython = Join-Path $DesktopVenv "Scripts\python.exe"
$OutputRoot = Join-Path $ProjectRoot "output\desktop\windows"
$WorkDir = Join-Path $OutputRoot "build"
$DistDir = Join-Path $OutputRoot "dist"
$PackageDir = Join-Path $ProjectRoot "packages"
$SpecPath = Join-Path $ProjectRoot "packaging\transcription_locale.spec"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    if ($PythonPath) {
        & $PythonPath -m venv $DesktopVenv
    } elseif (Get-Command py.exe -ErrorAction SilentlyContinue) {
        & py.exe -3.12 -m venv $DesktopVenv
    } else {
        & python.exe -m venv $DesktopVenv
    }
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Impossible de créer l'environnement Python bureau : $DesktopVenv"
}

if (-not $SkipDependencies) {
    & $VenvPython -m pip install --upgrade pip wheel
    & $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements-desktop.txt")
    # Le moteur recommandé sous Windows cible les GPU Intel Arc. Il reste
    # optionnel sur les autres plateformes, mais doit être présent dans un
    # paquet Windows neuf afin que l'application ne retombe pas immédiatement
    # sur Faster-Whisper chez un collègue.
    & $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements-openvino.txt")
}

New-Item -ItemType Directory -Force -Path $OutputRoot, $WorkDir, $DistDir, $PackageDir | Out-Null
$env:TRANSCRIPTION_DESKTOP_VERSION = $Version
& $VenvPython -m PyInstaller `
    --noconfirm `
    --clean `
    --workpath $WorkDir `
    --distpath $DistDir `
    $SpecPath

$AppDir = Join-Path $DistDir "TranscriptionLocale"
$Executable = Join-Path $AppDir "TranscriptionLocale.exe"
if (-not (Test-Path -LiteralPath $Executable)) {
    throw "Le build PyInstaller n'a pas produit l'exécutable attendu : $Executable"
}

$PortableZip = Join-Path $PackageDir "StudioAudio-Windows-x64-Portable-$Version.zip"
Compress-Archive -Path (Join-Path $AppDir "*") -DestinationPath $PortableZip -CompressionLevel Optimal -Force
Write-Host "Version portable créée : $PortableZip" -ForegroundColor Green

if (-not $SkipInstaller) {
    $Iscc = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if (-not $Iscc) {
        $Candidates = @(
            (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
            (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe")
        ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
        if ($Candidates.Count -gt 0) {
            $Iscc = Get-Item -LiteralPath $Candidates[0]
        }
    }
    if ($Iscc) {
        $IsccPath = if ($Iscc.Source) { $Iscc.Source } else { $Iscc.FullName }
        & $IsccPath `
            "/DSourceDir=$AppDir" `
            "/DOutputDir=$PackageDir" `
            "/DAppVersion=$Version" `
            "/DIconFile=$(Join-Path $ProjectRoot "assets\app-icon.ico")" `
            (Join-Path $ProjectRoot "packaging\windows\TranscriptionLocale.iss")
        Write-Host "Installateur Windows créé dans : $PackageDir" -ForegroundColor Green
    } else {
        Write-Warning "Inno Setup 6 absent : le ZIP portable est prêt, mais aucun Setup.exe n'a été généré."
    }
}
