param(
    [string]$Version = "0.3.2",
    [string]$PythonPath = "",
    [switch]$SkipDependencies,
    [switch]$SkipPortable,
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

# Valide le véritable binaire PyInstaller avant de créer les livrables. Ce
# contrôle démarre le serveur local embarqué, vérifie sa réponse HTTP puis son
# arrêt propre, sans ouvrir la fenêtre de l'application.
$SmokeDataDir = Join-Path $OutputRoot ("smoke-data-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $SmokeDataDir | Out-Null
$SmokeEnvironment = @{
    TRANSCRIPTION_APP_DATA_DIR = $env:TRANSCRIPTION_APP_DATA_DIR
    GRADIO_ANALYTICS_ENABLED = $env:GRADIO_ANALYTICS_ENABLED
    HF_HUB_DISABLE_TELEMETRY = $env:HF_HUB_DISABLE_TELEMETRY
    DO_NOT_TRACK = $env:DO_NOT_TRACK
    NO_PROXY = $env:NO_PROXY
}
$SmokeProcess = $null
try {
    $env:TRANSCRIPTION_APP_DATA_DIR = $SmokeDataDir
    $env:GRADIO_ANALYTICS_ENABLED = "False"
    $env:HF_HUB_DISABLE_TELEMETRY = "1"
    $env:DO_NOT_TRACK = "1"
    $env:NO_PROXY = "127.0.0.1,localhost"

    $SmokeProcess = Start-Process `
        -FilePath $Executable `
        -ArgumentList "--smoke-test" `
        -WorkingDirectory $AppDir `
        -WindowStyle Hidden `
        -PassThru
    $SmokeCompleted = $SmokeProcess.WaitForExit(240000)
    if (-not $SmokeCompleted) {
        Stop-Process -Id $SmokeProcess.Id -Force -ErrorAction SilentlyContinue
        throw "Le smoke test Windows a dépassé le délai de 4 minutes."
    }
    $SmokeProcess.Refresh()
    if ($SmokeProcess.ExitCode -ne 0) {
        throw "Le smoke test Windows a échoué avec le code $($SmokeProcess.ExitCode)."
    }

    $SmokeLog = Join-Path $SmokeDataDir "logs\application.log"
    if (-not (Test-Path -LiteralPath $SmokeLog)) {
        throw "Le smoke test Windows n'a produit aucun journal : $SmokeLog"
    }
    # Le journal est UTF-8 sans BOM. Windows PowerShell 5.1 peut afficher les
    # accents en mojibake ; cette sous-chaîne ASCII reste fiable dans pwsh et
    # Windows PowerShell.
    if (-not (Select-String -LiteralPath $SmokeLog -Pattern "smoke test bureau : ok" -Quiet)) {
        throw "Le journal ne confirme pas la réussite du smoke test Windows : $SmokeLog"
    }

    $LingeringProcess = Get-CimInstance Win32_Process -Filter "Name = 'TranscriptionLocale.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.ExecutablePath -eq $Executable }
    if ($LingeringProcess) {
        $LingeringProcess | ForEach-Object {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }
        throw "Le smoke test Windows a laissé un processus Studio Audio actif."
    }
    Write-Host "Smoke test du véritable exécutable Windows : OK" -ForegroundColor Green
} finally {
    foreach ($VariableName in $SmokeEnvironment.Keys) {
        Set-Item -Path "Env:$VariableName" -Value $SmokeEnvironment[$VariableName] -ErrorAction SilentlyContinue
        if ($null -eq $SmokeEnvironment[$VariableName]) {
            Remove-Item -Path "Env:$VariableName" -ErrorAction SilentlyContinue
        }
    }
}

if (-not $SkipPortable) {
    $PortableZip = Join-Path $PackageDir "StudioAudio-Windows-x64-Portable-$Version.zip"
    Compress-Archive -Path (Join-Path $AppDir "*") -DestinationPath $PortableZip -CompressionLevel Optimal -Force
    Write-Host "Version portable créée : $PortableZip" -ForegroundColor Green
}

if (-not $SkipInstaller) {
    $Iscc = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if (-not $Iscc) {
        $Candidates = @(
            (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"),
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
