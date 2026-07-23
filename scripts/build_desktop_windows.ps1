param(
    [string]$Version = "0.3.8",
    [string]$PythonPath = "",
    [switch]$SkipDependencies,
    [switch]$SkipPortable,
    [switch]$SkipInstaller,
    [switch]$SkipBundledAssistantModel
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
$BundledModelsRoot = Join-Path $OutputRoot "bundled-models"
$BundledAssistantModelDir = Join-Path $BundledModelsRoot "qwen3-8b-int4-ov"
$AssistantModelRepository = "OpenVINO/Qwen3-8B-int4-ov"
$AssistantModelRevision = "5c47abf4b8e12ebe8e99745bb0c1ec17e0c0abcc"
$AssistantModelWeightsSha256 = "f2e6960fde61d24a83c477c46f9ee7158748762089f2018fc2bb6a6d6c7f2a7e"
$AssistantModelRequiredFiles = @(
    "config.json",
    "chat_template.jinja",
    "openvino_config.json",
    "openvino_detokenizer.xml",
    "openvino_detokenizer.bin",
    "openvino_model.xml",
    "openvino_model.bin",
    "openvino_tokenizer.xml",
    "openvino_tokenizer.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "studio-audio-model.json"
)

function Assert-BundledAssistantModel {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ModelDirectory
    )

    foreach ($RelativePath in $AssistantModelRequiredFiles) {
        $RequiredPath = Join-Path $ModelDirectory $RelativePath
        if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) {
            throw "Le modèle assistant hors ligne est incomplet : $RequiredPath"
        }
    }

    $WeightsPath = Join-Path $ModelDirectory "openvino_model.bin"
    if ((Get-Item -LiteralPath $WeightsPath).Length -lt 4GB) {
        throw "Les poids OpenVINO semblent tronqués : $WeightsPath"
    }
    $WeightsSha256 = (
        Get-FileHash -LiteralPath $WeightsPath -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    if ($WeightsSha256 -ne $AssistantModelWeightsSha256) {
        throw (
            "Le SHA-256 des poids OpenVINO est invalide : " +
            "attendu $AssistantModelWeightsSha256, obtenu $WeightsSha256."
        )
    }

    $ManifestPath = Join-Path $ModelDirectory "studio-audio-model.json"
    $Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    if ($Manifest.repository -ne $AssistantModelRepository -or
        $Manifest.revision -ne $AssistantModelRevision -or
        $Manifest.runtime -ne "openvino" -or
        $Manifest.weights_sha256 -ne $AssistantModelWeightsSha256) {
        throw "Le manifeste du modèle assistant Windows ne correspond pas à la révision épinglée."
    }
}

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

New-Item -ItemType Directory -Force -Path $OutputRoot, $WorkDir, $DistDir, $PackageDir, $BundledModelsRoot | Out-Null

if (-not $SkipBundledAssistantModel) {
    New-Item -ItemType Directory -Force -Path $BundledAssistantModelDir | Out-Null
    $env:HF_HUB_DISABLE_TELEMETRY = "1"
    # Le transfert Xet peut rester bloqué avant le premier octet du poids de
    # 4,86 Go sur certains réseaux. Le téléchargement HTTP classique est
    # reprenable et plus fiable pour un build automatisé.
    $env:HF_HUB_DISABLE_XET = "1"
    & $VenvPython `
        (Join-Path $ProjectRoot "scripts\download_assistant_model.py") `
        --repository $AssistantModelRepository `
        --revision $AssistantModelRevision `
        --destination $BundledAssistantModelDir
    if ($LASTEXITCODE -ne 0) {
        throw "Le téléchargement du modèle assistant Windows épinglé a échoué."
    }

    $HuggingFaceLocalCache = Join-Path $BundledAssistantModelDir ".cache"
    if (Test-Path -LiteralPath $HuggingFaceLocalCache) {
        $ResolvedModelRoot = [System.IO.Path]::GetFullPath($BundledAssistantModelDir)
        $ResolvedCachePath = [System.IO.Path]::GetFullPath($HuggingFaceLocalCache)
        if (-not $ResolvedCachePath.StartsWith(
            $ResolvedModelRoot + [System.IO.Path]::DirectorySeparatorChar,
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
            throw "Refus de nettoyer un cache hors du dossier modèle : $ResolvedCachePath"
        }
        Remove-Item -LiteralPath $ResolvedCachePath -Recurse -Force
    }

    Assert-BundledAssistantModel -ModelDirectory $BundledAssistantModelDir
    $env:TRANSCRIPTION_BUNDLED_ASSISTANT_MODEL_DIR = $BundledAssistantModelDir
    $env:TRANSCRIPTION_BUNDLED_ASSISTANT_MODEL_KIND = "openvino"
    Write-Host "Modèle assistant hors ligne prêt : $BundledAssistantModelDir" -ForegroundColor Green
} else {
    Remove-Item Env:TRANSCRIPTION_BUNDLED_ASSISTANT_MODEL_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:TRANSCRIPTION_BUNDLED_ASSISTANT_MODEL_KIND -ErrorAction SilentlyContinue
    Write-Warning "Build sans modèle assistant embarqué demandé explicitement."
}

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

if (-not $SkipBundledAssistantModel) {
    $PackagedAssistantModelDir = Join-Path $AppDir "_internal\modeles\assistant\qwen3-8b-int4-ov"
    Assert-BundledAssistantModel -ModelDirectory $PackagedAssistantModelDir
    Write-Host "Ressources Qwen3 OpenVINO du binaire Windows : OK" -ForegroundColor Green
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

# Le smoke serveur ci-dessus prouve que l'interface démarre. Ce second contrôle
# charge réellement le Qwen empaqueté et exige une petite réponse JSON hors
# réseau avant que le Setup puisse être créé.
if (-not $SkipBundledAssistantModel) {
    $AssistantSmokeDataDir = Join-Path $OutputRoot (
        "assistant-smoke-data-" + [Guid]::NewGuid().ToString("N")
    )
    New-Item -ItemType Directory -Force -Path $AssistantSmokeDataDir | Out-Null
    $AssistantSmokeStdout = Join-Path $AssistantSmokeDataDir "assistant-smoke.stdout.log"
    $AssistantSmokeStderr = Join-Path $AssistantSmokeDataDir "assistant-smoke.stderr.log"
    $AssistantSmokeEnvironment = @{
        TRANSCRIPTION_APP_DATA_DIR = $env:TRANSCRIPTION_APP_DATA_DIR
        HF_HUB_OFFLINE = $env:HF_HUB_OFFLINE
        TRANSFORMERS_OFFLINE = $env:TRANSFORMERS_OFFLINE
        HF_HUB_DISABLE_TELEMETRY = $env:HF_HUB_DISABLE_TELEMETRY
        DO_NOT_TRACK = $env:DO_NOT_TRACK
        TOKENIZERS_PARALLELISM = $env:TOKENIZERS_PARALLELISM
    }
    $AssistantSmokeProcess = $null
    try {
        $env:TRANSCRIPTION_APP_DATA_DIR = $AssistantSmokeDataDir
        $env:HF_HUB_OFFLINE = "1"
        $env:TRANSFORMERS_OFFLINE = "1"
        $env:HF_HUB_DISABLE_TELEMETRY = "1"
        $env:DO_NOT_TRACK = "1"
        $env:TOKENIZERS_PARALLELISM = "false"

        $AssistantSmokeProcess = Start-Process `
            -FilePath $Executable `
            -ArgumentList "--assistant-smoke-test" `
            -WorkingDirectory $AppDir `
            -WindowStyle Hidden `
            -RedirectStandardOutput $AssistantSmokeStdout `
            -RedirectStandardError $AssistantSmokeStderr `
            -PassThru
        $AssistantSmokeCompleted = $AssistantSmokeProcess.WaitForExit(600000)
        if (-not $AssistantSmokeCompleted) {
            Stop-Process -Id $AssistantSmokeProcess.Id -Force -ErrorAction SilentlyContinue
            throw "Le smoke test Qwen Windows a dépassé le délai de 10 minutes."
        }
        # Windows PowerShell 5.1 can leave ExitCode at $null after the timed
        # overload when stdout/stderr are redirected. The final wait drains
        # both streams and makes the exit code reliable.
        $AssistantSmokeProcess.WaitForExit()
        $AssistantSmokeProcess.Refresh()
        # Start-Process peut encore renvoyer $null avec les redirections sous
        # Windows PowerShell 5.1. Dans ce cas, le rapport JSON et le journal
        # vérifiés juste après restent l'autorité ; un vrai code non nul échoue.
        if (
            $null -ne $AssistantSmokeProcess.ExitCode -and
            $AssistantSmokeProcess.ExitCode -ne 0
        ) {
            $AssistantSmokeDetail = ""
            if (Test-Path -LiteralPath $AssistantSmokeStderr) {
                $AssistantSmokeDetail = [string](
                    Get-Content -LiteralPath $AssistantSmokeStderr -Raw
                )
                $AssistantSmokeDetail = $AssistantSmokeDetail.Trim()
            }
            throw (
                "Le smoke test Qwen Windows a échoué avec le code " +
                "$($AssistantSmokeProcess.ExitCode). $AssistantSmokeDetail"
            )
        }

        $AssistantSmokeReportPath = Join-Path (
            Join-Path $AssistantSmokeDataDir "logs"
        ) "assistant-smoke.json"
        if (-not (Test-Path -LiteralPath $AssistantSmokeReportPath)) {
            throw "Le rapport JSON du smoke test Qwen est absent : $AssistantSmokeReportPath"
        }
        $AssistantSmokeReport = Get-Content `
            -LiteralPath $AssistantSmokeReportPath `
            -Raw | ConvertFrom-Json
        if (
            $AssistantSmokeReport.mode -ne "assistant-smoke" -or
            $AssistantSmokeReport.status -ne "ok" -or
            -not $AssistantSmokeReport.backend -or
            -not $AssistantSmokeReport.device -or
            -not $AssistantSmokeReport.model -or
            $AssistantSmokeReport.response.ready -ne $true
        ) {
            throw (
                "Le rapport du smoke test Qwen est invalide : " +
                (ConvertTo-Json $AssistantSmokeReport -Compress -Depth 8)
            )
        }

        $AssistantSmokeLog = Join-Path $AssistantSmokeDataDir "logs\application.log"
        if (
            -not (Test-Path -LiteralPath $AssistantSmokeLog) -or
            -not (
                Select-String `
                    -LiteralPath $AssistantSmokeLog `
                    -Pattern "smoke test assistant : ok" `
                    -Quiet
            )
        ) {
            throw "Le journal ne confirme pas le smoke test Qwen : $AssistantSmokeLog"
        }

        $AssistantLingeringProcess = Get-CimInstance `
            Win32_Process `
            -Filter "Name = 'TranscriptionLocale.exe'" `
            -ErrorAction SilentlyContinue |
            Where-Object { $_.ExecutablePath -eq $Executable }
        if ($AssistantLingeringProcess) {
            $AssistantLingeringProcess | ForEach-Object {
                Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
            }
            throw "Le smoke test Qwen a laissé un processus Studio Audio actif."
        }
        Write-Host (
            "Smoke test du Qwen empaqueté : OK " +
            "[$($AssistantSmokeReport.backend) / $($AssistantSmokeReport.device)]"
        ) -ForegroundColor Green
    } finally {
        foreach ($VariableName in $AssistantSmokeEnvironment.Keys) {
            Set-Item `
                -Path "Env:$VariableName" `
                -Value $AssistantSmokeEnvironment[$VariableName] `
                -ErrorAction SilentlyContinue
            if ($null -eq $AssistantSmokeEnvironment[$VariableName]) {
                Remove-Item `
                    -Path "Env:$VariableName" `
                    -ErrorAction SilentlyContinue
            }
        }
    }
}

if (-not $SkipPortable) {
    $PortableZip = Join-Path $PackageDir "StudioAudio-Windows-x64-Portable-$Version.zip"
    if ($SkipBundledAssistantModel) {
        Compress-Archive -Path (Join-Path $AppDir "*") -DestinationPath $PortableZip -CompressionLevel Optimal -Force
    } else {
        # Compress-Archive ne prend pas en charge un fichier individuel de plus
        # de 2 Go. zipfile utilise ZIP64 et peut donc inclure les poids INT4.
        $Zip64Script = @'
import sys
import zipfile
from pathlib import Path

source = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
destination.unlink(missing_ok=True)
with zipfile.ZipFile(
    destination,
    mode="w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=9,
    allowZip64=True,
) as archive:
    for path in sorted(source.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(source))
'@
        & $VenvPython -c $Zip64Script $AppDir $PortableZip
        if ($LASTEXITCODE -ne 0) {
            throw "La création du ZIP portable ZIP64 a échoué."
        }
    }
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
            $Iscc = Get-Item -LiteralPath @($Candidates)[0]
        }
    }
    if ($Iscc) {
        $IsccPath = if ($Iscc.Source) { $Iscc.Source } else { $Iscc.FullName }
        $InstallerBaseName = "StudioAudio-Windows-x64-Setup-$Version"
        Get-ChildItem -LiteralPath $PackageDir -File |
            Where-Object { $_.Name.StartsWith($InstallerBaseName, [System.StringComparison]::OrdinalIgnoreCase) } |
            Remove-Item -Force
        $InnoArguments = @(
            "/DSourceDir=$AppDir",
            "/DOutputDir=$PackageDir",
            "/DAppVersion=$Version",
            "/DIconFile=$(Join-Path $ProjectRoot "assets\app-icon.ico")"
        )
        if (-not $SkipBundledAssistantModel) {
            $InnoArguments += "/DBundledAssistantModel=1"
        }
        $InnoArguments += (Join-Path $ProjectRoot "packaging\windows\TranscriptionLocale.iss")
        & $IsccPath @InnoArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Inno Setup n'a pas pu créer l'installateur Windows hors ligne."
        }

        $InstallerLauncher = Join-Path $PackageDir "$InstallerBaseName.exe"
        $InstallerVolumes = @(
            Get-ChildItem -LiteralPath $PackageDir -File -Filter "$InstallerBaseName-*.bin"
        )
        if (-not (Test-Path -LiteralPath $InstallerLauncher -PathType Leaf)) {
            throw "Le lanceur Setup Windows est absent : $InstallerLauncher"
        }
        if (-not $SkipBundledAssistantModel -and $InstallerVolumes.Count -eq 0) {
            throw "Les volumes Inno Setup nécessaires au paquet hors ligne sont absents."
        }
        Write-Host "Installateur Windows créé dans : $PackageDir" -ForegroundColor Green
        Write-Host "  Lanceur : $InstallerLauncher"
        Write-Host "  Volumes : $($InstallerVolumes.Count) fichier(s) .bin à conserver avec le lanceur"
    } else {
        Write-Warning "Inno Setup 6 absent : le ZIP portable est prêt, mais aucun Setup.exe n'a été généré."
    }
}
