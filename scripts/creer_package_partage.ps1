$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PackageDir = Join-Path $ProjectRoot "packages"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Destination = Join-Path $PackageDir "StudioAudio-$Timestamp.zip"
$Staging = Join-Path $PackageDir "staging-$Timestamp"

New-Item -ItemType Directory -Force -Path $PackageDir | Out-Null
New-Item -ItemType Directory -Force -Path $Staging | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Staging "transcription_locale") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Staging "scripts") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Staging "assets\icons") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Staging "packaging\windows") | Out-Null

$RootFiles = @(
    "app.py",
    "desktop_app.py",
    "requirements.txt",
    "requirements-desktop.txt",
    "requirements-openvino.txt",
    "README.md",
    "INSTALLER.bat",
    "INSTALLER_OPENVINO.bat",
    "LANCER.bat",
    "CREER_PACKAGE_PARTAGE.bat"
)
foreach ($Name in $RootFiles) {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot $Name) -Destination (Join-Path $Staging $Name)
}

Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "transcription_locale") -Filter "*.py" -File |
    Copy-Item -Destination (Join-Path $Staging "transcription_locale")
Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "scripts") -Filter "*.ps1" -File |
    Copy-Item -Destination (Join-Path $Staging "scripts")
Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "scripts") -Filter "*.py" -File |
    Copy-Item -Destination (Join-Path $Staging "scripts")
Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "scripts") -Filter "*.sh" -File |
    Copy-Item -Destination (Join-Path $Staging "scripts")

# L'interface charge cette icone directement depuis le paquet distribue.
Copy-Item -LiteralPath (Join-Path $ProjectRoot "assets\icons\trash-2.svg") `
    -Destination (Join-Path $Staging "assets\icons\trash-2.svg")
Copy-Item -LiteralPath (Join-Path $ProjectRoot "assets\app-icon.png") `
    -Destination (Join-Path $Staging "assets\app-icon.png")
Copy-Item -LiteralPath (Join-Path $ProjectRoot "assets\app-icon.ico") `
    -Destination (Join-Path $Staging "assets\app-icon.ico")
Copy-Item -LiteralPath (Join-Path $ProjectRoot "assets\app-icon.icns") `
    -Destination (Join-Path $Staging "assets\app-icon.icns")
Copy-Item -LiteralPath (Join-Path $ProjectRoot "packaging\transcription_locale.spec") `
    -Destination (Join-Path $Staging "packaging\transcription_locale.spec")
Copy-Item -LiteralPath (Join-Path $ProjectRoot "packaging\README_DESKTOP.md") `
    -Destination (Join-Path $Staging "packaging\README_DESKTOP.md")
Copy-Item -LiteralPath (Join-Path $ProjectRoot "packaging\windows\TranscriptionLocale.iss") `
    -Destination (Join-Path $Staging "packaging\windows\TranscriptionLocale.iss")

Compress-Archive -Path (Join-Path $Staging "*") -DestinationPath $Destination -CompressionLevel Optimal

$ResolvedPackageDir = (Resolve-Path -LiteralPath $PackageDir).Path.TrimEnd('\') + '\'
$ResolvedStaging = (Resolve-Path -LiteralPath $Staging).Path
if (-not $ResolvedStaging.StartsWith($ResolvedPackageDir, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Dossier temporaire inattendu : $ResolvedStaging"
}
Remove-Item -LiteralPath $ResolvedStaging -Recurse -Force
Write-Host "Package cree : $Destination" -ForegroundColor Green
