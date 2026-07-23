#ifndef SourceDir
  #error "SourceDir doit pointer vers le dossier PyInstaller."
#endif
#ifndef OutputDir
  #define OutputDir "."
#endif
#ifndef AppVersion
  #define AppVersion "0.3.8"
#endif
#ifndef IconFile
  #error "IconFile doit pointer vers l'icône Windows."
#endif

[Setup]
AppId={{58F07D5E-8597-45D2-A517-972383BFB2F9}
AppName=Studio Audio
AppVersion={#AppVersion}
AppPublisher=Studio Audio
DefaultDirName={localappdata}\Programs\Studio Audio
DefaultGroupName=Studio Audio
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=StudioAudio-Windows-x64-Setup-{#AppVersion}
; Les poids INT4/Safetensors sont déjà denses et gagnent très peu avec LZMA
; maximal. Le profil fast évite plusieurs heures de compression inutile.
Compression=lzma2/fast
SolidCompression=yes
; Qwen3 INT4 dépasse à lui seul 4,2 Go. Inno Setup impose alors le disk
; spanning : le petit lanceur .exe et tous les volumes .bin doivent être
; distribués ensemble dans le même dossier.
#ifdef BundledAssistantModel
DiskSpanning=yes
DiskSliceSize=1900000000
SlicesPerDisk=1
#endif
WizardStyle=modern
SetupLogging=yes
CloseApplications=yes
RestartApplications=no
UninstallDisplayIcon={app}\TranscriptionLocale.exe
SetupIconFile={#IconFile}

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Studio Audio"; Filename: "{app}\TranscriptionLocale.exe"
Name: "{autodesktop}\Studio Audio"; Filename: "{app}\TranscriptionLocale.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Créer un raccourci sur le Bureau"; GroupDescription: "Raccourcis supplémentaires :"; Flags: unchecked

[Run]
Filename: "{app}\TranscriptionLocale.exe"; Description: "Ouvrir Studio Audio"; Flags: nowait postinstall skipifsilent
