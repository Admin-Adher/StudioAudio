#ifndef SourceDir
  #error "SourceDir doit pointer vers le dossier PyInstaller."
#endif
#ifndef OutputDir
  #define OutputDir "."
#endif
#ifndef AppVersion
  #define AppVersion "0.3.2"
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
Compression=lzma2/max
SolidCompression=yes
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
