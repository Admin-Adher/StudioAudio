# Distribution bureau Windows et macOS

L'application bureau conserve le moteur Python/Gradio existant, mais l'affiche
dans une vraie fenêtre native avec **pywebview** : Edge WebView2 sous Windows et
WKWebView sous macOS. Aucun serveur n'est exposé au réseau ; Gradio écoute sur un
port aléatoire de `127.0.0.1` et s'arrête avec la fenêtre.

## Données et mises à jour

Les exécutables n'écrivent jamais dans leur dossier d'installation :

- Windows : `%LOCALAPPDATA%\TranscriptionLocale`
- macOS : `~/Library/Application Support/TranscriptionLocale`

Les audios, modèles, checkpoints, résultats et discussions survivent donc à une
mise à jour ou à la désinstallation du binaire. Ils ne sont volontairement pas
supprimés par l'installateur. En mode source, les dossiers historiques du projet
restent utilisés ; `TRANSCRIPTION_APP_DATA_DIR` permet une migration explicite
ou un emplacement administré.

Un seul processus bureau peut accéder au workspace à la fois. Une fermeture
pendant une transcription laisse le dernier checkpoint sur disque ; la reprise
existante s'applique au lancement suivant.

## Construire sous Windows

Depuis PowerShell, à la racine du projet :

```powershell
.\scripts\build_desktop_windows.ps1
```

Le script crée un environnement isolé `.venv-desktop`, construit une version
PyInstaller **one-folder** et produit :

- `packages\StudioAudio-Windows-x64-Portable-0.3.2.zip`
- un `Setup.exe` si Inno Setup 6 est installé.

Le build Windows installe et embarque aussi le runtime OpenVINO requis par le
moteur Intel Arc recommandé. Le modèle lui-même reste préparé une seule fois
dans le dossier de données de chaque utilisateur afin de ne pas gonfler chaque
mise à jour du logiciel.

Le ZIP doit être entièrement décompressé avant d'ouvrir
`TranscriptionLocale.exe` (nom technique interne). Le format one-folder est volontaire : il démarre
plus vite qu'un gros exécutable one-file et évite une extraction temporaire de
Torch à chaque ouverture.

Windows 10/11 contient normalement WebView2. Sur un poste d'entreprise qui en
serait dépourvu, installer le runtime Evergreen WebView2 avant l'application.

Pour une diffusion externe sans alerte SmartScreen, signer le `Setup.exe` et
l'exécutable avec un certificat Authenticode au sein du processus de livraison.

## Construire sous macOS

Le `.app` macOS doit être compilé **sur un Mac** ; PyInstaller ne sait pas
cross-compiler un bundle Cocoa depuis Windows.

```bash
bash scripts/build_desktop_macos.sh 0.3.2 arm64
```

Le script utilise un environnement `.venv-desktop-macos-arm64`, génère le véritable
`Studio Audio.app`, puis l'unique livrable
`packages/StudioAudio-macOS-arm64-0.3.2.dmg`. Le workflow macOS actuel cible
uniquement les Mac Apple Silicon et refuse un runner ou un binaire qui ne serait
pas strictement `arm64`.

Avant de publier le DMG, le workflow monte l'image disque et lance directement
le binaire du `Studio Audio.app` qu'elle contient avec `--smoke-test`. La livraison
est bloquée si les ressources embarquées sont incomplètes, si le serveur local ne
répond pas, si son arrêt n'est pas propre ou si un processus reste actif.

Pour signer et notariser la livraison :

```bash
export APPLE_SIGNING_IDENTITY="Developer ID Application: Organisation (TEAMID)"
export APPLE_NOTARY_PROFILE="transcription-locale"
bash scripts/build_desktop_macos.sh 0.3.2 arm64
```

Sans Developer ID/notarisation, Gatekeeper peut afficher un avertissement aux
collègues, même si l'application est entièrement locale.

## Contrôles avant diffusion

Tester le DMG arm64 sur un Mac Apple Silicon réel, au minimum :

1. premier lancement et page de préparation des modèles ;
2. fenêtre initiale 1280×820, taille minimale 1024×720 et plein écran ;
3. ajout, renommage, suppression et réorganisation de plusieurs audios ;
4. transcription, fermeture en cours, puis reprise après relance ;
5. lecture audio, zoom de timeline, rôles Client/Master, discussions et exports ;
6. mise à jour par-dessus une version existante sans perte du workspace ;
7. lancement hors ligne une fois les modèles préparés.

Le journal de diagnostic rotatif se trouve dans `logs/application.log` sous le
dossier de données de l'application.
