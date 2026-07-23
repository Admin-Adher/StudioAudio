#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Ce build doit être exécuté sur macOS (localement ou via GitHub Actions)." >&2
  exit 1
fi

VERSION="${1:-${TRANSCRIPTION_DESKTOP_VERSION:-0.3.1}}"
REQUESTED_ARCH="${2:-${MACOS_TARGET_ARCH:-auto}}"

if [[ ! "$VERSION" =~ ^[0-9]+(\.[0-9]+){1,2}$ ]]; then
  echo "Version invalide : '$VERSION'. Utilisez une version numérique telle que 0.3.1." >&2
  exit 1
fi

normalize_arch() {
  NORMALIZED_ARCH_VALUE="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$NORMALIZED_ARCH_VALUE" in
    arm64|aarch64) echo "arm64" ;;
    x86_64|amd64|x64) echo "x86_64" ;;
    *) return 1 ;;
  esac
}

HOST_ARCH="$(normalize_arch "$(uname -m)")" || {
  echo "Architecture macOS non prise en charge : $(uname -m)" >&2
  exit 1
}

if [[ "$REQUESTED_ARCH" == "auto" || -z "$REQUESTED_ARCH" ]]; then
  TARGET_ARCH="$HOST_ARCH"
else
  TARGET_ARCH="$(normalize_arch "$REQUESTED_ARCH")" || {
    echo "Architecture demandée invalide : $REQUESTED_ARCH (arm64 ou x86_64 attendu)." >&2
    exit 1
  }
fi

if [[ "$TARGET_ARCH" != "$HOST_ARCH" ]]; then
  echo "Le build doit être natif : hôte $HOST_ARCH, cible $TARGET_ARCH." >&2
  echo "Utilisez un runner Apple Silicon pour arm64 et un runner Intel pour x86_64." >&2
  exit 1
fi

export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-12.0}"

if [[ "$TARGET_ARCH" == "x86_64" ]]; then
  PACKAGE_ARCH="x64"
else
  PACKAGE_ARCH="arm64"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DESKTOP_VENV="$PROJECT_ROOT/.venv-desktop-macos-$PACKAGE_ARCH"
OUTPUT_ROOT="$PROJECT_ROOT/output/desktop/macos/$PACKAGE_ARCH"
WORK_DIR="$OUTPUT_ROOT/build"
DIST_DIR="$OUTPUT_ROOT/dist"
DMG_STAGING="$OUTPUT_ROOT/dmg-staging"
NOTARY_ARCHIVE="$OUTPUT_ROOT/notarization-upload.zip"
PACKAGE_DIR="$PROJECT_ROOT/packages"
REQUIREMENTS_FILE="$PROJECT_ROOT/packaging/requirements-macos.txt"
SPEC_FILE="$PROJECT_ROOT/packaging/transcription_locale.spec"

case "$OUTPUT_ROOT" in
  "$PROJECT_ROOT"/output/desktop/macos/*) ;;
  *)
    echo "Chemin de sortie non sûr : $OUTPUT_ROOT" >&2
    exit 1
    ;;
esac

if [[ -n "${MACOS_PYTHON:-}" ]]; then
  PYTHON_EXECUTABLE="$MACOS_PYTHON"
elif command -v python3.12 >/dev/null 2>&1; then
  PYTHON_EXECUTABLE="$(command -v python3.12)"
else
  echo "Python 3.12 est requis pour reproduire les bundles macOS." >&2
  echo "Installez-le ou fournissez son chemin via MACOS_PYTHON." >&2
  exit 1
fi

PYTHON_ARCH="$("$PYTHON_EXECUTABLE" -c 'import platform; print(platform.machine())')"
PYTHON_ARCH="$(normalize_arch "$PYTHON_ARCH")" || {
  echo "Architecture Python non reconnue : $PYTHON_ARCH" >&2
  exit 1
}
if [[ "$PYTHON_ARCH" != "$TARGET_ARCH" ]]; then
  echo "Python est $PYTHON_ARCH mais le bundle demandé est $TARGET_ARCH." >&2
  exit 1
fi

if [[ ! -x "$DESKTOP_VENV/bin/python" ]]; then
  "$PYTHON_EXECUTABLE" -m venv "$DESKTOP_VENV"
fi

VENV_ARCH="$($DESKTOP_VENV/bin/python -c 'import platform; print(platform.machine())')"
VENV_ARCH="$(normalize_arch "$VENV_ARCH")" || {
  echo "Architecture de l'environnement virtuel non reconnue : $VENV_ARCH" >&2
  exit 1
}
if [[ "$VENV_ARCH" != "$TARGET_ARCH" ]]; then
  echo "L'environnement $DESKTOP_VENV est $VENV_ARCH au lieu de $TARGET_ARCH." >&2
  echo "Supprimez uniquement cet environnement puis relancez le build." >&2
  exit 1
fi

"$DESKTOP_VENV/bin/python" -m pip install --upgrade pip wheel setuptools
"$DESKTOP_VENV/bin/python" -m pip install -r "$REQUIREMENTS_FILE"
"$DESKTOP_VENV/bin/python" -m pip check

mkdir -p "$OUTPUT_ROOT" "$PACKAGE_DIR"
rm -rf -- "$WORK_DIR" "$DIST_DIR" "$DMG_STAGING"
rm -f -- "$NOTARY_ARCHIVE"
mkdir -p "$WORK_DIR" "$DIST_DIR"

export TRANSCRIPTION_DESKTOP_VERSION="$VERSION"
export TRANSCRIPTION_DESKTOP_ARCH="$TARGET_ARCH"

"$DESKTOP_VENV/bin/python" -m PyInstaller \
  --noconfirm \
  --clean \
  --workpath "$WORK_DIR" \
  --distpath "$DIST_DIR" \
  "$SPEC_FILE"

APP_PATH="$DIST_DIR/Studio Audio.app"
APP_EXECUTABLE="$APP_PATH/Contents/MacOS/TranscriptionLocale"
INFO_PLIST="$APP_PATH/Contents/Info.plist"

if [[ ! -x "$APP_EXECUTABLE" ]]; then
  echo "Le bundle attendu est absent ou incomplet : $APP_PATH" >&2
  exit 1
fi

plutil -lint "$INFO_PLIST"
BUNDLE_VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$INFO_PLIST")"
if [[ "$BUNDLE_VERSION" != "$VERSION" ]]; then
  echo "Version du bundle incorrecte : $BUNDLE_VERSION (attendu : $VERSION)." >&2
  exit 1
fi

BUILT_ARCHS="$(lipo -archs "$APP_EXECUTABLE")"
if [[ " $BUILT_ARCHS " != *" $TARGET_ARCH "* ]]; then
  echo "Le binaire contient '$BUILT_ARCHS' mais pas l'architecture $TARGET_ARCH." >&2
  exit 1
fi

# PyInstaller signe chaque binaire Mach-O puis le bundle. Sans Developer ID,
# cette signature est ad hoc : l'application reste native mais macOS affichera
# l'avertissement habituel pour un développeur non identifié.
if ! codesign --verify --deep --strict --verbose=2 "$APP_PATH"; then
  if [[ -n "${APPLE_SIGNING_IDENTITY:-}" ]]; then
    echo "La signature Developer ID produite par PyInstaller est invalide." >&2
    exit 1
  fi
  codesign --force --deep --sign - "$APP_PATH"
  codesign --verify --deep --strict --verbose=2 "$APP_PATH"
fi

if [[ -n "${APPLE_NOTARY_PROFILE:-}" ]]; then
  if [[ -z "${APPLE_SIGNING_IDENTITY:-}" ]]; then
    echo "APPLE_NOTARY_PROFILE exige également APPLE_SIGNING_IDENTITY." >&2
    exit 1
  fi
  ditto -c -k --sequesterRsrc --keepParent "$APP_PATH" "$NOTARY_ARCHIVE"
  xcrun notarytool submit "$NOTARY_ARCHIVE" \
    --keychain-profile "$APPLE_NOTARY_PROFILE" --wait
  xcrun stapler staple "$APP_PATH"
  xcrun stapler validate "$APP_PATH"
  rm -f -- "$NOTARY_ARCHIVE"
fi

PACKAGE_BASE="StudioAudio-macOS-$PACKAGE_ARCH-$VERSION"
DMG_PATH="$PACKAGE_DIR/$PACKAGE_BASE.dmg"

rm -f -- "$DMG_PATH"

mkdir -p "$DMG_STAGING"
ditto "$APP_PATH" "$DMG_STAGING/Studio Audio.app"
ln -s /Applications "$DMG_STAGING/Applications"
hdiutil create \
  -volname "Studio Audio" \
  -srcfolder "$DMG_STAGING" \
  -ov \
  -format UDZO \
  "$DMG_PATH"
hdiutil verify "$DMG_PATH"
rm -rf -- "$DMG_STAGING"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "architecture=$PACKAGE_ARCH"
    echo "app=$APP_PATH"
    echo "executable=$APP_EXECUTABLE"
    echo "dmg=$DMG_PATH"
  } >> "$GITHUB_OUTPUT"
fi

echo
echo "Bundle macOS natif prêt ($TARGET_ARCH) :"
echo "  Application : $APP_PATH"
echo "  Image disque: $DMG_PATH"
if [[ -z "${APPLE_SIGNING_IDENTITY:-}" ]]; then
  echo "  Signature    : ad hoc (clic droit > Ouvrir au premier lancement)."
fi
