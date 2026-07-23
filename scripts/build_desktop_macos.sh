#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Ce build doit être exécuté sur macOS (localement ou via GitHub Actions)." >&2
  exit 1
fi

VERSION="${1:-${TRANSCRIPTION_DESKTOP_VERSION:-0.3.8}}"
REQUESTED_ARCH="${2:-${MACOS_TARGET_ARCH:-auto}}"
BUNDLE_ASSISTANT_MODEL="${3:-${BUNDLE_ASSISTANT_MODEL:-1}}"

case "$(printf '%s' "$BUNDLE_ASSISTANT_MODEL" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on) BUNDLE_ASSISTANT_MODEL=1 ;;
  0|false|no|off) BUNDLE_ASSISTANT_MODEL=0 ;;
  *)
    echo "BUNDLE_ASSISTANT_MODEL doit valoir 1 ou 0." >&2
    exit 1
    ;;
esac

if [[ ! "$VERSION" =~ ^[0-9]+(\.[0-9]+){1,2}$ ]]; then
  echo "Version invalide : '$VERSION'. Utilisez une version numérique telle que 0.3.2." >&2
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

export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}"

if [[ "$TARGET_ARCH" == "x86_64" ]]; then
  PACKAGE_ARCH="x64"
else
  PACKAGE_ARCH="arm64"
fi

PRUNE_MODEL_SOURCE_AFTER_BUILD="$(
  printf '%s' "${MACOS_PRUNE_MODEL_SOURCE_AFTER_BUILD:-${GITHUB_ACTIONS:-false}}" \
    | tr '[:upper:]' '[:lower:]'
)"
case "$PRUNE_MODEL_SOURCE_AFTER_BUILD" in
  1|true|yes|on) PRUNE_MODEL_SOURCE_AFTER_BUILD=1 ;;
  0|false|no|off|"") PRUNE_MODEL_SOURCE_AFTER_BUILD=0 ;;
  *)
    echo "MACOS_PRUNE_MODEL_SOURCE_AFTER_BUILD doit valoir 1 ou 0." >&2
    exit 1
    ;;
esac

PRUNE_VENV_AFTER_BUILD="$(
  printf '%s' "${MACOS_PRUNE_VENV_AFTER_BUILD:-${GITHUB_ACTIONS:-false}}" \
    | tr '[:upper:]' '[:lower:]'
)"
case "$PRUNE_VENV_AFTER_BUILD" in
  1|true|yes|on) PRUNE_VENV_AFTER_BUILD=1 ;;
  0|false|no|off|"") PRUNE_VENV_AFTER_BUILD=0 ;;
  *)
    echo "MACOS_PRUNE_VENV_AFTER_BUILD doit valoir 1 ou 0." >&2
    exit 1
    ;;
esac

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
BUNDLED_MODELS_ROOT="$OUTPUT_ROOT/bundled-models"
BUNDLED_ASSISTANT_MODEL_DIR="$BUNDLED_MODELS_ROOT/qwen3-8b-mlx-4bit"
ASSISTANT_MODEL_REPOSITORY="mlx-community/Qwen3-8B-4bit"
ASSISTANT_MODEL_REVISION="545dc4251c05440727734bcd94334791f6ab0192"
ASSISTANT_MODEL_WEIGHTS_SIZE="4607835174"
ASSISTANT_MODEL_WEIGHTS_SHA256="f2d29621aab300336ad645567ff38c42aac755513006ef4e8a579cf7ef5256d8"

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

# Sur un runner macOS 15, pip préfère naturellement les roues MLX marquées
# macOS 15 même lorsqu'une roue macOS 13 existe. Réinstaller explicitement les
# deux roues macOS 13 évite de livrer un .app qui mentirait sur sa version
# minimale et refuserait de démarrer chez un collègue resté sous Ventura.
if [[ "$TARGET_ARCH" == "arm64" ]]; then
  MLX_WHEEL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/studio-audio-mlx-wheels.XXXXXX")"
  "$DESKTOP_VENV/bin/python" -m pip download \
    --dest "$MLX_WHEEL_DIR" \
    --only-binary=:all: \
    --no-deps \
    --platform macosx_13_0_arm64 \
    --implementation cp \
    --python-version 312 \
    --abi cp312 \
    "mlx==0.25.2"
  "$DESKTOP_VENV/bin/python" -m pip install \
    --force-reinstall \
    --no-deps \
    "$MLX_WHEEL_DIR"/mlx-0.25.2-*-macosx_13_0_arm64.whl
  rm -rf -- "$MLX_WHEEL_DIR"
fi

"$DESKTOP_VENV/bin/python" -m pip check

mkdir -p "$OUTPUT_ROOT" "$PACKAGE_DIR" "$BUNDLED_MODELS_ROOT"

if [[ "$BUNDLE_ASSISTANT_MODEL" -eq 1 ]]; then
  mkdir -p "$BUNDLED_ASSISTANT_MODEL_DIR"
  HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1 \
    "$DESKTOP_VENV/bin/python" - \
    "$ASSISTANT_MODEL_REPOSITORY" \
    "$ASSISTANT_MODEL_REVISION" \
    "$ASSISTANT_MODEL_WEIGHTS_SIZE" \
    "$ASSISTANT_MODEL_WEIGHTS_SHA256" \
    "$BUNDLED_ASSISTANT_MODEL_DIR" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repository, revision, expected_size_value, expected_sha256, destination = sys.argv[1:6]
expected_size = int(expected_size_value)
model_dir = Path(destination).resolve()
model_dir.mkdir(parents=True, exist_ok=True)
required_model_files = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)
weights_path = model_dir / "model.safetensors"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


actual_sha256 = ""
model_is_reusable = False
if (
    all((model_dir / name).is_file() for name in required_model_files)
    and weights_path.stat().st_size == expected_size
):
    actual_sha256 = file_sha256(weights_path)
    model_is_reusable = actual_sha256 == expected_sha256
if not model_is_reusable:
    shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repository,
        revision=revision,
        local_dir=model_dir,
    )
    weights_path = model_dir / "model.safetensors"
    if not weights_path.is_file() or weights_path.stat().st_size != expected_size:
        raise SystemExit(
            "La taille des poids MLX téléchargés ne correspond pas au modèle épinglé."
        )
    actual_sha256 = file_sha256(weights_path)
weights_path = model_dir / "model.safetensors"
if not weights_path.is_file() or weights_path.stat().st_size != expected_size:
    raise SystemExit(
        "La taille des poids MLX téléchargés ne correspond pas au modèle épinglé."
    )
if actual_sha256 != expected_sha256:
    raise SystemExit(
        "Le SHA-256 des poids MLX téléchargés ne correspond pas au modèle épinglé."
    )
(model_dir / "studio-audio-model.json").write_text(
    json.dumps(
        {
            "format": 1,
            "name": "Qwen3 8B MLX 4-bit",
            "repository": repository,
            "revision": revision,
            "runtime": "mlx",
            "weights_size": expected_size,
            "weights_sha256": expected_sha256,
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)
PY
  rm -rf -- "$BUNDLED_ASSISTANT_MODEL_DIR/.cache"

  "$DESKTOP_VENV/bin/python" - \
    "$BUNDLED_ASSISTANT_MODEL_DIR" \
    "$ASSISTANT_MODEL_REPOSITORY" \
    "$ASSISTANT_MODEL_REVISION" \
    "$ASSISTANT_MODEL_WEIGHTS_SIZE" \
    "$ASSISTANT_MODEL_WEIGHTS_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1]).resolve()
repository, revision, expected_size_value, expected_sha256 = sys.argv[2:6]
expected_size = int(expected_size_value)
required = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "studio-audio-model.json",
)
missing = [name for name in required if not (model_dir / name).is_file()]
if missing:
    raise SystemExit(
        "Le modèle assistant macOS est incomplet : " + ", ".join(missing)
    )
weights_path = model_dir / "model.safetensors"
if weights_path.stat().st_size != expected_size:
    raise SystemExit("La taille des poids MLX n'est pas celle du modèle épinglé.")
digest = hashlib.sha256()
with weights_path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != expected_sha256:
    raise SystemExit("Le SHA-256 des poids MLX n'est pas celui du modèle épinglé.")
manifest = json.loads(
    (model_dir / "studio-audio-model.json").read_text(encoding="utf-8")
)
if (
    manifest.get("repository") != repository
    or manifest.get("revision") != revision
    or manifest.get("runtime") != "mlx"
    or manifest.get("weights_size") != expected_size
    or manifest.get("weights_sha256") != expected_sha256
):
    raise SystemExit("Le manifeste du modèle MLX n'est pas celui épinglé.")
PY
  export TRANSCRIPTION_BUNDLED_ASSISTANT_MODEL_DIR="$BUNDLED_ASSISTANT_MODEL_DIR"
  export TRANSCRIPTION_BUNDLED_ASSISTANT_MODEL_KIND=mlx
  echo "Modèle assistant MLX hors ligne prêt : $BUNDLED_ASSISTANT_MODEL_DIR"
else
  unset TRANSCRIPTION_BUNDLED_ASSISTANT_MODEL_DIR
  unset TRANSCRIPTION_BUNDLED_ASSISTANT_MODEL_KIND
  echo "Attention : build macOS sans modèle assistant embarqué." >&2
fi

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

if [[ "$BUNDLE_ASSISTANT_MODEL" -eq 1 ]]; then
  APP_ASSISTANT_MODEL="$APP_PATH/Contents/Frameworks/modeles/assistant/qwen3-8b-mlx-4bit"
  "$DESKTOP_VENV/bin/python" - \
    "$APP_ASSISTANT_MODEL" \
    "$ASSISTANT_MODEL_REPOSITORY" \
    "$ASSISTANT_MODEL_REVISION" \
    "$ASSISTANT_MODEL_WEIGHTS_SIZE" \
    "$ASSISTANT_MODEL_WEIGHTS_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
repository, revision, expected_size_value, expected_sha256 = sys.argv[2:6]
expected_size = int(expected_size_value)
required = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "studio-audio-model.json",
)
missing = [name for name in required if not (model_dir / name).is_file()]
if missing:
    raise SystemExit(
        "Ressources Qwen3 MLX absentes du .app : " + ", ".join(missing)
    )
weights_path = model_dir / "model.safetensors"
if weights_path.stat().st_size != expected_size:
    raise SystemExit("La taille des poids Qwen3 MLX du .app est incorrecte.")
if weights_path.is_symlink():
    raise SystemExit("Les poids Qwen3 MLX du .app ne doivent pas être un lien.")
digest = hashlib.sha256()
with weights_path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != expected_sha256:
    raise SystemExit("Le SHA-256 des poids Qwen3 MLX du .app est incorrect.")
manifest = json.loads(
    (model_dir / "studio-audio-model.json").read_text(encoding="utf-8")
)
if (
    manifest.get("repository") != repository
    or manifest.get("revision") != revision
    or manifest.get("runtime") != "mlx"
    or manifest.get("weights_size") != expected_size
    or manifest.get("weights_sha256") != expected_sha256
):
    raise SystemExit("Manifeste Qwen3 MLX incorrect dans le .app.")
PY
  echo "Ressources Qwen3 MLX du véritable .app : OK"
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

# Le .app est désormais autonome, complet, de la bonne architecture et signé.
# Libérer le workdir avant le DMG abaisse le pic disque sans toucher à APP_PATH.
case "$WORK_DIR" in
  "$OUTPUT_ROOT"/build) rm -rf -- "$WORK_DIR" ;;
  *)
    echo "Refus de nettoyer un workdir inattendu : $WORK_DIR" >&2
    exit 1
    ;;
esac

# Les runners GitHub sont éphémères et disposent de peu de disque. Le modèle
# source de 4,6 Go peut donc disparaître une fois sa copie autonome validée
# dans le .app. En local il reste, par défaut, réutilisable au prochain build.
if [[
  "$PRUNE_MODEL_SOURCE_AFTER_BUILD" -eq 1
  && "$BUNDLE_ASSISTANT_MODEL" -eq 1
]]; then
  case "$BUNDLED_ASSISTANT_MODEL_DIR" in
    "$OUTPUT_ROOT"/bundled-models/qwen3-8b-mlx-4bit)
      rm -rf -- "$BUNDLED_ASSISTANT_MODEL_DIR"
      rmdir "$BUNDLED_MODELS_ROOT" 2>/dev/null || true
      ;;
    *)
      echo "Refus de nettoyer une source modèle inattendue : $BUNDLED_ASSISTANT_MODEL_DIR" >&2
      exit 1
      ;;
  esac
fi

# Le workflow de smoke-test utilise le Python 3.12 du runner, pas ce venv.
# Sur GitHub (ou sur demande explicite), il peut donc être supprimé après que
# toutes les validations Python et la signature du .app ont réussi.
if [[ "$PRUNE_VENV_AFTER_BUILD" -eq 1 ]]; then
  case "$DESKTOP_VENV" in
    "$PROJECT_ROOT"/.venv-desktop-macos-"$PACKAGE_ARCH")
      rm -rf -- "$DESKTOP_VENV"
      ;;
    *)
      echo "Refus de nettoyer un venv inattendu : $DESKTOP_VENV" >&2
      exit 1
      ;;
  esac
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
STAGED_APP_PATH="$DMG_STAGING/Studio Audio.app"

staged_app_is_complete() {
  [[
    -x "$STAGED_APP_PATH/Contents/MacOS/TranscriptionLocale"
    && -f "$STAGED_APP_PATH/Contents/Info.plist"
  ]] || return 1
  if [[ "$BUNDLE_ASSISTANT_MODEL" -eq 1 ]]; then
    STAGED_MODEL_WEIGHTS="$STAGED_APP_PATH/Contents/Frameworks/modeles/assistant/qwen3-8b-mlx-4bit/model.safetensors"
    [[
      -f "$STAGED_MODEL_WEIGHTS"
      && "$(stat -f '%z' "$STAGED_MODEL_WEIGHTS")" -eq "$ASSISTANT_MODEL_WEIGHTS_SIZE"
    ]] || return 1
  fi
}

# Sur APFS, cp -cR clone les blocs du .app au lieu de dupliquer environ 5 Go.
# Si le clone n'est pas disponible, supprimer toute copie partielle avant le
# repli ditto afin de ne jamais mélanger deux stagings incomplets.
if cp -cR "$APP_PATH" "$STAGED_APP_PATH" && staged_app_is_complete; then
  echo "Staging DMG créé par clone APFS."
else
  echo "Clone APFS indisponible ; repli vers ditto." >&2
  rm -rf -- "$STAGED_APP_PATH"
  ditto "$APP_PATH" "$STAGED_APP_PATH"
  if ! staged_app_is_complete; then
    echo "Le staging DMG obtenu par repli est incomplet." >&2
    exit 1
  fi
fi
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
