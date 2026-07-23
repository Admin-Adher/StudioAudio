# -*- mode: python ; coding: utf-8 -*-

import os
import platform
import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)


# ``SPECPATH`` est le dossier qui contient ce fichier spec (``packaging``),
# pas le chemin du fichier lui-même. Le projet se trouve donc un niveau plus haut.
project_root = Path(SPECPATH).resolve().parent
app_version = os.environ.get("TRANSCRIPTION_DESKTOP_VERSION", "0.3.8")
windows_icon = project_root / "assets" / "app-icon.ico"
macos_icon = project_root / "assets" / "app-icon.icns"
is_macos = sys.platform == "darwin"
executable_icon = macos_icon if is_macos else windows_icon


def normalize_macos_arch(value):
    normalized = (value or "").strip().lower()
    if normalized in {"", "auto"}:
        normalized = platform.machine().lower()
    aliases = {
        "aarch64": "arm64",
        "amd64": "x86_64",
        "x64": "x86_64",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"arm64", "x86_64"}:
        raise ValueError(
            "TRANSCRIPTION_DESKTOP_ARCH doit valoir arm64 ou x86_64 "
            f"(valeur reçue : {value!r})."
        )
    return normalized


if is_macos and not re.fullmatch(r"\d+(?:\.\d+){1,2}", app_version):
    raise ValueError(
        "TRANSCRIPTION_DESKTOP_VERSION doit être une version macOS numérique "
        "telle que 0.3.2."
    )

target_arch = (
    normalize_macos_arch(os.environ.get("TRANSCRIPTION_DESKTOP_ARCH", "auto"))
    if is_macos
    else None
)
codesign_identity = (
    os.environ.get("APPLE_SIGNING_IDENTITY")
    or os.environ.get("TRANSCRIPTION_CODESIGN_IDENTITY")
    or None
) if is_macos else None
entitlements_value = os.environ.get("TRANSCRIPTION_ENTITLEMENTS_FILE", "").strip()
entitlements_file = Path(entitlements_value).expanduser().resolve() if entitlements_value else None
if entitlements_file is not None and not entitlements_file.is_file():
    raise FileNotFoundError(f"Fichier d'entitlements introuvable : {entitlements_file}")
bundle_identifier = os.environ.get(
    "TRANSCRIPTION_BUNDLE_IDENTIFIER",
    "fr.transcriptionlocale.studio",
)

datas = [(str(project_root / "assets"), "assets")]
binaries = []
hiddenimports = []

bundled_assistant_value = os.environ.get(
    "TRANSCRIPTION_BUNDLED_ASSISTANT_MODEL_DIR",
    "",
).strip()
bundled_assistant_kind = os.environ.get(
    "TRANSCRIPTION_BUNDLED_ASSISTANT_MODEL_KIND",
    "",
).strip().lower()
if bundled_assistant_value:
    bundled_assistant_dir = Path(bundled_assistant_value).expanduser().resolve()
    expected_kind = "mlx" if is_macos else "openvino"
    if bundled_assistant_kind != expected_kind:
        raise ValueError(
            "Le runtime du modèle assistant embarqué doit être "
            f"{expected_kind!r} sur cette plateforme "
            f"(reçu : {bundled_assistant_kind!r})."
        )
    if expected_kind == "mlx":
        bundled_assistant_destination = (
            "modeles/assistant/qwen3-8b-mlx-4bit"
        )
        bundled_assistant_required = (
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "studio-audio-model.json",
        )
        bundled_weights = "model.safetensors"
    else:
        bundled_assistant_destination = (
            "modeles/assistant/qwen3-8b-int4-ov"
        )
        bundled_assistant_required = (
            "config.json",
            "openvino_model.xml",
            "openvino_model.bin",
            "tokenizer.json",
            "tokenizer_config.json",
            "studio-audio-model.json",
        )
        bundled_weights = "openvino_model.bin"

    missing_model_files = [
        name
        for name in bundled_assistant_required
        if not (bundled_assistant_dir / name).is_file()
    ]
    if missing_model_files:
        raise FileNotFoundError(
            "Le modèle assistant à embarquer est incomplet : "
            + ", ".join(missing_model_files)
        )
    if (bundled_assistant_dir / bundled_weights).stat().st_size < 4_000_000_000:
        raise ValueError(
            "Les poids du modèle assistant semblent tronqués : "
            f"{bundled_assistant_dir / bundled_weights}"
        )
    datas.append(
        (str(bundled_assistant_dir), bundled_assistant_destination)
    )

data_packages = (
    "gradio",
    "gradio_client",
    "groovy",
    "safehttpx",
    "webview",
    "faster_whisper",
    "diarize",
    "docx",
    "silero_vad",
    "wespeakerruntime",
)
dynamic_packages = (
    "ctranslate2",
    "av",
    "onnxruntime",
    "torch",
    "torchaudio",
)
submodule_packages = (
    "diarize",
    "faster_whisper",
    "silero_vad",
    "wespeakerruntime",
    "webview",
)
distribution_names = (
    "gradio",
    "gradio-client",
    "fastapi",
    "groovy",
    "safehttpx",
    "faster-whisper",
    "ctranslate2",
    "diarize",
    "pywebview",
    "python-docx",
    "torch",
    "torchaudio",
    "onnxruntime",
    "av",
)

# OpenVINO/Intel Arc reste la voie Windows. Sur macOS, la transcription repose
# sur Faster-Whisper et l'assistant sur MLX/Metal ; ne pas embarquer les
# runtimes Windows/C# inutiles réduit la taille du .app et évite des imports
# Python.NET non disponibles.
if not is_macos:
    data_packages += (
        "openvino",
        "openvino_genai",
        "openvino_tokenizers",
    )
    dynamic_packages += (
        "openvino",
        "openvino_genai",
        "openvino_tokenizers",
    )
    submodule_packages += (
        "clr_loader",
        "pythonnet",
        "openvino",
        "openvino_genai",
        "openvino_tokenizers",
    )
    distribution_names += (
        "pythonnet",
        "openvino",
        "openvino-genai",
        "openvino-tokenizers",
    )
else:
    # Le raisonneur premium macOS utilise Metal via MLX. Les extensions
    # natives et métadonnées doivent être présentes dans le véritable .app,
    # pas seulement dans l'environnement du runner de build.
    data_packages += (
        "mlx",
        "mlx_lm",
        "transformers",
        "tokenizers",
        "safetensors",
        "sentencepiece",
    )
    dynamic_packages += (
        "mlx",
        "tokenizers",
        "safetensors",
        "sentencepiece",
    )
    submodule_packages += (
        "mlx",
        "mlx_lm",
    )
    distribution_names += (
        "mlx",
        "mlx-lm",
        "transformers",
        "tokenizers",
        "safetensors",
        "sentencepiece",
        "huggingface-hub",
    )

for package in data_packages:
    try:
        # Gradio inspecte plusieurs fichiers source au démarrage pour fabriquer
        # les métadonnées d'événements. Ils doivent rester accessibles comme
        # fichiers, même si le bytecode est également archivé.
        datas += collect_data_files(
            package,
            include_py_files=package == "gradio",
        )
    except Exception:
        pass

for package in dynamic_packages:
    try:
        binaries += collect_dynamic_libs(package)
    except Exception:
        pass

for package in submodule_packages:
    try:
        if is_macos and package == "webview":
            hiddenimports += collect_submodules(
                package,
                filter=lambda name: not name.startswith("webview.platforms.")
                or name == "webview.platforms.cocoa",
            )
        else:
            hiddenimports += collect_submodules(package)
    except Exception:
        pass

for distribution in distribution_names:
    try:
        datas += copy_metadata(distribution)
    except Exception:
        pass

hiddenimports = sorted(set(hiddenimports))

excluded_modules = ["IPython", "jupyter", "notebook"]
if is_macos:
    excluded_modules += [
        "clr",
        "clr_loader",
        "pythonnet",
        "openvino",
        "openvino_genai",
        "openvino_tokenizers",
        "webview.platforms.android",
        "webview.platforms.cef",
        "webview.platforms.edgechromium",
        "webview.platforms.gtk",
        "webview.platforms.mshtml",
        "webview.platforms.qt",
        "webview.platforms.win32",
        "webview.platforms.winforms",
    ]

a = Analysis(
    [str(project_root / "desktop_app.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_modules,
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TranscriptionLocale",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(executable_icon) if executable_icon.is_file() else None,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=target_arch,
    codesign_identity=codesign_identity,
    entitlements_file=str(entitlements_file) if entitlements_file else None,
)

collection = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="TranscriptionLocale",
)

if is_macos:
    app = BUNDLE(
        collection,
        name="Studio Audio.app",
        icon=str(macos_icon) if macos_icon.is_file() else None,
        version=app_version,
        bundle_identifier=bundle_identifier,
        info_plist={
            "CFBundleDisplayName": "Studio Audio",
            "CFBundleName": "Studio Audio",
            "CFBundleShortVersionString": app_version,
            "CFBundleVersion": app_version,
            "CFBundlePackageType": "APPL",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "13.0",
            "LSApplicationCategoryType": "public.app-category.productivity",
            "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True},
        },
    )
