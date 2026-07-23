"""Préparation anticipée des modèles utilisés par l'application."""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .core import (
    MODEL_DIR,
    WESPEAKER_MODEL_DIR,
    WHISPER_MODEL_DIR,
    _load_whisper_model,
)
from .assistant_engine import (
    ASSISTANT_MODEL_REPOSITORY,
    ASSISTANT_MODEL_REVISION,
    ASSISTANT_REQUIRED_MODEL_FILES,
    DEFAULT_ASSISTANT_MODEL_DIR,
    resolve_openvino_assistant_model_path,
)
from .mlx_assistant_backend import (
    DEFAULT_MLX_ASSISTANT_MODEL_DIR,
    MLX_ASSISTANT_MODEL_REPOSITORY,
    MLX_ASSISTANT_MODEL_REVISION,
    MLX_REQUIRED_MODEL_FILES,
    resolve_mlx_model_path,
)
from .openvino_backend import (
    DEFAULT_OPENVINO_MODEL_DIR,
    OPENVINO_MODEL_REPOSITORY,
    OPENVINO_MODEL_REVISION,
    REQUIRED_MODEL_FILES,
    openvino_gpu_runtime_available,
)


SETUP_VERSION = 3
SETUP_MARKER = MODEL_DIR / "modeles-prepares.json"
WESPEAKER_PATH = WESPEAKER_MODEL_DIR / "en" / "model.onnx"
LEGACY_WESPEAKER_PATH = Path.home() / ".wespeaker" / "en" / "model.onnx"
EXPECTED_DOWNLOAD_BYTES = 8_100_000_000

MODEL_CACHE_DIRECTORIES = {
    "base": "models--Systran--faster-whisper-base",
    "small": "models--Systran--faster-whisper-small",
    "turbo": "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo",
}

MODEL_STAGES = [
    ("base", "Mode Très rapide"),
    ("turbo", "Mode Précis"),
    ("small", "Mode Équilibré"),
]

SetupCallback = Callable[[str, int, int], None]


def _model_is_cached(model_id: str) -> bool:
    directory_name = MODEL_CACHE_DIRECTORIES[model_id]
    directory = WHISPER_MODEL_DIR / directory_name
    return directory.is_dir() and any(directory.rglob("model.bin"))


def _wespeaker_model_path() -> Path:
    """Accepte le cache applicatif et l'ancien cache global de WeSpeaker."""

    if WESPEAKER_PATH.is_file():
        return WESPEAKER_PATH
    return LEGACY_WESPEAKER_PATH


def _openvino_is_required() -> bool:
    return sys.platform == "win32" and openvino_gpu_runtime_available()


def _openvino_model_is_cached() -> bool:
    return all(
        (DEFAULT_OPENVINO_MODEL_DIR / name).is_file()
        for name in REQUIRED_MODEL_FILES
    )


def _assistant_is_required() -> bool:
    return _assistant_platform_kind() is not None


def _assistant_model_is_cached() -> bool:
    kind = _assistant_platform_kind()
    if kind == "openvino":
        return resolve_openvino_assistant_model_path() is not None
    if kind == "mlx":
        return resolve_mlx_model_path() is not None
    return True


def _assistant_platform_kind() -> str | None:
    if sys.platform == "win32" and openvino_gpu_runtime_available():
        return "openvino"
    if sys.platform == "darwin" and platform.machine().lower() in {
        "arm64",
        "aarch64",
    }:
        return "mlx"
    return None


def setup_components_status() -> dict[str, bool]:
    openvino_required = _openvino_is_required()
    assistant_required = _assistant_is_required()
    return {
        "base": _model_is_cached("base"),
        "small": _model_is_cached("small"),
        "turbo": _model_is_cached("turbo"),
        "openvino": not openvino_required or _openvino_model_is_cached(),
        "assistant": not assistant_required or _assistant_model_is_cached(),
        "speakers": _wespeaker_model_path().is_file(),
    }


def models_ready() -> bool:
    if not SETUP_MARKER.is_file():
        return False
    try:
        marker = json.loads(SETUP_MARKER.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return (
        int(marker.get("version") or 0) == SETUP_VERSION
        and all(setup_components_status().values())
    )


def downloaded_bytes() -> int:
    """Taille des moteurs déjà disponibles, cache utilisateur ou bundle.

    ``EXPECTED_DOWNLOAD_BYTES`` représente l'ensemble d'un premier lancement
    complet. Un assistant livré dans les ressources PyInstaller ne doit donc
    pas rester invisible dans le numérateur : sinon l'interface semble figée
    pendant la préparation des autres moteurs alors que près de 5 Go sont déjà
    présents et immédiatement utilisables.
    """

    total = 0
    if WHISPER_MODEL_DIR.is_dir():
        for path in WHISPER_MODEL_DIR.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
    if DEFAULT_OPENVINO_MODEL_DIR.is_dir():
        for path in DEFAULT_OPENVINO_MODEL_DIR.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
    assistant_directories = {
        DEFAULT_ASSISTANT_MODEL_DIR.resolve(),
        DEFAULT_MLX_ASSISTANT_MODEL_DIR.resolve(),
    }
    # Les résolveurs donnent priorité au chemin explicite puis au modèle
    # embarqué dans RESOURCE_ROOT, avant le cache durable. L'ensemble évite de
    # compter deux fois un même dossier lorsqu'il correspond déjà au chemin par
    # défaut dans APP_DATA_ROOT.
    for resolver in (
        resolve_openvino_assistant_model_path,
        resolve_mlx_model_path,
    ):
        try:
            resolved_assistant = resolver()
        except (OSError, RuntimeError, ValueError):
            resolved_assistant = None
        if resolved_assistant is not None:
            assistant_directories.add(Path(resolved_assistant).resolve())

    for assistant_directory in assistant_directories:
        if not assistant_directory.is_dir():
            continue
        for path in assistant_directory.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
    try:
        speaker_path = _wespeaker_model_path()
        if speaker_path.is_file():
            total += speaker_path.stat().st_size
    except OSError:
        pass
    return total


def prepare_all_models(callback: SetupCallback | None = None) -> None:
    """Télécharge et valide tous les moteurs avant la première transcription."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    WHISPER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    prepare_openvino = _openvino_is_required()
    assistant_kind = _assistant_platform_kind()
    prepare_assistant = (
        assistant_kind is not None and not _assistant_model_is_cached()
    )
    total_stages = (
        len(MODEL_STAGES)
        + 2
        + int(prepare_openvino)
        + int(prepare_assistant)
    )

    for index, (model_id, label) in enumerate(MODEL_STAGES, start=1):
        if callback is not None:
            callback(label, index, total_stages)
        _load_whisper_model(model_id)

    next_stage = len(MODEL_STAGES) + 1
    if prepare_openvino:
        if callback is not None:
            callback("Intel Arc · pilote", next_stage, total_stages)
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("DO_NOT_TRACK", "1")
        from huggingface_hub import snapshot_download

        DEFAULT_OPENVINO_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=OPENVINO_MODEL_REPOSITORY,
            revision=OPENVINO_MODEL_REVISION,
            local_dir=DEFAULT_OPENVINO_MODEL_DIR,
        )
        missing = [
            name
            for name in REQUIRED_MODEL_FILES
            if not (DEFAULT_OPENVINO_MODEL_DIR / name).is_file()
        ]
        if missing:
            raise RuntimeError(
                "Fichiers OpenVINO manquants : " + ", ".join(missing)
            )
        next_stage += 1

    if prepare_assistant:
        if callback is not None:
            callback("Assistant Premium · Qwen3 8B", next_stage, total_stages)
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("DO_NOT_TRACK", "1")
        from huggingface_hub import snapshot_download

        if assistant_kind == "mlx":
            assistant_directory = DEFAULT_MLX_ASSISTANT_MODEL_DIR
            assistant_repository = MLX_ASSISTANT_MODEL_REPOSITORY
            assistant_revision = MLX_ASSISTANT_MODEL_REVISION
            assistant_required_files = MLX_REQUIRED_MODEL_FILES
        else:
            assistant_directory = DEFAULT_ASSISTANT_MODEL_DIR
            assistant_repository = ASSISTANT_MODEL_REPOSITORY
            assistant_revision = ASSISTANT_MODEL_REVISION
            assistant_required_files = ASSISTANT_REQUIRED_MODEL_FILES

        assistant_directory.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=assistant_repository,
            revision=assistant_revision,
            local_dir=assistant_directory,
        )
        missing = [
            name
            for name in assistant_required_files
            if not (assistant_directory / name).is_file()
        ]
        if assistant_kind == "mlx" and not any(
            assistant_directory.glob("*.safetensors")
        ):
            missing.append("*.safetensors")
        if missing:
            raise RuntimeError(
                "Fichiers de l'Assistant Premium manquants : "
                + ", ".join(missing)
            )
        next_stage += 1

    if callback is not None:
        callback("Séparation des interlocuteurs", next_stage, total_stages)
    import wespeakerruntime as wespeaker_rt

    wespeaker_rt.Speaker(lang="en")

    if callback is not None:
        callback("Détection de la parole", next_stage + 1, total_stages)
    from silero_vad import load_silero_vad

    load_silero_vad()

    marker = {
        "version": SETUP_VERSION,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "models": [model_id for model_id, _ in MODEL_STAGES],
        "openvino_model": (
            OPENVINO_MODEL_REPOSITORY if prepare_openvino else None
        ),
        "assistant_model": (
            (
                MLX_ASSISTANT_MODEL_REPOSITORY
                if assistant_kind == "mlx"
                else ASSISTANT_MODEL_REPOSITORY
            )
            if assistant_kind is not None
            else None
        ),
        "speaker_model": "wespeaker-resnet34-lm",
    }
    SETUP_MARKER.write_text(
        json.dumps(marker, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
