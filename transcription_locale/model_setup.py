"""Préparation anticipée des modèles utilisés par l'application."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .core import (
    MODEL_DIR,
    WESPEAKER_MODEL_DIR,
    WHISPER_MODEL_DIR,
    _load_whisper_model,
)


SETUP_VERSION = 1
SETUP_MARKER = MODEL_DIR / "modeles-prepares.json"
WESPEAKER_PATH = WESPEAKER_MODEL_DIR / "en" / "model.onnx"
LEGACY_WESPEAKER_PATH = Path.home() / ".wespeaker" / "en" / "model.onnx"
EXPECTED_DOWNLOAD_BYTES = 2_285_000_000

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


def setup_components_status() -> dict[str, bool]:
    return {
        "base": _model_is_cached("base"),
        "small": _model_is_cached("small"),
        "turbo": _model_is_cached("turbo"),
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
    total = 0
    if WHISPER_MODEL_DIR.is_dir():
        for path in WHISPER_MODEL_DIR.rglob("*"):
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
    total_stages = len(MODEL_STAGES) + 2

    for index, (model_id, label) in enumerate(MODEL_STAGES, start=1):
        if callback is not None:
            callback(label, index, total_stages)
        _load_whisper_model(model_id)

    if callback is not None:
        callback("Séparation des interlocuteurs", 4, total_stages)
    import wespeakerruntime as wespeaker_rt

    wespeaker_rt.Speaker(lang="en")

    if callback is not None:
        callback("Détection de la parole", 5, total_stages)
    from silero_vad import load_silero_vad

    load_silero_vad()

    marker = {
        "version": SETUP_VERSION,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "models": [model_id for model_id, _ in MODEL_STAGES],
        "speaker_model": "wespeaker-resnet34-lm",
    }
    SETUP_MARKER.write_text(
        json.dumps(marker, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
