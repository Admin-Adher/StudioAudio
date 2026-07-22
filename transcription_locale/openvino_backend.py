"""Backend OpenVINO optionnel pour le pilote Intel Arc.

Ce module ne télécharge rien et n'importe OpenVINO qu'au moment où le pilote
est inspecté ou utilisé. L'application principale reste donc utilisable avec
la seule installation Faster-Whisper.
"""

from __future__ import annotations

import importlib
import gc
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runtime_paths import APP_DATA_ROOT

ROOT_DIR = APP_DATA_ROOT
OPENVINO_BACKEND_ID = "openvino-arc-pilot"
OPENVINO_MODEL_ID = "openvino/whisper-large-v3-turbo-int8-ov"
OPENVINO_MODEL_ENV = "TRANSCRIPTION_OPENVINO_MODEL_DIR"
DEFAULT_OPENVINO_MODEL_DIR = (
    ROOT_DIR / "modeles" / "openvino" / "whisper-large-v3-turbo-int8-ov"
)
REQUIRED_MODEL_FILES = (
    "generation_config.json",
    "openvino_encoder_model.xml",
    "openvino_decoder_model.xml",
)


class OpenVINOBackendError(RuntimeError):
    """Erreur récupérable : le pipeline principal peut utiliser son fallback."""


class OpenVINOBackendUnavailable(OpenVINOBackendError):
    """Le runtime, le GPU ou le modèle OpenVINO n'est pas prêt."""


class OpenVINOInvalidOutput(OpenVINOBackendError):
    """OpenVINO a répondu, mais le résultat ne peut pas être utilisé."""


@dataclass(slots=True, frozen=True)
class OpenVINOStatus:
    available: bool
    reason: str
    device: str
    device_name: str | None
    model_path: Path


@dataclass(slots=True, frozen=True)
class OpenVINOWord:
    start: float
    end: float
    text: str
    probability: float | None = None


@dataclass(slots=True, frozen=True)
class OpenVINOTranscription:
    words: tuple[OpenVINOWord, ...]
    text: str
    language: str
    language_probability: float | None
    model_id: str = OPENVINO_MODEL_ID


_PIPELINE_LOCK = threading.RLock()
_PIPELINE: object | None = None
_PIPELINE_KEY: tuple[str, str] | None = None


def resolve_openvino_model_path(model_path: str | Path | None = None) -> Path:
    """Résout le modèle local sans provoquer de téléchargement réseau."""
    configured = model_path or os.environ.get(OPENVINO_MODEL_ENV)
    return Path(configured or DEFAULT_OPENVINO_MODEL_DIR).expanduser().resolve()


def _import_openvino_modules() -> tuple[Any, Any]:
    # Évite que les diagnostics bas niveau oneDNN polluent la console de
    # l'application. Une valeur explicitement fournie par l'utilisateur reste
    # prioritaire grâce à setdefault.
    os.environ.setdefault("ONEDNN_VERBOSE", "0")
    os.environ.setdefault("DNNL_VERBOSE", "0")
    try:
        openvino = importlib.import_module("openvino")
        openvino_genai = importlib.import_module("openvino_genai")
    except (ImportError, OSError) as exc:
        raise OpenVINOBackendUnavailable(
            "Le runtime OpenVINO optionnel n'est pas installé ou ne peut pas être chargé."
        ) from exc
    return openvino, openvino_genai


def _missing_model_files(model_path: Path) -> list[str]:
    if not model_path.is_dir():
        return list(REQUIRED_MODEL_FILES)
    return [name for name in REQUIRED_MODEL_FILES if not (model_path / name).is_file()]


def get_openvino_status(
    model_path: str | Path | None = None,
    *,
    device: str = "GPU",
) -> OpenVINOStatus:
    """Retourne un diagnostic léger et sans mutation de l'installation."""
    resolved_model = resolve_openvino_model_path(model_path)
    missing_files = _missing_model_files(resolved_model)
    if missing_files:
        return OpenVINOStatus(
            available=False,
            reason=(
                "Le modèle OpenVINO local est absent ou incomplet "
                f"({', '.join(missing_files)})."
            ),
            device=device,
            device_name=None,
            model_path=resolved_model,
        )

    try:
        openvino, _ = _import_openvino_modules()
        core = openvino.Core()
        available_devices = [str(value) for value in core.available_devices]
        selected_device = next(
            (
                value
                for value in available_devices
                if value == device or value.startswith(f"{device}.")
            ),
            None,
        )
        if selected_device is None:
            return OpenVINOStatus(
                available=False,
                reason=(
                    f"Le périphérique OpenVINO {device} n'est pas disponible "
                    f"({', '.join(available_devices) or 'aucun périphérique'})."
                ),
                device=device,
                device_name=None,
                model_path=resolved_model,
            )

        try:
            device_name = str(
                core.get_property(selected_device, "FULL_DEVICE_NAME")
            ).strip()
        except Exception:
            device_name = selected_device
        return OpenVINOStatus(
            available=True,
            reason=f"OpenVINO est prêt sur {device_name}.",
            device=device,
            device_name=device_name,
            model_path=resolved_model,
        )
    except OpenVINOBackendError as exc:
        return OpenVINOStatus(
            available=False,
            reason=str(exc),
            device=device,
            device_name=None,
            model_path=resolved_model,
        )
    except Exception as exc:
        return OpenVINOStatus(
            available=False,
            reason=f"OpenVINO n'a pas pu vérifier le GPU ({type(exc).__name__}: {exc}).",
            device=device,
            device_name=None,
            model_path=resolved_model,
        )


def reset_openvino_pipeline() -> None:
    """Libère la pipeline persistante, notamment pour les tests ou un arrêt propre."""
    global _PIPELINE, _PIPELINE_KEY
    with _PIPELINE_LOCK:
        _PIPELINE = None
        _PIPELINE_KEY = None
    gc.collect()


def _load_openvino_pipeline(model_path: Path, device: str) -> object:
    global _PIPELINE, _PIPELINE_KEY
    pipeline_key = (str(model_path), device)
    with _PIPELINE_LOCK:
        if _PIPELINE is not None and _PIPELINE_KEY == pipeline_key:
            return _PIPELINE

        status = get_openvino_status(model_path, device=device)
        if not status.available:
            raise OpenVINOBackendUnavailable(status.reason)

        _, openvino_genai = _import_openvino_modules()
        try:
            # CACHE_DIR vide est volontaire : le cache compilé a produit des
            # sorties invalides avec le modèle nightly mesuré sur l'Arc 140V.
            pipeline = openvino_genai.WhisperPipeline(
                str(model_path),
                device,
                word_timestamps=True,
                CACHE_DIR="",
            )
        except Exception as exc:
            raise OpenVINOBackendUnavailable(
                "La pipeline OpenVINO n'a pas pu être initialisée sur "
                f"{device} ({type(exc).__name__}: {exc})."
            ) from exc

        _PIPELINE = pipeline
        _PIPELINE_KEY = pipeline_key
        return pipeline


def _language_token(language: str | None) -> str | None:
    if not language:
        return None
    normalized = language.strip().lower()
    if normalized.startswith("<|") and normalized.endswith("|>"):
        return normalized
    normalized = normalized.split("-", 1)[0].split("_", 1)[0]
    return f"<|{normalized}|>" if normalized else None


def _language_code(value: object, fallback: str | None) -> str:
    raw = str(value or fallback or "inconnue").strip()
    if raw.startswith("<|") and raw.endswith("|>"):
        return raw[2:-2]
    return raw or "inconnue"


def transcribe_openvino(
    audio: object,
    *,
    model_path: str | Path | None = None,
    device: str = "GPU",
    language: str | None = None,
    initial_prompt: str = "",
    hotwords: str = "",
    resume_offset: float = 0.0,
) -> OpenVINOTranscription:
    """Transcrit un tableau mono 16 kHz et renvoie des mots horodatés absolus."""
    try:
        import numpy as np
    except ImportError as exc:
        raise OpenVINOBackendUnavailable(
            "NumPy est requis pour préparer l'audio OpenVINO."
        ) from exc

    try:
        samples = np.asarray(audio, dtype=np.float32)
    except Exception as exc:
        raise OpenVINOBackendError(
            f"L'audio ne peut pas être préparé pour OpenVINO ({exc})."
        ) from exc
    if samples.ndim != 1 or samples.size == 0:
        raise OpenVINOBackendError(
            "OpenVINO attend un audio mono 16 kHz non vide."
        )
    if not math.isfinite(float(resume_offset)) or resume_offset < 0:
        raise OpenVINOBackendError("L'offset de reprise OpenVINO n'est pas valide.")

    resolved_model = resolve_openvino_model_path(model_path)
    pipeline = _load_openvino_pipeline(resolved_model, device)
    generation_options: dict[str, object] = {
        "task": "transcribe",
        "return_timestamps": True,
        "word_timestamps": True,
    }
    token = _language_token(language)
    if token:
        generation_options["language"] = token
    prompt = initial_prompt.strip()
    # La boucle applicative appelle OpenVINO par tranches de 30 secondes.
    # Repasser le prompt a chaque tranche ferait croire a Whisper qu'il vient
    # d'entendre ce texte avant chaque fenetre et augmente les hallucinations.
    if prompt and resume_offset <= 0.001:
        generation_options["initial_prompt"] = prompt
    domain_hotwords = hotwords.strip()
    if domain_hotwords:
        generation_options["hotwords"] = domain_hotwords

    try:
        with _PIPELINE_LOCK:
            decoded = pipeline.generate(samples, **generation_options)
    except Exception as exc:
        raise OpenVINOBackendError(
            "L'inférence OpenVINO a échoué "
            f"({type(exc).__name__}: {exc})."
        ) from exc

    duration = float(samples.size) / 16000.0
    words: list[OpenVINOWord] = []
    for raw_word in list(getattr(decoded, "words", None) or []):
        text = str(getattr(raw_word, "word", "") or "")
        if not text.strip():
            continue
        try:
            relative_start = float(getattr(raw_word, "start_ts"))
            relative_end = float(getattr(raw_word, "end_ts"))
        except (AttributeError, TypeError, ValueError):
            continue
        if not math.isfinite(relative_start) or not math.isfinite(relative_end):
            continue
        relative_start = max(0.0, min(duration, relative_start))
        relative_end = max(relative_start, min(duration, relative_end))
        words.append(
            OpenVINOWord(
                start=float(resume_offset) + relative_start,
                end=float(resume_offset) + relative_end,
                text=text,
            )
        )

    texts = getattr(decoded, "texts", None) or []
    recognized_text = "".join(str(value) for value in texts).strip()
    if not recognized_text:
        recognized_text = "".join(word.text for word in words).strip()
    if not words or not any(character.isalnum() for character in recognized_text):
        raise OpenVINOInvalidOutput(
            "OpenVINO a renvoyé une sortie vide ou non exploitable ; "
            "le moteur de secours doit être utilisé."
        )

    words.sort(key=lambda word: (word.start, word.end))
    return OpenVINOTranscription(
        words=tuple(words),
        text=recognized_text,
        language=_language_code(getattr(decoded, "language", None), language),
        language_probability=None,
    )


__all__ = [
    "DEFAULT_OPENVINO_MODEL_DIR",
    "OPENVINO_BACKEND_ID",
    "OPENVINO_MODEL_ID",
    "OpenVINOBackendError",
    "OpenVINOBackendUnavailable",
    "OpenVINOInvalidOutput",
    "OpenVINOStatus",
    "OpenVINOTranscription",
    "OpenVINOWord",
    "get_openvino_status",
    "reset_openvino_pipeline",
    "resolve_openvino_model_path",
    "transcribe_openvino",
]
