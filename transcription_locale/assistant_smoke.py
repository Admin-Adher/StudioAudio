"""Validation hors ligne du véritable assistant empaqueté.

Ce module reste volontairement indépendant de Gradio. Il est importé par le
point d'entrée bureau uniquement lorsque ``--assistant-smoke-test`` est demandé
et vérifie le chemin complet : résolution du modèle embarqué, chargement du
runtime natif et génération d'un petit objet JSON par Qwen.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path


ASSISTANT_SMOKE_TEST_FLAG = "--assistant-smoke-test"
ASSISTANT_SMOKE_REPORT_FILENAME = "assistant-smoke.json"
ASSISTANT_SMOKE_MAX_TOKENS = 64
_SMOKE_SCHEMA: Mapping[str, object] = {
    "type": "object",
    "properties": {
        "ready": {"type": "boolean"},
    },
    "required": ["ready"],
    "additionalProperties": False,
}
_SMOKE_PROMPT = (
    'Test technique hors ligne. Retourne uniquement cet objet JSON exact : '
    '{"ready":true}'
)


class AssistantSmokeError(RuntimeError):
    """Le modèle assistant empaqueté n'a pas réussi son inférence minimale."""


@contextmanager
def _offline_environment() -> Iterator[None]:
    """Interdit tout accès Hugging Face pendant le contrôle du modèle local."""

    values = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
    }
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _render_openvino_smoke(reasoner: object) -> str:
    generator = getattr(reasoner, "_generate_json", None)
    if not callable(generator):
        raise AssistantSmokeError(
            "Le backend OpenVINO ne fournit pas la génération JSON attendue."
        )
    return str(generator(_SMOKE_PROMPT, json_schema=_SMOKE_SCHEMA))


def _render_mlx_smoke(reasoner: object) -> str:
    loader = getattr(reasoner, "_load", None)
    generator = getattr(reasoner, "_generator", None)
    if not callable(loader) or not callable(generator):
        raise AssistantSmokeError(
            "Le backend MLX ne fournit pas le chargement local attendu."
        )

    # Import tardif : ce module n'est jamais chargé sur Windows.
    from .mlx_assistant_backend import _extract_json, _render_chat_prompt

    model, tokenizer = loader()
    prompt = _render_chat_prompt(tokenizer, _SMOKE_PROMPT)
    try:
        rendered = generator(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=ASSISTANT_SMOKE_MAX_TOKENS,
            verbose=False,
        )
    except TypeError:
        rendered = generator(
            model,
            tokenizer,
            prompt,
            max_tokens=ASSISTANT_SMOKE_MAX_TOKENS,
            verbose=False,
        )
    return _extract_json(rendered)


def _generate_smoke_json(reasoner: object) -> dict[str, object]:
    """Charge le backend réel et valide une réponse JSON produite par Qwen."""

    # Les deux implémentations conservent ce réglage comme attribut public.
    # L'affectation après construction permet au smoke MLX de rester très court
    # même si son réglage normal impose davantage de tokens.
    if hasattr(reasoner, "max_new_tokens"):
        setattr(reasoner, "max_new_tokens", ASSISTANT_SMOKE_MAX_TOKENS)

    backend = str(getattr(reasoner, "name", type(reasoner).__name__))
    if "mlx" in backend.lower():
        raw = _render_mlx_smoke(reasoner)
    else:
        raw = _render_openvino_smoke(reasoner)
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise AssistantSmokeError(
            "Qwen n'a pas produit un objet JSON valide pendant le smoke test."
        ) from exc
    if not isinstance(payload, dict) or payload.get("ready") is not True:
        raise AssistantSmokeError(
            "La réponse structurée de Qwen ne confirme pas son état prêt."
        )
    return payload


def _assistant_device(reasoner: object, backend: str) -> str:
    active = getattr(reasoner, "active_device", None)
    requested = getattr(reasoner, "device", None)
    if active or requested:
        return str(active or requested)
    if "mlx" in backend.lower():
        return "METAL"
    return platform.machine() or "local"


def _write_report(report: Mapping[str, object], report_path: Path) -> None:
    rendered = json.dumps(report, ensure_ascii=False, sort_keys=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(rendered + "\n", encoding="utf-8")
    # Les exécutables PyInstaller sans console peuvent exposer stdout à None.
    if sys.stdout is not None:
        print(rendered, flush=True)


def run_assistant_smoke_test(
    directories: Mapping[str, Path],
    log_path: Path,
    *,
    reasoner_factory: Callable[[], object | None] | None = None,
) -> int:
    """Exécute une vraie inférence locale courte, sans démarrer l'interface."""

    report_path = Path(
        directories.get("logs", Path(log_path).resolve().parent)
    ) / ASSISTANT_SMOKE_REPORT_FILENAME
    report: dict[str, object] = {
        "mode": "assistant-smoke",
        "status": "error",
        "backend": None,
        "device": None,
        "model": None,
    }
    started_at = time.monotonic()
    release_reasoners: Callable[[], None] | None = None
    try:
        with _offline_environment():
            if reasoner_factory is None:
                from .assistant_engine import (
                    get_default_reasoner,
                    release_default_reasoners,
                )

                reasoner_factory = get_default_reasoner
                release_reasoners = release_default_reasoners
            reasoner = reasoner_factory()
            if reasoner is None:
                raise AssistantSmokeError(
                    "Aucun backend Qwen embarqué n'est disponible."
                )

            backend = str(getattr(reasoner, "name", type(reasoner).__name__))
            model_path = getattr(reasoner, "model_path", None)
            report["backend"] = backend
            report["model"] = str(model_path) if model_path is not None else None
            report["response"] = _generate_smoke_json(reasoner)
            report["device"] = _assistant_device(reasoner, backend)
            report["status"] = "ok"
    except Exception as exc:
        report["error"] = str(exc) or exc.__class__.__name__
        logging.exception("Échec du smoke test de l'assistant local empaqueté")
    finally:
        if release_reasoners is not None:
            release_reasoners()
        report["elapsed_seconds"] = round(time.monotonic() - started_at, 3)
        _write_report(report, report_path)

    logging.info("Résultat du smoke test assistant : %s", report["status"])
    return 0 if report["status"] == "ok" else 5


__all__ = [
    "ASSISTANT_SMOKE_MAX_TOKENS",
    "ASSISTANT_SMOKE_REPORT_FILENAME",
    "ASSISTANT_SMOKE_TEST_FLAG",
    "AssistantSmokeError",
    "run_assistant_smoke_test",
]
