"""Backend local MLX pour l'assistant de consultation sur Apple Silicon.

Ce module ne télécharge jamais de modèle. Il n'accepte qu'un dossier local
complet, fourni explicitement, installé dans les données de l'application ou
embarqué dans les ressources en lecture seule du bundle macOS.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .assistant_engine import (
    ASSISTANT_MODEL_ENV,
    ConsultationRequest,
    build_consultation_prompt,
)
from .runtime_paths import APP_DATA_ROOT, RESOURCE_ROOT


MLX_ASSISTANT_MODEL_NAME = "qwen3-8b-mlx-4bit"
MLX_ASSISTANT_MODEL_REPOSITORY = "mlx-community/Qwen3-8B-4bit"
MLX_ASSISTANT_MODEL_REVISION = "545dc4251c05440727734bcd94334791f6ab0192"
DEFAULT_MLX_ASSISTANT_MODEL_DIR = (
    APP_DATA_ROOT / "modeles" / "assistant" / MLX_ASSISTANT_MODEL_NAME
)
BUNDLED_MLX_ASSISTANT_MODEL_DIR = (
    RESOURCE_ROOT / "modeles" / "assistant" / MLX_ASSISTANT_MODEL_NAME
)
MLX_REQUIRED_MODEL_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

Loader = Callable[[str], tuple[object, object]]
Generator = Callable[..., object]


def _missing_model_files(path: Path) -> tuple[str, ...]:
    missing = tuple(name for name in MLX_REQUIRED_MODEL_FILES if not (path / name).is_file())
    if not any(path.glob("*.safetensors")):
        missing = (*missing, "*.safetensors")
    return missing


def mlx_model_is_complete(path: str | Path) -> bool:
    """Indique si ``path`` ressemble à un export MLX Qwen utilisable localement."""

    resolved = Path(path).expanduser()
    return resolved.is_dir() and not _missing_model_files(resolved)


def resolve_mlx_model_path(
    model_path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    app_data_root: Path = APP_DATA_ROOT,
    resource_root: Path = RESOURCE_ROOT,
) -> Path | None:
    """Résout un modèle local sans interpréter une valeur comme un dépôt distant.

    Priorité : chemin explicite/variable d'environnement, modèle embarqué dans
    le véritable ``.app``, puis modèle durable installé pour l'utilisateur.
    """

    environment = os.environ if environ is None else environ
    if model_path is not None:
        configured = str(model_path).strip()
        if not configured:
            return None
        candidate = Path(configured).expanduser().resolve()
        return candidate if mlx_model_is_complete(candidate) else None

    # Une variable héritée d'une ancienne installation ne doit pas masquer le
    # modèle complet inclus dans le nouveau .app.
    configured = str(environment.get(ASSISTANT_MODEL_ENV, "")).strip()
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if mlx_model_is_complete(candidate):
            return candidate

    candidates = (
        Path(resource_root) / "modeles" / "assistant" / MLX_ASSISTANT_MODEL_NAME,
        Path(app_data_root) / "modeles" / "assistant" / MLX_ASSISTANT_MODEL_NAME,
    )
    return next(
        (candidate.resolve() for candidate in candidates if mlx_model_is_complete(candidate)),
        None,
    )


@contextmanager
def _offline_model_loading():
    """Force les bibliothèques Hugging Face utilisées par MLX en mode hors ligne."""

    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update({name: "1" for name in names})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _default_loader(path: str) -> tuple[object, object]:
    try:
        from mlx_lm import load
    except ImportError as exc:  # pragma: no cover - dépend du bundle macOS
        raise RuntimeError("Le runtime MLX-LM n'est pas installé.") from exc
    with _offline_model_loading():
        # ``path`` est un dossier absolu préalablement validé. Il n'est jamais
        # remplacé par un identifiant Hugging Face.
        return load(path)


def _default_generator(*args: Any, **kwargs: Any) -> object:
    try:
        from mlx_lm import generate
    except ImportError as exc:  # pragma: no cover - dépend du bundle macOS
        raise RuntimeError("Le runtime MLX-LM n'est pas installé.") from exc
    return generate(*args, **kwargs)


def _render_chat_prompt(tokenizer: object, user_prompt: str) -> str:
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        raise RuntimeError("Le tokenizer MLX ne fournit pas de chat template.")

    messages = [
        {
            "role": "system",
            "content": (
                "Tu es l'assistant local de Studio Audio. Réponds uniquement "
                "avec l'objet JSON demandé, sans Markdown ni texte autour."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
    try:
        rendered = apply_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Compatibilité avec les versions de transformers qui ne connaissent pas
        # encore ``enable_thinking``. Le marqueur Qwen garde alors la réponse
        # directe et limite les blocs de raisonnement visibles.
        messages[-1]["content"] = f"{user_prompt}\n/no_think"
        rendered = apply_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    if not isinstance(rendered, str) or not rendered.strip():
        raise RuntimeError("Le chat template MLX a produit un prompt vide.")
    return rendered


def _extract_json(rendered: object) -> str:
    if isinstance(rendered, str):
        text = rendered
    elif hasattr(rendered, "text"):
        text = str(getattr(rendered, "text"))
    else:
        text = str(rendered)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    candidate = match.group(0) if match else text.strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Le modèle MLX n'a pas produit un JSON valide.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("La réponse JSON du modèle MLX doit être un objet.")
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class MLXJsonReasoner:
    """Raisonneur Qwen MLX chargé paresseusement depuis un dossier local."""

    name = "MLX-local"

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_new_tokens: int = 512,
        loader: Loader | None = None,
        generator: Generator | None = None,
    ) -> None:
        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Modèle d'assistant MLX introuvable : {path}")
        missing = _missing_model_files(path)
        if missing:
            raise FileNotFoundError(
                "Modèle d'assistant MLX incomplet : " + ", ".join(missing)
            )
        self.model_path = path
        self.max_new_tokens = max(128, int(max_new_tokens))
        self._loader = loader or _default_loader
        self._generator = generator or _default_generator
        self._model: object | None = None
        self._tokenizer: object | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def _load(self) -> tuple[object, object]:
        if self.loaded:
            return self._model, self._tokenizer  # type: ignore[return-value]
        model, tokenizer = self._loader(str(self.model_path))
        self._model = model
        self._tokenizer = tokenizer
        return model, tokenizer

    def answer(self, request: ConsultationRequest) -> str:
        model, tokenizer = self._load()
        prompt = _render_chat_prompt(tokenizer, build_consultation_prompt(request))
        try:
            rendered = self._generator(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=self.max_new_tokens,
                verbose=False,
            )
        except TypeError:
            # Certains wrappers de test ou anciennes versions MLX-LM prennent le
            # prompt comme troisième argument positionnel.
            rendered = self._generator(
                model,
                tokenizer,
                prompt,
                max_tokens=self.max_new_tokens,
                verbose=False,
            )
        return _extract_json(rendered)


def create_default_mlx_reasoner(
    model_path: str | Path | None = None,
    *,
    loader: Loader | None = None,
    generator: Generator | None = None,
    environ: Mapping[str, str] | None = None,
    app_data_root: Path = APP_DATA_ROOT,
    resource_root: Path = RESOURCE_ROOT,
) -> MLXJsonReasoner | None:
    """Crée le backend macOS uniquement si modèle et runtime sont déjà locaux."""

    resolved = resolve_mlx_model_path(
        model_path,
        environ=environ,
        app_data_root=app_data_root,
        resource_root=resource_root,
    )
    if resolved is None:
        return None
    if loader is None and importlib.util.find_spec("mlx_lm") is None:
        return None
    return MLXJsonReasoner(
        resolved,
        loader=loader,
        generator=generator,
    )


__all__ = [
    "BUNDLED_MLX_ASSISTANT_MODEL_DIR",
    "DEFAULT_MLX_ASSISTANT_MODEL_DIR",
    "MLX_ASSISTANT_MODEL_NAME",
    "MLX_ASSISTANT_MODEL_REPOSITORY",
    "MLX_ASSISTANT_MODEL_REVISION",
    "MLXJsonReasoner",
    "MLX_REQUIRED_MODEL_FILES",
    "create_default_mlx_reasoner",
    "mlx_model_is_complete",
    "resolve_mlx_model_path",
]
