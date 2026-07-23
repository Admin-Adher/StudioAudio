"""Assistant local, fondé exclusivement sur une transcription horodatée.

Le module est volontairement indépendant de Gradio et du stockage du workspace.
Il fournit trois briques réutilisables par l'interface :

* une proposition prudente des rôles ``Client`` et ``Master`` ;
* une recherche lexicale locale de passages pertinents ;
* une vérification structurée avec citations, utilisable sans aucun modèle.

Un raisonneur local optionnel peut être branché plus tard (OpenVINO, par
exemple). Il ne reçoit que des passages identifiés et ne peut retourner que
leurs identifiants. Les citations exposées par l'API sont toujours reconstruites
depuis les tours originaux : un modèle ne peut donc jamais en inventer le texte.
"""

from __future__ import annotations

import json
import importlib.util
import math
import os
import platform
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence, runtime_checkable

from .runtime_paths import APP_DATA_ROOT, RESOURCE_ROOT


RoleName = Literal["Client", "Master", "Inconnu"]
QueryIntent = Literal[
    "open_summary",
    "topic_presence",
    "claim_verification",
    "consultation_qa",
]
VerdictName = Literal[
    "Confirmé",
    "Partiel",
    "Contredit",
    "Non retrouvé",
    "À clarifier",
    "Synthèse",
    "Thème principal",
    "Thème présent",
    "Mention ponctuelle",
    "Réponse",
]

OPEN_SUMMARY: QueryIntent = "open_summary"
TOPIC_PRESENCE: QueryIntent = "topic_presence"
CLAIM_VERIFICATION: QueryIntent = "claim_verification"
CONSULTATION_QA: QueryIntent = "consultation_qa"

CONFIRMED: VerdictName = "Confirmé"
PARTIAL: VerdictName = "Partiel"
CONTRADICTED: VerdictName = "Contredit"
NOT_FOUND: VerdictName = "Non retrouvé"
CLARIFY: VerdictName = "À clarifier"
SUMMARY: VerdictName = "Synthèse"
TOPIC_PRIMARY: VerdictName = "Thème principal"
TOPIC_PRESENT: VerdictName = "Thème présent"
TOPIC_MENTIONED: VerdictName = "Mention ponctuelle"
ANSWERED: VerdictName = "Réponse"

ASSISTANT_MODEL_ENV = "TRANSCRIPTION_ASSISTANT_MODEL_DIR"
ASSISTANT_DEVICE_ENV = "TRANSCRIPTION_ASSISTANT_DEVICE"
ASSISTANT_MODEL_REPOSITORY = "OpenVINO/Qwen3-8B-int4-ov"
ASSISTANT_MODEL_REVISION = "5c47abf4b8e12ebe8e99745bb0c1ec17e0c0abcc"
ASSISTANT_MODEL_NAME = "qwen3-8b-int4-ov"
DEFAULT_ASSISTANT_MODEL_DIR = (
    APP_DATA_ROOT / "modeles" / "assistant" / ASSISTANT_MODEL_NAME
)
BUNDLED_ASSISTANT_MODEL_DIR = (
    RESOURCE_ROOT / "modeles" / "assistant" / ASSISTANT_MODEL_NAME
)
ASSISTANT_REQUIRED_MODEL_FILES = (
    "config.json",
    "chat_template.jinja",
    "openvino_config.json",
    "openvino_detokenizer.xml",
    "openvino_detokenizer.bin",
    "openvino_model.xml",
    "openvino_model.bin",
    "openvino_tokenizer.xml",
    "openvino_tokenizer.bin",
    "tokenizer.json",
    "tokenizer_config.json",
)
_CLAIM_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": [
                "Confirmé",
                "Partiel",
                "Contredit",
                "Non retrouvé",
                "À clarifier",
            ],
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "citation_ids": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 6,
        },
    },
    "required": ["verdict", "confidence", "citation_ids"],
    "additionalProperties": False,
}
_CONSULTATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": [
                "answered",
                "partial",
                "insufficient",
                "clarify",
                "confirmed",
                "contradicted",
                "topic_primary",
                "topic_present",
                "topic_mentioned",
            ],
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "answer": {"type": "string"},
        "key_points": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5,
        },
        "nuance": {"type": "string"},
        "citation_ids": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 6,
        },
    },
    "required": [
        "status",
        "confidence",
        "answer",
        "key_points",
        "nuance",
        "citation_ids",
    ],
    "additionalProperties": False,
}
_DEFAULT_REASONER_CACHE: dict[
    tuple[str, str, str, str], ConsultationReasoner | ClaimReasoner
] = {}


@dataclass(slots=True, frozen=True)
class SpeakerRole:
    """Rôle proposé pour une voix issue de la diarisation."""

    speaker_id: str
    label: str
    role: RoleName
    confidence: float
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class RoleProposal:
    """Proposition complète, avec un signal explicite de validation humaine."""

    assignments: tuple[SpeakerRole, ...]
    confidence: float
    needs_confirmation: bool
    method: str = "heuristique-locale"

    @property
    def client_speaker_id(self) -> str | None:
        return self.speaker_id_for("Client")

    @property
    def master_speaker_id(self) -> str | None:
        return self.speaker_id_for("Master")

    def speaker_id_for(self, role: RoleName) -> str | None:
        match = next((item for item in self.assignments if item.role == role), None)
        return match.speaker_id if match else None

    def role_for(self, speaker_id: str) -> RoleName:
        match = next(
            (item for item in self.assignments if item.speaker_id == speaker_id),
            None,
        )
        return match.role if match else "Inconnu"

    def to_dict(self) -> dict[str, object]:
        return {
            "assignments": [item.to_dict() for item in self.assignments],
            "confidence": self.confidence,
            "needs_confirmation": self.needs_confirmation,
            "method": self.method,
        }


@dataclass(slots=True, frozen=True)
class Citation:
    """Passage immuable, reconstruit depuis un tour réel de transcription."""

    id: str
    turn_index: int
    start: float
    end: float
    speaker_id: str
    speaker: str
    role: RoleName
    text: str
    relevance: float
    excerpt: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class ReasonerRequest:
    """Entrée bornée transmise à un éventuel petit modèle local."""

    question: str
    claim: str
    target_role: RoleName | None
    candidates: tuple[Citation, ...]


@dataclass(slots=True, frozen=True)
class ReasonerDecision:
    """Sortie minimale attendue d'un raisonneur optionnel.

    ``citation_ids`` est le seul moyen de citer une preuve. Tout champ de texte
    supplémentaire produit par un backend est volontairement ignoré.
    """

    verdict: VerdictName
    confidence: float
    citation_ids: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class ConsultationRequest:
    """Question libre et transcription bornée transmise au modèle local."""

    question: str
    intent: QueryIntent
    candidates: tuple[Citation, ...]
    role_mapping_confirmed: bool
    audio_observations: Mapping[str, object] | None = None


@dataclass(slots=True, frozen=True)
class ConsultationDecision:
    """Réponse détaillée d'un modèle local, toujours reliée à des passages."""

    status: str
    confidence: float
    answer: str
    citation_ids: tuple[str, ...] = ()
    key_points: tuple[str, ...] = ()
    nuance: str = ""


@runtime_checkable
class ClaimReasoner(Protocol):
    """Contrat léger pour un backend local, sans dépendance imposée."""

    def decide(
        self, request: ReasonerRequest
    ) -> ReasonerDecision | Mapping[str, object] | str:
        """Retourne une décision structurée ou son JSON équivalent."""


@runtime_checkable
class ConsultationReasoner(Protocol):
    """Contrat d'un assistant local capable de répondre à une question libre."""

    def answer(
        self, request: ConsultationRequest
    ) -> ConsultationDecision | Mapping[str, object] | str:
        """Retourne une réponse détaillée avec identifiants de citations."""


class OpenVINOJsonReasoner:
    """Petit backend OpenVINO GenAI, optionnel et chargé au premier appel.

    La classe ne télécharge jamais de modèle. Son constructeur exige un dossier
    local existant ; l'import ``openvino_genai`` et le chargement GPU n'ont lieu
    que lors de la première question. ``pipeline_factory`` sert aux tests et à
    l'adaptation à une version future du runtime sans coupler le cœur métier.
    """

    name = "OpenVINO-local"

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "GPU",
        max_new_tokens: int = 512,
        pipeline_factory: Callable[[str, str], object] | None = None,
    ) -> None:
        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Modèle d'assistant introuvable : {path}")
        self.model_path = path
        self.device = str(device or "GPU").upper()
        self.max_new_tokens = max(64, int(max_new_tokens))
        self._pipeline_factory = pipeline_factory
        self._pipeline: object | None = None
        self._active_device: str | None = None
        self._fallback_reason: str | None = None

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def active_device(self) -> str | None:
        """Périphérique réellement utilisé après l'éventuel repli."""

        return self._active_device

    @property
    def fallback_used(self) -> bool:
        return self._active_device == "CPU" and self.device != "CPU"

    @property
    def fallback_reason(self) -> str | None:
        return self._fallback_reason

    def _resolve_pipeline_factory(self) -> Callable[[str, str], object]:
        if self._pipeline_factory is not None:
            return self._pipeline_factory
        try:
            import openvino_genai
        except ImportError as exc:  # pragma: no cover - installation optionnelle
            raise RuntimeError(
                "Le runtime OpenVINO GenAI n'est pas installé."
            ) from exc
        return openvino_genai.LLMPipeline

    @staticmethod
    def _failure_summary(exc: Exception) -> str:
        detail = re.sub(r"\s+", " ", str(exc)).strip()
        return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__

    def _create_pipeline(
        self,
        factory: Callable[[str, str], object],
        device: str,
    ) -> object:
        return factory(str(self.model_path), device)

    def _load_pipeline(self) -> object:
        if self._pipeline is not None:
            return self._pipeline

        factory = self._resolve_pipeline_factory()
        try:
            pipeline = self._create_pipeline(factory, self.device)
        except Exception as primary_error:
            if self.device == "CPU":
                raise
            self._fallback_reason = self._failure_summary(primary_error)
            try:
                pipeline = self._create_pipeline(factory, "CPU")
            except Exception as cpu_error:
                raise RuntimeError(
                    "L'assistant OpenVINO n'a pu démarrer ni sur "
                    f"{self.device} ({self._failure_summary(primary_error)}), "
                    f"ni sur CPU ({self._failure_summary(cpu_error)})."
                ) from cpu_error
            self._active_device = "CPU"
        else:
            self._active_device = self.device
        self._pipeline = pipeline
        return pipeline

    def _generate_raw(
        self,
        pipeline: object,
        prompt: str,
        *,
        json_schema: Mapping[str, object],
    ) -> object:
        chat_started = False
        try:
            try:
                import openvino_genai

                structured = openvino_genai.StructuredOutputConfig(
                    json_schema=json.dumps(
                        json_schema,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                config = openvino_genai.GenerationConfig(
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    structured_output_config=structured,
                )
            except (ImportError, AttributeError, TypeError):
                config = None

            start_chat = getattr(pipeline, "start_chat", None)
            if callable(start_chat):
                start_chat(
                    "Tu es l'assistant local de Studio Audio. Réponds "
                    "uniquement avec l'objet JSON demandé."
                )
                chat_started = True
            prompt_without_thinking = f"{prompt}\n/no_think"
            if config is not None:
                try:
                    return pipeline.generate(
                        prompt_without_thinking,
                        generation_config=config,
                    )
                except TypeError:
                    return pipeline.generate(prompt_without_thinking, config)
            return pipeline.generate(
                prompt_without_thinking,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        finally:
            if chat_started:
                finish_chat = getattr(pipeline, "finish_chat", None)
                if callable(finish_chat):
                    finish_chat()

    def _generate_json(
        self,
        prompt: str,
        *,
        json_schema: Mapping[str, object],
    ) -> str:
        pipeline = self._load_pipeline()
        try:
            raw = self._generate_raw(
                pipeline,
                prompt,
                json_schema=json_schema,
            )
        except Exception as primary_error:
            if self._active_device == "CPU" or self.device == "CPU":
                raise
            self._fallback_reason = self._failure_summary(primary_error)
            self._pipeline = None
            factory = self._resolve_pipeline_factory()
            try:
                cpu_pipeline = self._create_pipeline(factory, "CPU")
                raw = self._generate_raw(
                    cpu_pipeline,
                    prompt,
                    json_schema=json_schema,
                )
            except Exception as cpu_error:
                self._active_device = None
                raise RuntimeError(
                    "L'assistant OpenVINO a échoué pendant l'inférence sur "
                    f"{self.device} ({self._failure_summary(primary_error)}) "
                    f"puis sur CPU ({self._failure_summary(cpu_error)})."
                ) from cpu_error
            self._pipeline = cpu_pipeline
            self._active_device = "CPU"
        if isinstance(raw, str):
            rendered = raw
        elif hasattr(raw, "texts") and getattr(raw, "texts"):
            rendered = str(getattr(raw, "texts")[0])
        elif hasattr(raw, "text"):
            rendered = str(getattr(raw, "text"))
        else:
            rendered = str(raw)
        # Les balises Markdown sont ignorées si un modèle en ajoute malgré le
        # contrat JSON. Le parseur strict validera ensuite les identifiants.
        match = re.search(r"\{.*\}", rendered, re.DOTALL)
        return match.group(0) if match else rendered

    def decide(self, request: ReasonerRequest) -> str:
        return self._generate_json(
            build_grounded_prompt(request),
            json_schema=_CLAIM_JSON_SCHEMA,
        )

    def answer(self, request: ConsultationRequest) -> str:
        return self._generate_json(
            build_consultation_prompt(request),
            json_schema=_CONSULTATION_JSON_SCHEMA,
        )


def get_default_reasoner(
    model_path: str | Path | None = None,
    *,
    device: str | None = None,
    platform_name: str | None = None,
    machine_name: str | None = None,
) -> ConsultationReasoner | ClaimReasoner | None:
    """Retourne le backend local configuré, sinon ``None`` immédiatement.

    Le chemin peut être fourni explicitement ou via
    ``TRANSCRIPTION_ASSISTANT_MODEL_DIR``. Un dossier absent ou un runtime non
    installé ne bloque jamais l'application : l'UI peut simplement conserver
    le moteur déterministe probant.
    """

    current_platform = sys.platform if platform_name is None else platform_name
    current_machine = (
        platform.machine() if machine_name is None else machine_name
    ).lower()
    if current_platform == "darwin":
        if current_machine not in {"arm64", "aarch64"}:
            return None
        # Import tardif : Windows ne dépend jamais de MLX, et ce module peut
        # réutiliser le contrat/prompt défini plus haut sans import circulaire.
        from .mlx_assistant_backend import create_default_mlx_reasoner

        reasoner = create_default_mlx_reasoner(model_path)
        if reasoner is None:
            return None
        cache_key = (
            current_platform,
            current_machine,
            str(getattr(reasoner, "model_path", model_path or "default")),
            "MLX",
        )
        cached = _DEFAULT_REASONER_CACHE.get(cache_key)
        if cached is not None:
            return cached
        _DEFAULT_REASONER_CACHE[cache_key] = reasoner
        return reasoner

    path = resolve_openvino_assistant_model_path(model_path)
    if (
        path is None
        or importlib.util.find_spec("openvino_genai") is None
    ):
        return None
    resolved_device = device or os.environ.get(ASSISTANT_DEVICE_ENV, "GPU")
    cache_key = (
        current_platform,
        current_machine,
        str(path),
        str(resolved_device).upper(),
    )
    cached = _DEFAULT_REASONER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    reasoner = OpenVINOJsonReasoner(
        path,
        device=resolved_device,
    )
    _DEFAULT_REASONER_CACHE[cache_key] = reasoner
    return reasoner


def release_default_reasoners() -> None:
    """Libère les modèles conversationnels mis en cache.

    Ce point d'extension permet au traitement audio de récupérer la mémoire GPU
    ou unifiée avant une nouvelle transcription longue.
    """

    _DEFAULT_REASONER_CACHE.clear()


def resolve_openvino_assistant_model_path(
    model_path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    app_data_root: Path = APP_DATA_ROOT,
    resource_root: Path = RESOURCE_ROOT,
) -> Path | None:
    """Résout le Qwen OpenVINO local sans jamais déclencher de téléchargement.

    L'installateur complet place le modèle dans les ressources en lecture seule.
    Les anciennes installations qui l'ont déjà téléchargé continuent d'utiliser
    leur copie durable dans les données utilisateur.
    """

    environment = os.environ if environ is None else environ

    def complete(candidate: Path) -> bool:
        return candidate.is_dir() and all(
            (candidate / name).is_file()
            for name in ASSISTANT_REQUIRED_MODEL_FILES
        )

    # Un chemin passé par l'appelant est une contrainte explicite : ne jamais
    # le remplacer silencieusement par un autre modèle s'il est invalide.
    if model_path is not None:
        configured = str(model_path).strip()
        if not configured:
            return None
        candidate = Path(configured).expanduser().resolve()
        return candidate if complete(candidate) else None

    # La variable d'environnement peut survivre à une ancienne installation.
    # Si sa cible n'existe plus, poursuivre vers les ressources embarquées au
    # lieu de désactiver l'assistant livré par le nouvel installateur.
    configured = str(environment.get(ASSISTANT_MODEL_ENV, "")).strip()
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if complete(candidate):
            return candidate

    candidates = (
        Path(resource_root) / "modeles" / "assistant" / ASSISTANT_MODEL_NAME,
        Path(app_data_root) / "modeles" / "assistant" / ASSISTANT_MODEL_NAME,
    )
    return next(
        (
            candidate.resolve()
            for candidate in candidates
            if complete(candidate)
        ),
        None,
    )


@dataclass(slots=True, frozen=True)
class VerificationResult:
    """Réponse prête à être persistée ou rendue dans la fiche audio."""

    question: str
    claim: str
    intent: QueryIntent
    target_role: RoleName | None
    verdict: VerdictName
    confidence: float
    explanation: str
    citations: tuple[Citation, ...]
    roles: RoleProposal
    backend: str
    answer: str = ""
    analysis_points: tuple[str, ...] = ()
    nuance: str = ""
    clarification_options: tuple[dict[str, str], ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "claim": self.claim,
            "intent": self.intent,
            "answer_type": self.intent,
            "engine_version": 2,
            "target_role": self.target_role,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "citations": [
                {
                    **item.to_dict(),
                    # Noms explicites consommés par l'interface et conservés
                    # en plus des attributs Python courts.
                    "quote": item.excerpt or item.text,
                    "start_seconds": item.start,
                    "end_seconds": item.end,
                }
                for item in self.citations
            ],
            "roles": self.roles.to_dict(),
            "backend": self.backend,
            "answer": self.answer or self.explanation,
            "analysis_points": list(self.analysis_points),
            "nuance": self.nuance,
            "clarification_options": [dict(item) for item in self.clarification_options],
            "warnings": list(self.warnings),
        }


@dataclass(slots=True, frozen=True)
class _Turn:
    index: int
    start: float
    end: float
    speaker_id: str
    speaker: str
    text: str


@dataclass(slots=True, frozen=True)
class _ScoredTurn:
    turn: _Turn
    role: RoleName
    score: float
    token_recall: float
    content_recall: float
    bigram_recall: float
    exact: bool
    polarity_conflict: bool
    excerpt: str


_MASTER_CUES: tuple[tuple[str, float], ...] = (
    ("je vois", 3.0),
    ("je ressens", 3.0),
    ("je percois", 3.0),
    ("on me dit", 3.0),
    ("les cartes", 3.0),
    ("je tire les cartes", 4.0),
    ("je vais regarder", 3.0),
    ("je regarde", 2.0),
    ("votre date de naissance", 4.0),
    ("donnez moi votre", 3.5),
    ("quel est votre prenom", 3.5),
    ("quelle est votre question", 3.5),
    ("que souhaitez vous savoir", 3.5),
    ("je vous conseille", 3.0),
    ("je vous confirme", 3.0),
    ("vous allez", 1.8),
    ("il va", 1.2),
    ("elle va", 1.2),
    ("dans les prochains", 1.5),
)

_CLIENT_CUES: tuple[tuple[str, float], ...] = (
    ("je voudrais savoir", 3.5),
    ("j aimerais savoir", 3.5),
    ("je voulais savoir", 3.0),
    ("ma question", 3.0),
    ("je vous appelle", 2.5),
    ("mon mari", 2.5),
    ("ma femme", 2.5),
    ("mon ex", 2.5),
    ("ma relation", 2.0),
    ("mon travail", 2.0),
    ("ma situation", 2.0),
    ("est ce que", 1.8),
    ("quand est ce", 2.0),
    ("je suis inquiet", 2.0),
    ("je suis inquiete", 2.0),
)

_MASTER_LABELS = {
    "master",
    "voyant",
    "voyante",
    "medium",
    "conseiller",
    "conseillere",
}
_CLIENT_LABELS = {"client", "cliente", "appelant", "appelante"}

_STOPWORDS = {
    "afin",
    "alors",
    "apres",
    "avec",
    "avoir",
    "cela",
    "cette",
    "comme",
    "comment",
    "considere",
    "consultation",
    "dans",
    "depuis",
    "elle",
    "est",
    "etre",
    "faire",
    "fait",
    "lors",
    "mais",
    "moi",
    "nous",
    "par",
    "pas",
    "peut",
    "pour",
    "pourquoi",
    "que",
    "quel",
    "quelle",
    "qui",
    "quoi",
    "ses",
    "son",
    "sont",
    "sur",
    "telephonique",
    "tout",
    "tres",
    "une",
    "vous",
    "vrai",
    "vraie",
    # Formulations de discours rapporté, peu utiles pour retrouver le fond.
    "affirme",
    "affirmee",
    "ca",
    "ce",
    "cet",
    "ces",
    "cliente",
    "client",
    "de",
    "des",
    "dit",
    "dire",
    "du",
    "en",
    "la",
    "le",
    "les",
    "master",
    "sa",
    "se",
    "son",
    "un",
    "va",
    "voyant",
    "voyante",
}

_NEGATION_WORDS = {
    "aucun",
    "aucune",
    "faux",
    "impossible",
    "jamais",
    "non",
    "nullement",
}


# Les consultations emploient de nombreuses formulations équivalentes pour
# parler de la fin d'une relation. Les ramener à deux concepts explicites évite
# qu'une question naturelle (« se terminer avec son conjoint ») ne soit classée
# uniquement grâce aux mots-outils « ça va ». Ce petit lexique reste local,
# auditable et volontairement limité au domaine du produit.
_RELATION_END_STEMS = {
    "arret",
    "cess",
    "fini",
    "fin",
    "romp",
    "rompr",
    "ruptur",
    "separ",
    "termin",
}
_RELATION_STEMS = {
    "conjoint",
    "coupl",
    "epoux",
    "femm",
    "lien",
    "mari",
    "partenair",
    "relation",
    "relat",
    "union",
}
_RELATION_END_PHRASES = (
    r"\bmettre\s+(?:un\s+terme|fin)\b",
    r"\bcouper\s+les\s+ponts\b",
    r"\bse\s+la(?:che|chent|cher)\b",
    r"\bse\s+laiss(?:e|ent|er)\s+partir\b",
    r"\bchacun\s+de\s+son\s+cote\b",
    r"\bpas\s+de\s+reprise\s+de\s+relation",
)


@dataclass(slots=True, frozen=True)
class _TopicDefinition:
    key: str
    label: str
    patterns: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class _TopicHit:
    turn: _Turn
    role: RoleName
    score: float
    matched_patterns: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class _TopicAnalysis:
    definition: _TopicDefinition
    hits: tuple[_TopicHit, ...]
    score: float
    coverage: float


_TOPIC_DEFINITIONS: tuple[_TopicDefinition, ...] = (
    _TopicDefinition(
        "sentimental",
        "la vie sentimentale et les relations",
        (
            r"\bsentiment\w*\b",
            r"\bamour\w*\b",
            r"\bamoureu\w*\b",
            r"\brelation\w*\b",
            r"\bcouple\w*\b",
            r"\bconjoint\w*\b",
            r"\bcompagnon\w*\b",
            r"\bpartenaire\w*\b",
            r"\blien\b",
            r"\bruptur\w*\b",
            r"\bsepar\w*\b",
            r"\brapproch\w*\b",
            r"\brenouveau\w*\b",
            r"\bstabilis\w*\b",
        ),
    ),
    _TopicDefinition(
        "travail",
        "le travail et la carrière",
        (
            r"\btravail\w*\b",
            r"\bemploi\w*\b",
            r"\bprofession\w*\b",
            r"\bcarriere\w*\b",
            r"\bposte\w*\b",
            r"\bentretien\w*\b",
            r"\bembauch\w*\b",
        ),
    ),
    _TopicDefinition(
        "famille",
        "la famille",
        (
            r"\bfamill\w*\b",
            r"\benfant\w*\b",
            r"\bfils\b",
            r"\bfille\w*\b",
            r"\bmere\b",
            r"\bpere\b",
            r"\bgrossess\w*\b",
        ),
    ),
    _TopicDefinition(
        "sante",
        "la santé et le bien-être",
        (
            r"\bsante\b",
            r"\bmaladi\w*\b",
            r"\bmedical\w*\b",
            r"\bmedecin\w*\b",
            r"\bsoin\w*\b",
            r"\bbien etre\b",
        ),
    ),
    _TopicDefinition(
        "finance",
        "l'argent et les finances",
        (
            r"\bargent\b",
            r"\bfinanc\w*\b",
            r"\bsalair\w*\b",
            r"\bdette\w*\b",
            r"\bcredit\w*\b",
            r"\bbudget\w*\b",
        ),
    ),
    _TopicDefinition(
        "voyage",
        "les voyages et les vacances",
        (
            r"\bvoyag\w*\b",
            r"\bvacanc\w*\b",
            r"\bdeplac\w*\b",
            r"\bweek end\b",
        ),
    ),
    _TopicDefinition(
        "logement",
        "le logement et l'immobilier",
        (
            r"\blogement\w*\b",
            r"\bimmobili\w*\b",
            r"\bmaison\w*\b",
            r"\bappartement\w*\b",
            r"\bdemenag\w*\b",
        ),
    ),
)


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    without_accents = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    without_accents = without_accents.lower().replace("’", "'")
    without_accents = re.sub(r"\b([cdjlmnstqu]+)'", r"\1 ", without_accents)
    without_accents = re.sub(r"[^a-z0-9']+", " ", without_accents)
    return re.sub(r"\s+", " ", without_accents).strip()


def _field(turn: object, name: str, default: object = None) -> object:
    if isinstance(turn, Mapping):
        return turn.get(name, default)
    return getattr(turn, name, default)


def _normalise_turns(turns: Sequence[object]) -> tuple[_Turn, ...]:
    normalised: list[_Turn] = []
    for index, raw in enumerate(turns):
        text = str(_field(raw, "text", "") or "").strip()
        if not text:
            continue
        try:
            start = max(0.0, float(_field(raw, "start", 0.0) or 0.0))
            end = max(start, float(_field(raw, "end", start) or start))
        except (TypeError, ValueError):
            start = 0.0
            end = 0.0
        label = str(_field(raw, "speaker", "") or "").strip()
        speaker_id = str(_field(raw, "speaker_id", "") or "").strip()
        speaker_id = speaker_id or label or "SPEAKER_00"
        normalised.append(
            _Turn(
                index=index,
                start=start,
                end=end,
                speaker_id=speaker_id,
                speaker=label or speaker_id,
                text=text,
            )
        )
    if not normalised:
        raise ValueError("La transcription ne contient aucun passage exploitable.")
    return tuple(normalised)


def _normalise_role(value: object) -> RoleName | None:
    folded = _fold(str(value or ""))
    if folded in _CLIENT_LABELS:
        return "Client"
    if folded in _MASTER_LABELS:
        return "Master"
    if folded in {"inconnu", "unknown"}:
        return "Inconnu"
    return None


def _cue_score(text: str, cues: Sequence[tuple[str, float]]) -> tuple[float, list[str]]:
    score = 0.0
    evidence: list[str] = []
    for phrase, weight in cues:
        occurrences = text.count(phrase)
        if occurrences:
            score += min(2, occurrences) * weight
            evidence.append(phrase)
    return score, evidence


def propose_speaker_roles(
    turns: Sequence[object],
    *,
    overrides: Mapping[str, str] | None = None,
    confirmation_threshold: float = 0.76,
) -> RoleProposal:
    """Propose ``Client`` et ``Master`` sans se fier à l'ordre des voix.

    Les corrections humaines passées via ``overrides`` sont prioritaires. Avec
    exactement deux voix, corriger un seul rôle suffit à déduire l'autre. Une
    conversation sans indice exploitable reste explicitement ``Inconnu``.
    """

    records = _normalise_turns(turns)
    by_speaker: dict[str, list[_Turn]] = defaultdict(list)
    for turn in records:
        by_speaker[turn.speaker_id].append(turn)
    speaker_ids = list(by_speaker)

    explicit: dict[str, RoleName] = {}
    for speaker_id, value in (overrides or {}).items():
        if speaker_id not in by_speaker:
            continue
        role = _normalise_role(value)
        if role in {"Client", "Master"}:
            explicit[speaker_id] = role
    if len(speaker_ids) == 2 and len(explicit) == 1:
        known_id, known_role = next(iter(explicit.items()))
        other_id = next(item for item in speaker_ids if item != known_id)
        explicit[other_id] = "Master" if known_role == "Client" else "Client"

    stats: dict[str, dict[str, Any]] = {}
    for speaker_id, speaker_turns in by_speaker.items():
        combined = " ".join(_fold(item.text) for item in speaker_turns)
        label = speaker_turns[0].speaker
        folded_label = _fold(label)
        master_score, master_evidence = _cue_score(combined, _MASTER_CUES)
        client_score, client_evidence = _cue_score(combined, _CLIENT_CUES)
        if folded_label in _MASTER_LABELS:
            master_score += 12.0
            master_evidence.insert(0, f"libellé : {label}")
        if folded_label in _CLIENT_LABELS:
            client_score += 12.0
            client_evidence.insert(0, f"libellé : {label}")

        question_turns = sum("?" in item.text for item in speaker_turns)
        client_score += min(3, question_turns) * 0.45
        if question_turns:
            client_evidence.append("questions personnelles")
        avg_words = sum(len(item.text.split()) for item in speaker_turns) / max(
            1, len(speaker_turns)
        )
        stats[speaker_id] = {
            "label": label,
            "master": master_score,
            "client": client_score,
            "delta": master_score - client_score,
            "avg_words": avg_words,
            "master_evidence": master_evidence,
            "client_evidence": client_evidence,
        }

    assignments: dict[str, RoleName] = dict(explicit)
    method = "correction-utilisateur" if explicit else "heuristique-locale"
    confidence = 1.0 if len(explicit) == len(speaker_ids) else 0.0

    remaining = [item for item in speaker_ids if item not in assignments]
    if remaining:
        if len(speaker_ids) == 2 and not explicit:
            first, second = speaker_ids
            first_delta = float(stats[first]["delta"])
            second_delta = float(stats[second]["delta"])
            total_signal = sum(
                float(stats[item]["master"]) + float(stats[item]["client"])
                for item in speaker_ids
            )
            separation = abs(first_delta - second_delta)
            if total_signal < 0.75 or separation < 0.35:
                assignments[first] = "Inconnu"
                assignments[second] = "Inconnu"
                confidence = 0.35 if total_signal else 0.2
            else:
                master_id = max(
                    speaker_ids,
                    key=lambda item: (
                        float(stats[item]["delta"]),
                        float(stats[item]["master"]),
                        float(stats[item]["avg_words"]),
                    ),
                )
                client_id = next(item for item in speaker_ids if item != master_id)
                assignments[master_id] = "Master"
                assignments[client_id] = "Client"
                confidence = min(
                    0.97,
                    0.50
                    + 0.075 * min(5.0, separation)
                    + 0.025 * min(5.0, math.log1p(total_signal)),
                )
        elif len(speaker_ids) == 1:
            speaker_id = speaker_ids[0]
            signal = max(
                float(stats[speaker_id]["master"]),
                float(stats[speaker_id]["client"]),
            )
            if signal >= 3.0:
                assignments[speaker_id] = (
                    "Master"
                    if stats[speaker_id]["master"] > stats[speaker_id]["client"]
                    else "Client"
                )
                confidence = 0.58
            else:
                assignments[speaker_id] = "Inconnu"
                confidence = 0.2
        else:
            # Le produit vise deux interlocuteurs. S'il y en a davantage, on
            # propose les deux rôles les plus caractéristiques et demande une
            # confirmation au lieu de forcer les autres voix.
            available = [item for item in speaker_ids if item not in assignments]
            if "Master" not in assignments.values() and available:
                master_id = max(available, key=lambda item: float(stats[item]["delta"]))
                assignments[master_id] = "Master"
                available.remove(master_id)
            if "Client" not in assignments.values() and available:
                client_id = min(available, key=lambda item: float(stats[item]["delta"]))
                assignments[client_id] = "Client"
                available.remove(client_id)
            for speaker_id in available:
                assignments[speaker_id] = "Inconnu"
            confidence = min(confidence or 0.55, 0.55)

    role_items: list[SpeakerRole] = []
    for speaker_id in speaker_ids:
        role = assignments.get(speaker_id, "Inconnu")
        evidence_key = "master_evidence" if role == "Master" else "client_evidence"
        evidence = tuple(dict.fromkeys(stats[speaker_id].get(evidence_key, [])))
        item_confidence = 1.0 if speaker_id in explicit else confidence
        role_items.append(
            SpeakerRole(
                speaker_id=speaker_id,
                label=str(stats[speaker_id]["label"]),
                role=role,
                confidence=round(max(0.0, min(1.0, item_confidence)), 3),
                evidence=evidence[:5],
            )
        )

    confidence = round(max(0.0, min(1.0, confidence)), 3)
    valid_pair = (
        len(speaker_ids) == 2
        and {item.role for item in role_items} == {"Client", "Master"}
    )
    return RoleProposal(
        assignments=tuple(role_items),
        confidence=confidence,
        needs_confirmation=(not valid_pair or confidence < confirmation_threshold),
        method=method,
    )


def classify_query_intent(question: str) -> QueryIntent:
    """Distingue une synthèse, une recherche de thème et une affirmation.

    Cette étape est volontairement déterministe : elle évite qu'une question
    ouverte soit transformée en proposition vrai/faux avant même la recherche
    de passages.
    """

    value = str(question or "").strip()
    if not value:
        raise ValueError("La question ne peut pas être vide.")
    folded = _fold(value)
    summary_patterns = (
        r"\bde quoi (?:parle|traite)\b",
        r"\b(?:quel|quels) (?:est|sont) (?:le |les )?(?:sujet|theme)s?\b",
        r"\b(?:sujet|theme)s? principa\w*\b",
        r"\b(?:resume|resumer|resumee|synthese)\b",
        r"\b(?:parle|traite) de quoi principa\w*\b",
        r"\bquoi principa\w*\b",
    )
    if any(re.search(pattern, folded) for pattern in summary_patterns):
        return OPEN_SUMMARY

    topic_patterns = (
        r"\b(?:est ce que|cela|ca|conversation|consultation|audio|echange)\b"
        r".{0,35}\b(?:parle|traite|porte|aborde)\b.{0,18}\b(?:de|du|des|sur)\b",
        r"\b(?:parle|traite|porte|aborde)\b.{0,18}\b(?:de|du|des|sur)\b",
        r"\b(?:theme|sujet)\b.{0,18}\b(?:present|aborde|evoque|mentionne)\b",
        r"\by a t il\b.{0,18}\b(?:de|du|des)\b",
    )
    if any(re.search(pattern, folded) for pattern in topic_patterns):
        return TOPIC_PRESENCE
    claim_patterns = (
        r"\b(?:master|voyant\w*|medium|client\w*)\b.{0,28}"
        r"\b(?:dit|annonce|affirme|explique|prevoit|confirme|voit|pense)\b",
        r"\b(?:a|aurait|avait)\s+(?:dit|annonce|affirme|explique|prevu|confirme)\b",
        r"\b(?:est ce vrai|est ce confirme|confirme dans l audio)\b",
        r"\b(?:est ce que|pensez vous que)\b.{0,80}"
        r"\b(?:va|vont|sera|seront|aura|auront|fera|feront|reviendra|terminera)\b",
    )
    if (
        any(re.search(pattern, folded) for pattern in claim_patterns)
        or bool(re.search(r"[«“\"].{3,}[»”\"]", value))
    ):
        return CLAIM_VERIFICATION
    return CONSULTATION_QA


def _topic_matches(definition: _TopicDefinition, value: str) -> tuple[str, ...]:
    folded = _fold(value)
    return tuple(
        pattern for pattern in definition.patterns if re.search(pattern, folded)
    )


def _analyse_topics(
    records: Sequence[_Turn],
    roles: RoleProposal,
) -> tuple[_TopicAnalysis, ...]:
    analyses: list[_TopicAnalysis] = []
    for definition in _TOPIC_DEFINITIONS:
        hits: list[_TopicHit] = []
        for turn in records:
            matches = _topic_matches(definition, turn.text)
            if not matches:
                continue
            duration = max(0.0, turn.end - turn.start)
            score = (
                1.0
                + 0.42 * min(4, len(matches) - 1)
                + 0.18 * min(1.0, duration / 35.0)
            )
            role = roles.role_for(turn.speaker_id)
            if role == "Master":
                score += 0.08
            hits.append(
                _TopicHit(
                    turn=turn,
                    role=role,
                    score=score,
                    matched_patterns=matches,
                )
            )
        total_score = sum(item.score for item in hits)
        analyses.append(
            _TopicAnalysis(
                definition=definition,
                hits=tuple(hits),
                score=total_score,
                coverage=len(hits) / max(1, len(records)),
            )
        )
    return tuple(
        sorted(
            analyses,
            key=lambda item: (-item.score, -len(item.hits), item.definition.key),
        )
    )


def _extract_topic_phrase(question: str) -> str:
    value = re.sub(r"\s+", " ", str(question or "")).strip(" .?!")
    patterns = (
        r"(?is)^.*?\b(?:parle|parler|traite|traiter|porte|porter|aborde|aborder)"
        r"\s+(?:principalement\s+)?(?:de|du|des|sur)\s+(.+)$",
        r"(?is)^.*?\b(?:theme|sujet)\s+(.+?)\s+"
        r"(?:est|soit|serait)\s+(?:present|aborde|evoque|mentionne).*$",
        r"(?is)^.*?\by\s+a[- ]t[- ]il\s+(?:de|du|des)\s+(.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, value)
        if match:
            topic = match.group(1).strip(" .?!:;,-")
            topic = re.sub(
                r"(?is)\s+(?:dans|pendant|au cours de)\s+"
                r"(?:la|cette)?\s*(?:consultation|conversation|audio).*$",
                "",
                topic,
            ).strip(" .?!:;,-")
            if topic:
                return topic
    return value


def _requested_topic_definitions(question: str) -> tuple[_TopicDefinition, ...]:
    phrase = _extract_topic_phrase(question)
    requested = tuple(
        definition
        for definition in _TOPIC_DEFINITIONS
        if _topic_matches(definition, phrase)
    )
    return requested


def _topic_excerpt(text: str, definition: _TopicDefinition) -> str:
    candidates: list[tuple[int, int, str]] = []
    for window in _text_windows(text):
        match_count = len(_topic_matches(definition, window))
        if match_count:
            candidates.append((match_count, -len(window.split()), window.strip()))
    if not candidates:
        return str(text or "").strip()
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _topic_citation(item: _TopicHit, definition: _TopicDefinition) -> Citation:
    turn = item.turn
    return Citation(
        id=f"turn-{turn.index:05d}",
        turn_index=turn.index,
        start=turn.start,
        end=turn.end,
        speaker_id=turn.speaker_id,
        speaker=turn.speaker,
        role=item.role,
        text=turn.text,
        relevance=round(min(1.0, item.score / 2.4), 4),
        excerpt=_topic_excerpt(turn.text, definition),
    )


def _representative_topic_hits(
    analysis: _TopicAnalysis,
    *,
    limit: int = 3,
) -> tuple[_TopicHit, ...]:
    if not analysis.hits or limit < 1:
        return ()
    chronological = sorted(
        analysis.hits,
        key=lambda item: (item.turn.start, item.turn.index),
    )
    selected: list[_TopicHit] = []

    # Le premier tour du Client explique souvent la raison de l'appel.
    client_hit = next((item for item in chronological if item.role == "Client"), None)
    if client_hit is not None:
        selected.append(client_hit)

    master_hits = [item for item in chronological if item.role == "Master"]
    duration = max(item.turn.end for item in chronological)
    for lower, upper in ((0.0, 0.55), (0.45, 1.01)):
        bucket = [
            item
            for item in master_hits
            if duration * lower <= item.turn.start < duration * upper
            and item not in selected
        ]
        if bucket:
            selected.append(
                max(
                    bucket,
                    key=lambda item: (
                        item.score,
                        len(item.matched_patterns),
                        -item.turn.start,
                    ),
                )
            )
        if len(selected) >= limit:
            break

    for item in sorted(
        chronological,
        key=lambda value: (-value.score, value.turn.start),
    ):
        if item not in selected:
            selected.append(item)
        if len(selected) >= limit:
            break
    return tuple(sorted(selected[:limit], key=lambda item: item.turn.start))


_IGNORED_PROPER_NAMES = {
    "allo",
    "alors",
    "apres",
    "avant",
    "bonjour",
    "bonsoir",
    "cedonc",
    "client",
    "cliente",
    "daccord",
    "donc",
    "interlocuteur",
    "master",
    "merci",
    "non",
    "oui",
    "voila",
}


def _proper_names(value: str) -> tuple[str, ...]:
    names: list[str] = []
    for match in re.finditer(
        r"(?<![A-Za-zÀ-ÖØ-öø-ÿ'’\-])"
        r"([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]{2,})",
        str(value or ""),
    ):
        name = match.group(1).strip(" .'’-")
        if _fold(name) not in _IGNORED_PROPER_NAMES:
            names.append(name)
    return tuple(names)


def _relationship_people(
    records: Sequence[_Turn],
    roles: RoleProposal,
) -> tuple[str | None, str | None]:
    client_names: set[str] = set()
    target: str | None = None
    counts: dict[str, tuple[str, int]] = {}

    for turn in records:
        if roles.role_for(turn.speaker_id) == "Client":
            intro = re.search(
                r"\b(?:je suis|je m['’]appelle)\s+"
                r"([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]{2,})",
                turn.text,
            )
            if intro:
                client_names.add(_fold(intro.group(1)))
            if target is None and re.search(
                r"\b(?:sentiment\w*|amour\w*|relation\w*|lien)\b",
                _fold(turn.text),
            ):
                match = re.search(
                    r"\b(?:avec|entre)\s+"
                    r"([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]{2,})",
                    turn.text,
                )
                if match:
                    target = match.group(1)
        for name in _proper_names(turn.text):
            key = _fold(name)
            if key in client_names:
                continue
            previous = counts.get(key)
            counts[key] = (name, (previous[1] if previous else 0) + 1)

    if target is None:
        ranked = sorted(
            counts.values(),
            key=lambda item: (-item[1], _fold(item[0])),
        )
        target = ranked[0][0] if ranked and ranked[0][1] >= 2 else None
    target_key = _fold(target or "")
    third = next(
        (
            name
            for name, count in sorted(
                counts.values(),
                key=lambda item: (-item[1], _fold(item[0])),
            )
            if count >= 2
            and _fold(name) != target_key
            and _fold(name) not in client_names
        ),
        None,
    )
    return target, third


def _sentimental_overview(
    records: Sequence[_Turn],
    roles: RoleProposal,
) -> str:
    target, third = _relationship_people(records, roles)
    combined = _fold(" ".join(turn.text for turn in records))
    details: list[str] = []
    if target and third:
        details.append(
            f"L'échange suit surtout l'évolution du lien entre la cliente et "
            f"{target}, tandis que la relation entre {target} et {third} "
            "constitue l'obstacle central."
        )
    elif target:
        details.append(
            f"La cliente cherche surtout à comprendre l'évolution de son lien "
            f"avec {target}."
        )
    has_end = bool(
        re.search(
            r"\b(?:fin\w*|ruptur\w*|separ\w*|termin\w*|couper les ponts)\b",
            combined,
        )
    )
    has_renewal = bool(
        re.search(r"\b(?:renouveau\w*|rapproch\w*|stabilis\w*|reprise)\b", combined)
    )
    if has_end and has_renewal and target and third:
        details.append(
            f"Les passages décrivent une fin progressive du lien {target}–{third} "
            f"et un rapprochement ou une stabilisation possibles entre la cliente "
            f"et {target}."
        )
    elif has_end and has_renewal:
        details.append(
            "Les passages abordent à la fois une séparation progressive et la "
            "possibilité d'un rapprochement ou d'une stabilisation."
        )
    elif has_end:
        details.append("Une séparation ou une fin de relation est largement évoquée.")
    elif has_renewal:
        details.append("Un rapprochement ou un renouveau du lien est évoqué.")
    return " ".join(details)


def extract_claim(question: str) -> str:
    """Extrait de préférence l'affirmation placée entre guillemets."""

    value = str(question or "").strip()
    if not value:
        raise ValueError("La question ne peut pas être vide.")
    quoted: list[str] = []
    patterns = (
        r"«\s*(.+?)\s*»",
        r"[“]\s*(.+?)\s*[”]",
        r'"\s*(.+?)\s*"',
    )
    for pattern in patterns:
        quoted.extend(match.strip() for match in re.findall(pattern, value, re.DOTALL))
    quoted = [item for item in quoted if len(_fold(item)) >= 3]
    if quoted:
        return max(quoted, key=len)

    # Sans guillemets, on retire le cadre interrogatif et le verbe de discours.
    # La formulation « est-ce que la master dit que ... » est celle du champ de
    # l'interface ; la laisser dans la requête donnait artificiellement beaucoup
    # de poids à « ça va » et aux libellés de rôle.
    cleaned = re.sub(
        r"(?is)^\s*(?:est[- ]ce\s+que\s+)?"
        r"(?:(?:la|le|un|une)\s+)?"
        r"(?:master|voyant(?:e)?|m[eé]dium|client(?:e)?)\s+"
        r"(?:(?:a|aurait)\s+)?"
        r"(?:dit|indique|annonce|explique|affirme|pense|voit|pr[eé]voit)\s*"
        r"(?:que|qu['’])?\s*",
        "",
        value,
        count=1,
    ).strip()

    # Formes rapportées plus longues, par exemple « la cliente affirme que la
    # voyante lui aurait dit ... ». Conserver le reste évite de réécrire
    # l'affirmation du dossier.
    cleaned = re.sub(
        r"(?is)^.*?(?:aurait\s+dit|a\s+dit|affirme\s+que|m['’]a\s+dit\s+que)\s*[:,-]?\s*",
        "",
        cleaned,
        count=1,
    ).strip()
    cleaned = re.sub(
        r"(?is)\s*[,;:-]?\s*(?:est[- ]ce\s+que\s+)?(?:tu\s+)?(?:consid[eè]res?|penses?)\s+.*$",
        "",
        cleaned,
        count=1,
    ).strip(" .?!:;,-")
    return cleaned or value


def _stem(token: str) -> str:
    # Stemming français volontairement conservateur pour rester explicable et
    # ne pas ajouter de dépendance au moteur hors ligne.
    for suffix in (
        "eraient",
        "erions",
        "erez",
        "erai",
        "eras",
        "erait",
        "aient",
        "ement",
        "ments",
        "ation",
        "ions",
        "ees",
        "es",
        "ent",
        "ait",
        "era",
        "er",
        "ee",
        "e",
        "s",
    ):
        if len(token) - len(suffix) >= 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _tokens(value: str, *, keep_stopwords: bool = False) -> tuple[str, ...]:
    result: list[str] = []
    for raw in _fold(value).split():
        if len(raw) < 2:
            continue
        if not keep_stopwords and raw in _STOPWORDS:
            continue
        result.append(_stem(raw))
    return tuple(result)


def _bigrams(tokens: Sequence[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:]))


def _semantic_token(token: str) -> str:
    if token in _RELATION_END_STEMS:
        return "__fin_relation__"
    if token in _RELATION_STEMS:
        return "__relation__"
    return token


def _semantic_tokens(value: str) -> tuple[str, ...]:
    """Normalise les synonymes métier sans générer de nouveau texte."""

    result = [_semantic_token(token) for token in _tokens(value)]
    folded = _fold(value)
    if any(re.search(pattern, folded) for pattern in _RELATION_END_PHRASES):
        result.append("__fin_relation__")
    return tuple(result)


def _text_windows(value: str) -> tuple[str, ...]:
    """Découpe un long tour en propositions locales et paires voisines.

    Les tours diarizés peuvent durer près d'une minute. Comparer la polarité de
    toute cette minute faisait qu'un « hors de question » sans rapport avec la
    phrase recherchée transformait une preuve affirmative en contradiction.
    """

    text = str(value or "").strip()
    if not text:
        return ("",)
    parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?;:])\s+|[\r\n]+", text)
        if part.strip()
    ]
    windows = list(parts)
    if len(parts) > 1:
        windows.extend(f"{left} {right}" for left, right in zip(parts, parts[1:]))
    # Certains résultats ASR ne comportent aucune ponctuation sur une minute
    # entière. Des fenêtres glissantes empêchent alors qu'une négation située
    # quarante mots avant la preuve ne soit considérée comme sa polarité.
    for part in parts:
        words = part.split()
        if len(words) <= 48:
            continue
        for start in range(0, len(words), 18):
            window = " ".join(words[start : start + 42]).strip()
            if len(window.split()) >= 12:
                windows.append(window)
            if start + 42 >= len(words):
                break
    # Le texte complet reste utile pour une courte citation qui aurait été
    # ponctuée de manière excessive par le moteur de transcription.
    if len(text.split()) <= 45:
        windows.append(text)
    return tuple(dict.fromkeys(windows))


def _relation_end_polarity(value: str) -> bool | None:
    """Retourne la polarité locale du concept « fin de relation ».

    ``True`` signifie qu'au moins une fin est affirmée, ``False`` que les seules
    occurrences sont explicitement niées, et ``None`` qu'aucune occurrence ne
    permet de décider. Les mots négatifs sans lien syntaxique avec la fin sont
    ignorés (par exemple « il n'est pas porté sur les choix »).
    """

    folded = _fold(value)
    cue_pattern = re.compile(
        r"\b(?:fin(?:i|ie|ir|ira|issent)?|ruptur\w*|separ\w*|"
        r"termin\w*|arret\w*|cess\w*|romp\w*)\b|"
        r"\bmettre\s+(?:un\s+terme|fin)\b|"
        r"\bcouper\s+les\s+ponts\b|"
        r"\bse\s+la(?:che|chent|cher)\b|"
        r"\bse\s+laiss(?:e|ent|er)\s+partir\b"
    )
    polarities: list[bool] = []
    for match in cue_pattern.finditer(folded):
        before = folded[max(0, match.start() - 42) : match.start()]
        after = folded[match.end() : min(len(folded), match.end() + 30)]
        negated = bool(
            re.search(r"\b(?:pas\s+de|aucun(?:e)?|sans)\s*$", before)
            or re.search(r"\bne\b[^.!?;]{0,35}$", before)
            and re.search(r"^.{0,24}\b(?:pas|plus|jamais)\b", after)
            or re.search(
                r"\bne\b[^.!?;]{0,30}\b(?:pas|plus|jamais)\b[^.!?;]{0,16}$",
                before,
            )
        )
        polarities.append(not negated)
    if not polarities:
        return None
    if any(polarities):
        return True
    return False


def _polarity_conflict(claim: str, evidence: str) -> bool:
    claim_concepts = set(_semantic_tokens(claim))
    evidence_concepts = set(_semantic_tokens(evidence))
    if "__fin_relation__" in claim_concepts.intersection(evidence_concepts):
        claim_polarity = _relation_end_polarity(claim)
        evidence_polarity = _relation_end_polarity(evidence)
        if claim_polarity is not None and evidence_polarity is not None:
            return claim_polarity != evidence_polarity
    return _has_negation(claim) != _has_negation(evidence)


def _has_negation(value: str) -> bool:
    folded = _fold(value)
    words = set(folded.split())
    if words.intersection(_NEGATION_WORDS):
        return True
    return bool(
        re.search(r"\bne\b.{0,45}\b(?:pas|plus|jamais|rien)\b", folded)
        or re.search(r"\bn\b.{0,45}\b(?:pas|plus|jamais|rien)\b", folded)
    )


def _target_role(question: str) -> RoleName | None:
    folded = _fold(question)
    # Dans « la cliente dit que la voyante lui a dit », la personne dont on
    # vérifie les propos est le Master : ce rôle est donc prioritaire.
    if any(label in folded.split() for label in _MASTER_LABELS):
        return "Master"
    if any(label in folded.split() for label in _CLIENT_LABELS):
        return "Client"
    return None


def _score_turn(turn: _Turn, claim: str, role: RoleName) -> _ScoredTurn:
    claim_folded = _fold(claim)
    text_folded = _fold(turn.text)
    claim_tokens = _semantic_tokens(claim)
    claim_set = set(claim_tokens)
    claim_bigrams = _bigrams(claim_tokens)
    exact = bool(claim_folded and claim_folded in text_folded)

    best_window = turn.text
    best_metrics = (-1.0, 0.0, 0.0, 0.0)
    for window in _text_windows(turn.text):
        text_tokens = _semantic_tokens(window)
        text_set = set(text_tokens)
        token_recall = len(claim_set.intersection(text_set)) / max(1, len(claim_set))
        # Le prédicat demandé est obligatoire : citer seulement les bons noms
        # et le mot « relation » ne prouve pas qu'une fin a été annoncée.
        if "__fin_relation__" in claim_set and "__fin_relation__" not in text_set:
            token_recall *= 0.55
        # ``content_recall`` est volontairement calculé sur les mêmes concepts
        # informatifs. Les déterminants et auxiliaires ne doivent jamais rendre
        # une contradiction « fiable ».
        content_recall = token_recall
        text_bigrams = _bigrams(text_tokens)
        bigram_recall = len(claim_bigrams.intersection(text_bigrams)) / max(
            1, len(claim_bigrams)
        )
        similarity = SequenceMatcher(None, claim_folded, _fold(window)).ratio()
        window_exact = bool(claim_folded and claim_folded in _fold(window))
        score = (
            1.0
            if window_exact
            else 0.62 * token_recall + 0.25 * bigram_recall + 0.13 * similarity
        )
        metrics = (score, token_recall, content_recall, bigram_recall)
        if metrics > best_metrics:
            best_metrics = metrics
            best_window = window

    score, token_recall, content_recall, bigram_recall = best_metrics
    if exact:
        score = 1.0
    # Une négation située ailleurs dans un long tour ne doit jamais annuler une
    # citation littérale. Les tours téléphoniques peuvent durer plusieurs
    # dizaines de secondes et contenir plusieurs idées indépendantes. La
    # polarité est donc comparée uniquement dans la meilleure fenêtre locale.
    polarity_conflict = (
        not exact
        and _polarity_conflict(claim, best_window)
        and content_recall >= 0.58
    )
    return _ScoredTurn(
        turn=turn,
        role=role,
        score=max(0.0, min(1.0, score)),
        token_recall=token_recall,
        content_recall=content_recall,
        bigram_recall=bigram_recall,
        exact=exact,
        polarity_conflict=polarity_conflict,
        excerpt=best_window,
    )


def _citation(item: _ScoredTurn) -> Citation:
    turn = item.turn
    return Citation(
        id=f"turn-{turn.index:05d}",
        turn_index=turn.index,
        start=turn.start,
        end=turn.end,
        speaker_id=turn.speaker_id,
        speaker=turn.speaker,
        role=item.role,
        text=turn.text,
        relevance=round(item.score, 4),
        excerpt=item.excerpt,
    )


def _scored_sort_key(item: _ScoredTurn) -> tuple[int, float, int]:
    """Classe les quasi-égalités dans l'ordre de la consultation.

    Une différence lexicale inférieure à deux points sur cent n'est pas assez
    significative pour envoyer l'utilisateur douze minutes plus loin dans
    l'audio. Le premier passage probant est alors plus naturel à relire.
    """

    relevance_bucket = int((item.score + 1e-9) * 50)
    return (-relevance_bucket, item.turn.start, item.turn.index)


@dataclass(slots=True, frozen=True)
class _RelationshipClarification:
    answer: str
    options: tuple[dict[str, str], ...]
    evidence: tuple[_ScoredTurn, ...]


def _named_relation_target(question: str) -> str | None:
    """Retrouve le prénom placé après « conjoint », « partenaire », etc."""

    match = re.search(
        r"\b(?:conjoint(?:e)?|compagnon|compagne|partenaire|mari|femme)\s+"
        r"(?:s['’]appelle\s+)?"
        r"([A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]{2,})",
        str(question or ""),
        re.IGNORECASE,
    )
    if not match:
        return None
    value = match.group(1).strip(" .'’-")
    return value or None


def _other_person_in_pair(text: str, target: str) -> str | None:
    """Trouve l'autre prénom dans une paire explicite « X et Y »."""

    name = r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]{2,}"
    ignored = {
        "alors",
        "apres",
        "avant",
        "client",
        "cliente",
        "donc",
        "master",
        "mais",
        "oui",
        "pour",
    }
    target_folded = _fold(target)
    for left, right in re.findall(rf"\b({name})\s+(?:et|avec)\s+({name})\b", text):
        left_folded = _fold(left)
        right_folded = _fold(right)
        if left_folded == target_folded and right_folded not in ignored:
            return right
        if right_folded == target_folded and left_folded not in ignored:
            return left
    return None


def _client_relation_is_positive(text: str, target: str) -> bool:
    folded = _fold(text)
    target_folded = _fold(target)
    if not re.search(rf"\b{re.escape(target_folded)}\b", folded):
        return False
    if not re.search(r"\b(?:vous|votre)\b", folded):
        return False
    return bool(
        re.search(
            r"\b(?:renouveau|stabilis\w*|rapproch\w*|reprise|reprendre|"
            r"nouvel\s+elan|evolu\w*\s+ensemble|relation\s+amoureuse)\b",
            folded,
        )
    )


def _client_relation_excerpt(text: str, target: str) -> str:
    for window in _text_windows(text):
        if _client_relation_is_positive(window, target):
            words = window.split()
            for index, word in enumerate(words):
                folded_word = _fold(word)
                if re.match(
                    r"^(?:renouveau|stabilis|rapproch|reprise|reprendre|nouvel|evolu)",
                    folded_word,
                ):
                    compact = " ".join(
                        words[max(0, index - 10) : min(len(words), index + 22)]
                    ).strip()
                    if _client_relation_is_positive(compact, target):
                        return compact
            return window
    return str(text or "").strip()


def _relationship_clarification(
    question: str,
    scored: Sequence[_ScoredTurn],
) -> _RelationshipClarification | None:
    """Évite de confondre le couple Client–X avec le couple X–tierce personne.

    La clarification n'est proposée que lorsque les deux lectures sont
    réellement présentes dans les propos du Master. Une simple question sur
    une rupture ne devient donc pas ambiguë par défaut.
    """

    target = _named_relation_target(question)
    folded_question = _fold(question)
    if not target or not re.search(r"\b(?:ca|cela|son|sa)\b", folded_question):
        return None

    separation_item: _ScoredTurn | None = None
    other_person: str | None = None
    client_item: _ScoredTurn | None = None
    for item in sorted(scored, key=lambda value: (value.turn.start, value.turn.index)):
        if item.role != "Master":
            continue
        if separation_item is None and _relation_end_polarity(item.turn.text) is True:
            other = _other_person_in_pair(item.turn.text, target)
            if other:
                separation_item = item
                other_person = other
        if client_item is None and _client_relation_is_positive(item.turn.text, target):
            client_item = item
        if separation_item is not None and client_item is not None:
            break

    if separation_item is None or client_item is None or not other_person:
        return None

    pair_label = f"{target} et {other_person}"
    client_label = f"La cliente et {target}"
    answer = (
        f"La question peut viser deux relations. Pour {pair_label}, les passages "
        "retenus annoncent une séparation progressive. Pour la cliente et "
        f"{target}, le Master évoque au contraire un renouveau ou une "
        "stabilisation du lien. Choisissez le couple visé pour obtenir un "
        "verdict unique."
    )
    options = (
        {
            "label": pair_label,
            "question": (
                f"Est-ce que le Master dit que la relation entre {target} et "
                f"{other_person} va se terminer ?"
            ),
        },
        {
            "label": client_label,
            "question": (
                f"Est-ce que le Master dit que la relation entre la cliente et "
                f"{target} va se terminer ?"
            ),
        },
    )
    client_evidence = replace(
        client_item,
        excerpt=_client_relation_excerpt(client_item.turn.text, target),
    )
    return _RelationshipClarification(
        answer=answer,
        options=options,
        evidence=(separation_item, client_evidence),
    )


def retrieve_passages(
    turns: Sequence[object],
    query: str,
    *,
    roles: RoleProposal | None = None,
    target_role: RoleName | None = None,
    top_k: int = 6,
) -> tuple[Citation, ...]:
    """Retourne les tours les plus proches, sans générer ni reformuler le texte."""

    if top_k < 1:
        raise ValueError("top_k doit être supérieur ou égal à 1.")
    claim = extract_claim(query)
    records = _normalise_turns(turns)
    role_map = roles or propose_speaker_roles(turns)
    scored = [
        _score_turn(turn, claim, role_map.role_for(turn.speaker_id))
        for turn in records
    ]
    role_filter = target_role or _target_role(query)
    if role_filter and not role_map.needs_confirmation:
        filtered = [item for item in scored if item.role == role_filter]
        if filtered:
            scored = filtered
    scored.sort(key=_scored_sort_key)
    return tuple(_citation(item) for item in scored[:top_k])


def build_grounded_prompt(request: ReasonerRequest) -> str:
    """Construit le prompt strict destiné à un futur backend local.

    La fonction ne lance aucun téléchargement et n'importe aucune bibliothèque
    de modèle. Elle permet aux backends OpenVINO/ONNX de partager le même
    contrat de sortie.
    """

    passages = "\n".join(
        (
            f"[{item.id}] {item.role} | {item.start:.2f}-{item.end:.2f} | "
            f"{item.text}"
        )
        for item in request.candidates
    )
    return (
        "Tu vérifies une affirmation uniquement à partir des passages fournis.\n"
        "N'utilise aucune connaissance externe. N'invente aucune citation.\n"
        "Réponds uniquement en JSON avec les clés verdict, confidence et "
        "citation_ids.\n"
        "verdict doit être l'une de ces valeurs exactes : Confirmé, Partiel, "
        "Contredit, Non retrouvé, À clarifier.\n"
        "citation_ids doit contenir uniquement des identifiants présents entre "
        "crochets ci-dessous. Si rien ne permet de conclure, utilise Non retrouvé "
        "et une liste vide.\n\n"
        f"<affirmation>{request.claim}</affirmation>\n"
        f"<role_cible>{request.target_role or 'Non précisé'}</role_cible>\n"
        f"<passages>\n{passages}\n</passages>"
    )


def build_consultation_prompt(request: ConsultationRequest) -> str:
    """Construit le contexte strict d'une question libre sur la consultation."""

    passages = "\n".join(
        (
            f"[{item.id}] {_answer_timecode(item.start)}–"
            f"{_answer_timecode(item.end)} | {item.role} | {item.text}"
        )
        for item in request.candidates
    )
    role_note = (
        "L'attribution Client/Master est confirmée."
        if request.role_mapping_confirmed
        else (
            "L'attribution Client/Master est provisoire : signale cette limite "
            "si elle influence la réponse."
        )
    )
    audio_context = (
        json.dumps(
            request.audio_observations,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if request.audio_observations
        else (
            '{"available":false,"limitations":'
            '"Aucune mesure acoustique fournie pour cette question."}'
        )
    )
    return (
        "Tu es l'assistant d'analyse d'une consultation téléphonique entre une "
        "Cliente et un Master. Réponds en français, de manière précise, utile et "
        "professionnelle.\n"
        "RÈGLES ABSOLUES:\n"
        "1. Utilise uniquement la transcription horodatée fournie. N'ajoute aucun "
        "fait extérieur et n'invente aucune citation.\n"
        "2. Réponds directement à la question, même si elle demande une synthèse, "
        "une comparaison, une chronologie, les thèmes, les positions des personnes "
        "ou l'évolution de l'échange.\n"
        "3. Distingue ce qui est explicitement dit, ce qui est une inférence "
        "prudente et ce qui ne peut pas être établi.\n"
        "4. Pour le ton vocal, le volume, l'intonation ou une émotion acoustique, "
        "la transcription seule ne suffit pas : tu peux analyser les mots, les "
        "désaccords et la dynamique des tours, mais tu dois nommer cette limite.\n"
        "5. Chaque conclusion factuelle doit être soutenue par un ou plusieurs "
        "citation_ids existants. Si les preuves sont insuffisantes, utilise le "
        "statut insufficient et ne force pas une réponse.\n"
        "6. Ne juge jamais si une prédiction de voyance est vraie dans le monde "
        "réel ; indique uniquement ce qui a été formulé dans la consultation.\n"
        "7. Si la question demande quand, à quel moment ou une chronologie, "
        "donne les timecodes utiles directement dans answer ou key_points et "
        "distingue clairement les personnes ou relations concernées.\n\n"
        "Réponds uniquement avec un objet JSON valide comportant exactement:\n"
        '{\n  "status": "answered|partial|insufficient|clarify|confirmed|'
        'contradicted|topic_primary|topic_present|topic_mentioned",\n'
        '  "confidence": 0.0,\n'
        '  "answer": "réponse complète en 2 à 6 phrases",\n'
        '  "key_points": ["point précis 1", "point précis 2"],\n'
        '  "nuance": "limite ou lecture prudente",\n'
        '  "citation_ids": ["turn-00001"]\n}\n\n'
        f"<intention>{request.intent}</intention>\n"
        f"<question>{request.question}</question>\n"
        f"<roles>{role_note}</roles>\n"
        f"<mesures_audio>{audio_context}</mesures_audio>\n"
        f"<transcription>\n{passages}\n</transcription>"
    )


def _coerce_verdict(value: object) -> VerdictName | None:
    folded = _fold(str(value or ""))
    aliases: dict[str, VerdictName] = {
        "confirme": CONFIRMED,
        "confirmed": CONFIRMED,
        "partiel": PARTIAL,
        "partiellement confirme": PARTIAL,
        "partial": PARTIAL,
        "contredit": CONTRADICTED,
        "contradicted": CONTRADICTED,
        "non retrouve": NOT_FOUND,
        "not found": NOT_FOUND,
        "insufficient evidence": NOT_FOUND,
        "a clarifier": CLARIFY,
        "question ambigue": CLARIFY,
        "ambiguous": CLARIFY,
    }
    return aliases.get(folded)


def _parse_reasoner_decision(
    value: ReasonerDecision | Mapping[str, object] | str,
    *,
    valid_ids: set[str],
) -> ReasonerDecision | None:
    if isinstance(value, ReasonerDecision):
        raw_verdict: object = value.verdict
        raw_confidence: object = value.confidence
        raw_ids: object = value.citation_ids
    else:
        raw: object = value
        if isinstance(value, str):
            try:
                raw = json.loads(value)
            except json.JSONDecodeError:
                return None
        if not isinstance(raw, Mapping):
            return None
        raw_verdict = raw.get("verdict")
        raw_confidence = raw.get("confidence", 0.5)
        raw_ids = raw.get("citation_ids", [])

    verdict = _coerce_verdict(raw_verdict)
    if verdict is None:
        return None
    try:
        confidence = max(0.0, min(1.0, float(raw_confidence)))
    except (TypeError, ValueError):
        confidence = 0.5
    if isinstance(raw_ids, str):
        ids = (raw_ids,)
    elif isinstance(raw_ids, Sequence):
        ids = tuple(str(item) for item in raw_ids)
    else:
        ids = ()
    safe_ids = tuple(
        dict.fromkeys(item for item in ids if item in valid_ids)
    )[:6]
    if verdict != NOT_FOUND and not safe_ids:
        # Décision non prouvée ou citation inventée : elle n'est jamais exposée.
        return None
    if verdict == NOT_FOUND:
        safe_ids = ()
    return ReasonerDecision(
        verdict=verdict,
        confidence=round(confidence, 3),
        citation_ids=safe_ids,
    )


def _model_context_citations(
    records: Sequence[_Turn],
    roles: RoleProposal,
    question: str,
    intent: QueryIntent,
    *,
    max_characters: int = 90_000,
) -> tuple[Citation, ...]:
    """Fournit le transcript complet tant qu'il tient dans le contexte.

    Pour les consultations exceptionnellement longues, les passages les plus
    proches sont combinés à un échantillon chronologique. Le modèle conserve
    ainsi à la fois le contexte global et les preuves directement pertinentes.
    """

    all_citations = tuple(
        Citation(
            id=f"turn-{turn.index:05d}",
            turn_index=turn.index,
            start=turn.start,
            end=turn.end,
            speaker_id=turn.speaker_id,
            speaker=turn.speaker,
            role=roles.role_for(turn.speaker_id),
            text=turn.text,
            relevance=1.0,
            excerpt=turn.text,
        )
        for turn in records
    )
    estimated = sum(len(item.text) + 80 for item in all_citations)
    if estimated <= max_characters:
        return all_citations

    selected_indexes: set[int] = set()
    sample_count = min(24, len(records))
    if sample_count:
        for sample_index in range(sample_count):
            index = round(sample_index * (len(records) - 1) / max(1, sample_count - 1))
            selected_indexes.add(index)

    if intent != OPEN_SUMMARY:
        query_for_retrieval = (
            _extract_topic_phrase(question)
            if intent == TOPIC_PRESENCE
            else question
        )
        claim = extract_claim(query_for_retrieval)
        ranked = sorted(
            (
                _score_turn(turn, claim, roles.role_for(turn.speaker_id))
                for turn in records
            ),
            key=_scored_sort_key,
        )
        selected_indexes.update(item.turn.index for item in ranked[:24])

    bounded: list[Citation] = []
    used = 0
    for index in sorted(selected_indexes):
        citation = all_citations[index]
        cost = len(citation.text) + 80
        if bounded and used + cost > max_characters:
            break
        bounded.append(citation)
        used += cost
    return tuple(bounded)


def _parse_consultation_decision(
    value: ConsultationDecision | Mapping[str, object] | str,
    *,
    valid_ids: set[str],
) -> ConsultationDecision | None:
    if isinstance(value, ConsultationDecision):
        raw_status: object = value.status
        raw_confidence: object = value.confidence
        raw_answer: object = value.answer
        raw_ids: object = value.citation_ids
        raw_points: object = value.key_points
        raw_nuance: object = value.nuance
    else:
        raw: object = value
        if isinstance(value, str):
            try:
                raw = json.loads(value)
            except json.JSONDecodeError:
                return None
        if not isinstance(raw, Mapping):
            return None
        raw_status = raw.get("status", "answered")
        raw_confidence = raw.get("confidence", 0.55)
        raw_answer = raw.get("answer", "")
        raw_ids = raw.get("citation_ids", [])
        raw_points = raw.get("key_points", [])
        raw_nuance = raw.get("nuance", "")

    status = _fold(str(raw_status or "answered")).replace(" ", "_")
    aliases = {
        "reponse": "answered",
        "answered": "answered",
        "confirme": "confirmed",
        "confirmed": "confirmed",
        "contredit": "contradicted",
        "contradicted": "contradicted",
        "partiel": "partial",
        "partial": "partial",
        "insuffisant": "insufficient",
        "insufficient": "insufficient",
        "non_retrouve": "insufficient",
        "clarifier": "clarify",
        "a_clarifier": "clarify",
        "topic_primary": "topic_primary",
        "theme_principal": "topic_primary",
        "topic_present": "topic_present",
        "theme_present": "topic_present",
        "topic_mentioned": "topic_mentioned",
        "mention_ponctuelle": "topic_mentioned",
    }
    status = aliases.get(status, status)
    if status not in {
        "answered",
        "confirmed",
        "contradicted",
        "partial",
        "insufficient",
        "clarify",
        "topic_primary",
        "topic_present",
        "topic_mentioned",
    }:
        status = "answered"

    answer = re.sub(r"\s+", " ", str(raw_answer or "")).strip()
    nuance = re.sub(r"\s+", " ", str(raw_nuance or "")).strip()
    try:
        confidence = max(0.0, min(1.0, float(raw_confidence)))
    except (TypeError, ValueError):
        confidence = 0.55
    if isinstance(raw_ids, str):
        ids = (raw_ids,)
    elif isinstance(raw_ids, Sequence):
        ids = tuple(str(item) for item in raw_ids)
    else:
        ids = ()
    safe_ids = tuple(
        dict.fromkeys(item for item in ids if item in valid_ids)
    )[:6]
    if isinstance(raw_points, str):
        points = (re.sub(r"\s+", " ", raw_points).strip(),)
    elif isinstance(raw_points, Sequence):
        points = tuple(
            re.sub(r"\s+", " ", str(item)).strip()
            for item in raw_points
            if str(item).strip()
        )
    else:
        points = ()

    if not answer:
        return None
    if status not in {"insufficient", "clarify"} and not safe_ids:
        # Une réponse factuelle sans passage réel n'est jamais affichée.
        return None
    return ConsultationDecision(
        status=status,
        confidence=round(confidence, 3),
        answer=answer[:6000],
        citation_ids=safe_ids,
        key_points=points[:5],
        nuance=nuance[:1800],
    )


def _model_verdict(intent: QueryIntent, status: str) -> VerdictName:
    if intent == OPEN_SUMMARY:
        return NOT_FOUND if status == "insufficient" else SUMMARY
    if intent == TOPIC_PRESENCE:
        return {
            "topic_primary": TOPIC_PRIMARY,
            "topic_present": TOPIC_PRESENT,
            "topic_mentioned": TOPIC_MENTIONED,
            "insufficient": NOT_FOUND,
            "clarify": CLARIFY,
        }.get(status, TOPIC_PRESENT)
    if intent == CLAIM_VERIFICATION:
        return {
            "confirmed": CONFIRMED,
            "contradicted": CONTRADICTED,
            "partial": PARTIAL,
            "insufficient": NOT_FOUND,
            "clarify": CLARIFY,
        }.get(status, PARTIAL)
    return {
        "partial": PARTIAL,
        "insufficient": NOT_FOUND,
        "clarify": CLARIFY,
    }.get(status, ANSWERED)


def _try_consultation_reasoner(
    reasoner: object,
    *,
    records: Sequence[_Turn],
    question: str,
    intent: QueryIntent,
    roles: RoleProposal,
    audio_observations: Mapping[str, object] | None = None,
) -> tuple[VerificationResult | None, str | None]:
    answer_method = getattr(reasoner, "answer", None)
    if not callable(answer_method):
        return None, None
    candidates = _model_context_citations(records, roles, question, intent)
    candidate_by_id = {item.id: item for item in candidates}
    request = ConsultationRequest(
        question=question,
        intent=intent,
        candidates=candidates,
        role_mapping_confirmed=not roles.needs_confirmation,
        audio_observations=audio_observations,
    )
    try:
        raw = answer_method(request)
        decision = _parse_consultation_decision(
            raw,
            valid_ids=set(candidate_by_id),
        )
    except Exception as exc:  # pragma: no cover - dépend du runtime local
        return (
            None,
            "Le modèle local polyvalent est indisponible ; l'analyse de secours "
            f"a pris le relais ({type(exc).__name__}).",
        )
    if decision is None:
        return (
            None,
            "La réponse du modèle local n'était pas suffisamment reliée aux "
            "passages ; l'analyse de secours a pris le relais.",
        )
    citations = tuple(
        candidate_by_id[citation_id]
        for citation_id in decision.citation_ids
        if citation_id in candidate_by_id
    )
    confidence_caps = {
        "insufficient": 0.6,
        "clarify": 0.65,
        "partial": 0.78,
        "contradicted": 0.9,
        "confirmed": 0.9,
        "topic_primary": 0.9,
        "topic_present": 0.9,
        "topic_mentioned": 0.85,
        "answered": 0.9,
    }
    calibrated_confidence = min(
        decision.confidence,
        confidence_caps.get(decision.status, 0.85),
    )
    if (
        isinstance(audio_observations, Mapping)
        and audio_observations.get("available")
    ):
        # Les métriques acoustiques décrivent l'intensité, le débit et les
        # alternances, mais ne prouvent jamais seules une émotion ou une
        # intention. La confiance affichée doit refléter cette limite.
        calibrated_confidence = min(calibrated_confidence, 0.82)
    if roles.needs_confirmation:
        calibrated_confidence = min(calibrated_confidence, 0.65)
    nuance = decision.nuance or (
        "La réponse porte uniquement sur le contenu de cette consultation."
    )
    if roles.needs_confirmation:
        nuance += " L'attribution Client/Master doit encore être confirmée."
    claim = (
        _extract_topic_phrase(question)
        if intent == TOPIC_PRESENCE
        else (extract_claim(question) if intent == CLAIM_VERIFICATION else question)
    )
    return (
        VerificationResult(
            question=question,
            claim=claim,
            intent=intent,
            target_role=(
                _target_role(question) if intent == CLAIM_VERIFICATION else None
            ),
            verdict=_model_verdict(intent, decision.status),
            confidence=round(calibrated_confidence, 3),
            explanation=decision.answer,
            citations=citations,
            roles=roles,
            backend=str(getattr(reasoner, "name", type(reasoner).__name__)),
            answer=decision.answer,
            analysis_points=decision.key_points,
            nuance=nuance,
        ),
        None,
    )


def _deterministic_decision(scored: Sequence[_ScoredTurn]) -> ReasonerDecision:
    if not scored:
        return ReasonerDecision(NOT_FOUND, 0.55, ())
    top = scored[0]
    citation_id = f"turn-{top.turn.index:05d}"
    if top.polarity_conflict and top.content_recall >= 0.62:
        confidence = min(0.96, 0.58 + 0.36 * top.content_recall)
        return ReasonerDecision(CONTRADICTED, round(confidence, 3), (citation_id,))
    if top.exact or (
        top.score >= 0.70 and top.token_recall >= 0.68
    ) or (
        top.token_recall >= 0.80 and top.bigram_recall >= 0.40
    ):
        confidence = min(0.98, 0.62 + 0.35 * top.score)
        eligible = [
            item
            for item in scored
            if item.score >= max(0.45, top.score * 0.72)
        ]
        selected_items = [top]
        # Une réponse de Master est souvent coupée par un très court « oui » du
        # Client. Si la suite immédiate complète le même concept, la conserver
        # avant de sauter vers un passage plus tardif rend la preuve lisible.
        continuation = next(
            (
                item
                for item in sorted(eligible, key=lambda value: value.turn.start)
                if item is not top
                and 0.0 <= item.turn.start - top.turn.end <= 75.0
                and item.role == top.role
                and item.token_recall >= 0.80
            ),
            None,
        )
        if continuation is not None:
            selected_items.append(continuation)
        for item in eligible:
            if item not in selected_items:
                selected_items.append(item)
            if len(selected_items) >= 3:
                break
        selected = tuple(
            f"turn-{item.turn.index:05d}" for item in selected_items[:3]
        )
        return ReasonerDecision(CONFIRMED, round(confidence, 3), selected)
    if top.score >= 0.34 and top.token_recall >= 0.32:
        confidence = min(0.84, 0.42 + 0.36 * top.score)
        selected = tuple(
            f"turn-{item.turn.index:05d}"
            for item in scored[:2]
            if item.score >= max(0.28, top.score * 0.72)
        )
        return ReasonerDecision(PARTIAL, round(confidence, 3), selected)
    # L'absence de correspondance est moins certaine qu'une citation exacte.
    confidence = max(0.5, min(0.78, 0.76 - 0.25 * top.score))
    return ReasonerDecision(NOT_FOUND, round(confidence, 3), ())


def _explanation(
    verdict: VerdictName,
    *,
    target_role: RoleName | None,
    roles: RoleProposal,
) -> str:
    subject = f"dans les propos du {target_role}" if target_role else "dans la transcription"
    if verdict == CONFIRMED:
        text = f"L'affirmation est appuyée par un passage correspondant {subject}."
    elif verdict == PARTIAL:
        text = (
            f"Une partie de l'affirmation est retrouvée {subject}, mais le passage "
            "ne permet pas de confirmer l'ensemble de la formulation."
        )
    elif verdict == CONTRADICTED:
        text = f"Un passage pertinent {subject} formule une négation incompatible."
    elif verdict == CLARIFY:
        text = (
            "La formulation peut désigner plusieurs personnes ou relations. "
            "Précisez le référent avant de retenir un verdict."
        )
    else:
        text = (
            f"Aucun passage suffisamment proche n'a été retrouvé {subject}. "
            "Cela ne prouve pas que l'affirmation est fausse."
        )
    if target_role and roles.needs_confirmation:
        text += " L'attribution Client/Master doit encore être confirmée."
    return text


def _answer_timecode(value: object) -> str:
    try:
        total = max(0, int(float(value or 0.0)))
    except (TypeError, ValueError):
        total = 0
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _citation_field(citation: Citation | Mapping[str, object], name: str) -> object:
    if isinstance(citation, Citation):
        return getattr(citation, name, None)
    return citation.get(name)


def _citation_text(citation: Citation | Mapping[str, object]) -> str:
    full = str(_citation_field(citation, "text") or "").strip()
    excerpt = str(
        _citation_field(citation, "excerpt")
        or _citation_field(citation, "quote")
        or full
    ).strip()
    return excerpt or full


def _compact_quote(value: str, *, limit: int = 220) -> str:
    compact = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(compact) <= limit:
        return compact
    shortened = compact[: limit + 1].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{shortened}…"


def _evidence_insight(citation: Citation | Mapping[str, object]) -> str:
    text = _citation_text(citation)
    folded = _fold(text)
    role = str(_citation_field(citation, "role") or "Interlocuteur")
    start = _citation_field(citation, "start")
    if start is None:
        start = _citation_field(citation, "start_seconds")
    role_subject = "la Cliente" if role == "Client" else f"le {role}"
    prefix = f"À {_answer_timecode(start)}, {role_subject}"

    ideas: list[str] = []
    relation_polarity = _relation_end_polarity(text)
    if relation_polarity is True:
        ideas.append("évoque explicitement une fin, une séparation ou un éloignement dans la relation")
    elif relation_polarity is False:
        ideas.append("indique explicitement que la relation ne se terminera pas")

    if re.search(
        r"\b(?:renouveau|stabilis\w*|rapproch\w*|reprise|reprendre|"
        r"nouvel\s+elan|relation\s+amoureuse)\b",
        folded,
    ):
        ideas.append("mentionne aussi un rapprochement, un renouveau ou une stabilisation du lien")
    if re.search(
        r"\b(?:reprise|reprendre)\b.{0,50}\b(?:pas\s+encore|pas\s+de\s+suite|"
        r"pas\s+pour\s+le\s+moment)\b",
        folded,
    ) or re.search(
        r"\b(?:pas\s+encore|pas\s+de\s+suite|pas\s+pour\s+le\s+moment)\b"
        r".{0,50}\b(?:reprise|reprendre)\b",
        folded,
    ):
        ideas.append("précise qu'aucune reprise immédiate n'est annoncée")
    if re.search(r"\b(?:voyage|vacances|deplacement)\w*\b", folded):
        ideas.append("mentionne un voyage, un déplacement ou des vacances")
    if re.search(
        r"\b(?:travail|emploi|profession|carriere)\b",
        folded,
    ) and re.search(r"\b(?:chang\w*|evolu\w*|nouveau|nouvelle)\b", folded):
        ideas.append("parle d'un changement ou d'une évolution professionnelle")

    if ideas:
        return prefix + " " + "; ".join(dict.fromkeys(ideas)) + "."
    quote = _compact_quote(
        str(
            _citation_field(citation, "excerpt")
            or _citation_field(citation, "quote")
            or text
        )
    )
    if quote.endswith("."):
        quote = quote[:-1]
    return f"{prefix} dit : « {quote} »." if quote else f"{prefix} apporte un passage pertinent."


def build_answer_sections(
    verdict: VerdictName | str,
    claim: str,
    target_role: RoleName | str | None,
    citations: Sequence[Citation | Mapping[str, object]],
    *,
    clarification_answer: str = "",
    role_mapping_confirmed: bool = True,
) -> tuple[str, tuple[str, ...], str]:
    """Produit une réponse locale détaillée sans inventer de faits.

    La conclusion reformule le verdict, les points d'analyse restent ancrés
    dans les citations et la nuance rappelle ce que l'outil peut réellement
    établir. La fonction accepte aussi les dictionnaires persistés afin que les
    anciennes analyses profitent du nouveau rendu sans migration.
    """

    clean_claim = re.sub(r"\s+", " ", str(claim or "")).strip(" .?!")
    clean_claim = re.sub(r"^(?:est[- ]ce\s+que|estceque)\s+", "", clean_claim, flags=re.IGNORECASE)
    if len(clean_claim) > 260:
        clean_claim = _compact_quote(clean_claim, limit=260)
    quoted_claim = f"« {clean_claim} »" if clean_claim else "l'affirmation formulée"
    role_label = str(target_role or "").strip()
    subject = f" dans les propos du {role_label}" if role_label else " dans la transcription"
    count = len(citations)
    passage_label = f"{count} passage{'s' if count != 1 else ''}"

    if str(verdict) == CLARIFY and clarification_answer.strip():
        conclusion = clarification_answer.strip()
    elif str(verdict) == CONFIRMED:
        supports_verb = "appuient" if count != 1 else "appuie"
        conclusion = (
            f"Oui. {passage_label.capitalize()}{subject} {supports_verb} "
            f"clairement {quoted_claim}."
        )
    elif str(verdict) == PARTIAL:
        contains_verb = "contiennent" if count != 1 else "contient"
        conclusion = (
            f"Partiellement. {passage_label.capitalize()}{subject} {contains_verb} "
            f"des éléments liés à {quoted_claim}, "
            "mais pas assez pour confirmer chaque détail de la formulation."
        )
    elif str(verdict) == CONTRADICTED:
        conclusion = (
            f"Non. Le passage le plus pertinent{subject} formule un élément "
            f"incompatible avec {quoted_claim}."
        )
    elif str(verdict) == CLARIFY:
        conclusion = (
            f"La question autour de {quoted_claim} peut viser plusieurs personnes "
            "ou relations. Une précision est nécessaire avant de retenir un verdict unique."
        )
    else:
        conclusion = (
            f"Aucun passage suffisamment probant n'a été retrouvé pour établir "
            f"{quoted_claim}. Cette absence ne prouve pas que l'affirmation est fausse."
        )

    points = tuple(_evidence_insight(citation) for citation in citations[:3])
    combined_text = " ".join(_citation_text(citation) for citation in citations)
    has_other_relation_signal = bool(
        "__fin_relation__" in set(_semantic_tokens(clean_claim))
        and re.search(
            r"\b(?:renouveau|stabilis\w*|rapproch\w*|reprise|reprendre)\b",
            _fold(combined_text),
        )
    )
    if str(verdict) == CLARIFY:
        nuance = (
            "Les passages décrivent des relations différentes. Choisissez la relation "
            "visée pour relancer une analyse qui ne mélange pas les deux situations."
        )
    elif str(verdict) == PARTIAL:
        nuance = (
            "Le thème est présent, mais la transcription ne répond pas à tous les "
            "éléments de la question. Le verdict reste donc volontairement nuancé."
        )
    elif str(verdict) == CONTRADICTED:
        nuance = (
            "Le verdict porte sur ce qui a été formulé dans la consultation : le "
            "passage retenu va dans le sens inverse de l'affirmation vérifiée."
        )
    elif str(verdict) == NOT_FOUND:
        nuance = (
            "Un passage peut avoir été mal transcrit ou formulé autrement. Il est "
            "préférable d'écouter l'audio avant de conclure à une absence définitive."
        )
    elif has_other_relation_signal:
        nuance = (
            "Certains extraits évoquent aussi un rapprochement ou une stabilisation "
            "pour un autre lien. La conclusion ci-dessus porte uniquement sur la "
            "relation nommée dans la question."
        )
    else:
        nuance = (
            "Cette conclusion confirme uniquement ce qui a été dit dans l'audio ; "
            "elle ne permet pas d'affirmer que la prédiction se réalisera."
        )
    if not role_mapping_confirmed:
        nuance += " L'attribution Client/Master doit encore être confirmée."
    return conclusion, points, nuance


def _summary_result(
    records: Sequence[_Turn],
    question: str,
    roles: RoleProposal,
) -> VerificationResult:
    analyses = _analyse_topics(records, roles)
    primary = next((item for item in analyses if item.score > 0.0), None)
    if primary is None:
        return VerificationResult(
            question=question,
            claim=question,
            intent=OPEN_SUMMARY,
            target_role=None,
            verdict=NOT_FOUND,
            confidence=0.45,
            explanation="Aucun thème dominant n'a pu être isolé dans la transcription.",
            citations=(),
            roles=roles,
            backend="analyse-thématique-locale",
            answer=(
                "La transcription est exploitable, mais aucun thème suffisamment "
                "récurrent ne permet d'en produire une synthèse fiable."
            ),
            nuance=(
                "Cette absence de synthèse automatique ne signifie pas que l'audio "
                "est vide ; une relecture de la transcription reste nécessaire."
            ),
        )

    selected_hits = _representative_topic_hits(primary)
    citations = tuple(
        _topic_citation(item, primary.definition) for item in selected_hits
    )
    answer_parts = [
        f"La consultation porte principalement sur {primary.definition.label}."
    ]
    if primary.definition.key == "sentimental":
        overview = _sentimental_overview(records, roles)
        if overview:
            answer_parts.append(overview)

    secondary = next(
        (
            item
            for item in analyses
            if item is not primary
            and len(item.hits) >= 2
            and item.score >= max(2.4, primary.score * 0.28)
        ),
        None,
    )
    if secondary is not None:
        answer_parts.append(
            f"Un sujet secondaire concerne {secondary.definition.label}."
        )

    total_topic_score = sum(item.score for item in analyses)
    dominance = primary.score / max(primary.score, total_topic_score)
    confidence = min(
        0.94,
        0.62 + 0.18 * min(1.0, primary.coverage / 0.25) + 0.14 * dominance,
    )
    points = tuple(_evidence_insight(citation) for citation in citations)
    nuance = (
        "Cette synthèse hiérarchise les thèmes réellement présents dans la "
        "transcription. Elle restitue les propos tenus, sans garantir que les "
        "prédictions évoquées se réaliseront."
    )
    if roles.needs_confirmation:
        nuance += " L'attribution Client/Master doit encore être confirmée."
    return VerificationResult(
        question=question,
        claim=question,
        intent=OPEN_SUMMARY,
        target_role=None,
        verdict=SUMMARY,
        confidence=round(confidence, 3),
        explanation="Synthèse thématique de la consultation complète.",
        citations=citations,
        roles=roles,
        backend="analyse-thématique-locale",
        answer=" ".join(answer_parts),
        analysis_points=points,
        nuance=nuance,
    )


def _ad_hoc_topic_analysis(
    records: Sequence[_Turn],
    roles: RoleProposal,
    phrase: str,
) -> _TopicAnalysis:
    tokens = tuple(dict.fromkeys(_tokens(phrase)))
    definition = _TopicDefinition(
        key="custom",
        label=f"le thème « {phrase} »",
        patterns=tuple(rf"\b{re.escape(token)}\w*\b" for token in tokens),
    )
    hits: list[_TopicHit] = []
    for turn in records:
        matches = _topic_matches(definition, turn.text)
        if not matches:
            continue
        recall = len(matches) / max(1, len(definition.patterns))
        if recall < (0.5 if len(definition.patterns) > 1 else 1.0):
            continue
        hits.append(
            _TopicHit(
                turn=turn,
                role=roles.role_for(turn.speaker_id),
                score=0.7 + 0.8 * recall,
                matched_patterns=matches,
            )
        )
    return _TopicAnalysis(
        definition=definition,
        hits=tuple(hits),
        score=sum(item.score for item in hits),
        coverage=len(hits) / max(1, len(records)),
    )


def _topic_presence_result(
    records: Sequence[_Turn],
    question: str,
    roles: RoleProposal,
) -> VerificationResult:
    phrase = _extract_topic_phrase(question)
    all_analyses = _analyse_topics(records, roles)
    by_key = {item.definition.key: item for item in all_analyses}
    requested_definitions = _requested_topic_definitions(question)
    requested = tuple(
        by_key[definition.key]
        for definition in requested_definitions
        if definition.key in by_key
    )
    if not requested:
        requested = (_ad_hoc_topic_analysis(records, roles, phrase),)

    matching = tuple(item for item in requested if item.hits)
    top_overall = next((item for item in all_analyses if item.score > 0.0), None)
    if not matching:
        answer = (
            f"Le thème « {phrase} » n'a pas été retrouvé de façon suffisamment "
            "claire dans la transcription."
        )
        return VerificationResult(
            question=question,
            claim=phrase,
            intent=TOPIC_PRESENCE,
            target_role=None,
            verdict=NOT_FOUND,
            confidence=0.72,
            explanation=answer,
            citations=(),
            roles=roles,
            backend="analyse-thématique-locale",
            answer=answer,
            nuance=(
                "Une formulation différente ou une erreur de transcription reste "
                "possible. Cette absence ne constitue pas une preuve sur la réalité "
                "extérieure."
            ),
        )

    strongest = max(matching, key=lambda item: item.score)
    is_primary = bool(
        top_overall is not None
        and strongest.definition.key == top_overall.definition.key
        and (
            strongest.coverage >= 0.14
            or strongest.score >= max(4.5, top_overall.score * 0.72)
        )
    )
    is_mentioned = bool(
        not is_primary
        and (
            len(strongest.hits) == 1
            or (
                top_overall is not None
                and strongest.score < max(2.5, top_overall.score * 0.24)
            )
        )
    )
    if is_primary:
        verdict = TOPIC_PRIMARY
        answer = (
            f"Oui. Le thème « {phrase} » constitue le sujet principal de la "
            "consultation."
        )
    elif is_mentioned:
        verdict = TOPIC_MENTIONED
        answer = (
            f"Oui, mais de façon ponctuelle. Le thème « {phrase} » apparaît "
            "surtout comme un élément de contexte, pas comme le sujet principal."
        )
    else:
        verdict = TOPIC_PRESENT
        answer = (
            f"Oui. Le thème « {phrase} » est bien abordé dans la consultation, "
            "sans être son sujet principal."
        )

    if strongest.definition.key == "sentimental":
        overview = _sentimental_overview(records, roles)
        if overview:
            answer += " " + overview

    selected_hits = _representative_topic_hits(strongest)
    citations = tuple(
        _topic_citation(item, strongest.definition) for item in selected_hits
    )
    points = tuple(_evidence_insight(citation) for citation in citations)
    relative_score = (
        strongest.score / max(strongest.score, top_overall.score)
        if top_overall is not None
        else 1.0
    )
    confidence = min(
        0.96,
        0.66
        + 0.18 * min(1.0, strongest.coverage / 0.20)
        + 0.10 * relative_score,
    )
    nuance = (
        "La réponse indique si le thème est présent et quelle place il occupe "
        "dans l'échange. Une phrase négative sur ce thème compte comme une "
        "mention du thème, pas comme son absence."
    )
    if roles.needs_confirmation:
        nuance += " L'attribution Client/Master doit encore être confirmée."
    return VerificationResult(
        question=question,
        claim=phrase,
        intent=TOPIC_PRESENCE,
        target_role=None,
        verdict=verdict,
        confidence=round(confidence, 3),
        explanation=answer,
        citations=citations,
        roles=roles,
        backend="analyse-thématique-locale",
        answer=answer,
        analysis_points=points,
        nuance=nuance,
    )


def _general_fallback_result(
    records: Sequence[_Turn],
    question: str,
    roles: RoleProposal,
    audio_observations: Mapping[str, object] | None = None,
) -> VerificationResult:
    """Réponse prudente lorsque le modèle polyvalent n'est pas disponible."""

    folded = _fold(question)
    tone_question = bool(
        re.search(
            r"\b(?:ton|voix|monte|degener\w*|tension|disput\w*|"
            r"agress\w*|coler\w*|enerve\w*)\b",
            folded,
        )
    )
    if tone_question:
        acoustic_available = bool(
            isinstance(audio_observations, Mapping)
            and audio_observations.get("available")
        )
        signal_pattern = re.compile(
            r"\b(?:hors de question|conflit\w*|disput\w*|agress\w*|"
            r"coler\w*|enerve\w*|marre|insult\w*|cri\w*|atroce|"
            r"cingl\w*|tension\w*)\b"
        )
        signal_turns = [
            turn for turn in records if signal_pattern.search(_fold(turn.text))
        ]
        if acoustic_available:
            by_id = {f"turn-{turn.index:05d}": turn for turn in records}
            for group_name in ("loudest_turns", "fastest_turns"):
                raw_group = audio_observations.get(group_name)
                if not isinstance(raw_group, Sequence):
                    continue
                for metric in raw_group[:3]:
                    if not isinstance(metric, Mapping):
                        continue
                    turn = by_id.get(str(metric.get("turn_id") or ""))
                    if turn is not None and turn not in signal_turns:
                        signal_turns.append(turn)
        citations = tuple(
            Citation(
                id=f"turn-{turn.index:05d}",
                turn_index=turn.index,
                start=turn.start,
                end=turn.end,
                speaker_id=turn.speaker_id,
                speaker=turn.speaker,
                role=roles.role_for(turn.speaker_id),
                text=turn.text,
                relevance=0.7,
                excerpt=turn.text,
            )
            for turn in signal_turns[:3]
        )
        if acoustic_available:
            change = float(audio_observations.get("loudness_change_db") or 0.0)
            assessment = str(audio_observations.get("assessment") or "")
            rapid = int(audio_observations.get("rapid_alternation_count") or 0)
            overlaps = int(audio_observations.get("overlap_count") or 0)
            if assessment == "rising":
                opening = (
                    f"Les mesures montrent une hausse globale d'intensité "
                    f"d'environ {change:.1f} dB entre le début et la fin."
                )
            elif assessment == "falling":
                opening = (
                    f"Les mesures ne montrent pas une montée globale : l'intensité "
                    f"baisse d'environ {abs(change):.1f} dB."
                )
            else:
                opening = (
                    f"Les mesures ne montrent pas de hausse globale nette de "
                    f"l'intensité ({change:+.1f} dB)."
                )
            answer = (
                f"{opening} La conversation compte {rapid} alternances rapides "
                f"et {overlaps} chevauchements détectés. "
                + (
                    "Certains passages contiennent aussi des formulations de "
                    "tension, mais l'ensemble ne suffit pas à qualifier "
                    "automatiquement l'échange de dégénérescence."
                    if signal_turns
                    else (
                        "La transcription ne contient pas non plus d'escalade "
                        "verbale explicite."
                    )
                )
            )
            points = tuple(_evidence_insight(citation) for citation in citations)
        elif citations:
            answer = (
                "La transcription contient quelques formulations de tension ou "
                "de désaccord, mais elle ne montre pas à elle seule que le ton "
                "vocal monte continuellement ou que la consultation dégénère."
            )
            points = tuple(_evidence_insight(citation) for citation in citations)
        else:
            answer = (
                "Je ne retrouve pas d'escalade verbale explicite dans la "
                "transcription. Je ne peux toutefois pas conclure sur le volume, "
                "l'intonation ou la colère sans analyse acoustique de l'audio."
            )
            points = ()
        return VerificationResult(
            question=question,
            claim=question,
            intent=CONSULTATION_QA,
            target_role=None,
            verdict=ANSWERED,
            confidence=0.58 if citations else 0.45,
            explanation=answer,
            citations=citations,
            roles=roles,
            backend="analyse-conversationnelle-de-secours",
            answer=answer,
            analysis_points=points,
            nuance=(
                (
                    str(audio_observations.get("limitations") or "")
                    if acoustic_available and isinstance(audio_observations, Mapping)
                    else (
                        "Cette lecture porte sur les mots et la dynamique des tours. "
                        "Une mesure du volume, de l'intonation et du débit est "
                        "nécessaire pour évaluer précisément le ton vocal."
                    )
                )
            ),
            warnings=(
                "Le modèle local polyvalent n'est pas installé ou configuré.",
            ),
        )

    if re.search(r"\bqui\b.{0,25}\bparle\b.{0,15}\bplus\b", folded):
        counts: dict[RoleName, int] = defaultdict(int)
        for turn in records:
            counts[roles.role_for(turn.speaker_id)] += len(turn.text.split())
        ranked = sorted(counts.items(), key=lambda item: -item[1])
        leading_role, leading_words = ranked[0]
        total_words = sum(counts.values())
        share = leading_words / max(1, total_words)
        answer = (
            f"Le {leading_role} parle le plus dans la transcription, avec environ "
            f"{share:.0%} des mots transcrits ({leading_words} sur {total_words})."
        )
        representative = next(
            (
                turn
                for turn in records
                if roles.role_for(turn.speaker_id) == leading_role
            ),
            records[0],
        )
        citation = Citation(
            id=f"turn-{representative.index:05d}",
            turn_index=representative.index,
            start=representative.start,
            end=representative.end,
            speaker_id=representative.speaker_id,
            speaker=representative.speaker,
            role=leading_role,
            text=representative.text,
            relevance=1.0,
            excerpt=representative.text,
        )
        return VerificationResult(
            question=question,
            claim=question,
            intent=CONSULTATION_QA,
            target_role=None,
            verdict=ANSWERED,
            confidence=0.9,
            explanation=answer,
            citations=(citation,),
            roles=roles,
            backend="statistiques-locales",
            answer=answer,
            analysis_points=(
                f"Client : {counts.get('Client', 0)} mots ; "
                f"Master : {counts.get('Master', 0)} mots.",
            ),
            nuance=(
                "Le calcul utilise les mots transcrits ; il peut différer "
                "légèrement du temps de parole acoustique réel."
            ),
        )

    claim = extract_claim(question)
    scored = sorted(
        (
            _score_turn(turn, claim, roles.role_for(turn.speaker_id))
            for turn in records
        ),
        key=_scored_sort_key,
    )
    useful = [item for item in scored[:3] if item.score >= 0.30]
    citations = tuple(_citation(item) for item in useful)
    if citations:
        answer = (
            "J'ai retrouvé des passages liés à la question, mais le moteur de "
            "secours ne permet pas d'en tirer une réponse suffisamment complète. "
            "Les extraits ci-dessous sont proposés pour relecture."
        )
        verdict = PARTIAL
        confidence = 0.45
    else:
        answer = (
            "Je ne peux pas répondre de manière fiable à cette question avec le "
            "moteur de secours. Installez l'Assistant Premium ou reformulez la "
            "demande autour d'un fait, d'un thème ou d'une synthèse."
        )
        verdict = NOT_FOUND
        confidence = 0.35
    return VerificationResult(
        question=question,
        claim=question,
        intent=CONSULTATION_QA,
        target_role=None,
        verdict=verdict,
        confidence=confidence,
        explanation=answer,
        citations=citations,
        roles=roles,
        backend="analyse-conversationnelle-de-secours",
        answer=answer,
        analysis_points=tuple(_evidence_insight(item) for item in citations),
        nuance=(
            "Aucun fait n'est inventé : la réponse reste limitée aux passages "
            "réellement retrouvés."
        ),
        warnings=(
            "Le modèle local polyvalent n'est pas installé ou configuré.",
        ),
    )


def verify_claim_result(
    turns: Sequence[object],
    question: str,
    *,
    roles: RoleProposal | None = None,
    role_overrides: Mapping[str, str] | None = None,
    reasoner: ClaimReasoner | None = None,
    top_k: int = 6,
    audio_observations: Mapping[str, object] | None = None,
) -> VerificationResult:
    """Vérifie une affirmation avec un repli déterministe toujours disponible.

    Cette fonction vérifie uniquement ce qui a été dit dans la consultation.
    Elle ne peut pas établir la vérité d'une prédiction dans le monde réel.
    """

    question = str(question or "").strip()
    intent = classify_query_intent(question)
    records = _normalise_turns(turns)
    role_proposal = roles or propose_speaker_roles(turns, overrides=role_overrides)
    model_warning: str | None = None
    if reasoner is not None:
        model_result, model_warning = _try_consultation_reasoner(
            reasoner,
            records=records,
            question=question,
            intent=intent,
            roles=role_proposal,
            audio_observations=audio_observations,
        )
        if model_result is not None:
            return model_result
    if intent == OPEN_SUMMARY:
        fallback = _summary_result(records, question, role_proposal)
        return (
            replace(fallback, warnings=(model_warning,))
            if model_warning
            else fallback
        )
    if intent == TOPIC_PRESENCE:
        fallback = _topic_presence_result(records, question, role_proposal)
        return (
            replace(fallback, warnings=(model_warning,))
            if model_warning
            else fallback
        )
    if intent == CONSULTATION_QA:
        fallback = _general_fallback_result(
            records,
            question,
            role_proposal,
            audio_observations=audio_observations,
        )
        return (
            replace(
                fallback,
                warnings=tuple(
                    dict.fromkeys((*fallback.warnings, model_warning))
                ),
            )
            if model_warning
            else fallback
        )

    claim = extract_claim(question)
    target_role = _target_role(question)
    scored = [
        _score_turn(turn, claim, role_proposal.role_for(turn.speaker_id))
        for turn in records
    ]
    if target_role and not role_proposal.needs_confirmation:
        role_scored = [item for item in scored if item.role == target_role]
        if role_scored:
            scored = role_scored
    clarification = _relationship_clarification(question, scored)
    scored.sort(key=_scored_sort_key)
    limited_scored = list(scored[: max(1, int(top_k))])
    if clarification is not None:
        existing_indexes = {item.turn.index for item in limited_scored}
        for evidence_item in clarification.evidence:
            if evidence_item.turn.index not in existing_indexes:
                limited_scored.append(evidence_item)
                existing_indexes.add(evidence_item.turn.index)
    scored = limited_scored
    candidates = tuple(_citation(item) for item in scored)
    candidate_by_id = {item.id: item for item in candidates}

    warnings: list[str] = []
    backend_name = "déterministe-local"
    decision: ReasonerDecision | None = None
    if clarification is not None:
        decision = ReasonerDecision(
            CLARIFY,
            0.5,
            tuple(f"turn-{item.turn.index:05d}" for item in clarification.evidence),
        )
    elif reasoner is not None:
        request = ReasonerRequest(
            question=question,
            claim=claim,
            target_role=target_role,
            candidates=candidates,
        )
        try:
            raw_decision = reasoner.decide(request)
            decision = _parse_reasoner_decision(
                raw_decision,
                valid_ids=set(candidate_by_id),
            )
            if decision is None:
                warnings.append(
                    "La réponse du modèle local n'était pas suffisamment ancrée ; "
                    "le vérificateur déterministe a pris le relais."
                )
            else:
                backend_name = str(
                    getattr(reasoner, "name", type(reasoner).__name__)
                )
        except Exception as exc:  # pragma: no cover - le type dépend du backend futur
            warnings.append(
                "Le modèle local est indisponible ; le vérificateur déterministe "
                f"a pris le relais ({type(exc).__name__})."
            )

    if decision is None:
        decision = _deterministic_decision(scored)

    citations = tuple(
        candidate_by_id[citation_id]
        for citation_id in decision.citation_ids
        if citation_id in candidate_by_id
    )
    # Double garde-fou : même après parsing, aucun verdict positif ne sort sans
    # passage réel. Ce cas ne devrait être atteignable qu'avec un backend fautif.
    if decision.verdict != NOT_FOUND and not citations:
        decision = _deterministic_decision(scored)
        citations = tuple(
            candidate_by_id[citation_id]
            for citation_id in decision.citation_ids
            if citation_id in candidate_by_id
        )
        backend_name = "déterministe-local"

    explanation = (
        clarification.answer
        if clarification is not None
        else _explanation(
            decision.verdict,
            target_role=target_role,
            roles=role_proposal,
        )
    )
    answer, analysis_points, nuance = build_answer_sections(
        decision.verdict,
        claim,
        target_role,
        citations,
        clarification_answer=(clarification.answer if clarification else ""),
        role_mapping_confirmed=not role_proposal.needs_confirmation,
    )
    return VerificationResult(
        question=question,
        claim=claim,
        intent=CLAIM_VERIFICATION,
        target_role=target_role,
        verdict=decision.verdict,
        confidence=decision.confidence,
        explanation=explanation,
        citations=citations,
        roles=role_proposal,
        backend=backend_name,
        answer=answer,
        analysis_points=analysis_points,
        nuance=nuance,
        clarification_options=(clarification.options if clarification else ()),
        warnings=tuple(warnings),
    )


def _role_overrides_from_payload(
    value: Mapping[str, object] | None,
) -> dict[str, str]:
    if not value:
        return {}
    raw_roles = value.get("roles")
    if isinstance(raw_roles, Mapping):
        return {str(key): str(role) for key, role in raw_roles.items()}
    raw_assignments = value.get("assignments")
    if isinstance(raw_assignments, Sequence) and not isinstance(
        raw_assignments, (str, bytes)
    ):
        result: dict[str, str] = {}
        for raw in raw_assignments:
            if not isinstance(raw, Mapping):
                continue
            speaker_id = str(raw.get("speaker_id") or "")
            role = str(raw.get("role") or "")
            if speaker_id and _normalise_role(role) in {"Client", "Master"}:
                result[speaker_id] = role
        if result:
            return result
    client_id = str(value.get("client_speaker_id") or "")
    master_id = str(value.get("master_speaker_id") or "")
    if client_id or master_id:
        result = {}
        if client_id:
            result[client_id] = "Client"
        if master_id:
            result[master_id] = "Master"
        return result
    return {
        str(key): str(role)
        for key, role in value.items()
        if _normalise_role(role) in {"Client", "Master"}
    }


def suggest_speaker_roles(
    turns: Sequence[object],
    existing_roles: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Façade JSON pour la fiche audio.

    ``existing_roles`` accepte aussi bien la sortie de cette fonction qu'une
    simple table ``speaker_id -> rôle``. Les choix déjà confirmés restent ainsi
    prioritaires lors d'un rechargement de la fiche.
    """

    proposal = propose_speaker_roles(
        turns,
        overrides=_role_overrides_from_payload(existing_roles),
    )
    payload = proposal.to_dict()
    payload["roles"] = {
        item.speaker_id: item.role for item in proposal.assignments
    }
    payload["client_speaker_id"] = proposal.client_speaker_id
    payload["master_speaker_id"] = proposal.master_speaker_id
    return payload


def verify_claim(
    question: str,
    turns: Sequence[object],
    speaker_roles: RoleProposal | Mapping[str, object] | None = None,
    *,
    reasoner: ClaimReasoner | None = None,
    top_k: int = 6,
    audio_observations: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Façade JSON stable destinée à l'interface de la fiche audio."""

    if isinstance(speaker_roles, RoleProposal):
        roles = speaker_roles
    else:
        roles = propose_speaker_roles(
            turns,
            overrides=_role_overrides_from_payload(speaker_roles),
        )
    result = verify_claim_result(
        turns,
        question,
        roles=roles,
        reasoner=reasoner,
        top_k=top_k,
        audio_observations=audio_observations,
    )
    payload = result.to_dict()
    records = _normalise_turns(turns)
    start_seconds = min(item.start for item in records)
    end_seconds = max(item.end for item in records)
    payload["coverage"] = {
        "scope": "transcription-complete",
        "turns_analyzed": len(records),
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "duration_seconds": max(0.0, end_seconds - start_seconds),
        "target_role": result.target_role,
        "role_mapping_confirmed": not roles.needs_confirmation,
    }
    return payload


__all__ = [
    "ANSWERED",
    "ASSISTANT_MODEL_REPOSITORY",
    "ASSISTANT_MODEL_REVISION",
    "ASSISTANT_MODEL_NAME",
    "ASSISTANT_REQUIRED_MODEL_FILES",
    "BUNDLED_ASSISTANT_MODEL_DIR",
    "CLAIM_VERIFICATION",
    "CLARIFY",
    "CONFIRMED",
    "CONSULTATION_QA",
    "CONTRADICTED",
    "ConsultationDecision",
    "ConsultationRequest",
    "ConsultationReasoner",
    "DEFAULT_ASSISTANT_MODEL_DIR",
    "NOT_FOUND",
    "OPEN_SUMMARY",
    "PARTIAL",
    "SUMMARY",
    "TOPIC_MENTIONED",
    "TOPIC_PRESENCE",
    "TOPIC_PRESENT",
    "TOPIC_PRIMARY",
    "Citation",
    "ClaimReasoner",
    "OpenVINOJsonReasoner",
    "ReasonerDecision",
    "ReasonerRequest",
    "RoleProposal",
    "SpeakerRole",
    "VerificationResult",
    "build_answer_sections",
    "build_consultation_prompt",
    "build_grounded_prompt",
    "classify_query_intent",
    "extract_claim",
    "get_default_reasoner",
    "resolve_openvino_assistant_model_path",
    "release_default_reasoners",
    "propose_speaker_roles",
    "retrieve_passages",
    "suggest_speaker_roles",
    "verify_claim",
    "verify_claim_result",
]
