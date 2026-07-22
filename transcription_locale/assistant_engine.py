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
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence, runtime_checkable


RoleName = Literal["Client", "Master", "Inconnu"]
VerdictName = Literal[
    "Confirmé",
    "Partiel",
    "Contredit",
    "Non retrouvé",
    "À clarifier",
]

CONFIRMED: VerdictName = "Confirmé"
PARTIAL: VerdictName = "Partiel"
CONTRADICTED: VerdictName = "Contredit"
NOT_FOUND: VerdictName = "Non retrouvé"
CLARIFY: VerdictName = "À clarifier"

ASSISTANT_MODEL_ENV = "TRANSCRIPTION_ASSISTANT_MODEL_DIR"
ASSISTANT_DEVICE_ENV = "TRANSCRIPTION_ASSISTANT_DEVICE"


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


@runtime_checkable
class ClaimReasoner(Protocol):
    """Contrat léger pour un backend local, sans dépendance imposée."""

    def decide(
        self, request: ReasonerRequest
    ) -> ReasonerDecision | Mapping[str, object] | str:
        """Retourne une décision structurée ou son JSON équivalent."""


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
        max_new_tokens: int = 256,
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

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def _load_pipeline(self) -> object:
        if self._pipeline is not None:
            return self._pipeline
        if self._pipeline_factory is not None:
            factory = self._pipeline_factory
        else:
            try:
                import openvino_genai
            except ImportError as exc:  # pragma: no cover - installation optionnelle
                raise RuntimeError(
                    "Le runtime OpenVINO GenAI n'est pas installé."
                ) from exc
            factory = openvino_genai.LLMPipeline
        self._pipeline = factory(str(self.model_path), self.device)
        return self._pipeline

    def decide(self, request: ReasonerRequest) -> str:
        pipeline = self._load_pipeline()
        prompt = build_grounded_prompt(request)
        try:
            raw = pipeline.generate(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        except TypeError:  # Compatibilité avec les runtimes à config positionnelle.
            try:
                import openvino_genai

                config = openvino_genai.GenerationConfig()
                config.max_new_tokens = self.max_new_tokens
                config.do_sample = False
                raw = pipeline.generate(prompt, config)
            except (ImportError, AttributeError) as exc:
                raise RuntimeError(
                    "Version OpenVINO GenAI incompatible avec l'assistant."
                ) from exc
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


def get_default_reasoner(
    model_path: str | Path | None = None,
    *,
    device: str | None = None,
) -> OpenVINOJsonReasoner | None:
    """Retourne le backend local configuré, sinon ``None`` immédiatement.

    Le chemin peut être fourni explicitement ou via
    ``TRANSCRIPTION_ASSISTANT_MODEL_DIR``. Un dossier absent ou un runtime non
    installé ne bloque jamais l'application : l'UI peut simplement conserver
    le moteur déterministe probant.
    """

    configured = str(model_path or os.environ.get(ASSISTANT_MODEL_ENV, "")).strip()
    if not configured:
        return None
    path = Path(configured).expanduser()
    if not path.is_dir() or importlib.util.find_spec("openvino_genai") is None:
        return None
    return OpenVINOJsonReasoner(
        path,
        device=device or os.environ.get(ASSISTANT_DEVICE_ENV, "GPU"),
    )


@dataclass(slots=True, frozen=True)
class VerificationResult:
    """Réponse prête à être persistée ou rendue dans la fiche audio."""

    question: str
    claim: str
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
    safe_ids = tuple(dict.fromkeys(item for item in ids if item in valid_ids))
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
    prefix = f"À {_answer_timecode(start)}, le {role}"

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


def verify_claim_result(
    turns: Sequence[object],
    question: str,
    *,
    roles: RoleProposal | None = None,
    role_overrides: Mapping[str, str] | None = None,
    reasoner: ClaimReasoner | None = None,
    top_k: int = 6,
) -> VerificationResult:
    """Vérifie une affirmation avec un repli déterministe toujours disponible.

    Cette fonction vérifie uniquement ce qui a été dit dans la consultation.
    Elle ne peut pas établir la vérité d'une prédiction dans le monde réel.
    """

    question = str(question or "").strip()
    claim = extract_claim(question)
    records = _normalise_turns(turns)
    role_proposal = roles or propose_speaker_roles(turns, overrides=role_overrides)
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
    "CLARIFY",
    "CONFIRMED",
    "CONTRADICTED",
    "NOT_FOUND",
    "PARTIAL",
    "Citation",
    "ClaimReasoner",
    "OpenVINOJsonReasoner",
    "ReasonerDecision",
    "ReasonerRequest",
    "RoleProposal",
    "SpeakerRole",
    "VerificationResult",
    "build_answer_sections",
    "build_grounded_prompt",
    "extract_claim",
    "get_default_reasoner",
    "propose_speaker_roles",
    "retrieve_passages",
    "suggest_speaker_roles",
    "verify_claim",
    "verify_claim_result",
]
