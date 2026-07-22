"""Vocabulaire métier pour les consultations de voyance en français.

La couche reste volontairement légère : elle guide le décodeur avec un
contexte et des mots attendus, puis ne corrige dans le texte final que des
graphies certaines (accents/casse d'un terme explicitement fourni). Elle ne
tente jamais de deviner un prénom, un nombre ou une date de naissance.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable


MAX_USER_TERMS = 32
MAX_USER_TERM_LENGTH = 64
MAX_USER_PROMPT_LENGTH = 640
MAX_PRIORITY_TERMS_LENGTH = 160
MAX_PROMPT_PRIORITY_LENGTH = 120
MAX_DOMAIN_PROMPT_LENGTH = 240
MAX_DOMAIN_HOTWORDS_LENGTH = 640


ASTROLOGY_TERMS = (
    "astrologie",
    "thème astral",
    "carte du ciel",
    "horoscope",
    "signe astrologique",
    "ascendant",
    "maisons astrologiques",
    "transit",
    "rétrograde",
    "conjonction",
    "opposition",
    "carré",
    "trigone",
    "sextile",
    "Bélier",
    "Taureau",
    "Gémeaux",
    "Cancer",
    "Lion",
    "Vierge",
    "Balance",
    "Scorpion",
    "Sagittaire",
    "Capricorne",
    "Verseau",
    "Poissons",
    "Soleil",
    "Lune",
    "Mercure",
    "Vénus",
    "Mars",
    "Jupiter",
    "Saturne",
    "Uranus",
    "Neptune",
    "Pluton",
)

DIVINATION_TERMS = (
    "voyance",
    "Master",
    "consultation",
    "tirage",
    "tarot de Marseille",
    "oracle de Belline",
    "arcane majeur",
    "arcane mineur",
    "lame",
    "coupe",
    "denier",
    "bâton",
    "épée",
    "pendule",
    "cartomancie",
    "tarologie",
    "médiumnité",
    "clairvoyance",
    "guidance",
    "intuition",
    "numérologie",
    "chemin de vie",
)

BIRTH_DATE_TERMS = (
    "date de naissance",
    "né le",
    "née le",
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
)

# Les hotwords ne sont pas un dictionnaire exhaustif. Une longue liste de mots
# courants (notamment les douze mois) augmente les répétitions de Whisper sur
# les passages sans rapport. On ne garde donc ici que les expressions métier
# les plus distinctives ; le modèle général connaît déjà les mois, planètes et
# signes courants.
DOMAIN_HOTWORD_TERMS = (
    "date de naissance",
    "heure de naissance",
    "lieu de naissance",
    "thème astral",
    "carte du ciel",
    "signe astrologique",
    "ascendant",
    "maisons astrologiques",
    "transit astrologique",
    "rétrograde",
    "Bélier",
    "Gémeaux",
    "Sagittaire",
    "Capricorne",
    "Verseau",
    "tarot de Marseille",
    "oracle de Belline",
    "arcane majeur",
    "arcane mineur",
    "pendule divinatoire",
    "cartomancie",
    "tarologie",
    "médiumnité",
    "clairvoyance",
    "numérologie",
    "chemin de vie",
)

# Ces remplacements changent uniquement la casse ou les accents d'une graphie
# déjà reconnue. Les homophones (par exemple « Marcelle » / « Marseille ») et
# les nombres restent intacts, car les corriger sans l'audio serait risqué.
SAFE_CANONICAL_FORMS = (
    "thème astral",
    "rétrograde",
    "tarot de Marseille",
    "oracle de Belline",
    "bâton",
    "épée",
    "médiumnité",
    "numérologie",
)


def _compact(value: object, *, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit].rstrip()


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def extract_priority_terms(value: object) -> tuple[str, ...]:
    """Extrait les noms et termes courts saisis dans le champ facultatif.

    Les entrées trop longues ressemblent davantage à des consignes qu'à des
    mots métier ; elles restent dans le prompt brut, mais ne deviennent pas des
    hotwords ni des règles de normalisation.
    """

    raw = str(value or "")[:MAX_USER_PROMPT_LENGTH]
    candidates = re.split(r"[,;\n|]+", raw)
    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        term = re.sub(r"^[\s\-–—•·]+|[\s.]+$", "", candidate)
        term = re.sub(r"\s+", " ", term).strip(" \"'“”«»")
        if not term or len(term) > MAX_USER_TERM_LENGTH:
            continue
        if len(term.split()) > 6 or not any(char.isalpha() for char in term):
            continue
        key = _fold(term)
        if key in seen:
            continue
        seen.add(key)
        result.append(term)
        if len(result) >= MAX_USER_TERMS:
            break
    return tuple(result)


def _is_french(language: str | None) -> bool:
    if language is None:
        return True
    normalized = str(language).strip().casefold().replace("_", "-")
    return normalized in {"", "fr", "fr-fr", "français", "francais", "french"}


def _terms_within_budget(terms: Iterable[str], budget: int) -> tuple[str, ...]:
    selected: list[str] = []
    length = 0
    for term in terms:
        extra = len(term) + (2 if selected else 0)
        if length + extra > budget:
            break
        selected.append(term)
        length += extra
    return tuple(selected)


def build_domain_prompt(user_vocabulary: object = "", language: str | None = "fr") -> str:
    """Construit le contexte de décodage, avec les termes utilisateur en tête.

    Mettre les prénoms et noms fournis en premier leur évite d'être évincés si
    le moteur tronque un prompt long. Pour une langue explicitement non
    française, le lexique français n'est pas injecté.
    """

    raw_user_prompt = _compact(user_vocabulary, limit=MAX_USER_PROMPT_LENGTH)
    if not _is_french(language):
        return raw_user_prompt

    priority_terms = _terms_within_budget(
        extract_priority_terms(raw_user_prompt), MAX_PROMPT_PRIORITY_LENGTH
    )
    sections: list[str] = []
    if priority_terms:
        sections.append("Noms et termes : " + ", ".join(priority_terms) + ".")
    # Cette phrase courte a un double intérêt : elle ancre le secteur sans
    # remplir le contexte du décodeur, et distingue explicitement « date de
    # naissance » de la graphie acoustiquement proche « table de naissance ».
    # Le reste du domaine est transmis par hotwords.
    sections.append(
        "Consultation de voyance entre un Client et un Master. "
        "Quelle est votre date de naissance, s'il vous plaît ?"
    )
    prompt = " ".join(sections)
    # Garde-fou déterministe pour les versions OpenVINO dont la validation du
    # prompt est particulièrement stricte.
    if len(prompt) > MAX_DOMAIN_PROMPT_LENGTH:
        context = (
            "Consultation de voyance. "
            "Quelle est votre date de naissance, s'il vous plaît ?"
        )
        available = MAX_DOMAIN_PROMPT_LENGTH - len(context) - 1
        prefix = sections[0][:available].rsplit(",", 1)[0].rstrip(" ,") + "."
        prompt = prefix + " " + context
    return prompt


def build_domain_hotwords(
    user_vocabulary: object = "", language: str | None = "fr"
) -> str:
    """Renvoie les hotwords Faster-Whisper, sans doublons ni phrases libres."""

    if not _is_french(language):
        return ", ".join(extract_priority_terms(user_vocabulary))

    priority_terms = _terms_within_budget(
        extract_priority_terms(user_vocabulary), MAX_PRIORITY_TERMS_LENGTH
    )
    result: list[str] = []
    seen: set[str] = set()
    length = 0
    for term in (*priority_terms, *DOMAIN_HOTWORD_TERMS):
        key = _fold(term)
        if key in seen:
            continue
        extra = len(term) + (2 if result else 0)
        if length + extra > MAX_DOMAIN_HOTWORDS_LENGTH:
            break
        seen.add(key)
        result.append(term)
        length += extra
    return ", ".join(result)


_ACCENT_EQUIVALENTS: dict[str, str] = {
    "a": "aàâäáãå",
    "c": "cç",
    "e": "eéèêë",
    "i": "iîïíì",
    "o": "oôöóòõ",
    "u": "uùûüú",
    "y": "yÿý",
}


def _exact_term_pattern(term: str) -> re.Pattern[str]:
    pieces: list[str] = []
    for char in term:
        folded = _fold(char)
        if char.isspace():
            pieces.append(r"\s+")
        elif folded in _ACCENT_EQUIVALENTS:
            pieces.append("[" + _ACCENT_EQUIVALENTS[folded] + "]")
        else:
            pieces.append(re.escape(char))
    return re.compile(r"(?<!\w)" + "".join(pieces) + r"(?!\w)", re.IGNORECASE)


def _canonical_terms(user_vocabulary: object) -> Iterable[str]:
    yield from SAFE_CANONICAL_FORMS
    # Un terme fourni explicitement par l'utilisateur est une référence de
    # casse et d'accent, pas une invitation à une correction phonétique.
    yield from extract_priority_terms(user_vocabulary)


def normalize_domain_text(text: object, user_vocabulary: object = "") -> str:
    """Normalise seulement les graphies exactes et ne devine jamais le sens.

    Exemples autorisés : ``lyrax`` -> ``LyraX`` si l'utilisateur a fourni
    « LyraX », ou ``theme astral`` -> ``thème astral``. Une
    date fictive comme ``31 02 2099`` ou un identifiant phonétiquement voisin
    ne sont jamais réécrits.
    """

    normalized = str(text or "")
    seen: set[str] = set()
    for canonical in _canonical_terms(user_vocabulary):
        key = _fold(canonical)
        if not canonical or key in seen:
            continue
        seen.add(key)
        normalized = _exact_term_pattern(canonical).sub(canonical, normalized)
    return normalized


__all__ = [
    "ASTROLOGY_TERMS",
    "BIRTH_DATE_TERMS",
    "DIVINATION_TERMS",
    "DOMAIN_HOTWORD_TERMS",
    "build_domain_hotwords",
    "build_domain_prompt",
    "extract_priority_terms",
    "normalize_domain_text",
]
