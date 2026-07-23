"""Persistance locale de la bibliothèque audio et des fiches de travail."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .core import (
    FASTER_WHISPER_BACKEND,
    OPENVINO_ARC_BACKEND,
    ROOT_DIR,
    default_asr_backend,
    platform_compatible_asr_backend,
)


DATA_DIR = ROOT_DIR / "donnees"
AUDIO_LIBRARY_DIR = DATA_DIR / "audios"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
WORKSPACE_PATH = DATA_DIR / "espace-travail.json"
WORKSPACE_VERSION = 6

STANDARD_TRANSCRIPTION_ENGINE = FASTER_WHISPER_BACKEND
OPENVINO_TRANSCRIPTION_ENGINE = OPENVINO_ARC_BACKEND
DEFAULT_TRANSCRIPTION_ENGINE = default_asr_backend()
TRANSCRIPTION_ENGINES = {
    STANDARD_TRANSCRIPTION_ENGINE,
    OPENVINO_TRANSCRIPTION_ENGINE,
}

_WORKSPACE_LOCK = threading.RLock()


def blank_workspace() -> dict[str, Any]:
    default_engine = default_asr_backend()
    return {
        "version": WORKSPACE_VERSION,
        "order": [],
        "queue_order": [],
        "items": {},
        "batch_zip": None,
        "preferences": {
            "transcription_engine": default_engine,
        },
    }


def _safe_filename(name: str) -> str:
    source = Path(name)
    stem = re.sub(r"[^\w\-. ]+", "-", source.stem, flags=re.UNICODE)
    stem = re.sub(r"\s+", " ", stem).strip(" .-") or "audio"
    suffix = re.sub(r"[^A-Za-z0-9.]", "", source.suffix.lower())
    return f"{stem[:80]}{suffix}"


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_audio(source: str | os.PathLike[str]) -> tuple[str, Path]:
    """Copie un audio dans la bibliothèque durable et renvoie son identifiant."""

    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    AUDIO_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    digest = _file_digest(source_path)
    item_id = digest[:24]
    item_directory = (AUDIO_LIBRARY_DIR / item_id).resolve()
    if source_path.parent == item_directory:
        return item_id, source_path

    existing = sorted(item_directory.glob("*"), key=lambda path: path.name)
    if existing:
        return item_id, existing[0].resolve()

    item_directory.mkdir(parents=True, exist_ok=True)
    original_name = source_path.name
    while original_name.startswith(f"{item_id}-"):
        original_name = original_name[len(item_id) + 1 :]
    destination = item_directory / _safe_filename(original_name)
    if source_path != destination.resolve() and not destination.exists():
        try:
            source_path.relative_to(AUDIO_LIBRARY_DIR.resolve())
        except ValueError:
            shutil.copy2(source_path, destination)
        else:
            shutil.move(str(source_path), str(destination))
    return item_id, destination.resolve()


def rename_library_audio(
    item_id: str,
    source: str | os.PathLike[str],
    display_name: str,
) -> Path:
    """Renomme uniquement la copie gérée afin que l'uploader affiche le nom métier."""

    item_directory = (AUDIO_LIBRARY_DIR / item_id).resolve()
    source_path = Path(source).resolve()
    if source_path.parent != item_directory or not source_path.is_file():
        raise ValueError("Le fichier à renommer n'appartient pas à la bibliothèque locale.")

    destination = (item_directory / _safe_filename(display_name)).resolve()
    if destination.parent != item_directory:
        raise ValueError("Nom de fichier non valide.")
    if destination == source_path:
        return source_path
    if destination.exists():
        raise FileExistsError(destination)

    source_path.replace(destination)
    return destination


def delete_workspace_item(
    state: dict[str, Any],
    item_id: str,
) -> tuple[dict[str, Any], str]:
    """Supprime une fiche et uniquement ses fichiers gérés par l'application."""

    raw_items = state.get("items") if isinstance(state, dict) else None
    raw_item = raw_items.get(item_id) if isinstance(raw_items, dict) else None
    if isinstance(raw_item, dict) and str(raw_item.get("status") or "") == "En cours":
        raise ValueError("Une transcription en cours ne peut pas être supprimée.")

    normalised = normalise_workspace(state)
    items = normalised["items"]
    item = items.get(item_id)
    if not isinstance(item, dict):
        raise KeyError(item_id)
    display_name = str(item.get("name") or "Audio")
    library_root = AUDIO_LIBRARY_DIR.resolve()
    item_directory = (library_root / item_id).resolve()
    if item_directory.parent != library_root:
        raise ValueError("Identifiant de fiche non valide.")
    if item_directory.is_dir():
        shutil.rmtree(item_directory)
    delete_checkpoint(item_id)

    results_root = (ROOT_DIR / "resultats").resolve()
    result_parents: set[Path] = set()
    for raw_path in item.get("files", []):
        try:
            result_path = Path(str(raw_path)).resolve()
            result_path.relative_to(results_root)
        except (OSError, ValueError):
            continue
        if result_path.is_file():
            result_path.unlink()
            result_parents.add(result_path.parent)

    batch_zip = normalised.get("batch_zip")
    if batch_zip:
        try:
            batch_path = Path(str(batch_zip)).resolve()
            batch_path.relative_to(results_root)
        except (OSError, ValueError):
            pass
        else:
            if batch_path.is_file():
                batch_path.unlink()
            result_parents.add(batch_path.parent)
    normalised["batch_zip"] = None

    for directory in sorted(result_parents, key=lambda path: len(path.parts), reverse=True):
        if directory != results_root:
            try:
                directory.rmdir()
            except OSError:
                pass

    items.pop(item_id, None)
    normalised["order"] = [
        current_id
        for current_id in normalised["order"]
        if current_id != item_id
    ]
    normalised["queue_order"] = [
        current_id
        for current_id in normalised.get("queue_order", [])
        if current_id != item_id
    ]
    return save_workspace(normalised), display_name


def _stable_discussion_id(
    prefix: str,
    item_id: str,
    index: int,
    raw: dict[str, Any],
) -> str:
    """Construit un identifiant reproductible pour les anciennes discussions."""

    seed = "\x1f".join(
        (
            item_id,
            str(index),
            str(raw.get("author") or ""),
            str(raw.get("passage") or ""),
            str(raw.get("body") or ""),
            str(raw.get("created_at") or ""),
        )
    )
    return f"{prefix}-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:24]}"


def _optional_seconds(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if not (seconds >= 0.0 and seconds < float("inf")):
        return None
    return seconds


def _optional_confidence(value: object) -> float | None:
    """Normalise un score de confiance sans laisser passer NaN ou l'infini."""

    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if not (confidence >= 0.0 and confidence < float("inf")):
        return None
    return min(1.0, confidence)


def _normalise_role_name(value: object) -> str:
    role = str(value or "").strip()
    aliases = {
        "client": "Client",
        "customer": "Client",
        "master": "Master",
        "voyant": "Master",
        "voyante": "Master",
        "unknown": "À confirmer",
        "unassigned": "À confirmer",
        "a confirmer": "À confirmer",
        "à confirmer": "À confirmer",
    }
    return aliases.get(role.casefold(), role or "À confirmer")


def normalise_speaker_roles(raw: object) -> dict[str, dict[str, Any]]:
    """Retourne le mapping durable des voix vers les rôles Client/Master.

    Le format accepte également les valeurs texte historiques, par exemple
    ``{"SPEAKER_00": "Client"}``. Les champs additionnels sont conservés afin
    qu'une version plus récente de l'application puisse enrichir le schéma.
    """

    if not isinstance(raw, dict):
        return {}

    roles: dict[str, dict[str, Any]] = {}
    for raw_speaker_id, raw_role in raw.items():
        speaker_id = str(raw_speaker_id).strip()
        if not speaker_id:
            continue
        if isinstance(raw_role, dict):
            role = dict(raw_role)
        elif isinstance(raw_role, str):
            role = {"role": raw_role}
        else:
            continue

        confidence = _optional_confidence(role.get("confidence"))
        confirmed = role.get("confirmed")
        if not isinstance(confirmed, bool):
            confirmed = False
        role.update(
            {
                "speaker_id": speaker_id,
                "role": _normalise_role_name(role.get("role")),
                "confidence": confidence,
                "source": str(role.get("source") or "").strip(),
                "confirmed": confirmed,
                "updated_at": str(role.get("updated_at") or ""),
            }
        )
        roles[speaker_id] = role
    return roles


def _stable_assistant_id(
    prefix: str,
    item_id: str,
    index: int,
    raw: dict[str, Any],
) -> str:
    seed = "\x1f".join(
        (
            item_id,
            str(index),
            str(raw.get("question") or raw.get("quote") or ""),
            str(raw.get("created_at") or ""),
        )
    )
    return f"{prefix}-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:24]}"


def _normalise_assistant_citation(
    item_id: str,
    query_id: str,
    index: int,
    raw: object,
) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    citation = dict(raw)
    citation_id = str(raw.get("id") or "").strip()
    if not citation_id:
        citation_id = _stable_assistant_id(
            "citation",
            f"{item_id}:{query_id}",
            index,
            raw,
        )
    start_seconds = _optional_seconds(raw.get("start_seconds"))
    end_seconds = _optional_seconds(raw.get("end_seconds"))
    if start_seconds is None:
        end_seconds = None
    elif end_seconds is not None:
        end_seconds = max(start_seconds, end_seconds)
    citation.update(
        {
            "id": citation_id,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "speaker_id": str(raw.get("speaker_id") or "").strip(),
            "role": _normalise_role_name(raw.get("role")),
            "quote": str(raw.get("quote") or "").strip(),
        }
    )
    return citation


def _normalise_assistant_result(
    item_id: str,
    query_id: str,
    raw: object,
) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    result = dict(raw)
    raw_citations = raw.get("citations")
    citations: list[dict[str, Any]] = []
    if isinstance(raw_citations, list):
        for citation_index, raw_citation in enumerate(raw_citations):
            citation = _normalise_assistant_citation(
                item_id,
                query_id,
                citation_index,
                raw_citation,
            )
            if citation is not None:
                citations.append(citation)
    result.update(
        {
            "verdict": str(raw.get("verdict") or "").strip(),
            "confidence": _optional_confidence(raw.get("confidence")),
            "answer": str(raw.get("answer") or "").strip(),
            "citations": citations,
        }
    )
    return result


def normalise_assistant_queries(
    item_id: str,
    raw: object,
    *,
    fallback_created_at: str = "",
) -> list[dict[str, Any]]:
    """Normalise l'historique IA d'une fiche sans supprimer ses extensions."""

    if not isinstance(raw, list):
        return []

    queries: list[dict[str, Any]] = []
    for query_index, raw_query in enumerate(raw):
        if not isinstance(raw_query, dict):
            continue
        query = dict(raw_query)
        query_id = str(raw_query.get("id") or "").strip()
        if not query_id:
            query_id = _stable_assistant_id(
                "query",
                item_id,
                query_index,
                raw_query,
            )
        created_at = str(raw_query.get("created_at") or fallback_created_at)
        result = _normalise_assistant_result(
            item_id,
            query_id,
            raw_query.get("result"),
        )
        query.update(
            {
                "id": query_id,
                "question": str(raw_query.get("question") or "").strip(),
                "status": str(
                    raw_query.get("status")
                    or ("completed" if result is not None else "pending")
                ).strip(),
                "result": result,
                "created_at": created_at,
                "updated_at": str(raw_query.get("updated_at") or created_at),
            }
        )
        queries.append(query)
    return queries


def _json_export_safe(value: object) -> Any:
    """Copie récursivement une valeur vers des types sérialisables en JSON."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {
            str(key): _json_export_safe(child)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_export_safe(child) for child in value]
    if isinstance(value, (Path, datetime)):
        return str(value)
    return str(value)


def assistant_data_for_export(item: object) -> dict[str, Any]:
    """Produit une copie JSON sûre des rôles et analyses d'une fiche audio."""

    if not isinstance(item, dict):
        return {"speaker_roles": {}, "assistant_queries": []}
    item_id = str(item.get("id") or "")
    payload = {
        "speaker_roles": normalise_speaker_roles(item.get("speaker_roles")),
        "assistant_queries": normalise_assistant_queries(
            item_id,
            item.get("assistant_queries"),
            fallback_created_at=str(item.get("created_at") or ""),
        ),
    }
    return _json_export_safe(payload)


def _normalise_reply(
    item_id: str,
    comment_id: str,
    index: int,
    raw: object,
    *,
    fallback_created_at: str,
) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    # La copie conserve les futurs champs que cette version ne connaît pas.
    reply = dict(raw)
    reply_id = str(raw.get("id") or "").strip()
    if not reply_id:
        reply_id = _stable_discussion_id(
            "reply",
            f"{item_id}:{comment_id}",
            index,
            raw,
        )
    created_at = str(raw.get("created_at") or fallback_created_at)
    reply.update(
        {
            "id": reply_id,
            "author": str(raw.get("author") or "Moi").strip() or "Moi",
            "body": str(raw.get("body") or "").strip(),
            "created_at": created_at,
            "updated_at": str(raw.get("updated_at") or created_at),
        }
    )
    return reply


def _normalise_comment(
    item_id: str,
    index: int,
    raw: object,
    *,
    fallback_created_at: str,
) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    # Schéma additif : passage/body/status restent présents pour que les
    # anciennes interfaces puissent encore lire un workspace version 4.
    comment = dict(raw)
    comment_id = str(raw.get("id") or "").strip()
    if not comment_id:
        comment_id = _stable_discussion_id("comment", item_id, index, raw)

    raw_anchor = raw.get("anchor")
    anchor = dict(raw_anchor) if isinstance(raw_anchor, dict) else {}
    passage = str(raw.get("passage") or anchor.get("quote") or "").strip()
    quote = str(anchor.get("quote") or passage).strip()
    start_seconds = _optional_seconds(anchor.get("start_seconds"))
    end_seconds = _optional_seconds(anchor.get("end_seconds"))
    if start_seconds is None:
        end_seconds = None
    elif end_seconds is not None:
        end_seconds = max(start_seconds, end_seconds)
    anchor.update(
        {
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "quote": quote,
            "source": str(
                anchor.get("source")
                or ("transcript" if start_seconds is not None else "legacy-text")
            ),
        }
    )

    raw_status = str(raw.get("status") or "À discuter").strip()
    status = (
        "Résolu"
        if raw_status.casefold() in {"résolu", "resolu", "resolved", "closed"}
        else "À discuter"
    )
    created_at = str(raw.get("created_at") or fallback_created_at)
    updated_at = str(raw.get("updated_at") or created_at)
    raw_replies = raw.get("replies")
    replies: list[dict[str, Any]] = []
    if isinstance(raw_replies, list):
        for reply_index, raw_reply in enumerate(raw_replies):
            reply = _normalise_reply(
                item_id,
                comment_id,
                reply_index,
                raw_reply,
                fallback_created_at=created_at,
            )
            if reply is not None:
                replies.append(reply)

    resolved_at = raw.get("resolved_at")
    if status == "Résolu" and not resolved_at:
        resolved_at = updated_at or created_at or None
    elif status != "Résolu":
        resolved_at = None

    comment.update(
        {
            "id": comment_id,
            "author": str(raw.get("author") or "Moi").strip() or "Moi",
            "passage": passage,
            "body": str(raw.get("body") or "").strip(),
            "status": status,
            "anchor": anchor,
            "replies": replies,
            "created_at": created_at,
            "updated_at": updated_at,
            "resolved_at": resolved_at,
        }
    )
    return comment


def _normalise_item(item_id: str, raw: object) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    path = str(raw.get("path") or "")
    name = str(raw.get("name") or Path(path).name or "Audio")
    status = str(raw.get("status") or "En attente")
    comments = raw.get("comments")
    if not isinstance(comments, list):
        comments = []
    result = raw.get("result") if isinstance(raw.get("result"), dict) else None
    files = raw.get("files") if isinstance(raw.get("files"), list) else []
    created_at = str(raw.get("created_at") or "")
    if not created_at:
        try:
            created_at = datetime.fromtimestamp(Path(path).stat().st_mtime).isoformat(
                timespec="seconds"
            )
        except OSError:
            created_at = datetime.now().isoformat(timespec="seconds")
    updated_at = str(raw.get("updated_at") or created_at)
    normalised_comments: list[dict[str, Any]] = []
    for comment_index, raw_comment in enumerate(comments):
        comment = _normalise_comment(
            item_id,
            comment_index,
            raw_comment,
            fallback_created_at=created_at,
        )
        if comment is not None:
            normalised_comments.append(comment)
    # La normalisation est additive : elle ne retire pas les champs introduits
    # par de futures versions ou par un module optionnel.
    item = dict(raw)
    item.update({
        "id": item_id,
        "path": path,
        "source_path": str(raw.get("source_path") or path),
        "original_name": str(raw.get("original_name") or Path(path).name or name),
        "name": name,
        "status": status,
        "progress": float(raw.get("progress") or (1.0 if status == "Terminé" else 0.0)),
        "processing_time": str(raw.get("processing_time") or "—"),
        "processing_seconds": float(raw.get("processing_seconds") or 0.0),
        "partial": str(raw.get("partial") or ""),
        "stage": str(raw.get("stage") or ""),
        "settings": dict(raw.get("settings")) if isinstance(raw.get("settings"), dict) else {},
        "run_id": str(raw.get("run_id") or ""),
        "checkpoint_position": float(raw.get("checkpoint_position") or 0.0),
        "resume_count": int(raw.get("resume_count") or 0),
        "heartbeat": str(raw.get("heartbeat") or ""),
        "result": result,
        "files": [str(value) for value in files],
        "comments": normalised_comments,
        "speaker_roles": normalise_speaker_roles(raw.get("speaker_roles")),
        "assistant_queries": normalise_assistant_queries(
            item_id,
            raw.get("assistant_queries"),
            fallback_created_at=created_at,
        ),
        "error": str(raw.get("error") or ""),
        "created_at": created_at,
        "updated_at": updated_at,
    })
    return item


def normalise_workspace(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return blank_workspace()

    # Conserver les clés inconnues permet une lecture/écriture sans perte avec
    # les extensions et les versions futures du workspace.
    state = dict(raw)
    state["version"] = WORKSPACE_VERSION
    state["order"] = []
    state["queue_order"] = []
    state["items"] = {}
    state["batch_zip"] = None
    default_engine = default_asr_backend()
    state["preferences"] = {"transcription_engine": default_engine}

    raw_items = raw.get("items")
    if isinstance(raw_items, dict):
        for raw_id, raw_item in raw_items.items():
            item_id = str(raw_id)
            item = _normalise_item(item_id, raw_item)
            if item is not None:
                state["items"][item_id] = item

    raw_order = raw.get("order")
    if isinstance(raw_order, list):
        state["order"] = [
            str(item_id)
            for item_id in raw_order
            if str(item_id) in state["items"]
        ]
    for item_id in state["items"]:
        if item_id not in state["order"]:
            state["order"].append(item_id)

    raw_queue_order = raw.get("queue_order")
    if isinstance(raw_queue_order, list):
        queue_candidates = [str(item_id) for item_id in raw_queue_order]
    else:
        # Migration des workspaces antérieurs : tout élément non terminé était
        # implicitement considéré comme appartenant à la file de traitement.
        queue_candidates = list(state["order"])
    for item_id in queue_candidates:
        item = state["items"].get(item_id)
        if not isinstance(item, dict) or item_id in state["queue_order"]:
            continue
        completed = (
            str(item.get("status") or "") == "Terminé"
            and isinstance(item.get("result"), dict)
        )
        if not completed:
            state["queue_order"].append(item_id)

    # Un traitement interrompu doit rester récupérable même si une ancienne
    # écriture n'avait pas encore persisté son appartenance à la file.
    for item_id in state["order"]:
        item = state["items"].get(item_id)
        if (
            isinstance(item, dict)
            and str(item.get("status") or "") in {"En cours", "À reprendre"}
            and item_id not in state["queue_order"]
        ):
            state["queue_order"].append(item_id)

    batch_zip = raw.get("batch_zip")
    state["batch_zip"] = str(batch_zip) if batch_zip else None

    raw_preferences = raw.get("preferences")
    if isinstance(raw_preferences, dict):
        state["preferences"] = dict(raw_preferences)
        engine = str(
            raw_preferences.get("transcription_engine")
            or default_engine
        )
        state["preferences"]["transcription_engine"] = (
            platform_compatible_asr_backend(engine)
            if engine in TRANSCRIPTION_ENGINES
            else default_engine
        )
    return state


def load_workspace() -> dict[str, Any]:
    with _WORKSPACE_LOCK:
        if not WORKSPACE_PATH.is_file():
            return blank_workspace()
        try:
            with WORKSPACE_PATH.open("r", encoding="utf-8") as handle:
                return normalise_workspace(json.load(handle))
        except (OSError, ValueError, TypeError):
            return blank_workspace()


def save_workspace(state: dict[str, Any]) -> dict[str, Any]:
    with _WORKSPACE_LOCK:
        normalised = normalise_workspace(state)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        temporary = WORKSPACE_PATH.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(normalised, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, WORKSPACE_PATH)
        return normalised


def mutate_workspace(mutator: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Applique une modification lecture-écriture sans perdre une mise à jour concurrente."""

    with _WORKSPACE_LOCK:
        state = load_workspace()
        mutator(state)
        return save_workspace(state)


def update_workspace_item(item_id: str, **updates: object) -> dict[str, Any]:
    """Fusionne atomiquement des champs dans une fiche existante."""

    def apply(state: dict[str, Any]) -> None:
        item = state.get("items", {}).get(item_id)
        if not isinstance(item, dict):
            raise KeyError(item_id)
        item.update(updates)
        item["updated_at"] = datetime.now().isoformat(timespec="seconds")
        queue_order = state.setdefault("queue_order", [])
        if not isinstance(queue_order, list):
            queue_order = []
            state["queue_order"] = queue_order
        status = str(item.get("status") or "")
        completed = status == "Terminé" and isinstance(item.get("result"), dict)
        if completed:
            state["queue_order"] = [
                current_id for current_id in queue_order if current_id != item_id
            ]
        elif status in {"En cours", "À reprendre", "Échec"} and item_id not in queue_order:
            queue_order.append(item_id)

    return mutate_workspace(apply)


def checkpoint_path(item_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "", str(item_id))
    if not safe_id or safe_id != str(item_id):
        raise ValueError("Identifiant de checkpoint non valide.")
    return CHECKPOINT_DIR / f"{safe_id}.json"


def load_checkpoint(item_id: str) -> dict[str, Any] | None:
    path = checkpoint_path(item_id)
    with _WORKSPACE_LOCK:
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, ValueError, TypeError):
            return None
    return value if isinstance(value, dict) else None


def save_checkpoint(item_id: str, checkpoint: dict[str, Any]) -> Path:
    """Écrit un checkpoint séparé pour garder le workspace léger et robuste."""

    path = checkpoint_path(item_id)
    with _WORKSPACE_LOCK:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(checkpoint, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    return path


def delete_checkpoint(item_id: str) -> None:
    path = checkpoint_path(item_id)
    with _WORKSPACE_LOCK:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
