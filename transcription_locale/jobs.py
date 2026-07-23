"""File de transcription durable, indépendante de la connexion du navigateur."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable

from .core import (
    FASTER_WHISPER_BACKEND,
    PROFILE_CONFIG,
    platform_compatible_asr_backend,
    run_pipeline,
)
from .exports import format_duration, plain_transcript, write_batch_archive, write_exports
from .workspace import (
    delete_checkpoint,
    load_checkpoint,
    load_workspace,
    mutate_workspace,
    save_checkpoint,
    update_workspace_item,
)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _engine_version(engine: str = "faster-whisper") -> str:
    engine = platform_compatible_asr_backend(engine)
    package = (
        "openvino-genai"
        if engine == "openvino-arc-pilot"
        else "faster-whisper"
    )
    try:
        return version(package)
    except PackageNotFoundError:
        return "indisponible" if package == "openvino-genai" else "inconnue"


@dataclass(slots=True, frozen=True)
class JobSettings:
    profile: str
    language: str | None
    requested_speakers: int
    speaker_names: tuple[str, ...]
    initial_prompt: str = ""
    engine: str = "faster-whisper"
    engine_version: str = ""

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        normalised_engine = platform_compatible_asr_backend(self.engine)
        data["engine"] = normalised_engine
        data["speaker_names"] = list(self.speaker_names)
        profile_config = PROFILE_CONFIG[self.profile]
        data["model"] = str(profile_config["model"])
        data["beam_size"] = int(profile_config["beam_size"])
        data["batch_size"] = int(profile_config["batch_size"])
        data["engine_version"] = (
            self.engine_version
            if self.engine == normalised_engine and self.engine_version
            else _engine_version(normalised_engine)
        )
        return data

    @classmethod
    def from_dict(cls, raw: object) -> JobSettings | None:
        if not isinstance(raw, dict):
            return None
        profile = str(raw.get("profile") or "")
        if profile not in PROFILE_CONFIG:
            return None
        raw_names = raw.get("speaker_names")
        names = (
            tuple(str(value) for value in raw_names)
            if isinstance(raw_names, list)
            else ("Interlocuteur 1", "Interlocuteur 2")
        )
        try:
            speakers = max(1, min(8, int(raw.get("requested_speakers") or 2)))
        except (TypeError, ValueError):
            speakers = 2
        language = raw.get("language")
        raw_engine = str(raw.get("engine") or FASTER_WHISPER_BACKEND)
        engine = platform_compatible_asr_backend(raw_engine)
        raw_engine_version = str(raw.get("engine_version") or "")
        return cls(
            profile=profile,
            language=str(language) if language else None,
            requested_speakers=speakers,
            speaker_names=names,
            initial_prompt=str(raw.get("initial_prompt") or ""),
            engine=engine,
            engine_version=str(
                raw_engine_version
                if raw_engine == engine and raw_engine_version
                else _engine_version(engine)
            ),
        )

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class TranscriptionJobManager:
    """Exécute une seule file locale et conserve chaque étape sur disque."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pending: deque[str] = deque()
        self._settings: dict[str, JobSettings] = {}
        self._active_item: str | None = None
        self._thread: threading.Thread | None = None

    def active_item_ids(self) -> set[str]:
        with self._lock:
            active = set(self._pending)
            if self._active_item:
                active.add(self._active_item)
            return active

    def is_active(self, item_id: str | None = None) -> bool:
        active = self.active_item_ids()
        return bool(active if item_id is None else item_id in active)

    def enqueue(self, item_ids: Iterable[str], settings: JobSettings) -> list[str]:
        state = load_workspace()
        items = state.get("items") if isinstance(state.get("items"), dict) else {}
        candidates: list[str] = []
        for item_id in item_ids:
            item = items.get(item_id) if isinstance(items, dict) else None
            if not isinstance(item, dict) or not Path(str(item.get("path") or "")).is_file():
                continue
            if item.get("status") == "Terminé" and isinstance(item.get("result"), dict):
                continue
            candidates.append(item_id)

        with self._lock:
            known = set(self._pending)
            if self._active_item:
                known.add(self._active_item)
            candidates = [item_id for item_id in candidates if item_id not in known]
            if not candidates:
                return []
            settings_data = settings.to_dict()

            def persist(state_to_update: dict[str, object]) -> None:
                current_items = state_to_update.get("items")
                if not isinstance(current_items, dict):
                    return
                queue_order = state_to_update.setdefault("queue_order", [])
                if not isinstance(queue_order, list):
                    queue_order = []
                    state_to_update["queue_order"] = queue_order
                for item_id in candidates:
                    item = current_items.get(item_id)
                    if not isinstance(item, dict):
                        continue
                    item["settings"] = settings_data
                    if item.get("status") not in {"En cours", "À reprendre"}:
                        item["status"] = "En attente"
                    item["error"] = ""
                    item["updated_at"] = _now()
                    if item_id not in queue_order:
                        queue_order.append(item_id)

            mutate_workspace(persist)
            added: list[str] = []
            for item_id in candidates:
                self._settings[item_id] = settings
                self._pending.append(item_id)
                added.append(item_id)
            self._ensure_worker_locked()
        return added

    def recover_orphaned(self) -> list[str]:
        """Récupère les jobs laissés `En cours` par un arrêt du processus Python."""

        active = self.active_item_ids()
        state = load_workspace()
        order = state.get("order") if isinstance(state.get("order"), list) else []
        items = state.get("items") if isinstance(state.get("items"), dict) else {}
        recovered: list[tuple[str, JobSettings]] = []
        waiting_for_settings: list[str] = []
        for item_id in order:
            item = items.get(item_id) if isinstance(items, dict) else None
            if (
                not isinstance(item, dict)
                or item.get("status") != "En cours"
                or item_id in active
            ):
                continue
            settings = JobSettings.from_dict(item.get("settings"))
            if settings is None:
                waiting_for_settings.append(item_id)
            else:
                recovered.append((item_id, settings))

        orphaned_ids = waiting_for_settings + [item_id for item_id, _ in recovered]
        if orphaned_ids:
            def mark(state_to_update: dict[str, object]) -> None:
                current_items = state_to_update.get("items")
                if not isinstance(current_items, dict):
                    return
                for item_id in orphaned_ids:
                    item = current_items.get(item_id)
                    if not isinstance(item, dict):
                        continue
                    item["status"] = "À reprendre"
                    item["stage"] = "Traitement interrompu — reprise disponible"
                    item["updated_at"] = _now()

            mutate_workspace(mark)

        with self._lock:
            known = set(self._pending)
            if self._active_item:
                known.add(self._active_item)
            for item_id, settings in recovered:
                self._settings[item_id] = settings
                if item_id not in known:
                    self._pending.append(item_id)
                    known.add(item_id)
            if recovered:
                self._ensure_worker_locked()
        return [item_id for item_id, _ in recovered]

    def wait_until_idle(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_active():
                return True
            time.sleep(0.02)
        return not self.is_active()

    def _ensure_worker_locked(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="transcription-job-manager",
        )
        self._thread.start()

    def _worker_loop(self) -> None:
        while True:
            with self._lock:
                if not self._pending:
                    self._active_item = None
                    self._thread = None
                    break
                item_id = self._pending.popleft()
                settings = self._settings.pop(item_id)
                self._active_item = item_id
            try:
                self._process_item(item_id, settings)
            except Exception as exc:
                try:
                    update_workspace_item(
                        item_id,
                        status="Échec",
                        stage="Échec interne du gestionnaire de tâches",
                        error=f"{type(exc).__name__}: {exc}",
                        heartbeat=_now(),
                    )
                except Exception:
                    pass
            finally:
                with self._lock:
                    if self._active_item == item_id:
                        self._active_item = None
        self._refresh_batch_archive()

    def _process_item(self, item_id: str, settings: JobSettings) -> None:
        state = load_workspace()
        item = state.get("items", {}).get(item_id)
        if not isinstance(item, dict):
            return
        if item.get("status") == "Terminé" and isinstance(item.get("result"), dict):
            return
        source = Path(str(item.get("path") or ""))
        if not source.is_file():
            update_workspace_item(
                item_id,
                status="Échec",
                stage="Fichier audio introuvable",
                error=f"Fichier audio introuvable : {source}",
            )
            return

        display_name = str(item.get("name") or source.name)
        base_elapsed = max(0.0, float(item.get("processing_seconds") or 0.0))
        settings_hash = settings.fingerprint()
        saved_checkpoint = load_checkpoint(item_id)
        if saved_checkpoint and (
            saved_checkpoint.get("audio_id") != item_id
            or saved_checkpoint.get("settings_hash") != settings_hash
        ):
            delete_checkpoint(item_id)
            saved_checkpoint = None

        run_id = uuid.uuid4().hex
        resumed = bool(saved_checkpoint)
        resume_count = int(item.get("resume_count") or 0) + (1 if resumed else 0)
        started_at = time.monotonic()
        update_workspace_item(
            item_id,
            status="En cours",
            progress=(float(item.get("progress") or 0.0) if saved_checkpoint else 0.0),
            stage=(
                f"Reprise à {float(saved_checkpoint.get('committed_until_seconds') or 0.0):.1f} s"
                if saved_checkpoint
                else "Préparation de la transcription"
            ),
            run_id=run_id,
            checkpoint_position=(
                float(saved_checkpoint.get("committed_until_seconds") or 0.0)
                if saved_checkpoint
                else 0.0
            ),
            resume_count=resume_count,
            heartbeat=_now(),
            settings=settings.to_dict(),
            error="",
        )

        def elapsed() -> float:
            return base_elapsed + (time.monotonic() - started_at)

        def progress(value: float, message: str) -> None:
            update_workspace_item(
                item_id,
                status="En cours",
                progress=max(0.0, min(1.0, float(value))),
                processing_seconds=elapsed(),
                processing_time=format_duration(elapsed()),
                stage=str(message),
                heartbeat=_now(),
            )

        def partial(text: str, position: float) -> None:
            # Le callback de checkpoint qui suit immédiatement réalise l'écriture
            # durable. Celui-ci garde la compatibilité avec d'autres backends.
            if not text:
                return

        def checkpoint(payload: dict[str, object]) -> None:
            durable = dict(payload)
            durable.update(
                {
                    "audio_id": item_id,
                    "run_id": run_id,
                    "settings_hash": settings_hash,
                    "settings": settings.to_dict(),
                    "heartbeat": _now(),
                }
            )
            save_checkpoint(item_id, durable)
            position = float(durable.get("committed_until_seconds") or 0.0)
            update_workspace_item(
                item_id,
                status="En cours",
                partial=str(durable.get("partial") or ""),
                checkpoint_position=position,
                stage=str(durable.get("stage") or "Transcription"),
                processing_seconds=elapsed(),
                processing_time=format_duration(elapsed()),
                heartbeat=_now(),
            )

        try:
            result = run_pipeline(
                source,
                profile=settings.profile,
                language=settings.language,
                requested_speakers=settings.requested_speakers,
                speaker_names=settings.speaker_names,
                initial_prompt=settings.initial_prompt,
                progress=progress,
                partial=partial,
                resume_checkpoint=saved_checkpoint,
                checkpoint=checkpoint,
                asr_backend=platform_compatible_asr_backend(settings.engine),
            )
            result.audio_name = display_name
            result.processing_seconds = elapsed()
            files = write_exports(result)
            update_workspace_item(
                item_id,
                name=display_name,
                status="Terminé",
                progress=1.0,
                processing_seconds=result.processing_seconds,
                processing_time=format_duration(result.processing_seconds),
                partial=plain_transcript(result.turns),
                checkpoint_position=result.duration,
                stage="Transcription terminée",
                result=result.to_dict(),
                files=[str(path) for path in files],
                heartbeat=_now(),
                error="",
            )
            delete_checkpoint(item_id)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            update_workspace_item(
                item_id,
                status="Échec",
                stage=(
                    "Échec — reprise disponible depuis le dernier checkpoint"
                    if load_checkpoint(item_id)
                    else "Échec du traitement"
                ),
                processing_seconds=elapsed(),
                processing_time=format_duration(elapsed()),
                heartbeat=_now(),
                error=detail,
            )

    def _refresh_batch_archive(self) -> None:
        state = load_workspace()
        order = state.get("order") if isinstance(state.get("order"), list) else []
        items = state.get("items") if isinstance(state.get("items"), dict) else {}
        archives: list[Path] = []
        for item_id in order:
            item = items.get(item_id) if isinstance(items, dict) else None
            if not isinstance(item, dict) or item.get("status") != "Terminé":
                continue
            files = item.get("files")
            if not isinstance(files, list) or not files:
                continue
            archive = Path(str(files[0]))
            if archive.is_file():
                archives.append(archive)
        if len(archives) < 2:
            return
        try:
            batch_zip = write_batch_archive(archives)
        except Exception:
            return

        def persist(state_to_update: dict[str, object]) -> None:
            state_to_update["batch_zip"] = str(batch_zip)

        mutate_workspace(persist)


JOB_MANAGER = TranscriptionJobManager()
