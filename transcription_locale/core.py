"""Pipeline local : décodage, transcription, diarisation et alignement."""

from __future__ import annotations

import gc
import logging
import os
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .domain_vocabulary import (
    build_domain_hotwords,
    build_domain_prompt,
    normalize_domain_text,
)
from .runtime_paths import APP_DATA_ROOT

ROOT_DIR = APP_DATA_ROOT
MODEL_DIR = ROOT_DIR / "modeles"
WHISPER_MODEL_DIR = MODEL_DIR / "whisper"
WESPEAKER_MODEL_DIR = MODEL_DIR / "wespeaker"

FASTER_WHISPER_BACKEND = "faster-whisper"
OPENVINO_ARC_BACKEND = "openvino-arc-pilot"
OPENVINO_CHUNK_SECONDS = 30.0
OPENVINO_CHUNK_OVERLAP_SECONDS = 2.0

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("WESPEAKER_HOME", str(WESPEAKER_MODEL_DIR))
LOGGER = logging.getLogger(__name__)


def openvino_arc_supported(platform: str | None = None) -> bool:
    """Indique si le pilote Intel Arc peut être proposé sur cette plateforme.

    Le backend actuel s'appuie sur le runtime et le parcours d'installation
    validés sous Windows. Le garder hors de l'interface et du pipeline sur
    macOS/Linux évite qu'une ancienne préférence tente de charger OpenVINO.
    """

    return str(platform or sys.platform).casefold() == "win32"


def default_asr_backend(platform: str | None = None) -> str:
    """Retourne le moteur par défaut réellement pris en charge."""

    return (
        OPENVINO_ARC_BACKEND
        if openvino_arc_supported(platform)
        else FASTER_WHISPER_BACKEND
    )


def platform_compatible_asr_backend(
    backend: object,
    platform: str | None = None,
) -> str:
    """Neutralise explicitement une demande Intel Arc hors de Windows."""

    selected = str(backend or default_asr_backend(platform))
    if selected == OPENVINO_ARC_BACKEND and not openvino_arc_supported(platform):
        return FASTER_WHISPER_BACKEND
    return selected


PROFILE_CONFIG: dict[str, dict[str, object]] = {
    "Très rapide": {
        "model": "base",
        "beam_size": 1,
        "batch_size": 8,
        "description": "brouillon rapide",
    },
    "Équilibré": {
        "model": "small",
        "beam_size": 2,
        "batch_size": 8,
        "description": "meilleur compromis, traitement batch optimisé",
    },
    "Précis": {
        "model": "turbo",
        "beam_size": 3,
        "batch_size": 4,
        "description": "plus lourd, meilleure reconnaissance",
    },
}


ProgressCallback = Callable[[float, str], None]
PartialCallback = Callable[[str, float], None]
CheckpointCallback = Callable[[dict[str, object]], None]


@dataclass(slots=True, frozen=True)
class WordStamp:
    start: float
    end: float
    text: str
    probability: float | None = None


@dataclass(slots=True, frozen=True)
class SpeakerSegment:
    start: float
    end: float
    speaker: str


@dataclass(slots=True)
class TranscriptTurn:
    start: float
    end: float
    speaker_id: str
    speaker: str
    text: str


@dataclass(slots=True)
class TranscriptionResult:
    audio_name: str
    duration: float
    processing_seconds: float
    profile: str
    model: str
    language: str
    language_probability: float | None
    requested_speakers: int
    detected_speakers: int
    turns: list[TranscriptTurn]
    warnings: list[str] = field(default_factory=list)
    engine: str = FASTER_WHISPER_BACKEND

    def to_metadata(self) -> dict[str, object]:
        return {
            "audio_name": self.audio_name,
            "duration": self.duration,
            "processing_seconds": self.processing_seconds,
            "profile": self.profile,
            "model": self.model,
            "language": self.language,
            "language_probability": self.language_probability,
            "requested_speakers": self.requested_speakers,
            "detected_speakers": self.detected_speakers,
            "warnings": list(self.warnings),
            "engine": self.engine,
        }

    def to_dict(self) -> dict[str, object]:
        data = self.to_metadata()
        data["turns"] = [asdict(turn) for turn in self.turns]
        return data


_MODEL_LOCK = threading.RLock()
_WHISPER_MODEL: object | None = None
_WHISPER_MODEL_ID: str | None = None
_DIARIZATION_LOCK = threading.Lock()


def _progress(callback: ProgressCallback | None, value: float, message: str) -> None:
    if callback is not None:
        callback(max(0.0, min(1.0, value)), message)


def _load_whisper_model(model_id: str):
    """Charge un seul modèle à la fois pour limiter l'usage mémoire."""
    global _WHISPER_MODEL, _WHISPER_MODEL_ID

    with _MODEL_LOCK:
        if _WHISPER_MODEL is not None and _WHISPER_MODEL_ID == model_id:
            return _WHISPER_MODEL

        if _WHISPER_MODEL is not None:
            _WHISPER_MODEL = None
            _WHISPER_MODEL_ID = None
            gc.collect()

        from faster_whisper import WhisperModel

        WHISPER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        threads = max(1, min(os.cpu_count() or 4, 8))
        _WHISPER_MODEL = WhisperModel(
            model_id,
            device="cpu",
            compute_type="int8",
            cpu_threads=threads,
            num_workers=1,
            download_root=str(WHISPER_MODEL_DIR),
        )
        _WHISPER_MODEL_ID = model_id
        return _WHISPER_MODEL


def _clean_names(names: Sequence[str] | None, count: int) -> list[str]:
    result: list[str] = []
    for index in range(count):
        candidate = ""
        if names is not None and index < len(names):
            candidate = str(names[index]).strip()
        result.append(candidate or f"Interlocuteur {index + 1}")
    return result


def _join_word_text(words: Iterable[WordStamp]) -> str:
    tokens = [word.text for word in words if word.text and word.text.strip()]
    if not tokens:
        return ""

    text = tokens[0].strip()
    closing_punctuation = tuple(".,…)]}»")
    for raw in tokens[1:]:
        if not raw:
            continue
        if raw[0].isspace():
            text += raw
        elif raw.startswith(closing_punctuation) or text.endswith(("'", "’", "-", "(", "[", "{", "«")):
            text += raw
        else:
            text += " " + raw

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([.,…\)\]\}»])", r"\1", text)
    # Les horodatages par mot de Whisper peuvent isoler l'apostrophe ou le
    # trait d'union. On rétablit ici les formes françaises les plus courantes.
    text = re.sub(
        r"(?i)\b(qu|[cdjlmnst])\s+['’]\s*(?=\w)",
        lambda match: match.group(1) + "'",
        text,
    )
    text = re.sub(
        r"(?i)(?<=\w)\s*-\s*(?=(?:je|tu|il|elle|on|nous|vous|ils|elles|ce|t|moi|toi|lui|leur|y|en|à|être|ci|là|bas)\b)",
        "-",
        text,
    )
    return text.strip()


def _words_from_checkpoint(checkpoint: Mapping[str, object] | None) -> list[WordStamp]:
    if not checkpoint:
        return []
    raw_words = checkpoint.get("words")
    if not isinstance(raw_words, list):
        return []
    words: list[WordStamp] = []
    for raw in raw_words:
        if not isinstance(raw, Mapping):
            continue
        try:
            probability = raw.get("probability")
            words.append(
                WordStamp(
                    start=float(raw.get("start") or 0.0),
                    end=float(raw.get("end") or 0.0),
                    text=str(raw.get("text") or ""),
                    probability=float(probability) if probability is not None else None,
                )
            )
        except (TypeError, ValueError):
            continue
    return sorted(words, key=lambda word: (word.start, word.end))


def _speaker_segments_from_checkpoint(
    checkpoint: Mapping[str, object] | None,
) -> list[SpeakerSegment]:
    if not checkpoint:
        return []
    raw_segments = checkpoint.get("speaker_segments")
    if not isinstance(raw_segments, list):
        return []
    segments: list[SpeakerSegment] = []
    for raw in raw_segments:
        if not isinstance(raw, Mapping):
            continue
        try:
            segments.append(
                SpeakerSegment(
                    start=float(raw.get("start") or 0.0),
                    end=float(raw.get("end") or 0.0),
                    speaker=str(raw.get("speaker") or "SPEAKER_00"),
                )
            )
        except (TypeError, ValueError):
            continue
    return segments


def _transcription_checkpoint(
    *,
    stage: str,
    words: Sequence[WordStamp],
    position: float,
    model_id: str,
    language: str,
    language_probability: float | None,
) -> dict[str, object]:
    return {
        "version": 1,
        "stage": stage,
        "committed_until_seconds": max(0.0, float(position)),
        "model": model_id,
        "language": language,
        "language_probability": language_probability,
        "partial": _join_word_text(words),
        "words": [asdict(word) for word in words],
    }


def _speaker_for_word(
    word: WordStamp,
    segments: Sequence[SpeakerSegment],
) -> str:
    """Choisit la voix avec le plus grand chevauchement temporel."""
    if not segments:
        return "SPEAKER_00"

    start = max(0.0, word.start)
    end = max(start, word.end)
    midpoint = (start + end) / 2
    best_speaker: str | None = None
    best_overlap = 0.0
    nearest_speaker = segments[0].speaker
    nearest_distance = float("inf")

    for segment in segments:
        overlap = max(0.0, min(end, segment.end) - max(start, segment.start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = segment.speaker

        if midpoint < segment.start:
            distance = segment.start - midpoint
        elif midpoint > segment.end:
            distance = midpoint - segment.end
        else:
            distance = 0.0
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_speaker = segment.speaker

    return best_speaker or nearest_speaker


def align_words_to_speakers(
    words: Sequence[WordStamp],
    speaker_segments: Sequence[SpeakerSegment],
    speaker_names: Sequence[str] | None = None,
    *,
    split_silence_seconds: float = 1.8,
    max_turn_seconds: float = 50.0,
) -> list[TranscriptTurn]:
    """Aligne les mots aux voix et construit des tours de parole lisibles."""
    if not words:
        return []

    ordered_words = sorted(words, key=lambda item: (item.start, item.end))
    ordered_segments = sorted(speaker_segments, key=lambda item: (item.start, item.end))

    raw_speakers: list[str] = []
    assignments: list[tuple[WordStamp, str]] = []
    for word in ordered_words:
        speaker = _speaker_for_word(word, ordered_segments)
        assignments.append((word, speaker))
        if speaker not in raw_speakers:
            raw_speakers.append(speaker)

    names = _clean_names(speaker_names, len(raw_speakers))
    name_map = dict(zip(raw_speakers, names))

    turns: list[TranscriptTurn] = []
    current_words: list[WordStamp] = []
    current_speaker = assignments[0][1]
    current_start = assignments[0][0].start
    current_end = assignments[0][0].end

    def flush() -> None:
        nonlocal current_words
        text = _join_word_text(current_words)
        if text:
            turns.append(
                TranscriptTurn(
                    start=max(0.0, current_start),
                    end=max(current_start, current_end),
                    speaker_id=current_speaker,
                    speaker=name_map.get(current_speaker, current_speaker),
                    text=text,
                )
            )
        current_words = []

    for word, speaker in assignments:
        if not current_words:
            current_words = [word]
            current_speaker = speaker
            current_start = word.start
            current_end = word.end
            continue

        silence = max(0.0, word.start - current_end)
        too_long = (word.end - current_start) >= max_turn_seconds
        previous_ends_sentence = bool(re.search(r"[.!?…][\"'’»)]?$", current_words[-1].text.strip()))
        should_split = (
            speaker != current_speaker
            or silence > split_silence_seconds
            or (too_long and previous_ends_sentence)
        )

        if should_split:
            flush()
            current_speaker = speaker
            current_start = word.start
            current_words = [word]
        else:
            current_words.append(word)
        current_end = word.end

    flush()
    return turns


def _checkpoint_for_model(
    checkpoint: Mapping[str, object] | None,
    model_id: str,
) -> dict[str, object] | None:
    """Rend un checkpoint de mots réutilisable par un moteur de secours."""
    if not checkpoint or not _words_from_checkpoint(checkpoint):
        return None
    converted = dict(checkpoint)
    converted["model"] = model_id
    # Un checkpoint complet ne doit pas court-circuiter un repli déclenché par
    # une erreur. Les checkpoints partiels conservent naturellement leur état.
    if str(converted.get("stage") or "") == "transcription_complete":
        converted["stage"] = "transcription"
    return converted


def _transcribe_words_openvino(
    audio,
    *,
    duration: float,
    profile: str,
    language: str | None,
    initial_prompt: str,
    hotwords: str | None,
    progress: ProgressCallback | None,
    partial: PartialCallback | None,
    resume_checkpoint: Mapping[str, object] | None,
    checkpoint: CheckpointCallback | None,
    backend_warnings: list[str] | None,
    before_cpu_fallback: Callable[[], None] | None = None,
) -> tuple[list[WordStamp], str, float | None, str]:
    """Pilote OpenVINO fenêtré, progressif et reprenable."""
    from .openvino_backend import (
        OPENVINO_MODEL_ID,
        OpenVINOBackendError,
        transcribe_openvino,
    )

    model_id = OPENVINO_MODEL_ID
    saved_words = _words_from_checkpoint(resume_checkpoint)
    saved_stage = str((resume_checkpoint or {}).get("stage") or "")
    saved_model = str((resume_checkpoint or {}).get("model") or "")
    if saved_words and saved_model == model_id and saved_stage in {
        "transcription_complete",
        "diarization_complete",
    }:
        detected_language = str(
            (resume_checkpoint or {}).get("language") or language or "inconnue"
        )
        raw_probability = (resume_checkpoint or {}).get("language_probability")
        detected_probability = (
            float(raw_probability) if raw_probability is not None else None
        )
        position = float(
            (resume_checkpoint or {}).get("committed_until_seconds") or duration
        )
        if partial is not None:
            partial(_join_word_text(saved_words), position)
        _progress(progress, 0.63, "Transcription OpenVINO restaurée depuis le checkpoint.")
        return saved_words, detected_language, detected_probability, model_id

    resume_start = 0.0
    words: list[WordStamp] = []
    if saved_words and saved_model == model_id:
        committed_until = float(
            (resume_checkpoint or {}).get("committed_until_seconds") or 0.0
        )
        resume_start = max(0.0, committed_until - OPENVINO_CHUNK_OVERLAP_SECONDS)
        words = [word for word in saved_words if word.end <= resume_start + 0.05]
        if partial is not None and words:
            partial(_join_word_text(words), resume_start)

    _progress(
        progress,
        0.12,
        "Préparation du pilote OpenVINO Intel Arc (chargement au premier usage)…",
    )
    resume_fraction = resume_start / duration if duration > 0 else 0.0
    _progress(
        progress,
        0.20 + min(0.43, resume_fraction * 0.43),
        (
            f"Reprise OpenVINO à {resume_start:.1f} s…"
            if resume_start > 0
            else "Transcription OpenVINO par fenêtres…"
        ),
    )

    detected_language = str(language or "inconnue")
    detected_probability: float | None = None
    processed_until = resume_start
    first_window = True
    latest_checkpoint: Mapping[str, object] | None = resume_checkpoint

    try:
        while processed_until < duration - 0.001:
            window_start = (
                resume_start
                if first_window
                else max(
                    resume_start,
                    processed_until - OPENVINO_CHUNK_OVERLAP_SECONDS,
                )
            )
            window_end = min(duration, processed_until + OPENVINO_CHUNK_SECONDS)
            start_sample = max(0, min(len(audio), int(window_start * 16000)))
            end_sample = max(start_sample, min(len(audio), int(window_end * 16000)))
            window_audio = audio[start_sample:end_sample]

            result = transcribe_openvino(
                window_audio,
                device="GPU",
                language=(
                    detected_language
                    if detected_language != "inconnue"
                    else language
                ),
                initial_prompt=initial_prompt,
                hotwords=hotwords or "",
                resume_offset=window_start,
            )
            detected_language = str(result.language or language or "inconnue")
            detected_probability = result.language_probability

            # La nouvelle fenêtre remplace intégralement la zone de
            # chevauchement. Cela évite les mots dupliqués sans comparer des
            # graphies potentiellement différentes entre deux inférences.
            words = [word for word in words if word.end <= window_start + 0.05]
            words.extend(
                WordStamp(
                    start=float(word.start),
                    end=float(word.end),
                    text=str(word.text),
                    probability=word.probability,
                )
                for word in result.words
            )
            words.sort(key=lambda word: (word.start, word.end))

            processed_until = window_end
            first_window = False
            stage = (
                "transcription_complete"
                if processed_until >= duration - 0.001
                else "transcription"
            )
            latest_checkpoint = _transcription_checkpoint(
                stage=stage,
                words=words,
                position=processed_until,
                model_id=model_id,
                language=detected_language,
                language_probability=detected_probability,
            )
            if partial is not None:
                partial(_join_word_text(words), processed_until)
            if checkpoint is not None:
                checkpoint(dict(latest_checkpoint))
            fraction = processed_until / duration if duration > 0 else 0.0
            _progress(
                progress,
                0.20 + min(0.43, fraction * 0.43),
                "Transcription OpenVINO par fenêtres…",
            )
    except OpenVINOBackendError as exc:
        from .openvino_backend import reset_openvino_pipeline

        reset_openvino_pipeline()
        fallback_model = str(PROFILE_CONFIG[profile]["model"])
        fallback_checkpoint = _checkpoint_for_model(latest_checkpoint, fallback_model)
        warning = (
            "Le pilote OpenVINO Intel Arc n'a pas pu terminer la transcription ; "
            "Faster-Whisper a pris le relais depuis le dernier checkpoint "
            f"({type(exc).__name__}: {exc})."
        )
        if backend_warnings is not None:
            backend_warnings.append(warning)
        LOGGER.warning(warning)
        _progress(progress, 0.12, "Repli automatique vers Faster-Whisper…")
        if before_cpu_fallback is not None:
            before_cpu_fallback()
        return _transcribe_words(
            audio,
            duration=duration,
            profile=profile,
            language=language,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            progress=progress,
            partial=partial,
            resume_checkpoint=fallback_checkpoint,
            checkpoint=checkpoint,
            asr_backend=FASTER_WHISPER_BACKEND,
            backend_warnings=backend_warnings,
        )

    return words, detected_language, detected_probability, model_id


def _transcribe_words(
    audio,
    *,
    duration: float,
    profile: str,
    language: str | None,
    initial_prompt: str,
    progress: ProgressCallback | None,
    partial: PartialCallback | None,
    hotwords: str | None = None,
    resume_checkpoint: Mapping[str, object] | None = None,
    checkpoint: CheckpointCallback | None = None,
    asr_backend: str = FASTER_WHISPER_BACKEND,
    backend_warnings: list[str] | None = None,
    before_cpu_fallback: Callable[[], None] | None = None,
) -> tuple[list[WordStamp], str, float | None, str]:
    requested_backend = str(asr_backend or "")
    asr_backend = platform_compatible_asr_backend(asr_backend)
    if (
        requested_backend == OPENVINO_ARC_BACKEND
        and requested_backend != asr_backend
        and backend_warnings is not None
    ):
        backend_warnings.append(
            "Le pilote Intel Arc est réservé à Windows ; "
            "Faster-Whisper poursuit la transcription sur cette plateforme."
        )
    if asr_backend == OPENVINO_ARC_BACKEND:
        saved_model = str((resume_checkpoint or {}).get("model") or "")
        saved_stage = str((resume_checkpoint or {}).get("stage") or "")
        if (
            saved_model
            and not saved_model.startswith("openvino/")
            and _words_from_checkpoint(resume_checkpoint)
            and saved_stage in {
                "transcription",
                "transcription_complete",
                "diarization_complete",
            }
        ):
            warning = (
                "La reprise conserve Faster-Whisper, moteur du dernier "
                "checkpoint validé ; le pilote OpenVINO sera retenté au "
                "prochain audio."
            )
            if backend_warnings is not None:
                backend_warnings.append(warning)
            _progress(progress, 0.12, "Reprise du moteur de secours depuis le checkpoint…")
            if before_cpu_fallback is not None:
                before_cpu_fallback()
            return _transcribe_words(
                audio,
                duration=duration,
                profile=profile,
                language=language,
                initial_prompt=initial_prompt,
                hotwords=hotwords,
                progress=progress,
                partial=partial,
                resume_checkpoint=resume_checkpoint,
                checkpoint=checkpoint,
                asr_backend=FASTER_WHISPER_BACKEND,
                backend_warnings=backend_warnings,
            )
        return _transcribe_words_openvino(
            audio,
            duration=duration,
            profile=profile,
            language=language,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            progress=progress,
            partial=partial,
            resume_checkpoint=resume_checkpoint,
            checkpoint=checkpoint,
            backend_warnings=backend_warnings,
            before_cpu_fallback=before_cpu_fallback,
        )
    if asr_backend != FASTER_WHISPER_BACKEND:
        raise ValueError(f"Moteur de transcription inconnu : {asr_backend}")

    config = PROFILE_CONFIG[profile]
    model_id = str(config["model"])
    beam_size = int(config["beam_size"])
    batch_size = int(config["batch_size"])

    saved_words = _words_from_checkpoint(resume_checkpoint)
    saved_stage = str((resume_checkpoint or {}).get("stage") or "")
    saved_model = str((resume_checkpoint or {}).get("model") or "")
    if saved_words and saved_model == model_id and saved_stage in {
        "transcription_complete",
        "diarization_complete",
    }:
        detected_language = str(
            (resume_checkpoint or {}).get("language") or language or "inconnue"
        )
        raw_probability = (resume_checkpoint or {}).get("language_probability")
        detected_probability = (
            float(raw_probability) if raw_probability is not None else None
        )
        position = float(
            (resume_checkpoint or {}).get("committed_until_seconds") or duration
        )
        if partial is not None:
            partial(_join_word_text(saved_words), position)
        _progress(progress, 0.63, "Transcription restaurée depuis le checkpoint.")
        return saved_words, detected_language, detected_probability, model_id

    committed_until = 0.0
    resume_start = 0.0
    words: list[WordStamp] = []
    if saved_words and saved_model == model_id:
        committed_until = float(
            (resume_checkpoint or {}).get("committed_until_seconds") or 0.0
        )
        resume_start = max(0.0, committed_until - 2.0)
        words = [word for word in saved_words if word.end <= resume_start + 0.05]
        if partial is not None and words:
            partial(_join_word_text(words), resume_start)

    _progress(progress, 0.12, f"Chargement du modèle {model_id} (téléchargement au premier usage)…")
    model = _load_whisper_model(model_id)
    resume_fraction = resume_start / duration if duration > 0 else 0.0
    _progress(
        progress,
        0.20 + min(0.43, resume_fraction * 0.43),
        (
            f"Reprise de la transcription à {resume_start:.1f} s…"
            if resume_start > 0
            else "Transcription de l'audio…"
        ),
    )

    from faster_whisper import BatchedInferencePipeline

    batched_model = BatchedInferencePipeline(model=model)
    sample_rate = 16000
    resume_sample = max(0, min(len(audio), int(resume_start * sample_rate)))
    transcription_audio = audio[resume_sample:]
    segments, info = batched_model.transcribe(
        transcription_audio,
        language=language,
        task="transcribe",
        beam_size=beam_size,
        batch_size=batch_size,
        word_timestamps=True,
        without_timestamps=False,
        vad_filter=True,
        vad_parameters={
            "threshold": 0.45,
            "min_speech_duration_ms": 180,
            "min_silence_duration_ms": 450,
            "speech_pad_ms": 180,
        },
        initial_prompt=(initial_prompt or "").strip() or None,
        hotwords=(hotwords or "").strip() or None,
        condition_on_previous_text=False,
        temperature=0.0,
    )

    for segment in segments:
        segment_words = list(segment.words or [])
        if segment_words:
            for word in segment_words:
                if word.start is None or word.end is None:
                    continue
                words.append(
                    WordStamp(
                        start=resume_start + float(word.start),
                        end=resume_start + float(word.end),
                        text=str(word.word),
                        probability=(
                            float(word.probability)
                            if word.probability is not None
                            else None
                        ),
                    )
                )
        elif segment.text and segment.text.strip():
            words.append(
                WordStamp(
                    start=resume_start + float(segment.start),
                    end=resume_start + float(segment.end),
                    text=str(segment.text),
                )
            )

        segment_text = str(segment.text or "").strip()
        if segment_text:
            position = resume_start + float(segment.end)
            if partial is not None:
                partial(_join_word_text(words), position)
            if checkpoint is not None:
                checkpoint(
                    _transcription_checkpoint(
                        stage="transcription",
                        words=words,
                        position=position,
                        model_id=model_id,
                        language=str(getattr(info, "language", None) or language or "inconnue"),
                        language_probability=(
                            float(getattr(info, "language_probability"))
                            if getattr(info, "language_probability", None) is not None
                            else None
                        ),
                    )
                )

        absolute_end = resume_start + float(segment.end)
        fraction = absolute_end / duration if duration > 0 else 0.0
        _progress(progress, 0.20 + min(0.43, fraction * 0.43), "Transcription de l'audio…")

    detected_language = str(getattr(info, "language", None) or language or "inconnue")
    probability_value = getattr(info, "language_probability", None)
    detected_probability = (
        float(probability_value) if probability_value is not None else None
    )
    if checkpoint is not None:
        checkpoint(
            _transcription_checkpoint(
                stage="transcription_complete",
                words=words,
                position=duration,
                model_id=model_id,
                language=detected_language,
                language_probability=detected_probability,
            )
        )
    return words, detected_language, detected_probability, model_id


def _diarize_audio(
    normalized_path: Path,
    *,
    duration: float,
    requested_speakers: int,
    progress: ProgressCallback | None,
    raise_on_error: bool = False,
) -> tuple[list[SpeakerSegment], list[str]]:
    if requested_speakers <= 1:
        return [SpeakerSegment(0.0, duration, "SPEAKER_00")], []

    _progress(progress, 0.65, f"Séparation des {requested_speakers} voix…")
    try:
        # Les modèles de diarisation ne sont pas garantis réentrants. Le verrou
        # évite qu'un ancien flux UI et le gestionnaire de jobs les chargent
        # simultanément, tout en laissant l'ASR Intel Arc travailler en parallèle.
        with _DIARIZATION_LOCK:
            from diarize import diarize as diarize_file

            result = diarize_file(
                str(normalized_path),
                num_speakers=requested_speakers,
            )
        segments = [
            SpeakerSegment(float(item.start), float(item.end), str(item.speaker))
            for item in result.segments
        ]
        if not segments:
            raise RuntimeError("aucun segment de voix détecté")
        detected = len({segment.speaker for segment in segments})
        warnings: list[str] = []
        if detected != requested_speakers:
            warnings.append(
                f"La séparation a trouvé {detected} voix au lieu de {requested_speakers}."
            )
        return segments, warnings
    except Exception as exc:
        if raise_on_error:
            raise
        warning = (
            "La séparation des voix n'a pas abouti ; la transcription reste disponible "
            f"avec une seule étiquette ({type(exc).__name__}: {exc})."
        )
        return [SpeakerSegment(0.0, duration, "SPEAKER_00")], [warning]


def run_pipeline(
    audio_path: str | Path,
    *,
    profile: str = "Équilibré",
    language: str | None = "fr",
    requested_speakers: int = 2,
    speaker_names: Sequence[str] | None = None,
    initial_prompt: str = "",
    progress: ProgressCallback | None = None,
    partial: PartialCallback | None = None,
    resume_checkpoint: Mapping[str, object] | None = None,
    checkpoint: CheckpointCallback | None = None,
    asr_backend: str = FASTER_WHISPER_BACKEND,
) -> TranscriptionResult:
    """Exécute le pipeline complet sans envoyer l'audio sur Internet."""
    started_at = time.perf_counter()
    source = Path(audio_path)
    if not source.is_file():
        raise FileNotFoundError(f"Fichier audio introuvable : {source}")
    if profile not in PROFILE_CONFIG:
        raise ValueError(f"Profil inconnu : {profile}")
    if asr_backend not in {FASTER_WHISPER_BACKEND, OPENVINO_ARC_BACKEND}:
        raise ValueError(f"Moteur de transcription inconnu : {asr_backend}")
    if not 1 <= int(requested_speakers) <= 8:
        raise ValueError("Le nombre d'interlocuteurs doit être compris entre 1 et 8.")

    progress_lock = threading.Lock()
    last_progress = 0.0

    def report_progress(value: float, message: str) -> None:
        """Garantit une progression monotone, y compris lors d'un repli ASR."""

        nonlocal last_progress
        with progress_lock:
            last_progress = max(last_progress, max(0.0, min(1.0, float(value))))
            current = last_progress
        _progress(progress, current, message)

    checkpoint_lock = threading.RLock()
    checkpoint_state: dict[str, object] = dict(resume_checkpoint or {})

    def publish_checkpoint(update: Mapping[str, object]) -> None:
        if checkpoint is None:
            return
        # Le callback est volontairement exécuté sous le même verrou : un
        # résultat de diarisation terminé ne peut ainsi être écrasé par un
        # checkpoint ASR plus ancien publié au même instant.
        with checkpoint_lock:
            checkpoint_state.update(dict(update))
            checkpoint_state["version"] = 2
            checkpoint(dict(checkpoint_state))

    def publish_diarization(
        segments: Sequence[SpeakerSegment],
        diarization_warnings: Sequence[str],
    ) -> None:
        publish_checkpoint(
            {
                "diarization_stage": "complete",
                "speaker_segments": [asdict(segment) for segment in segments],
                "diarization_warnings": list(diarization_warnings),
            }
        )

    _progress(report_progress, 0.02, "Décodage et préparation de l'audio…")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    WESPEAKER_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    from faster_whisper.audio import decode_audio
    import soundfile as sf

    audio = decode_audio(str(source), sampling_rate=16000)
    duration = float(len(audio)) / 16000.0
    if duration <= 0:
        raise ValueError("Le fichier ne contient aucun son exploitable.")

    with tempfile.TemporaryDirectory(prefix="transcription-locale-") as temp_dir:
        normalized_path = Path(temp_dir) / "audio-16khz-mono.wav"
        sf.write(str(normalized_path), audio, 16000, subtype="PCM_16")
        _progress(report_progress, 0.08, "Audio prêt (mono, 16 kHz).")

        speaker_segments = _speaker_segments_from_checkpoint(resume_checkpoint)
        resume_diarization_stage = str(
            (resume_checkpoint or {}).get("diarization_stage") or ""
        )
        legacy_stage = str((resume_checkpoint or {}).get("stage") or "")
        diarization_restored = bool(
            speaker_segments
            and (
                resume_diarization_stage == "complete"
                or legacy_stage == "diarization_complete"
            )
        )
        raw_diarization_warnings = (resume_checkpoint or {}).get(
            "diarization_warnings"
        )
        if not isinstance(raw_diarization_warnings, list):
            raw_diarization_warnings = (
                (resume_checkpoint or {}).get("warnings")
                if legacy_stage == "diarization_complete"
                else []
            )
        restored_diarization_warnings = (
            [str(value) for value in raw_diarization_warnings]
            if isinstance(raw_diarization_warnings, list)
            else []
        )

        backend_warnings: list[str] = []
        domain_prompt = build_domain_prompt(initial_prompt, language)
        domain_hotwords = build_domain_hotwords(initial_prompt, language)
        diarization_future: Future[
            tuple[list[SpeakerSegment], list[str]]
        ] | None = None
        diarization_executor: ThreadPoolExecutor | None = None
        diarization_published = threading.Event()
        diarization_publish_errors: list[BaseException] = []
        resume_model = str((resume_checkpoint or {}).get("model") or "")
        resume_uses_cpu_fallback = bool(
            _words_from_checkpoint(resume_checkpoint)
            and resume_model
            and not resume_model.startswith("openvino/")
        )
        parallel_diarization = bool(
            not diarization_restored
            and not resume_uses_cpu_fallback
            and int(requested_speakers) > 1
            and platform_compatible_asr_backend(asr_backend)
            == OPENVINO_ARC_BACKEND
        )

        def diarization_finished(
            completed: Future[tuple[list[SpeakerSegment], list[str]]],
        ) -> None:
            try:
                completed_segments, completed_warnings = completed.result()
                # Ce checkpoint peut arriver avant la fin de l'ASR. Le
                # coordinateur conserve ensuite ces champs dans chaque
                # checkpoint de transcription.
                publish_diarization(completed_segments, completed_warnings)
            except BaseException as exc:
                # L'erreur d'inférence reste attachée au Future et sera traitée
                # par le repli séquentiel. Seule une erreur de publication doit
                # être transportée séparément jusqu'au thread principal.
                if not completed.cancelled() and completed.exception() is None:
                    diarization_publish_errors.append(exc)
            finally:
                diarization_published.set()

        if parallel_diarization:
            try:
                diarization_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="studio-audio-diarisation",
                )
                diarization_future = diarization_executor.submit(
                    _diarize_audio,
                    normalized_path,
                    duration=duration,
                    requested_speakers=int(requested_speakers),
                    progress=None,
                    raise_on_error=True,
                )
                diarization_future.add_done_callback(diarization_finished)
                _progress(
                    report_progress,
                    0.10,
                    "Transcription et séparation des voix en parallèle…",
                )
            except Exception:
                if diarization_executor is not None:
                    diarization_executor.shutdown(wait=True, cancel_futures=True)
                diarization_executor = None
                diarization_future = None
                _progress(
                    report_progress,
                    0.10,
                    "Transcription en cours ; séparation séquentielle prévue…",
                )

        parallel_diarization_result: tuple[
            list[SpeakerSegment],
            list[str],
        ] | None = None

        def resolve_parallel_diarization(
            *,
            wait_progress: float,
            wait_message: str,
        ) -> tuple[list[SpeakerSegment], list[str]]:
            nonlocal parallel_diarization_result
            if parallel_diarization_result is not None:
                return parallel_diarization_result
            if diarization_future is None:
                raise RuntimeError("Aucune séparation parallèle à finaliser.")
            if not diarization_future.done():
                _progress(report_progress, wait_progress, wait_message)
            try:
                while True:
                    try:
                        parallel_diarization_result = diarization_future.result(
                            timeout=1.0
                        )
                        break
                    except FutureTimeoutError:
                        _progress(report_progress, wait_progress, wait_message)
                diarization_published.wait()
            except Exception:
                _progress(
                    report_progress,
                    wait_progress,
                    "Repli vers la séparation séquentielle des voix…",
                )
                parallel_diarization_result = _diarize_audio(
                    normalized_path,
                    duration=duration,
                    requested_speakers=int(requested_speakers),
                    progress=report_progress,
                )
                publish_diarization(*parallel_diarization_result)
            if diarization_publish_errors:
                raise diarization_publish_errors[0]
            return parallel_diarization_result

        def finish_diarization_before_cpu_fallback() -> None:
            if diarization_future is None:
                return
            resolve_parallel_diarization(
                wait_progress=0.20,
                wait_message=(
                    "Intel Arc indisponible ; séparation des voix en cours "
                    "avant le repli CPU…"
                ),
            )

        try:
            words, detected_language, language_probability, model_id = _transcribe_words(
                audio,
                duration=duration,
                profile=profile,
                language=language,
                initial_prompt=domain_prompt,
                hotwords=domain_hotwords,
                progress=report_progress,
                partial=partial,
                resume_checkpoint=resume_checkpoint,
                checkpoint=publish_checkpoint,
                asr_backend=asr_backend,
                backend_warnings=backend_warnings,
                before_cpu_fallback=finish_diarization_before_cpu_fallback,
            )
            if not words:
                raise ValueError("Aucune parole n'a été reconnue dans ce fichier.")

            if diarization_restored:
                diarization_warnings = restored_diarization_warnings
                _progress(
                    report_progress,
                    0.86,
                    "Séparation des voix restaurée depuis le checkpoint.",
                )
            elif diarization_future is not None:
                speaker_segments, diarization_warnings = (
                    resolve_parallel_diarization(
                        wait_progress=0.65,
                        wait_message=(
                            "Transcription terminée ; finalisation de la "
                            "séparation des voix…"
                        ),
                    )
                )
                _progress(
                    report_progress,
                    0.86,
                    "Transcription et séparation des voix terminées.",
                )
            else:
                speaker_segments, diarization_warnings = _diarize_audio(
                    normalized_path,
                    duration=duration,
                    requested_speakers=int(requested_speakers),
                    progress=report_progress,
                )
                publish_diarization(speaker_segments, diarization_warnings)

            warnings = [*backend_warnings]
            warnings.extend(
                warning
                for warning in diarization_warnings
                if warning not in warnings
            )
            publish_checkpoint({"warnings": list(warnings)})
        except Exception:
            if diarization_future is not None and not diarization_future.done():
                _progress(
                    report_progress,
                    0.20,
                    "Arrêt de la transcription ; sécurisation de la séparation en cours…",
                )
            raise
        finally:
            if diarization_executor is not None:
                diarization_executor.shutdown(wait=True, cancel_futures=True)

    _progress(report_progress, 0.88, "Alignement des mots avec les voix…")
    turns = align_words_to_speakers(words, speaker_segments, speaker_names)
    for turn in turns:
        turn.text = normalize_domain_text(turn.text, initial_prompt)
    if not turns:
        raise ValueError("La transcription n'a produit aucun tour de parole.")

    detected_speakers = len({turn.speaker_id for turn in turns})
    elapsed = time.perf_counter() - started_at
    _progress(report_progress, 0.94, "Création des fichiers à télécharger…")
    effective_engine = (
        OPENVINO_ARC_BACKEND
        if model_id.startswith("openvino/")
        else FASTER_WHISPER_BACKEND
    )
    return TranscriptionResult(
        audio_name=source.name,
        duration=duration,
        processing_seconds=elapsed,
        profile=profile,
        model=model_id,
        language=detected_language,
        language_probability=language_probability,
        requested_speakers=int(requested_speakers),
        detected_speakers=detected_speakers,
        turns=turns,
        warnings=warnings,
        engine=effective_engine,
    )
