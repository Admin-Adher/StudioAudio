"""Mesures acoustiques légères pour les questions sur la dynamique d'un appel.

Ces mesures ne prétendent pas reconnaître une émotion. Elles décrivent des
signaux observables (intensité, débit, alternances rapides et chevauchements)
que l'assistant peut ensuite interpréter avec prudence et citer par timecode.
"""

from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import soundfile as sf


def _fold(value: object) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    ).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def question_needs_audio_analysis(question: object) -> bool:
    folded = _fold(question)
    return bool(
        re.search(
            r"\b(?:ton|voix|volume|crie\w*|hausse\w*|monte\w*|"
            r"degener\w*|tension\w*|disput\w*|agress\w*|coler\w*|"
            r"enerve\w*|emotion\w*|debit|interromp\w*|interruption\w*)\b",
            folded,
        )
    )


def _field(turn: object, name: str, default: object = None) -> object:
    if isinstance(turn, Mapping):
        return turn.get(name, default)
    return getattr(turn, name, default)


def _safe_seconds(turn: object, name: str, default: float = 0.0) -> float:
    try:
        return max(0.0, float(_field(turn, name, default) or default))
    except (TypeError, ValueError):
        return default


def _dbfs(samples: np.ndarray) -> float:
    if not samples.size:
        return -80.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return max(-80.0, 20.0 * math.log10(max(rms, 1e-8)))


def _turn_audio_metrics(
    audio: sf.SoundFile,
    turn: object,
    index: int,
) -> dict[str, object]:
    start = _safe_seconds(turn, "start")
    end = max(start, _safe_seconds(turn, "end", start))
    frame_start = min(len(audio), max(0, int(start * audio.samplerate)))
    frame_end = min(len(audio), max(frame_start, int(end * audio.samplerate)))
    audio.seek(frame_start)
    samples = audio.read(
        frames=max(0, frame_end - frame_start),
        dtype="float32",
        always_2d=True,
    )
    if samples.size:
        samples = samples.mean(axis=1)
        stride = max(1, int(audio.samplerate // 16_000))
        samples = samples[::stride]
    else:
        samples = np.asarray([], dtype=np.float32)
    text = str(_field(turn, "text", "") or "").strip()
    words = len(text.split())
    duration = max(0.25, end - start)
    return {
        "turn_id": f"turn-{index:05d}",
        "start_seconds": round(start, 3),
        "end_seconds": round(end, 3),
        "speaker_id": str(_field(turn, "speaker_id", "") or ""),
        "speaker": str(_field(turn, "speaker", "") or ""),
        "loudness_dbfs": round(_dbfs(samples), 2),
        "peak": round(float(np.max(np.abs(samples))) if samples.size else 0.0, 4),
        "speech_rate_wps": round(words / duration, 2),
        "word_count": words,
    }


def analyze_audio_dynamics(
    audio_path: str | Path,
    turns: Sequence[object],
) -> dict[str, object]:
    """Calcule une synthèse acoustique bornée et JSON-sérialisable."""

    path = Path(audio_path).expanduser()
    if not path.is_file() or not turns:
        return {
            "available": False,
            "limitation": "Fichier audio ou tours horodatés indisponibles.",
        }
    try:
        with sf.SoundFile(path) as audio:
            turn_metrics = [
                _turn_audio_metrics(audio, turn, index)
                for index, turn in enumerate(turns)
                if str(_field(turn, "text", "") or "").strip()
            ]
            audio_duration = len(audio) / max(1, audio.samplerate)
    except (OSError, RuntimeError, ValueError) as exc:
        return {
            "available": False,
            "limitation": (
                "L'audio n'a pas pu être mesuré "
                f"({type(exc).__name__})."
            ),
        }
    if not turn_metrics:
        return {
            "available": False,
            "limitation": "Aucun tour de parole mesurable.",
        }

    chronological = sorted(
        turn_metrics,
        key=lambda item: (float(item["start_seconds"]), str(item["turn_id"])),
    )
    split_time = max(1.0, audio_duration * 0.33)
    late_time = audio_duration * 0.67
    early = [
        float(item["loudness_dbfs"])
        for item in chronological
        if float(item["start_seconds"]) <= split_time
    ]
    late = [
        float(item["loudness_dbfs"])
        for item in chronological
        if float(item["start_seconds"]) >= late_time
    ]
    early_level = float(np.median(early or [item["loudness_dbfs"] for item in chronological]))
    late_level = float(np.median(late or [item["loudness_dbfs"] for item in chronological]))
    change_db = late_level - early_level

    overlaps = 0
    rapid_alternations = 0
    for previous, current in zip(chronological, chronological[1:]):
        gap = float(current["start_seconds"]) - float(previous["end_seconds"])
        speaker_changed = current["speaker_id"] != previous["speaker_id"]
        if gap < -0.05 and speaker_changed:
            overlaps += 1
        if -0.05 <= gap <= 0.45 and speaker_changed:
            rapid_alternations += 1

    loudest = sorted(
        chronological,
        key=lambda item: (
            -float(item["loudness_dbfs"]),
            float(item["start_seconds"]),
        ),
    )[:5]
    fastest = sorted(
        (item for item in chronological if int(item["word_count"]) >= 4),
        key=lambda item: (
            -float(item["speech_rate_wps"]),
            float(item["start_seconds"]),
        ),
    )[:5]
    if change_db >= 3.0:
        assessment = "rising"
    elif change_db <= -3.0:
        assessment = "falling"
    else:
        assessment = "stable_or_mixed"
    return {
        "available": True,
        "method": "RMS par tour + débit lexical + dynamique des alternances",
        "audio_duration_seconds": round(audio_duration, 3),
        "assessment": assessment,
        "early_median_loudness_dbfs": round(early_level, 2),
        "late_median_loudness_dbfs": round(late_level, 2),
        "loudness_change_db": round(change_db, 2),
        "overlap_count": overlaps,
        "rapid_alternation_count": rapid_alternations,
        "loudest_turns": loudest,
        "fastest_turns": fastest,
        "limitations": (
            "Ces mesures décrivent l'intensité et le débit ; elles ne détectent "
            "pas à elles seules la colère, l'ironie ou l'intention."
        ),
    }


__all__ = ["analyze_audio_dynamics", "question_needs_audio_analysis"]
