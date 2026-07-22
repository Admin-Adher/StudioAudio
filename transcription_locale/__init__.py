"""Transcription locale et diarisation de conversations."""

from .core import (
    PROFILE_CONFIG,
    SpeakerSegment,
    TranscriptTurn,
    TranscriptionResult,
    WordStamp,
    align_words_to_speakers,
    run_pipeline,
)

__all__ = [
    "PROFILE_CONFIG",
    "SpeakerSegment",
    "TranscriptTurn",
    "TranscriptionResult",
    "WordStamp",
    "align_words_to_speakers",
    "run_pipeline",
]
