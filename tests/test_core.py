from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from transcription_locale.core import (
    FASTER_WHISPER_BACKEND,
    OPENVINO_ARC_BACKEND,
    SpeakerSegment,
    WordStamp,
    _transcribe_words,
    align_words_to_speakers,
    default_asr_backend,
    openvino_arc_supported,
    platform_compatible_asr_backend,
)
from transcription_locale.exports import format_timestamp, parse_timestamp, rows_to_turns


class PlatformBackendTests(unittest.TestCase):
    def test_intel_arc_is_available_and_default_only_on_windows(self) -> None:
        self.assertTrue(openvino_arc_supported("win32"))
        self.assertFalse(openvino_arc_supported("darwin"))
        self.assertFalse(openvino_arc_supported("linux"))
        self.assertEqual(default_asr_backend("win32"), OPENVINO_ARC_BACKEND)
        self.assertEqual(default_asr_backend("darwin"), FASTER_WHISPER_BACKEND)
        self.assertEqual(default_asr_backend("linux"), FASTER_WHISPER_BACKEND)

    def test_arc_request_is_neutralised_outside_windows(self) -> None:
        self.assertEqual(
            platform_compatible_asr_backend(OPENVINO_ARC_BACKEND, "darwin"),
            FASTER_WHISPER_BACKEND,
        )
        self.assertEqual(
            platform_compatible_asr_backend(OPENVINO_ARC_BACKEND, "linux"),
            FASTER_WHISPER_BACKEND,
        )
        self.assertEqual(
            platform_compatible_asr_backend(OPENVINO_ARC_BACKEND, "win32"),
            OPENVINO_ARC_BACKEND,
        )


class AlignmentTests(unittest.TestCase):
    def test_words_follow_two_speakers(self) -> None:
        words = [
            WordStamp(0.2, 0.6, " Bonjour"),
            WordStamp(0.6, 1.0, " monsieur."),
            WordStamp(1.5, 1.8, " Bonjour"),
            WordStamp(1.8, 2.3, " madame."),
        ]
        speakers = [
            SpeakerSegment(0.0, 1.2, "SPEAKER_A"),
            SpeakerSegment(1.3, 2.5, "SPEAKER_B"),
        ]

        turns = align_words_to_speakers(words, speakers, ["Conseillère", "Client"])

        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0].speaker, "Conseillère")
        self.assertEqual(turns[0].text, "Bonjour monsieur.")
        self.assertEqual(turns[1].speaker, "Client")
        self.assertEqual(turns[1].text, "Bonjour madame.")

    def test_nearest_speaker_fills_diarization_gap(self) -> None:
        words = [WordStamp(1.1, 1.2, " Oui.")]
        speakers = [
            SpeakerSegment(0.0, 1.0, "SPEAKER_A"),
            SpeakerSegment(2.0, 3.0, "SPEAKER_B"),
        ]
        turns = align_words_to_speakers(words, speakers)
        self.assertEqual(turns[0].speaker_id, "SPEAKER_A")

    def test_long_silence_splits_same_speaker(self) -> None:
        words = [
            WordStamp(0.0, 0.3, " Premier."),
            WordStamp(3.0, 3.4, " Deuxième."),
        ]
        speakers = [SpeakerSegment(0.0, 4.0, "SPEAKER_A")]
        turns = align_words_to_speakers(words, speakers)
        self.assertEqual(len(turns), 2)

    def test_french_apostrophes_and_hyphens_are_rejoined(self) -> None:
        words = [
            WordStamp(0.0, 0.2, " Qu"),
            WordStamp(0.2, 0.3, " '"),
            WordStamp(0.3, 0.6, "est"),
            WordStamp(0.6, 0.7, " -"),
            WordStamp(0.7, 0.9, "ce"),
            WordStamp(0.9, 1.0, " qu"),
            WordStamp(1.0, 1.1, " '"),
            WordStamp(1.1, 1.3, "il"),
            WordStamp(1.3, 1.4, " dit ?"),
        ]
        turns = align_words_to_speakers(
            words,
            [SpeakerSegment(0.0, 2.0, "SPEAKER_A")],
        )
        self.assertEqual(turns[0].text, "Qu'est-ce qu'il dit ?")

    def test_transcription_resume_replaces_overlap_without_duplicate_words(self) -> None:
        resume = {
            "stage": "transcription",
            "model": "small",
            "committed_until_seconds": 3.0,
            "words": [
                {"start": 0.0, "end": 0.8, "text": " Ancien", "probability": 0.9},
                {"start": 1.2, "end": 1.5, "text": " doublon", "probability": 0.9},
            ],
        }
        segment = SimpleNamespace(
            start=0.1,
            end=0.5,
            text=" nouveau",
            words=[
                SimpleNamespace(start=0.1, end=0.5, word=" nouveau", probability=0.95)
            ],
        )
        info = SimpleNamespace(language="fr", language_probability=0.99)
        checkpoints: list[dict[str, object]] = []
        partials: list[tuple[str, float]] = []

        class FakeBatchedPipeline:
            def __init__(self, model):
                self.model = model

            def transcribe(self, audio, **kwargs):
                self.audio = audio
                return iter([segment]), info

        with (
            patch("transcription_locale.core._load_whisper_model", return_value=object()),
            patch("faster_whisper.BatchedInferencePipeline", FakeBatchedPipeline),
        ):
            words, language, probability, model = _transcribe_words(
                [0.0] * 160_000,
                duration=10.0,
                profile="Équilibré",
                language="fr",
                initial_prompt="",
                progress=None,
                partial=lambda text, position: partials.append((text, position)),
                resume_checkpoint=resume,
                checkpoint=checkpoints.append,
            )

        self.assertEqual([word.text for word in words], [" Ancien", " nouveau"])
        self.assertEqual(language, "fr")
        self.assertEqual(probability, 0.99)
        self.assertEqual(model, "small")
        self.assertNotIn("doublon", partials[-1][0])
        self.assertEqual(checkpoints[-1]["stage"], "transcription_complete")


class ExportHelpersTests(unittest.TestCase):
    def test_timestamp_round_trip(self) -> None:
        value = 3661.25
        rendered = format_timestamp(value, milliseconds=True)
        self.assertAlmostEqual(parse_timestamp(rendered), value, places=2)

    def test_rows_can_be_edited(self) -> None:
        turns = rows_to_turns([["00:00:01", "00:00:02", "Alice", "Texte corrigé"]])
        self.assertEqual(turns[0].speaker, "Alice")
        self.assertEqual(turns[0].text, "Texte corrigé")


if __name__ == "__main__":
    unittest.main()
