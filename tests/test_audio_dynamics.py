from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from transcription_locale.audio_dynamics import (
    analyze_audio_dynamics,
    question_needs_audio_analysis,
)
from transcription_locale.core import TranscriptTurn


class AudioDynamicsTests(unittest.TestCase):
    def test_tone_questions_are_detected_without_routing_content_questions(self) -> None:
        self.assertTrue(
            question_needs_audio_analysis(
                "Est-ce que le ton monte et dégénère durant la consultation ?"
            )
        )
        self.assertTrue(question_needs_audio_analysis("La Cliente crie-t-elle ?"))
        self.assertFalse(
            question_needs_audio_analysis(
                "Est-ce que le Master annonce une séparation ?"
            )
        )

    def test_analysis_reports_a_rising_loudness_with_timecoded_turns(self) -> None:
        sample_rate = 16_000
        seconds = 4
        time = np.arange(sample_rate * seconds, dtype=np.float32) / sample_rate
        signal = np.concatenate(
            (
                0.03 * np.sin(2 * np.pi * 220 * time[: sample_rate * 2]),
                0.30 * np.sin(2 * np.pi * 220 * time[: sample_rate * 2]),
            )
        ).astype(np.float32)
        turns = [
            TranscriptTurn(0.0, 2.0, "A", "Client", "Je parle calmement."),
            TranscriptTurn(
                2.0,
                4.0,
                "B",
                "Master",
                "La réponse devient beaucoup plus appuyée maintenant.",
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.wav"
            sf.write(path, signal, sample_rate)
            result = analyze_audio_dynamics(path, turns)

        self.assertTrue(result["available"])
        self.assertEqual(result["assessment"], "rising")
        self.assertGreaterEqual(result["loudness_change_db"], 10.0)
        self.assertEqual(result["loudest_turns"][0]["turn_id"], "turn-00001")

    def test_missing_audio_returns_an_explicit_limitation(self) -> None:
        result = analyze_audio_dynamics(
            "audio-introuvable.wav",
            [TranscriptTurn(0.0, 1.0, "A", "Client", "Bonjour.")],
        )

        self.assertFalse(result["available"])
        self.assertIn("indisponibles", result["limitation"])


if __name__ == "__main__":
    unittest.main()
