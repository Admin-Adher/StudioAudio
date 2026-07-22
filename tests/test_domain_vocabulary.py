from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from transcription_locale.core import SpeakerSegment, WordStamp, _transcribe_words, run_pipeline
from transcription_locale.domain_vocabulary import (
    build_domain_hotwords,
    build_domain_prompt,
    extract_priority_terms,
    normalize_domain_text,
)


class DomainVocabularyTests(unittest.TestCase):
    def test_short_prompt_and_hotwords_cover_the_business_domain(self) -> None:
        prompt = build_domain_prompt("LyraX, OrionX, Oracle Lumière", "fr")
        hotwords = build_domain_hotwords("LyraX, OrionX, Oracle Lumière", "fr")

        self.assertLess(prompt.index("LyraX"), prompt.index("date de naissance"))
        self.assertLessEqual(len(prompt), 240)
        self.assertLessEqual(len(hotwords), 640)
        self.assertIn("date de naissance", prompt)
        for expected in (
            "thème astral",
            "ascendant",
            "Bélier",
            "Gémeaux",
            "Sagittaire",
            "tarot de Marseille",
            "oracle de Belline",
            "médiumnité",
            "numérologie",
        ):
            self.assertIn(expected, hotwords)
        self.assertTrue(
            prompt.endswith("Quelle est votre date de naissance, s'il vous plaît ?")
        )
        for common_month in ("janvier", "février", "août", "décembre"):
            self.assertNotIn(common_month, hotwords)

    def test_explicit_non_french_language_does_not_receive_french_domain_bias(self) -> None:
        prompt = build_domain_prompt("ClienteX, birth chart", "en")
        hotwords = build_domain_hotwords("ClienteX, birth chart", "en")

        self.assertEqual(prompt, "ClienteX, birth chart")
        self.assertEqual(hotwords, "ClienteX, birth chart")
        self.assertNotIn("Bélier", prompt)

    def test_user_terms_are_deduplicated_and_free_form_instructions_are_not_hotwords(self) -> None:
        terms = extract_priority_terms(
            "LyraX, lyrax; OrionX\n"
            "Merci de corriger absolument tous les nombres entendus dans cet audio"
        )

        self.assertEqual(terms, ("LyraX", "OrionX"))
        hotwords = build_domain_hotwords("LyraX, lyrax, OrionX")
        self.assertEqual(hotwords.count("LyraX"), 1)
        self.assertTrue(hotwords.startswith("LyraX, OrionX"))

    def test_normalization_only_restores_exact_graphies(self) -> None:
        text = (
            "lyrax consulte le tarot de marseille en fevrier. "
            "Elle parle de Marcelle, de venus hier et de la date 31 02 2099."
        )

        normalized = normalize_domain_text(text, "LyraX")

        self.assertEqual(
            normalized,
            "LyraX consulte le tarot de Marseille en fevrier. "
            "Elle parle de Marcelle, de venus hier et de la date 31 02 2099.",
        )

    def test_user_term_requires_the_same_letters_not_a_phonetic_guess(self) -> None:
        normalized = normalize_domain_text(
            "LyraX parle avec OrionXylophone et orionx.",
            "OrionX",
        )
        self.assertEqual(
            normalized,
            "LyraX parle avec OrionXylophone et OrionX.",
        )


class DomainPipelineIntegrationTests(unittest.TestCase):
    def test_run_pipeline_injects_prompt_hotwords_and_normalizes_final_turns(self) -> None:
        received: dict[str, object] = {}

        def fake_transcribe(audio, **kwargs):
            received.update(kwargs)
            return (
                [
                    WordStamp(0.0, 0.4, " lyrax"),
                    WordStamp(0.4, 0.8, " consulte"),
                    WordStamp(0.8, 1.2, " le tarot de marseille"),
                    WordStamp(1.2, 1.6, " le 31 02 2099."),
                ],
                "fr",
                0.99,
                "openvino/whisper-large-v3-turbo-int8-ov",
            )

        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            with (
                patch("faster_whisper.audio.decode_audio", return_value=[0.0] * 32_000),
                patch("soundfile.write"),
                patch("transcription_locale.core._transcribe_words", side_effect=fake_transcribe),
                patch(
                    "transcription_locale.core._diarize_audio",
                    return_value=([SpeakerSegment(0.0, 2.0, "SPEAKER_00")], []),
                ),
            ):
                result = run_pipeline(
                    audio_file.name,
                    initial_prompt="LyraX, OrionX",
                    speaker_names=["Master"],
                    asr_backend="openvino-arc-pilot",
                )

        self.assertIn("LyraX", str(received["initial_prompt"]))
        self.assertIn("date de naissance", str(received["initial_prompt"]))
        self.assertLessEqual(len(str(received["initial_prompt"])), 240)
        self.assertIn("thème astral", str(received["hotwords"]))
        self.assertTrue(
            str(received["hotwords"]).startswith("LyraX, OrionX")
        )
        self.assertEqual(
            result.turns[0].text,
            "LyraX consulte le tarot de Marseille le 31 02 2099.",
        )

    def test_faster_whisper_receives_hotwords_separately_from_the_prompt(self) -> None:
        captured: dict[str, object] = {}
        segment = SimpleNamespace(
            start=0.0,
            end=1.0,
            text=" thème astral",
            words=[
                SimpleNamespace(
                    start=0.0,
                    end=1.0,
                    word=" thème astral",
                    probability=0.9,
                )
            ],
        )
        info = SimpleNamespace(language="fr", language_probability=0.99)

        class FakeBatchedPipeline:
            def __init__(self, model):
                self.model = model

            def transcribe(self, audio, **kwargs):
                captured.update(kwargs)
                return iter([segment]), info

        with (
            patch("transcription_locale.core._load_whisper_model", return_value=object()),
            patch("faster_whisper.BatchedInferencePipeline", FakeBatchedPipeline),
        ):
            _transcribe_words(
                [0.0] * 16_000,
                duration=1.0,
                profile="Équilibré",
                language="fr",
                initial_prompt="Contexte de voyance.",
                hotwords="LyraX, thème astral",
                progress=None,
                partial=None,
            )

        self.assertEqual(captured["initial_prompt"], "Contexte de voyance.")
        self.assertEqual(captured["hotwords"], "LyraX, thème astral")


if __name__ == "__main__":
    unittest.main()
