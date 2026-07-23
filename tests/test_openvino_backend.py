from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from transcription_locale.core import (
    OPENVINO_ARC_BACKEND,
    _transcribe_words,
)
from transcription_locale.openvino_backend import (
    OPENVINO_MODEL_ID,
    REQUIRED_MODEL_FILES,
    OpenVINOBackendUnavailable,
    OpenVINOBackendError,
    OpenVINOInvalidOutput,
    OpenVINOTranscription,
    OpenVINOWord,
    get_openvino_status,
    reset_openvino_pipeline,
    transcribe_openvino,
)


def _create_fake_model(root: Path) -> Path:
    model = root / "model"
    model.mkdir()
    for name in REQUIRED_MODEL_FILES:
        (model / name).write_text("test", encoding="utf-8")
    return model


class _FakeCore:
    available_devices = ["CPU", "GPU.0"]

    def get_property(self, device: str, name: str) -> str:
        if device == "GPU.0" and name == "FULL_DEVICE_NAME":
            return "Intel Arc Test GPU"
        raise KeyError((device, name))


class OpenVINOAdapterTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_openvino_pipeline()

    def test_status_is_lazy_when_model_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "absent"
            with patch(
                "transcription_locale.openvino_backend._import_openvino_modules",
                side_effect=AssertionError("OpenVINO ne doit pas être importé"),
            ):
                status = get_openvino_status(missing)

        self.assertFalse(status.available)
        self.assertIn("absent ou incomplet", status.reason)

    def test_status_detects_an_intel_gpu_without_loading_a_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = _create_fake_model(Path(temporary))
            fake_openvino = SimpleNamespace(Core=_FakeCore)
            with patch(
                "transcription_locale.openvino_backend._import_openvino_modules",
                return_value=(fake_openvino, object()),
            ):
                status = get_openvino_status(model)

        self.assertTrue(status.available)
        self.assertEqual(status.device_name, "Intel Arc Test GPU")

    def test_pipeline_is_persistent_and_returns_absolute_word_timestamps(self) -> None:
        class FakePipeline:
            instances: list["FakePipeline"] = []

            def __init__(self, model_path: str, device: str, **kwargs):
                self.model_path = model_path
                self.device = device
                self.options = kwargs
                self.calls: list[dict[str, object]] = []
                self.__class__.instances.append(self)

            def generate(self, samples, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    texts=[" Bonjour."],
                    language="<|fr|>",
                    words=[
                        SimpleNamespace(
                            start_ts=0.25,
                            end_ts=0.75,
                            word=" Bonjour.",
                        )
                    ],
                )

        with tempfile.TemporaryDirectory() as temporary:
            model = _create_fake_model(Path(temporary))
            fake_openvino = SimpleNamespace(Core=_FakeCore)
            fake_genai = SimpleNamespace(WhisperPipeline=FakePipeline)
            with patch(
                "transcription_locale.openvino_backend._import_openvino_modules",
                return_value=(fake_openvino, fake_genai),
            ):
                first = transcribe_openvino(
                    [0.0] * 16_000,
                    model_path=model,
                    language="fr-FR",
                    initial_prompt="ClienteX",
                    hotwords="ClienteX, date de naissance, thème astral",
                )
                resumed = transcribe_openvino(
                    [0.0] * 8_000,
                    model_path=model,
                    language="fr",
                    initial_prompt="ClienteX",
                    hotwords="ClienteX, date de naissance, thème astral",
                    resume_offset=12.0,
                )

        self.assertEqual(len(FakePipeline.instances), 1)
        pipeline = FakePipeline.instances[0]
        self.assertTrue(pipeline.options["word_timestamps"])
        self.assertEqual(pipeline.options["CACHE_DIR"], "")
        self.assertEqual(pipeline.calls[0]["language"], "<|fr|>")
        self.assertEqual(pipeline.calls[0]["initial_prompt"], "ClienteX")
        self.assertIn("ClienteX", pipeline.calls[0]["hotwords"])
        self.assertIn("date de naissance", pipeline.calls[0]["hotwords"])
        self.assertNotIn("initial_prompt", pipeline.calls[1])
        self.assertIn("thème astral", pipeline.calls[1]["hotwords"])
        self.assertEqual(first.language, "fr")
        self.assertAlmostEqual(resumed.words[0].start, 12.25)
        self.assertAlmostEqual(resumed.words[0].end, 12.5)

    def test_unsupported_hotwords_are_wrapped_for_the_existing_fallback(self) -> None:
        class FakePipeline:
            def __init__(self, *args, **kwargs):
                pass

            def generate(self, samples, **kwargs):
                if "hotwords" in kwargs:
                    raise TypeError("unexpected keyword argument 'hotwords'")
                raise AssertionError("Le test doit transmettre les hotwords")

        with tempfile.TemporaryDirectory() as temporary:
            model = _create_fake_model(Path(temporary))
            fake_openvino = SimpleNamespace(Core=_FakeCore)
            fake_genai = SimpleNamespace(WhisperPipeline=FakePipeline)
            with patch(
                "transcription_locale.openvino_backend._import_openvino_modules",
                return_value=(fake_openvino, fake_genai),
            ):
                with self.assertRaises(OpenVINOBackendError) as error:
                    transcribe_openvino(
                        [0.0] * 16_000,
                        model_path=model,
                        hotwords="date de naissance",
                    )

        self.assertIn("TypeError", str(error.exception))

    def test_punctuation_only_output_is_rejected_for_fallback(self) -> None:
        class FakePipeline:
            def __init__(self, *args, **kwargs):
                pass

            def generate(self, samples, **kwargs):
                return SimpleNamespace(
                    texts=["..."],
                    language="fr",
                    words=[
                        SimpleNamespace(start_ts=0.0, end_ts=0.5, word="...")
                    ],
                )

        with tempfile.TemporaryDirectory() as temporary:
            model = _create_fake_model(Path(temporary))
            fake_openvino = SimpleNamespace(Core=_FakeCore)
            fake_genai = SimpleNamespace(WhisperPipeline=FakePipeline)
            with patch(
                "transcription_locale.openvino_backend._import_openvino_modules",
                return_value=(fake_openvino, fake_genai),
            ):
                with self.assertRaises(OpenVINOInvalidOutput):
                    transcribe_openvino([0.0] * 16_000, model_path=model)


class OpenVINOCoreIntegrationTests(unittest.TestCase):
    def test_windows_are_progressive_and_replace_the_overlap(self) -> None:
        calls: list[float] = []
        call_options: list[dict[str, object]] = []

        def fake_openvino(audio, **kwargs):
            offset = float(kwargs["resume_offset"])
            calls.append(offset)
            call_options.append(dict(kwargs))
            if offset == 0.0:
                words = (
                    OpenVINOWord(1.0, 1.5, " Début"),
                    OpenVINOWord(28.2, 29.0, " ancien-chevauchement"),
                )
            elif offset == 28.0:
                words = (
                    OpenVINOWord(28.1, 28.8, " nouveau-chevauchement"),
                    OpenVINOWord(45.0, 45.5, " Milieu"),
                    OpenVINOWord(58.2, 59.0, " ancienne-fin"),
                )
            else:
                words = (
                    OpenVINOWord(58.1, 58.8, " nouvelle-fin"),
                    OpenVINOWord(64.0, 64.5, " Fin"),
                )
            return OpenVINOTranscription(
                words=words,
                text="".join(word.text for word in words),
                language="fr",
                language_probability=None,
            )

        checkpoints: list[dict[str, object]] = []
        partials: list[tuple[str, float]] = []
        with patch(
            "transcription_locale.openvino_backend.transcribe_openvino",
            side_effect=fake_openvino,
        ):
            words, language, probability, model = _transcribe_words(
                [0.0] * (65 * 16_000),
                duration=65.0,
                profile="Équilibré",
                language="fr",
                initial_prompt="Quelle est votre date de naissance ?",
                hotwords="date de naissance, thème astral, oracle",
                progress=None,
                partial=lambda text, position: partials.append((text, position)),
                checkpoint=checkpoints.append,
                asr_backend=OPENVINO_ARC_BACKEND,
            )

        text = "".join(word.text for word in words)
        self.assertEqual(calls, [0.0, 28.0, 58.0])
        self.assertTrue(
            all(
                options["hotwords"]
                == "date de naissance, thème astral, oracle"
                for options in call_options
            )
        )
        self.assertNotIn("ancien-chevauchement", text)
        self.assertNotIn("ancienne-fin", text)
        self.assertIn("nouveau-chevauchement", text)
        self.assertIn("nouvelle-fin", text)
        self.assertEqual([item["committed_until_seconds"] for item in checkpoints], [30.0, 60.0, 65.0])
        self.assertEqual([position for _, position in partials], [30.0, 60.0, 65.0])
        self.assertEqual(checkpoints[-1]["stage"], "transcription_complete")
        self.assertEqual(language, "fr")
        self.assertIsNone(probability)
        self.assertEqual(model, OPENVINO_MODEL_ID)

    def test_failure_falls_back_and_reuses_the_last_checkpoint(self) -> None:
        resume = {
            "stage": "transcription",
            "model": OPENVINO_MODEL_ID,
            "committed_until_seconds": 3.0,
            "words": [
                {"start": 0.0, "end": 0.8, "text": " Conservé", "probability": None},
                {"start": 1.2, "end": 1.6, "text": " chevauchement", "probability": None},
            ],
        }
        segment = SimpleNamespace(
            start=0.1,
            end=0.5,
            text=" repris",
            words=[SimpleNamespace(start=0.1, end=0.5, word=" repris", probability=0.9)],
        )
        info = SimpleNamespace(language="fr", language_probability=0.99)
        warnings: list[str] = []
        fallback_order: list[str] = []

        class FakeBatchedPipeline:
            def __init__(self, model):
                self.model = model

            def transcribe(self, audio, **kwargs):
                return iter([segment]), info

        with (
            patch(
                "transcription_locale.openvino_backend.transcribe_openvino",
                side_effect=OpenVINOBackendUnavailable("GPU indisponible"),
            ),
            patch("transcription_locale.core._load_whisper_model", return_value=object()),
            patch("faster_whisper.BatchedInferencePipeline", FakeBatchedPipeline),
        ):
            words, _, _, model = _transcribe_words(
                [0.0] * 160_000,
                duration=10.0,
                profile="Équilibré",
                language="fr",
                initial_prompt="",
                progress=None,
                partial=None,
                resume_checkpoint=resume,
                checkpoint=None,
                asr_backend=OPENVINO_ARC_BACKEND,
                backend_warnings=warnings,
                before_cpu_fallback=lambda: fallback_order.append("diarisation-finie"),
            )

        self.assertEqual([word.text for word in words], [" Conservé", " repris"])
        self.assertEqual(model, "small")
        self.assertEqual(fallback_order, ["diarisation-finie"])
        self.assertEqual(len(warnings), 1)
        self.assertIn("Faster-Whisper a pris le relais", warnings[0])

    def test_restart_keeps_the_effective_engine_from_a_fallback_checkpoint(self) -> None:
        resume = {
            "stage": "transcription_complete",
            "model": "small",
            "committed_until_seconds": 10.0,
            "language": "fr",
            "language_probability": 0.98,
            "words": [
                {"start": 0.0, "end": 1.0, "text": " Déjà sauvegardé.", "probability": 0.9}
            ],
        }
        warnings: list[str] = []
        with patch(
            "transcription_locale.openvino_backend.transcribe_openvino",
            side_effect=AssertionError("OpenVINO ne doit pas redémarrer cet audio"),
        ):
            words, language, probability, model = _transcribe_words(
                [0.0] * 160_000,
                duration=10.0,
                profile="Équilibré",
                language="fr",
                initial_prompt="",
                progress=None,
                partial=None,
                resume_checkpoint=resume,
                checkpoint=None,
                asr_backend=OPENVINO_ARC_BACKEND,
                backend_warnings=warnings,
            )

        self.assertEqual([word.text for word in words], [" Déjà sauvegardé."])
        self.assertEqual((language, probability, model), ("fr", 0.98, "small"))
        self.assertIn("dernier checkpoint", warnings[0])


if __name__ == "__main__":
    unittest.main()
