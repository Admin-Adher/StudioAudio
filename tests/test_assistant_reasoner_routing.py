from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from transcription_locale.assistant_engine import (
    ASSISTANT_REQUIRED_MODEL_FILES,
    OpenVINOJsonReasoner,
    ReasonerRequest,
    get_default_reasoner,
    release_default_reasoners,
)


class _WorkingPipeline:
    def generate(self, prompt, **kwargs):
        return (
            '{"verdict":"Non retrouvé","confidence":0.7,'
            '"citation_ids":[]}'
        )


class _FailingGenerationPipeline:
    def generate(self, prompt, **kwargs):
        raise RuntimeError("Intel Arc indisponible pendant l'inférence")


class OpenVINODeviceFallbackTests(unittest.TestCase):
    def test_gpu_load_failure_falls_back_to_cpu_once(self) -> None:
        calls: list[str] = []

        def factory(path: str, device: str):
            calls.append(device)
            if device == "GPU":
                raise RuntimeError("Intel Arc non détectée")
            return _WorkingPipeline()

        with TemporaryDirectory() as directory:
            reasoner = OpenVINOJsonReasoner(
                Path(directory),
                device="GPU",
                pipeline_factory=factory,
            )
            result = reasoner.decide(
                ReasonerRequest(
                    question="Question",
                    claim="Question",
                    target_role=None,
                    candidates=(),
                )
            )
            reasoner.decide(
                ReasonerRequest(
                    question="Deuxième question",
                    claim="Deuxième question",
                    target_role=None,
                    candidates=(),
                )
            )

        self.assertIn('"verdict":"Non retrouvé"', result)
        self.assertEqual(calls, ["GPU", "CPU"])
        self.assertEqual(reasoner.active_device, "CPU")
        self.assertTrue(reasoner.fallback_used)
        self.assertIn("Intel Arc non détectée", reasoner.fallback_reason)

    def test_gpu_generation_failure_recreates_pipeline_on_cpu(self) -> None:
        calls: list[str] = []

        def factory(path: str, device: str):
            calls.append(device)
            return (
                _FailingGenerationPipeline()
                if device == "GPU"
                else _WorkingPipeline()
            )

        with TemporaryDirectory() as directory:
            reasoner = OpenVINOJsonReasoner(
                Path(directory),
                device="GPU",
                pipeline_factory=factory,
            )
            result = reasoner.decide(
                ReasonerRequest(
                    question="Question",
                    claim="Question",
                    target_role=None,
                    candidates=(),
                )
            )

        self.assertIn('"confidence":0.7', result)
        self.assertEqual(calls, ["GPU", "CPU"])
        self.assertEqual(reasoner.active_device, "CPU")
        self.assertTrue(reasoner.fallback_used)
        self.assertIn("inférence", reasoner.fallback_reason)

    def test_explicit_cpu_never_attempts_another_device(self) -> None:
        calls: list[str] = []

        def factory(path: str, device: str):
            calls.append(device)
            raise RuntimeError("CPU indisponible")

        with TemporaryDirectory() as directory:
            reasoner = OpenVINOJsonReasoner(
                Path(directory),
                device="CPU",
                pipeline_factory=factory,
            )
            with self.assertRaisesRegex(RuntimeError, "CPU indisponible"):
                reasoner.decide(
                    ReasonerRequest(
                        question="Question",
                        claim="Question",
                        target_role=None,
                        candidates=(),
                    )
                )

        self.assertEqual(calls, ["CPU"])
        self.assertFalse(reasoner.fallback_used)


class PlatformReasonerRoutingTests(unittest.TestCase):
    def tearDown(self) -> None:
        release_default_reasoners()

    def test_windows_routes_to_openvino_with_gpu_then_cpu_capability(self) -> None:
        with TemporaryDirectory() as directory:
            model_path = Path(directory)
            for name in ASSISTANT_REQUIRED_MODEL_FILES:
                (model_path / name).write_text("local", encoding="utf-8")
            with patch(
                "transcription_locale.assistant_engine.importlib.util.find_spec",
                return_value=object(),
            ):
                reasoner = get_default_reasoner(
                    model_path,
                    platform_name="win32",
                    machine_name="AMD64",
                    device="GPU",
                )

        self.assertIsInstance(reasoner, OpenVINOJsonReasoner)
        self.assertEqual(reasoner.device, "GPU")

    def test_intel_macos_does_not_route_to_windows_openvino_backend(self) -> None:
        with patch(
            "transcription_locale.assistant_engine.resolve_openvino_assistant_model_path",
            side_effect=AssertionError("OpenVINO ne doit pas être résolu sur macOS"),
        ):
            reasoner = get_default_reasoner(
                platform_name="darwin",
                machine_name="x86_64",
            )

        self.assertIsNone(reasoner)


if __name__ == "__main__":
    unittest.main()
