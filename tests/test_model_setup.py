from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import transcription_locale.model_setup as model_setup


class ModelSetupTests(unittest.TestCase):
    def test_windows_intel_gpu_requires_the_openvino_model(self) -> None:
        with (
            patch.object(model_setup.sys, "platform", "win32"),
            patch.object(
                model_setup,
                "openvino_gpu_runtime_available",
                return_value=True,
            ),
            patch.object(model_setup, "_openvino_model_is_cached", return_value=False),
            patch.object(model_setup, "_model_is_cached", return_value=True),
            patch.object(
                model_setup,
                "_wespeaker_model_path",
                return_value=Path(__file__),
            ),
        ):
            status = model_setup.setup_components_status()

        self.assertFalse(status["openvino"])

    def test_preparation_downloads_and_validates_openvino_before_marking_ready(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "openvino"
            assistant_dir = root / "assistant"
            marker = root / "modeles-prepares.json"
            stages: list[str] = []

            def fake_download(**kwargs):
                destination = kwargs["local_dir"]
                if kwargs["repo_id"] == model_setup.OPENVINO_MODEL_REPOSITORY:
                    self.assertEqual(destination, model_dir)
                    required = model_setup.REQUIRED_MODEL_FILES
                else:
                    self.assertEqual(
                        kwargs["repo_id"],
                        model_setup.ASSISTANT_MODEL_REPOSITORY,
                    )
                    self.assertEqual(destination, assistant_dir)
                    required = model_setup.ASSISTANT_REQUIRED_MODEL_FILES
                destination.mkdir(parents=True, exist_ok=True)
                for name in required:
                    (destination / name).write_text("test", encoding="utf-8")

            fake_wespeaker = SimpleNamespace(
                Speaker=lambda **_kwargs: object(),
            )
            fake_silero = SimpleNamespace(load_silero_vad=lambda: object())
            with (
                patch.object(model_setup, "MODEL_DIR", root),
                patch.object(model_setup, "WHISPER_MODEL_DIR", root / "whisper"),
                patch.object(model_setup, "SETUP_MARKER", marker),
                patch.object(model_setup, "DEFAULT_OPENVINO_MODEL_DIR", model_dir),
                patch.object(
                    model_setup,
                    "DEFAULT_ASSISTANT_MODEL_DIR",
                    assistant_dir,
                ),
                patch.object(model_setup, "_openvino_is_required", return_value=True),
                patch.object(
                    model_setup,
                    "_assistant_platform_kind",
                    return_value="openvino",
                ),
                patch.object(
                    model_setup,
                    "_assistant_model_is_cached",
                    return_value=False,
                ),
                patch.object(model_setup, "_load_whisper_model"),
                patch(
                    "huggingface_hub.snapshot_download",
                    side_effect=fake_download,
                ),
                patch.dict(
                    sys.modules,
                    {
                        "wespeakerruntime": fake_wespeaker,
                        "silero_vad": fake_silero,
                    },
                ),
            ):
                model_setup.prepare_all_models(
                    lambda label, _index, _total: stages.append(label)
                )

            payload = json.loads(marker.read_text(encoding="utf-8"))
            self.assertIn("Intel Arc · pilote", stages)
            self.assertIn("Assistant Premium · Qwen3 8B", stages)
            self.assertEqual(
                payload["openvino_model"],
                model_setup.OPENVINO_MODEL_REPOSITORY,
            )
            self.assertEqual(
                payload["assistant_model"],
                model_setup.ASSISTANT_MODEL_REPOSITORY,
            )
            self.assertEqual(payload["version"], model_setup.SETUP_VERSION)

    def test_bundled_mlx_model_satisfies_the_macos_assistant_stage(self) -> None:
        with (
            patch.object(model_setup.sys, "platform", "darwin"),
            patch.object(model_setup.platform, "machine", return_value="arm64"),
            patch.object(
                model_setup,
                "resolve_mlx_model_path",
                return_value=Path("/Studio Audio.app/qwen3"),
            ),
            patch.object(model_setup, "_model_is_cached", return_value=True),
            patch.object(
                model_setup,
                "_wespeaker_model_path",
                return_value=Path(__file__),
            ),
        ):
            status = model_setup.setup_components_status()

        self.assertTrue(status["assistant"])

    def test_downloaded_bytes_counts_an_assistant_embedded_in_resources(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            whisper = root / "whisper"
            openvino = root / "openvino"
            bundled_assistant = root / "resources" / "assistant"
            speaker = root / "speaker.onnx"
            whisper.mkdir()
            openvino.mkdir()
            bundled_assistant.mkdir(parents=True)
            (whisper / "model.bin").write_bytes(b"w" * 11)
            (openvino / "model.bin").write_bytes(b"o" * 13)
            (bundled_assistant / "config.json").write_bytes(b"c" * 17)
            (bundled_assistant / "model.bin").write_bytes(b"m" * 19)
            speaker.write_bytes(b"s" * 23)

            with (
                patch.object(model_setup, "WHISPER_MODEL_DIR", whisper),
                patch.object(
                    model_setup,
                    "DEFAULT_OPENVINO_MODEL_DIR",
                    openvino,
                ),
                patch.object(
                    model_setup,
                    "DEFAULT_ASSISTANT_MODEL_DIR",
                    root / "app-data-openvino-assistant",
                ),
                patch.object(
                    model_setup,
                    "DEFAULT_MLX_ASSISTANT_MODEL_DIR",
                    root / "app-data-mlx-assistant",
                ),
                patch.object(
                    model_setup,
                    "resolve_openvino_assistant_model_path",
                    return_value=bundled_assistant,
                ),
                patch.object(
                    model_setup,
                    "resolve_mlx_model_path",
                    return_value=None,
                ),
                patch.object(
                    model_setup,
                    "_wespeaker_model_path",
                    return_value=speaker,
                ),
            ):
                total = model_setup.downloaded_bytes()

        self.assertEqual(total, 11 + 13 + 17 + 19 + 23)

    def test_downloaded_bytes_does_not_count_the_same_assistant_twice(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assistant = root / "assistant"
            assistant.mkdir()
            (assistant / "model.bin").write_bytes(b"a" * 29)

            with (
                patch.object(
                    model_setup,
                    "WHISPER_MODEL_DIR",
                    root / "missing-whisper",
                ),
                patch.object(
                    model_setup,
                    "DEFAULT_OPENVINO_MODEL_DIR",
                    root / "missing-openvino",
                ),
                patch.object(
                    model_setup,
                    "DEFAULT_ASSISTANT_MODEL_DIR",
                    assistant,
                ),
                patch.object(
                    model_setup,
                    "DEFAULT_MLX_ASSISTANT_MODEL_DIR",
                    root / "missing-mlx",
                ),
                patch.object(
                    model_setup,
                    "resolve_openvino_assistant_model_path",
                    return_value=assistant,
                ),
                patch.object(
                    model_setup,
                    "resolve_mlx_model_path",
                    return_value=None,
                ),
                patch.object(
                    model_setup,
                    "_wespeaker_model_path",
                    return_value=root / "missing-speaker",
                ),
            ):
                total = model_setup.downloaded_bytes()

        self.assertEqual(total, 29)


if __name__ == "__main__":
    unittest.main()
