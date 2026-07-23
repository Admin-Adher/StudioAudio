from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from transcription_locale.assistant_engine import (
    ASSISTANT_MODEL_ENV,
    ConsultationRequest,
    get_default_reasoner,
)
from transcription_locale.mlx_assistant_backend import (
    MLX_ASSISTANT_MODEL_NAME,
    MLXJsonReasoner,
    create_default_mlx_reasoner,
    mlx_model_is_complete,
    resolve_mlx_model_path,
)


def _write_fake_mlx_model(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (root / name).write_text("{}", encoding="utf-8")
    (root / "model-00001-of-00001.safetensors").write_bytes(b"local-model")
    return root


class FakeTokenizer:
    def __init__(self) -> None:
        self.messages = None
        self.options = None

    def apply_chat_template(self, messages, **options):
        self.messages = messages
        self.options = options
        return "<chat>" + messages[-1]["content"] + "</chat>"


class MLXAssistantBackendTests(unittest.TestCase):
    def test_model_must_be_complete_and_local(self) -> None:
        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "model"
            model_path.mkdir()
            self.assertFalse(mlx_model_is_complete(model_path))
            with self.assertRaises(FileNotFoundError):
                MLXJsonReasoner(model_path, loader=lambda path: (object(), object()))

            _write_fake_mlx_model(model_path)
            self.assertTrue(mlx_model_is_complete(model_path))

    def test_resolver_prefers_bundled_model_then_installed_model(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            app_root = root / "app-data"
            resource_root = root / "bundle"
            installed = _write_fake_mlx_model(
                app_root / "modeles" / "assistant" / MLX_ASSISTANT_MODEL_NAME
            )
            bundled = _write_fake_mlx_model(
                resource_root / "modeles" / "assistant" / MLX_ASSISTANT_MODEL_NAME
            )

            resolved = resolve_mlx_model_path(
                environ={},
                app_data_root=app_root,
                resource_root=resource_root,
            )
            self.assertEqual(resolved, bundled.resolve())

            for child in bundled.iterdir():
                child.unlink()
            bundled.rmdir()
            resolved = resolve_mlx_model_path(
                environ={},
                app_data_root=app_root,
                resource_root=resource_root,
            )
            self.assertEqual(resolved, installed.resolve())

    def test_explicit_remote_repository_is_rejected_without_calling_loader(self) -> None:
        calls: list[str] = []
        reasoner = create_default_mlx_reasoner(
            "mlx-community/Qwen3-8B-4bit",
            loader=lambda path: calls.append(path) or (object(), FakeTokenizer()),
            generator=lambda *args, **kwargs: "{}",
            environ={},
        )
        self.assertIsNone(reasoner)
        self.assertEqual(calls, [])

    def test_stale_environment_path_falls_back_to_bundled_model(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            resource_root = root / "bundle"
            bundled = _write_fake_mlx_model(
                resource_root
                / "modeles"
                / "assistant"
                / MLX_ASSISTANT_MODEL_NAME
            )
            stale = root / "ancienne-installation" / MLX_ASSISTANT_MODEL_NAME

            resolved = resolve_mlx_model_path(
                environ={ASSISTANT_MODEL_ENV: str(stale)},
                app_data_root=root / "app-data",
                resource_root=resource_root,
            )
            explicitly_invalid = resolve_mlx_model_path(
                stale,
                environ={},
                app_data_root=root / "app-data",
                resource_root=resource_root,
            )

        self.assertEqual(resolved, bundled.resolve())
        self.assertIsNone(explicitly_invalid)

    def test_reasoner_is_lazy_applies_qwen_chat_template_and_returns_json(self) -> None:
        load_calls: list[str] = []
        generation_calls: list[dict[str, object]] = []
        tokenizer = FakeTokenizer()

        def loader(path: str):
            load_calls.append(path)
            return object(), tokenizer

        def generator(model, received_tokenizer, **kwargs):
            self.assertIs(received_tokenizer, tokenizer)
            generation_calls.append(kwargs)
            return (
                "<think>raisonnement privé</think>\n"
                '{"status":"answered","confidence":0.91,'
                '"answer":"La consultation reste calme.",'
                '"key_points":["Aucune escalade explicite."],'
                '"nuance":"Les mesures acoustiques sont nécessaires.",'
                '"citation_ids":["turn-00001"]}'
            )

        with TemporaryDirectory() as directory:
            model_path = _write_fake_mlx_model(Path(directory) / "model")
            reasoner = MLXJsonReasoner(
                model_path,
                loader=loader,
                generator=generator,
            )
            self.assertFalse(reasoner.loaded)
            request = ConsultationRequest(
                question="Est-ce que le ton monte ?",
                intent="consultation_qa",
                candidates=(),
                role_mapping_confirmed=True,
                audio_observations={"available": True, "loudness_change_db": 1.0},
            )
            raw = reasoner.answer(request)

        self.assertTrue(reasoner.loaded)
        self.assertEqual(load_calls, [str(model_path.resolve())])
        self.assertEqual(len(generation_calls), 1)
        self.assertEqual(generation_calls[0]["max_tokens"], 512)
        self.assertFalse(generation_calls[0]["verbose"])
        self.assertEqual(tokenizer.options["tokenize"], False)
        self.assertEqual(tokenizer.options["add_generation_prompt"], True)
        self.assertEqual(tokenizer.options["enable_thinking"], False)
        self.assertEqual(tokenizer.messages[0]["role"], "system")
        self.assertIn("objet JSON", tokenizer.messages[0]["content"])
        self.assertIn("Est-ce que le ton monte", tokenizer.messages[1]["content"])
        payload = json.loads(raw)
        self.assertEqual(payload["status"], "answered")
        self.assertNotIn("raisonnement privé", raw)

    def test_factory_accepts_injected_runtime_but_never_loads_eagerly(self) -> None:
        calls: list[str] = []
        with TemporaryDirectory() as directory:
            model_path = _write_fake_mlx_model(Path(directory) / "model")
            reasoner = create_default_mlx_reasoner(
                model_path,
                loader=lambda path: calls.append(path) or (object(), FakeTokenizer()),
                generator=lambda *args, **kwargs: '{"status":"insufficient"}',
                environ={},
            )
            self.assertIsInstance(reasoner, MLXJsonReasoner)
            self.assertFalse(reasoner.loaded)
            self.assertEqual(calls, [])

    def test_cross_platform_factory_dispatches_apple_silicon_to_mlx(self) -> None:
        sentinel = object()
        with patch(
            "transcription_locale.mlx_assistant_backend.create_default_mlx_reasoner",
            return_value=sentinel,
        ) as create_mlx:
            result = get_default_reasoner(
                "/tmp/local-qwen",
                platform_name="darwin",
                machine_name="arm64",
            )

        self.assertIs(result, sentinel)
        create_mlx.assert_called_once_with("/tmp/local-qwen")

    def test_offline_environment_is_not_modified_by_injected_loader(self) -> None:
        with TemporaryDirectory() as directory:
            model_path = _write_fake_mlx_model(Path(directory) / "model")
            with patch.dict(os.environ, {"HF_HUB_OFFLINE": "custom"}, clear=False):
                reasoner = MLXJsonReasoner(
                    model_path,
                    loader=lambda path: (object(), FakeTokenizer()),
                    generator=lambda *args, **kwargs: (
                        '{"status":"insufficient","confidence":0.5,'
                        '"answer":"Information insuffisante.",'
                        '"citation_ids":[]}'
                    ),
                )
                reasoner.answer(
                    ConsultationRequest(
                        question="Question",
                        intent="consultation_qa",
                        candidates=(),
                        role_mapping_confirmed=False,
                    )
                )
                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "custom")


if __name__ == "__main__":
    unittest.main()
