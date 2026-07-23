from __future__ import annotations

import http.server
import io
import json
import os
import socket
import sys
import tempfile
import threading
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from transcription_locale.desktop import (
    ASSISTANT_SMOKE_TEST_FLAG,
    AlreadyRunningError,
    AppInstanceLock,
    DesktopController,
    DesktopSmokeError,
    SMOKE_IMPORTS,
    configure_desktop_environment,
    find_available_port,
    main,
    run_desktop_smoke_test,
    run_assistant_smoke_test,
    validate_desktop_runtime,
    wait_for_loopback_server,
)


class DesktopRuntimeTests(unittest.TestCase):
    def test_desktop_environment_uses_only_the_selected_data_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict(os.environ, {}, clear=True):
                directories = configure_desktop_environment(root)
                self.assertEqual(directories["root"], root)
                self.assertEqual(os.environ["TRANSCRIPTION_APP_DATA_DIR"], str(root))
                self.assertEqual(
                    os.environ["WESPEAKER_HOME"],
                    str(root / "modeles" / "wespeaker"),
                )
                self.assertEqual(
                    os.environ["GRADIO_TEMP_DIR"],
                    str(root / "cache" / "gradio"),
                )
            for name in ("donnees", "modeles", "resultats", "cache", "logs"):
                self.assertTrue((root / name).is_dir())

    def test_selected_port_is_bound_to_loopback_and_available(self) -> None:
        port = find_available_port()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", port))

    def test_instance_lock_rejects_a_second_process_handle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "desktop.lock"
            first = AppInstanceLock(path)
            second = AppInstanceLock(path)
            first.acquire()
            try:
                with self.assertRaises(AlreadyRunningError):
                    second.acquire()
            finally:
                first.release()
            second.acquire()
            second.release()

    def test_runtime_validation_reports_a_missing_groovy_version(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "assets").mkdir()
            (root / "assets" / "app-icon.png").write_bytes(b"icon")
            groovy_directory = root / "groovy"
            groovy_directory.mkdir()
            with patch(
                "transcription_locale.desktop._package_directory",
                return_value=groovy_directory,
            ):
                with self.assertRaisesRegex(DesktopSmokeError, "groovy.*version.txt"):
                    validate_desktop_runtime(root)

    def test_runtime_validation_imports_every_desktop_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "assets").mkdir()
            (root / "assets" / "app-icon.png").write_bytes(b"icon")
            groovy_directory = root / "groovy"
            groovy_directory.mkdir()
            (groovy_directory / "version.txt").write_text("0.1.2", encoding="utf-8")
            with (
                patch(
                    "transcription_locale.desktop._package_directory",
                    return_value=groovy_directory,
                ),
                patch("transcription_locale.desktop.importlib.import_module") as importer,
            ):
                report = validate_desktop_runtime(root)
            self.assertEqual(report["groovy_version"], "0.1.2")
            self.assertEqual(
                [call.args[0] for call in importer.call_args_list],
                list(SMOKE_IMPORTS),
            )

    def test_controller_starts_the_real_loopback_configuration_and_closes(self) -> None:
        class FakeApp:
            def __init__(self) -> None:
                self.queue_kwargs: dict[str, object] = {}
                self.launch_kwargs: dict[str, object] = {}
                self.close_count = 0

            def queue(self, **kwargs: object) -> None:
                self.queue_kwargs = kwargs

            def launch(self, **kwargs: object) -> None:
                self.launch_kwargs = kwargs

            def close(self) -> None:
                self.close_count += 1

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = configure_desktop_environment(root)
            results_dir = root / "resultats"
            data_dir = root / "donnees"
            assets_dir = root / "assets"
            assets_dir.mkdir()
            fake_app = FakeApp()

            exports_module = types.ModuleType("transcription_locale.exports")
            exports_module.RESULTS_DIR = results_dir
            ui_module = types.ModuleType("transcription_locale.ui")
            ui_module.ASSETS_DIR = assets_dir
            ui_module.CUSTOM_CSS = "/* test */"
            ui_module.build_app = lambda: fake_app
            workspace_module = types.ModuleType("transcription_locale.workspace")
            workspace_module.DATA_DIR = data_dir

            with patch.dict(
                sys.modules,
                {
                    "transcription_locale.exports": exports_module,
                    "transcription_locale.ui": ui_module,
                    "transcription_locale.workspace": workspace_module,
                },
            ):
                controller = DesktopController(directories, root / "application.log")
                self.assertTrue(controller.start_server())

            self.assertEqual(fake_app.queue_kwargs["default_concurrency_limit"], 1)
            self.assertEqual(fake_app.launch_kwargs["server_name"], "127.0.0.1")
            self.assertEqual(fake_app.launch_kwargs["server_port"], controller.port)
            self.assertFalse(fake_app.launch_kwargs["inbrowser"])
            self.assertFalse(fake_app.launch_kwargs["share"])
            self.assertTrue(fake_app.launch_kwargs["prevent_thread_lock"])
            self.assertTrue(controller.session_path.is_file())
            session = json.loads(controller.session_path.read_text(encoding="utf-8"))
            self.assertEqual(session["url"], controller.url)
            self.assertTrue(controller.stop())
            self.assertEqual(fake_app.close_count, 1)
            self.assertFalse(controller.session_path.exists())

    def test_loopback_probe_reaches_a_local_http_server(self) -> None:
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - API de BaseHTTPRequestHandler
                self.send_response(204)
                self.end_headers()

            def log_message(self, *_: object) -> None:
                return

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/"
            self.assertEqual(wait_for_loopback_server(url, timeout_seconds=2), 204)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_loopback_probe_refuses_a_non_local_url(self) -> None:
        with self.assertRaisesRegex(DesktopSmokeError, "boucle locale"):
            wait_for_loopback_server("https://example.com", timeout_seconds=0.1)

    def test_main_smoke_mode_does_not_start_a_native_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = {"root": root, "logs": root}
            with (
                patch(
                    "transcription_locale.desktop.configure_desktop_environment",
                    return_value=directories,
                ),
                patch(
                    "transcription_locale.desktop.configure_logging",
                    return_value=root / "application.log",
                ),
                patch(
                    "transcription_locale.desktop.run_desktop_smoke_test",
                    return_value=0,
                ) as smoke,
            ):
                self.assertEqual(main(["--smoke-test"]), 0)
            smoke.assert_called_once_with(directories, root / "application.log")

    def test_main_assistant_smoke_mode_does_not_start_a_native_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = {"root": root, "logs": root / "logs"}
            with (
                patch(
                    "transcription_locale.desktop.configure_desktop_environment",
                    return_value=directories,
                ),
                patch(
                    "transcription_locale.desktop.configure_logging",
                    return_value=root / "application.log",
                ),
                patch(
                    "transcription_locale.desktop.run_assistant_smoke_test",
                    return_value=0,
                ) as assistant_smoke,
                patch(
                    "transcription_locale.desktop.run_desktop_smoke_test",
                ) as desktop_smoke,
            ):
                self.assertEqual(main([ASSISTANT_SMOKE_TEST_FLAG]), 0)
            assistant_smoke.assert_called_once_with(
                directories,
                root / "application.log",
            )
            desktop_smoke.assert_not_called()

    def test_assistant_smoke_loads_qwen_and_writes_a_structured_report(self) -> None:
        class FakeOpenVINOReasoner:
            name = "OpenVINO-local"
            model_path = Path("C:/modeles/qwen")
            device = "GPU"
            active_device = "GPU"
            max_new_tokens = 512

            def __init__(self) -> None:
                self.prompt = ""
                self.schema: object | None = None

            def _generate_json(
                self,
                prompt: str,
                *,
                json_schema: object,
            ) -> str:
                self.prompt = prompt
                self.schema = json_schema
                self.assert_offline = os.environ["HF_HUB_OFFLINE"]
                return '{"ready":true}'

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reasoner = FakeOpenVINOReasoner()
            output = io.StringIO()
            with redirect_stdout(output):
                code = run_assistant_smoke_test(
                    {"logs": root / "logs"},
                    root / "application.log",
                    reasoner_factory=lambda: reasoner,
                )

            self.assertEqual(code, 0)
            self.assertEqual(reasoner.max_new_tokens, 64)
            self.assertEqual(reasoner.assert_offline, "1")
            self.assertIn('{"ready":true}', reasoner.prompt)
            self.assertIsNotNone(reasoner.schema)
            report = json.loads(
                (root / "logs" / "assistant-smoke.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["mode"], "assistant-smoke")
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["backend"], "OpenVINO-local")
            self.assertEqual(report["device"], "GPU")
            self.assertEqual(report["model"], str(reasoner.model_path))
            self.assertEqual(report["response"], {"ready": True})
            self.assertEqual(json.loads(output.getvalue()), report)

    def test_assistant_smoke_reports_an_unavailable_packaged_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with redirect_stdout(io.StringIO()):
                code = run_assistant_smoke_test(
                    {"logs": root / "logs"},
                    root / "application.log",
                    reasoner_factory=lambda: None,
                )
            report = json.loads(
                (root / "logs" / "assistant-smoke.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(code, 5)
            self.assertEqual(report["status"], "error")
            self.assertIn("Aucun backend Qwen", report["error"])

    def test_assistant_smoke_uses_the_short_mlx_generation_path(self) -> None:
        class FakeTokenizer:
            def apply_chat_template(
                self,
                _messages: object,
                **_kwargs: object,
            ) -> str:
                return "prompt-local"

        class FakeMLXReasoner:
            name = "MLX-local"
            model_path = Path("/Applications/Studio Audio/Qwen")
            max_new_tokens = 512

            def __init__(self) -> None:
                self.generator_kwargs: dict[str, object] = {}
                self._generator = self.generate

            def _load(self) -> tuple[object, object]:
                return object(), FakeTokenizer()

            def generate(
                self,
                _model: object,
                _tokenizer: object,
                **kwargs: object,
            ) -> str:
                self.generator_kwargs = kwargs
                return '{"ready":true}'

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reasoner = FakeMLXReasoner()
            with redirect_stdout(io.StringIO()):
                code = run_assistant_smoke_test(
                    {"logs": root / "logs"},
                    root / "application.log",
                    reasoner_factory=lambda: reasoner,
                )
            report = json.loads(
                (root / "logs" / "assistant-smoke.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(code, 0)
            self.assertEqual(reasoner.generator_kwargs["max_tokens"], 64)
            self.assertEqual(reasoner.generator_kwargs["prompt"], "prompt-local")
            self.assertEqual(report["backend"], "MLX-local")
            self.assertEqual(report["device"], "METAL")

    def test_smoke_runner_reports_success_only_after_clean_shutdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            class FakeController:
                url = "http://127.0.0.1:54321"
                session_path = root / "desktop-session.json"

                def __init__(self, *_: object) -> None:
                    self.started = False
                    self.stopped = False

                def start_server(self) -> bool:
                    self.started = True
                    return True

                def stop(self) -> bool:
                    self.stopped = True
                    return True

            controller = FakeController()
            output = io.StringIO()
            with (
                patch(
                    "transcription_locale.desktop.DesktopController",
                    return_value=controller,
                ),
                patch(
                    "transcription_locale.desktop.validate_desktop_runtime",
                    return_value={"groovy_version": "0.1.2"},
                ),
                patch(
                    "transcription_locale.desktop.wait_for_loopback_server",
                    return_value=200,
                ),
                redirect_stdout(output),
            ):
                code = run_desktop_smoke_test(
                    {"root": root},
                    root / "application.log",
                    timeout_seconds=1,
                )
            report = json.loads(output.getvalue())
            self.assertEqual(code, 0)
            self.assertTrue(controller.started)
            self.assertTrue(controller.stopped)
            self.assertEqual(report["status"], "ok")
            self.assertTrue(report["clean_shutdown"])


if __name__ == "__main__":
    unittest.main()
