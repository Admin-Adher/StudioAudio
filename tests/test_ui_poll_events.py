from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr

from transcription_locale.ui import (
    build_app,
    poll_workspace_queue_ui,
    sync_workspace_ui,
)


class WorkspacePollEventTests(unittest.TestCase):
    @staticmethod
    def _queue_state(
        paths: dict[str, Path],
        *,
        queue_order: list[str] | None = None,
        completed: set[str] | None = None,
    ) -> dict[str, object]:
        completed_ids = completed or set()
        items: dict[str, object] = {}
        for item_id, path in paths.items():
            is_completed = item_id in completed_ids
            items[item_id] = {
                "id": item_id,
                "path": str(path),
                "source_path": str(path),
                "original_name": path.name,
                "name": path.name,
                "status": "Terminé" if is_completed else "En attente",
                "progress": 1.0 if is_completed else 0.0,
                "processing_time": "1 s" if is_completed else "—",
                "processing_seconds": 1.0 if is_completed else 0.0,
                "result": {"turns": []} if is_completed else None,
                "files": [],
                "comments": [],
            }
        return {
            "version": 6,
            "order": list(paths),
            "queue_order": list(queue_order if queue_order is not None else paths),
            "items": items,
        }

    def test_active_job_never_resynchronises_the_file_uploader(self) -> None:
        with (
            patch(
                "transcription_locale.ui.JOB_MANAGER.is_active",
                return_value=True,
            ),
            patch("transcription_locale.ui.load_workspace") as load_workspace,
        ):
            outputs = poll_workspace_queue_ui(["C:/cache/audio.wav"])

        self.assertEqual(outputs, (gr.skip(), gr.skip(), gr.skip()))
        load_workspace.assert_not_called()

    def test_first_upload_is_not_erased_before_it_reaches_the_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            uploaded = Path(directory) / "premier.wav"
            uploaded.write_bytes(b"audio")
            state = self._queue_state({}, queue_order=[])

            with (
                patch(
                    "transcription_locale.ui.JOB_MANAGER.is_active",
                    return_value=False,
                ),
                patch(
                    "transcription_locale.ui.JOB_MANAGER.active_item_ids",
                    return_value=set(),
                ),
                patch("transcription_locale.ui.load_workspace", return_value=state),
            ):
                outputs = poll_workspace_queue_ui([str(uploaded)])

        self.assertEqual(outputs, (gr.skip(), gr.skip(), gr.skip()))

    def test_first_upload_is_persisted_on_its_first_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            uploaded = Path(directory) / "premier.wav"
            uploaded.write_bytes(b"audio")
            state = self._queue_state({}, queue_order=[])

            with (
                patch("transcription_locale.ui.load_workspace", return_value=state),
                patch(
                    "transcription_locale.ui.import_audio",
                    return_value=("audio-1", uploaded),
                ) as import_audio,
                patch(
                    "transcription_locale.ui._save_workspace_from_ui",
                    side_effect=lambda current: current,
                ) as save_workspace,
                patch(
                    "transcription_locale.ui.JOB_MANAGER.active_item_ids",
                    return_value=set(),
                ),
            ):
                outputs = sync_workspace_ui(
                    [str(uploaded)],
                    {str(uploaded): "Premier essai.wav"},
                    state,
                )

        import_audio.assert_called_once_with(uploaded)
        saved_state = save_workspace.call_args.args[0]
        self.assertEqual(saved_state["queue_order"], ["audio-1"])
        self.assertEqual(saved_state["items"]["audio-1"]["name"], "Premier essai.wav")
        self.assertEqual(outputs[1][0][1], "Premier essai.wav")

    def test_deleted_file_is_never_reinserted_by_the_periodic_prune(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "a.wav"
            second = Path(directory) / "b.wav"
            first.write_bytes(b"a")
            second.write_bytes(b"b")
            state = self._queue_state({"a": first, "b": second})

            with (
                patch(
                    "transcription_locale.ui.JOB_MANAGER.is_active",
                    return_value=False,
                ),
                patch(
                    "transcription_locale.ui.JOB_MANAGER.active_item_ids",
                    return_value=set(),
                ),
                patch("transcription_locale.ui.load_workspace", return_value=state),
            ):
                outputs = poll_workspace_queue_ui([str(second)])

        self.assertEqual(outputs, (gr.skip(), gr.skip(), gr.skip()))

    def test_drag_reorder_is_never_undone_by_the_periodic_prune(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "a.wav"
            second = Path(directory) / "b.wav"
            first.write_bytes(b"a")
            second.write_bytes(b"b")
            state = self._queue_state({"a": first, "b": second})

            with (
                patch(
                    "transcription_locale.ui.JOB_MANAGER.is_active",
                    return_value=False,
                ),
                patch(
                    "transcription_locale.ui.JOB_MANAGER.active_item_ids",
                    return_value=set(),
                ),
                patch("transcription_locale.ui.load_workspace", return_value=state),
            ):
                outputs = poll_workspace_queue_ui([str(second), str(first)])

        self.assertEqual(outputs, (gr.skip(), gr.skip(), gr.skip()))

    def test_periodic_prune_only_removes_completed_visible_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed = Path(directory) / "managed-termine.wav"
            waiting = Path(directory) / "managed-attente.wav"
            uploaded_completed = Path(directory) / "upload-termine.wav"
            uploaded_waiting = Path(directory) / "upload-attente.wav"
            for path, payload in (
                (completed, b"done"),
                (waiting, b"waiting"),
                (uploaded_completed, b"done"),
                (uploaded_waiting, b"waiting"),
            ):
                path.write_bytes(payload)
            state = self._queue_state(
                {"done": completed, "waiting": waiting},
                completed={"done"},
            )
            state["items"]["done"]["source_path"] = str(uploaded_completed)
            state["items"]["waiting"]["source_path"] = str(uploaded_waiting)

            with (
                patch(
                    "transcription_locale.ui.JOB_MANAGER.is_active",
                    return_value=False,
                ),
                patch(
                    "transcription_locale.ui.JOB_MANAGER.active_item_ids",
                    return_value=set(),
                ),
                patch("transcription_locale.ui.load_workspace", return_value=state),
            ):
                outputs = poll_workspace_queue_ui(
                    [str(uploaded_completed), str(uploaded_waiting)]
                )

        self.assertEqual(outputs[0], [str(waiting)])
        self.assertIn("1 audio", outputs[1])

    def test_stale_ui_state_cannot_overwrite_a_completed_durable_result(self) -> None:
        audio_path = "C:/managed/audio.wav"
        result = {
            "duration": 42.0,
            "processing_seconds": 10.0,
            "turns": [],
        }
        durable_state = {
            "version": 4,
            "order": ["audio-1"],
            "queue_order": [],
            "items": {
                "audio-1": {
                    "id": "audio-1",
                    "path": audio_path,
                    "source_path": audio_path,
                    "original_name": "audio.wav",
                    "name": "Audio terminé.wav",
                    "status": "Terminé",
                    "progress": 1.0,
                    "processing_time": "10 s",
                    "processing_seconds": 10.0,
                    "result": result,
                    "files": ["C:/results/audio.zip"],
                    "comments": [],
                }
            },
        }
        stale_ui_state = {
            "version": 4,
            "order": ["audio-1"],
            "queue_order": ["audio-1"],
            "items": {
                "audio-1": {
                    "id": "audio-1",
                    "path": audio_path,
                    "name": "audio.wav",
                    "status": "En cours",
                    "progress": 0.02,
                    "processing_time": "0 s",
                    "processing_seconds": 0.0,
                    "result": None,
                    "files": [],
                    "comments": [],
                }
            },
        }

        with (
            patch(
                "transcription_locale.ui.load_workspace",
                return_value=durable_state,
            ),
            patch(
                "transcription_locale.ui.import_audio",
                return_value=("audio-1", audio_path),
            ),
            patch(
                "transcription_locale.ui._save_workspace_from_ui",
                side_effect=lambda state: state,
            ) as save_workspace,
            patch(
                "transcription_locale.ui.JOB_MANAGER.active_item_ids",
                return_value=set(),
            ),
        ):
            outputs = sync_workspace_ui(
                [audio_path],
                {audio_path: "audio.wav"},
                stale_ui_state,
            )

        saved_state = save_workspace.call_args.args[0]
        saved_item = saved_state["items"]["audio-1"]
        self.assertEqual(saved_item["status"], "Terminé")
        self.assertEqual(saved_item["progress"], 1.0)
        self.assertEqual(saved_item["processing_time"], "10 s")
        self.assertEqual(saved_item["result"], result)
        self.assertEqual(outputs[0]["items"]["audio-1"]["result"], result)

    def test_periodic_and_programmatic_queue_events_hide_gradio_progress(self) -> None:
        config = build_app().get_config_file()
        dependencies = config["dependencies"]
        components = {component["id"]: component for component in config["components"]}
        audio_queue_id = next(
            component_id
            for component_id, component in components.items()
            if component.get("props", {}).get("elem_id") == "audio-queue"
        )

        timer_dependencies = [
            dependency
            for dependency in dependencies
            if any(
                target_id in components
                and components[target_id].get("type") == "timer"
                and event_name == "tick"
                for target_id, event_name in dependency.get("targets", [])
            )
        ]
        self.assertTrue(timer_dependencies)
        for dependency in timer_dependencies:
            self.assertEqual(dependency.get("show_progress"), "hidden")
            self.assertEqual(dependency.get("show_progress_on"), [])

        queue_sync = next(
            dependency
            for dependency in dependencies
            if dependency.get("api_name") == "sync_queue_ui"
            and (audio_queue_id, "change") in dependency.get("targets", [])
        )
        workspace_sync = next(
            dependency
            for dependency in dependencies
            if dependency.get("api_name") == "sync_workspace_ui"
            and dependency.get("trigger_after") == queue_sync["id"]
        )
        for dependency in (queue_sync, workspace_sync):
            self.assertEqual(dependency.get("show_progress"), "hidden")
            self.assertEqual(dependency.get("show_progress_on"), [])


if __name__ == "__main__":
    unittest.main()
