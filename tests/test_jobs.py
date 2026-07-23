from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from transcription_locale import jobs, workspace
from transcription_locale.core import TranscriptTurn, TranscriptionResult


class DurableJobManagerTests(unittest.TestCase):
    @patch("transcription_locale.core.sys.platform", "win32")
    def test_openvino_settings_record_the_requested_runtime_version(self) -> None:
        settings = jobs.JobSettings(
            profile="Équilibré",
            language="fr",
            requested_speakers=2,
            speaker_names=("Interlocuteur 1", "Interlocuteur 2"),
            engine="openvino-arc-pilot",
        )
        with patch("transcription_locale.jobs.version", return_value="2026.2.1") as mocked:
            serialized = settings.to_dict()

        mocked.assert_called_once_with("openvino-genai")
        self.assertEqual(serialized["engine"], "openvino-arc-pilot")
        self.assertEqual(serialized["engine_version"], "2026.2.1")

    @patch("transcription_locale.core.sys.platform", "darwin")
    def test_macos_normalises_saved_and_new_arc_job_settings(self) -> None:
        settings = jobs.JobSettings(
            profile="Équilibré",
            language="fr",
            requested_speakers=2,
            speaker_names=("Interlocuteur 1", "Interlocuteur 2"),
            engine="openvino-arc-pilot",
            engine_version="ancienne-version-openvino",
        )
        with patch("transcription_locale.jobs.version", return_value="1.2.3") as mocked:
            serialized = settings.to_dict()
            restored = jobs.JobSettings.from_dict(serialized)

        self.assertEqual(serialized["engine"], "faster-whisper")
        self.assertEqual(serialized["engine_version"], "1.2.3")
        self.assertIsNotNone(restored)
        self.assertEqual(restored.engine, "faster-whisper")
        self.assertEqual(restored.engine_version, "1.2.3")
        mocked.assert_called_once_with("faster-whisper")

    def test_job_persists_a_live_processing_anchor_before_pipeline_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "donnees"
            state_path = data_dir / "espace-travail.json"
            checkpoint_dir = data_dir / "checkpoints"
            source = root / "entretien.wav"
            source.write_bytes(b"RIFF-audio")
            item_id = "audio-clock"
            state = workspace.blank_workspace()
            state["order"] = [item_id]
            state["items"] = {
                item_id: {
                    "id": item_id,
                    "path": str(source),
                    "source_path": str(source),
                    "name": "Entretien.wav",
                    "status": "En attente",
                    "processing_seconds": 37.5,
                    "processing_time": "38 s",
                }
            }
            pipeline_started = threading.Event()
            release_pipeline = threading.Event()

            def fake_pipeline(path, **kwargs):
                pipeline_started.set()
                release_pipeline.wait(5)
                return TranscriptionResult(
                    audio_name=Path(path).name,
                    duration=10.0,
                    processing_seconds=1.0,
                    profile="Équilibré",
                    model="small",
                    language="fr",
                    language_probability=1.0,
                    requested_speakers=2,
                    detected_speakers=1,
                    turns=[
                        TranscriptTurn(
                            0.0,
                            10.0,
                            "SPEAKER_00",
                            "Interlocuteur 1",
                            "Texte terminé.",
                        )
                    ],
                )

            settings = jobs.JobSettings(
                profile="Équilibré",
                language="fr",
                requested_speakers=2,
                speaker_names=("Interlocuteur 1", "Interlocuteur 2"),
            )
            manager = jobs.TranscriptionJobManager()
            with (
                patch.object(workspace, "DATA_DIR", data_dir),
                patch.object(workspace, "WORKSPACE_PATH", state_path),
                patch.object(workspace, "CHECKPOINT_DIR", checkpoint_dir),
                patch("transcription_locale.jobs.run_pipeline", side_effect=fake_pipeline),
                patch(
                    "transcription_locale.jobs.release_default_reasoners"
                ) as release_reasoners,
                patch(
                    "transcription_locale.jobs.write_exports",
                    return_value=[root / "entretien.zip"],
                ),
            ):
                workspace.save_workspace(state)
                self.assertEqual(manager.enqueue([item_id], settings), [item_id])
                self.assertTrue(pipeline_started.wait(5))

                running = workspace.load_workspace()["items"][item_id]
                self.assertEqual(running["status"], "En cours")
                self.assertEqual(running["processing_seconds"], 37.5)
                self.assertEqual(running["processing_base_seconds"], 37.5)
                self.assertTrue(running["processing_started_at"].endswith("+00:00"))

                release_pipeline.set()
                self.assertTrue(manager.wait_until_idle(5))
                completed = workspace.load_workspace()["items"][item_id]

            release_reasoners.assert_called_once_with()
            self.assertEqual(completed["status"], "Terminé")
            self.assertGreaterEqual(completed["processing_seconds"], 37.5)
            self.assertEqual(
                completed["processing_base_seconds"],
                completed["processing_seconds"],
            )
            self.assertEqual(completed["processing_started_at"], "")

    @patch("transcription_locale.core.sys.platform", "win32")
    def test_background_job_survives_ui_detach_and_rejects_duplicate_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "donnees"
            state_path = data_dir / "espace-travail.json"
            checkpoint_dir = data_dir / "checkpoints"
            source = root / "entretien.wav"
            source.write_bytes(b"RIFF-audio")
            item_id = "audio-1"
            state = workspace.blank_workspace()
            state["order"] = [item_id]
            state["items"] = {
                item_id: {
                    "id": item_id,
                    "path": str(source),
                    "source_path": str(source),
                    "name": "Entretien.wav",
                    "status": "En attente",
                }
            }
            started = threading.Event()
            release = threading.Event()
            received_engines: list[object] = []

            def fake_pipeline(path, **kwargs):
                received_engines.append(kwargs.get("asr_backend"))
                kwargs["progress"](0.25, "Transcription")
                kwargs["checkpoint"](
                    {
                        "version": 1,
                        "stage": "transcription",
                        "committed_until_seconds": 12.0,
                        "model": "small",
                        "language": "fr",
                        "language_probability": 1.0,
                        "partial": "Bonjour, ceci est sauvegardé.",
                        "words": [
                            {
                                "start": 11.0,
                                "end": 12.0,
                                "text": " sauvegardé.",
                                "probability": 0.95,
                            }
                        ],
                    }
                )
                started.set()
                release.wait(5)
                return TranscriptionResult(
                    audio_name=Path(path).name,
                    duration=20.0,
                    processing_seconds=1.0,
                    profile="Équilibré",
                    model="small",
                    language="fr",
                    language_probability=1.0,
                    requested_speakers=2,
                    detected_speakers=1,
                    turns=[
                        TranscriptTurn(
                            0.0,
                            20.0,
                            "SPEAKER_00",
                            "Interlocuteur 1",
                            "Bonjour, ceci est terminé.",
                        )
                    ],
                )

            settings = jobs.JobSettings(
                profile="Équilibré",
                language="fr",
                requested_speakers=2,
                speaker_names=("Interlocuteur 1", "Interlocuteur 2"),
                engine="openvino-arc-pilot",
            )
            manager = jobs.TranscriptionJobManager()
            with (
                patch.object(workspace, "DATA_DIR", data_dir),
                patch.object(workspace, "WORKSPACE_PATH", state_path),
                patch.object(workspace, "CHECKPOINT_DIR", checkpoint_dir),
                patch("transcription_locale.jobs.run_pipeline", side_effect=fake_pipeline),
                patch(
                    "transcription_locale.jobs.write_exports",
                    return_value=[root / "entretien.zip"],
                ),
            ):
                workspace.save_workspace(state)
                self.assertEqual(manager.enqueue([item_id], settings), [item_id])
                self.assertTrue(started.wait(5))

                # Simule une nouvelle page qui relit le disque pendant que le
                # worker, indépendant de la requête UI, poursuit son calcul.
                refreshed = workspace.load_workspace()["items"][item_id]
                self.assertEqual(refreshed["status"], "En cours")
                self.assertEqual(refreshed["checkpoint_position"], 12.0)
                self.assertIn("sauvegardé", refreshed["partial"])
                self.assertEqual(manager.enqueue([item_id], settings), [])

                release.set()
                self.assertTrue(manager.wait_until_idle(5))
                completed_workspace = workspace.load_workspace()
                completed = completed_workspace["items"][item_id]
                self.assertEqual(completed["status"], "Terminé")
                self.assertEqual(completed["progress"], 1.0)
                self.assertIn("terminé", completed["partial"])
                self.assertNotIn(item_id, completed_workspace["queue_order"])
                self.assertIsNone(workspace.load_checkpoint(item_id))
                self.assertEqual(received_engines, ["openvino-arc-pilot"])

    def test_orphaned_job_resumes_from_checkpoint_after_process_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "donnees"
            state_path = data_dir / "espace-travail.json"
            checkpoint_dir = data_dir / "checkpoints"
            source = root / "entretien.wav"
            source.write_bytes(b"RIFF-audio")
            item_id = "audio-2"
            settings = jobs.JobSettings(
                profile="Équilibré",
                language="fr",
                requested_speakers=2,
                speaker_names=("Interlocuteur 1", "Interlocuteur 2"),
            )
            state = workspace.blank_workspace()
            state["order"] = [item_id]
            state["items"] = {
                item_id: {
                    "id": item_id,
                    "path": str(source),
                    "name": "Entretien.wav",
                    "status": "En cours",
                    "progress": 0.35,
                    "settings": settings.to_dict(),
                }
            }
            checkpoint = {
                "version": 1,
                "audio_id": item_id,
                "settings_hash": settings.fingerprint(),
                "stage": "transcription",
                "committed_until_seconds": 8.0,
                "model": "small",
                "language": "fr",
                "partial": "Début conservé.",
                "words": [],
            }
            received: list[object] = []

            def fake_pipeline(path, **kwargs):
                received.append(kwargs.get("resume_checkpoint"))
                return TranscriptionResult(
                    audio_name=Path(path).name,
                    duration=10.0,
                    processing_seconds=1.0,
                    profile="Équilibré",
                    model="small",
                    language="fr",
                    language_probability=1.0,
                    requested_speakers=2,
                    detected_speakers=1,
                    turns=[
                        TranscriptTurn(
                            0.0,
                            10.0,
                            "SPEAKER_00",
                            "Interlocuteur 1",
                            "Texte repris.",
                        )
                    ],
                )

            manager = jobs.TranscriptionJobManager()
            with (
                patch.object(workspace, "DATA_DIR", data_dir),
                patch.object(workspace, "WORKSPACE_PATH", state_path),
                patch.object(workspace, "CHECKPOINT_DIR", checkpoint_dir),
                patch("transcription_locale.jobs.run_pipeline", side_effect=fake_pipeline),
                patch(
                    "transcription_locale.jobs.write_exports",
                    return_value=[root / "entretien.zip"],
                ),
            ):
                workspace.save_workspace(state)
                workspace.save_checkpoint(item_id, checkpoint)
                self.assertEqual(manager.recover_orphaned(), [item_id])
                self.assertTrue(manager.wait_until_idle(5))
                completed = workspace.load_workspace()["items"][item_id]

            self.assertEqual(received[0]["committed_until_seconds"], 8.0)
            self.assertEqual(completed["status"], "Terminé")
            self.assertEqual(completed["resume_count"], 1)

    def test_recovery_does_not_interrupt_a_job_owned_by_the_current_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "donnees"
            state_path = data_dir / "espace-travail.json"
            checkpoint_dir = data_dir / "checkpoints"
            source = root / "entretien.wav"
            source.write_bytes(b"RIFF-audio")
            item_id = "audio-live"
            settings = jobs.JobSettings(
                profile="Équilibré",
                language="fr",
                requested_speakers=2,
                speaker_names=("Interlocuteur 1", "Interlocuteur 2"),
            )
            state = workspace.blank_workspace()
            state["order"] = [item_id]
            state["items"] = {
                item_id: {
                    "id": item_id,
                    "path": str(source),
                    "name": "Entretien.wav",
                    "status": "En cours",
                    "progress": 0.94,
                    "settings": settings.to_dict(),
                    "owner_pid": jobs.os.getpid(),
                    "owner_session": jobs._PROCESS_SESSION_ID,
                }
            }
            manager = jobs.TranscriptionJobManager()
            with (
                patch.object(workspace, "DATA_DIR", data_dir),
                patch.object(workspace, "WORKSPACE_PATH", state_path),
                patch.object(workspace, "CHECKPOINT_DIR", checkpoint_dir),
            ):
                workspace.save_workspace(state)
                self.assertEqual(manager.recover_orphaned(), [])
                current = workspace.load_workspace()["items"][item_id]

            self.assertEqual(current["status"], "En cours")
            self.assertEqual(current["progress"], 0.94)


if __name__ == "__main__":
    unittest.main()
