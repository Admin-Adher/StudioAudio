from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from transcription_locale import workspace


class WorkspacePersistenceTests(unittest.TestCase):
    @patch("transcription_locale.core.sys.platform", "win32")
    def test_new_workspace_defaults_to_the_openvino_arc_pilot(self) -> None:
        state = workspace.blank_workspace()

        self.assertEqual(
            state["preferences"]["transcription_engine"],
            "openvino-arc-pilot",
        )
        self.assertEqual(state["queue_order"], [])

    def test_legacy_workspace_migrates_only_actionable_items_into_the_queue(self) -> None:
        migrated = workspace.normalise_workspace(
            {
                "version": 5,
                "order": ["done", "waiting"],
                "items": {
                    "done": {
                        "name": "Terminé.wav",
                        "status": "Terminé",
                        "result": {"turns": []},
                    },
                    "waiting": {
                        "name": "À traiter.wav",
                        "status": "En attente",
                        "result": None,
                    },
                },
            }
        )

        self.assertEqual(migrated["order"], ["done", "waiting"])
        self.assertEqual(migrated["queue_order"], ["waiting"])

    @patch("transcription_locale.core.sys.platform", "win32")
    def test_engine_preference_is_persisted_and_unknown_values_use_the_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "donnees"
            state_path = data_dir / "espace-travail.json"
            state = workspace.blank_workspace()
            state["preferences"]["transcription_engine"] = "openvino-arc-pilot"

            with (
                patch.object(workspace, "DATA_DIR", data_dir),
                patch.object(workspace, "WORKSPACE_PATH", state_path),
            ):
                workspace.save_workspace(state)
                restored = workspace.load_workspace()

            self.assertEqual(
                restored["preferences"]["transcription_engine"],
                "openvino-arc-pilot",
            )
            restored["preferences"]["transcription_engine"] = "moteur-inconnu"
            self.assertEqual(
                workspace.normalise_workspace(restored)["preferences"][
                    "transcription_engine"
                ],
                "openvino-arc-pilot",
            )

    @patch("transcription_locale.core.sys.platform", "win32")
    def test_existing_explicit_standard_choice_is_not_migrated_to_arc(self) -> None:
        legacy = {
            "version": 4,
            "order": [],
            "items": {},
            "preferences": {"transcription_engine": "faster-whisper"},
        }

        migrated = workspace.normalise_workspace(legacy)

        self.assertEqual(
            migrated["preferences"]["transcription_engine"],
            "faster-whisper",
        )

    @patch("transcription_locale.core.sys.platform", "win32")
    def test_existing_workspace_without_engine_preference_adopts_arc_default(self) -> None:
        legacy = {
            "version": 4,
            "order": [],
            "items": {},
            "preferences": {"future_preference": "kept"},
        }

        migrated = workspace.normalise_workspace(legacy)

        self.assertEqual(
            migrated["preferences"]["transcription_engine"],
            "openvino-arc-pilot",
        )
        self.assertEqual(migrated["preferences"]["future_preference"], "kept")

    @patch("transcription_locale.core.sys.platform", "darwin")
    def test_macos_defaults_to_standard_and_migrates_an_old_arc_preference(self) -> None:
        self.assertEqual(
            workspace.blank_workspace()["preferences"]["transcription_engine"],
            "faster-whisper",
        )
        legacy = {
            "version": 5,
            "order": [],
            "items": {},
            "preferences": {
                "transcription_engine": "openvino-arc-pilot",
                "future_preference": "kept",
            },
        }

        migrated = workspace.normalise_workspace(legacy)

        self.assertEqual(
            migrated["preferences"]["transcription_engine"],
            "faster-whisper",
        )
        self.assertEqual(migrated["preferences"]["future_preference"], "kept")

    def test_audio_import_is_deduplicated_by_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_a = root / "appel-original.wav"
            source_b = root / "appel-copie.wav"
            source_a.write_bytes(b"RIFF-audio-identique")
            source_b.write_bytes(source_a.read_bytes())
            library = root / "donnees" / "audios"

            with patch.object(workspace, "AUDIO_LIBRARY_DIR", library):
                first_id, first_path = workspace.import_audio(source_a)
                second_id, second_path = workspace.import_audio(source_b)
                stored_id, stored_path = workspace.import_audio(first_path)

            self.assertEqual(first_id, second_id)
            self.assertEqual(first_id, stored_id)
            self.assertEqual(first_path, second_path)
            self.assertEqual(first_path, stored_path)
            self.assertEqual(len(list(library.iterdir())), 1)

    def test_library_copy_can_be_renamed_without_touching_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "appel-original.wav"
            source.write_bytes(b"RIFF-audio")
            library = root / "donnees" / "audios"

            with patch.object(workspace, "AUDIO_LIBRARY_DIR", library):
                item_id, managed = workspace.import_audio(source)
                renamed = workspace.rename_library_audio(
                    item_id,
                    managed,
                    "Entretien client.wav",
                )

            self.assertTrue(source.is_file())
            self.assertFalse(managed.exists())
            self.assertEqual(renamed.name, "Entretien client.wav")
            self.assertEqual(renamed.read_bytes(), source.read_bytes())

    def test_workspace_item_deletion_keeps_the_original_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source-originale.wav"
            source.write_bytes(b"RIFF-audio")
            data_dir = root / "donnees"
            library = data_dir / "audios"
            state_path = data_dir / "espace-travail.json"

            with (
                patch.object(workspace, "DATA_DIR", data_dir),
                patch.object(workspace, "AUDIO_LIBRARY_DIR", library),
                patch.object(workspace, "WORKSPACE_PATH", state_path),
            ):
                item_id, managed = workspace.import_audio(source)
                state = workspace.blank_workspace()
                state["order"] = [item_id]
                state["items"] = {
                    item_id: {
                        "id": item_id,
                        "path": str(managed),
                        "source_path": str(source),
                        "name": "Entretien.wav",
                        "status": "En attente",
                    }
                }
                saved, deleted_name = workspace.delete_workspace_item(state, item_id)

            self.assertEqual(deleted_name, "Entretien.wav")
            self.assertTrue(source.is_file())
            self.assertFalse(managed.exists())
            self.assertEqual(saved["order"], [])
            self.assertEqual(saved["items"], {})

    def test_running_workspace_item_cannot_be_deleted(self) -> None:
        state = workspace.blank_workspace()
        state["order"] = ["audio-1"]
        state["items"] = {
            "audio-1": {
                "id": "audio-1",
                "path": "C:/managed/audio.wav",
                "name": "Entretien.wav",
                "status": "En cours",
            }
        }

        with self.assertRaisesRegex(ValueError, "en cours"):
            workspace.delete_workspace_item(state, "audio-1")

    def test_workspace_round_trip_preserves_running_state_for_job_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "donnees"
            state_path = data_dir / "espace-travail.json"
            state = workspace.blank_workspace()
            state["order"] = ["audio-1"]
            state["items"] = {
                "audio-1": {
                    "id": "audio-1",
                    "path": "C:/audio.wav",
                    "name": "Entretien client.wav",
                    "status": "En cours",
                    "progress": 0.42,
                    "comments": [
                        {
                            "id": "comment-1",
                            "author": "AuteurX",
                            "passage": "Un passage",
                            "body": "À vérifier",
                            "status": "À discuter",
                        }
                    ],
                }
            }

            with (
                patch.object(workspace, "DATA_DIR", data_dir),
                patch.object(workspace, "WORKSPACE_PATH", state_path),
            ):
                workspace.save_workspace(state)
                restored = workspace.load_workspace()

            item = restored["items"]["audio-1"]
            self.assertEqual(item["status"], "En cours")
            self.assertEqual(item["name"], "Entretien client.wav")
            self.assertEqual(item["comments"][0]["body"], "À vérifier")

    def test_version_three_comment_without_id_is_migrated_deterministically(self) -> None:
        legacy = {
            "version": 3,
            "order": ["audio-1"],
            "items": {
                "audio-1": {
                    "id": "audio-1",
                    "path": "C:/audio.wav",
                    "name": "Entretien.wav",
                    "status": "Terminé",
                    "created_at": "2026-07-21T09:00:00",
                    "comments": [
                        {
                            "author": "AuteurX",
                            "passage": "Un passage historique",
                            "body": "À confirmer",
                            "status": "À discuter",
                            "created_at": "21/07/2026 09:30",
                            "priority": "haute",
                        }
                    ],
                }
            },
        }

        first = workspace.normalise_workspace(legacy)
        second = workspace.normalise_workspace(legacy)
        migrated = first["items"]["audio-1"]["comments"][0]

        self.assertEqual(first["version"], workspace.WORKSPACE_VERSION)
        self.assertEqual(migrated["id"], second["items"]["audio-1"]["comments"][0]["id"])
        self.assertTrue(migrated["id"].startswith("comment-"))
        self.assertEqual(migrated["passage"], "Un passage historique")
        self.assertEqual(migrated["body"], "À confirmer")
        self.assertEqual(migrated["author"], "AuteurX")
        self.assertEqual(migrated["status"], "À discuter")
        self.assertEqual(
            migrated["anchor"],
            {
                "start_seconds": None,
                "end_seconds": None,
                "quote": "Un passage historique",
                "source": "legacy-text",
            },
        )
        self.assertEqual(migrated["replies"], [])
        self.assertEqual(migrated["updated_at"], "21/07/2026 09:30")
        self.assertIsNone(migrated["resolved_at"])
        self.assertEqual(migrated["priority"], "haute")

    def test_discussion_schema_round_trip_preserves_anchor_replies_and_unknown_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "donnees"
            state_path = data_dir / "espace-travail.json"
            state = workspace.blank_workspace()
            state["order"] = ["audio-1"]
            state["items"] = {
                "audio-1": {
                    "id": "audio-1",
                    "path": "C:/audio.wav",
                    "name": "Entretien.wav",
                    "status": "Terminé",
                    "created_at": "2026-07-22T10:00:00+02:00",
                    "comments": [
                        {
                            "id": "discussion-1",
                            "author": "AuteurX",
                            "passage": "Passage horodaté",
                            "body": "Votre avis ?",
                            "status": "resolved",
                            "anchor": {
                                "start_seconds": "12.5",
                                "end_seconds": 18.75,
                                "quote": "Passage horodaté",
                                "source": "transcript",
                                "speaker": "Client",
                            },
                            "created_at": "2026-07-22T10:05:00+02:00",
                            "updated_at": "2026-07-22T10:10:00+02:00",
                            "resolved_at": "2026-07-22T10:10:00+02:00",
                            "priority": "haute",
                            "replies": [
                                {
                                    "author": "Marie",
                                    "body": "Je confirme.",
                                    "created_at": "2026-07-22T10:07:00+02:00",
                                    "mentions": ["AuteurX"],
                                }
                            ],
                        }
                    ],
                }
            }

            with (
                patch.object(workspace, "DATA_DIR", data_dir),
                patch.object(workspace, "WORKSPACE_PATH", state_path),
            ):
                saved = workspace.save_workspace(state)
                restored = workspace.load_workspace()

        saved_comment = saved["items"]["audio-1"]["comments"][0]
        restored_comment = restored["items"]["audio-1"]["comments"][0]
        self.assertEqual(restored["version"], workspace.WORKSPACE_VERSION)
        self.assertEqual(restored_comment, saved_comment)
        self.assertEqual(restored_comment["status"], "Résolu")
        self.assertEqual(restored_comment["anchor"]["start_seconds"], 12.5)
        self.assertEqual(restored_comment["anchor"]["end_seconds"], 18.75)
        self.assertEqual(restored_comment["anchor"]["speaker"], "Client")
        self.assertEqual(restored_comment["priority"], "haute")
        reply = restored_comment["replies"][0]
        self.assertTrue(reply["id"].startswith("reply-"))
        self.assertEqual(reply["updated_at"], reply["created_at"])
        self.assertEqual(reply["mentions"], ["AuteurX"])

    def test_version_four_migration_adds_assistant_fields_without_data_loss(self) -> None:
        legacy = {
            "version": 4,
            "order": ["audio-1"],
            "items": {
                "audio-1": {
                    "id": "audio-1",
                    "path": "C:/audio.wav",
                    "name": "Consultation.wav",
                    "status": "TerminÃ©",
                    "future_item_field": {"kept": True},
                }
            },
            "preferences": {
                "transcription_engine": "faster-whisper",
                "future_preference": "kept",
            },
            "future_workspace_field": ["kept"],
        }

        migrated = workspace.normalise_workspace(legacy)
        item = migrated["items"]["audio-1"]

        self.assertEqual(migrated["version"], workspace.WORKSPACE_VERSION)
        self.assertEqual(migrated["future_workspace_field"], ["kept"])
        self.assertEqual(migrated["preferences"]["future_preference"], "kept")
        self.assertEqual(item["future_item_field"], {"kept": True})
        self.assertEqual(item["speaker_roles"], {})
        self.assertEqual(item["assistant_queries"], [])

    def test_assistant_history_round_trip_preserves_roles_results_and_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "donnees"
            state_path = data_dir / "espace-travail.json"
            state = workspace.blank_workspace()
            state["order"] = ["audio-1"]
            state["items"] = {
                "audio-1": {
                    "id": "audio-1",
                    "path": "C:/audio.wav",
                    "name": "Consultation.wav",
                    "status": "TerminÃ©",
                    "created_at": "2026-07-22T11:00:00+02:00",
                    "speaker_roles": {
                        "SPEAKER_00": "Client",
                        "SPEAKER_01": {
                            "role": "voyante",
                            "confidence": "0.94",
                            "source": "automatic",
                            "confirmed": True,
                            "badge_color": "sable",
                        },
                    },
                    "assistant_queries": [
                        {
                            "question": "Le Master a-t-il annoncé un déménagement ?",
                            "status": "completed",
                            "provider": "local",
                            "result": {
                                "verdict": "partially_confirmed",
                                "confidence": "0.82",
                                "answer": "Le déménagement est évoqué, sans date précise.",
                                "model": "qwen-local",
                                "citations": [
                                    {
                                        "start_seconds": "42.5",
                                        "end_seconds": 49,
                                        "speaker_id": "SPEAKER_01",
                                        "role": "Master",
                                        "quote": "Je vois un changement de logement.",
                                        "turn_index": 12,
                                    }
                                ],
                            },
                        }
                    ],
                }
            }

            with (
                patch.object(workspace, "DATA_DIR", data_dir),
                patch.object(workspace, "WORKSPACE_PATH", state_path),
            ):
                saved = workspace.save_workspace(state)
                restored = workspace.load_workspace()

        self.assertEqual(restored, saved)
        item = restored["items"]["audio-1"]
        self.assertEqual(item["speaker_roles"]["SPEAKER_00"]["role"], "Client")
        master = item["speaker_roles"]["SPEAKER_01"]
        self.assertEqual(master["role"], "Master")
        self.assertEqual(master["confidence"], 0.94)
        self.assertTrue(master["confirmed"])
        self.assertEqual(master["badge_color"], "sable")

        query = item["assistant_queries"][0]
        self.assertTrue(query["id"].startswith("query-"))
        self.assertEqual(query["provider"], "local")
        self.assertEqual(query["result"]["confidence"], 0.82)
        self.assertEqual(query["result"]["model"], "qwen-local")
        citation = query["result"]["citations"][0]
        self.assertTrue(citation["id"].startswith("citation-"))
        self.assertEqual(citation["start_seconds"], 42.5)
        self.assertEqual(citation["end_seconds"], 49.0)
        self.assertEqual(citation["turn_index"], 12)

    def test_assistant_export_helper_returns_json_safe_copy(self) -> None:
        item = {
            "id": "audio-1",
            "created_at": "2026-07-22T11:00:00+02:00",
            "speaker_roles": {"A": {"role": "client"}},
            "assistant_queries": [
                {
                    "question": "Question",
                    "debug_path": Path("C:/model/cache"),
                    "debug_score": float("nan"),
                    "result": {
                        "verdict": "not_found",
                        "answer": "Passage non retrouvé.",
                        "citations": [],
                    },
                }
            ],
        }

        exported = workspace.assistant_data_for_export(item)

        json.dumps(exported, ensure_ascii=False, allow_nan=False)
        self.assertEqual(exported["speaker_roles"]["A"]["role"], "Client")
        self.assertEqual(
            exported["assistant_queries"][0]["debug_path"],
            "C:\\model\\cache",
        )
        self.assertIsNone(exported["assistant_queries"][0]["debug_score"])

    def test_checkpoint_round_trip_keeps_words_and_position(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary) / "checkpoints"
            payload = {
                "version": 1,
                "stage": "transcription",
                "committed_until_seconds": 42.5,
                "words": [
                    {"start": 41.0, "end": 42.5, "text": " Bonjour", "probability": 0.98}
                ],
            }
            with patch.object(workspace, "CHECKPOINT_DIR", checkpoint_dir):
                path = workspace.save_checkpoint("audio-1", payload)
                restored = workspace.load_checkpoint("audio-1")
                workspace.delete_checkpoint("audio-1")

            self.assertEqual(restored, payload)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
