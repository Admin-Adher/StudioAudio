from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr

from transcription_locale.core import TranscriptTurn, TranscriptionResult
from transcription_locale.ui import (
    DESKTOP_VISIBILITY_SYNC_JS,
    LIBRARY_SORTS,
    OPENVINO_PILOT_ENGINE,
    PROCESSING_CLOCK_JS,
    _assistant_empty_html,
    _assistant_history_html,
    _assistant_result_html,
    _assistant_role_proposal,
    _comment_anchor,
    _comments_feed_html,
    _effective_engine_choice,
    _engine_choices,
    _engine_settings_copy,
    _engine_status_html,
    _fiche_header_html,
    _item_transcript,
    _role_summary_html,
    _role_voice_excerpt_html,
    _workspace_queue_rows,
    _workspace_status_markdown,
    _workspace_library_rows,
    add_comment_ui,
    add_reply_ui,
    capture_selected_passage_ui,
    delete_audio_by_target_ui,
    handle_comment_feed_action_ui,
    poll_workspace_ui,
    prepare_discussion_from_assistant_ui,
    preview_queue_ui,
    rename_audio_by_target_ui,
    rename_audio_ui,
    restore_engine_preference_ui,
    run_assistant_verification_ui,
    save_comment_ui,
    save_engine_preference_ui,
    swap_roles_ui,
    sync_queue_ui,
    toggle_comment_ui,
    transcribe_ui,
)


class StreamingUiTests(unittest.TestCase):
    @staticmethod
    def _discussion_state(*item_ids: str) -> dict[str, object]:
        items: dict[str, object] = {}
        for item_id in item_ids or ("audio-1",):
            items[item_id] = {
                "id": item_id,
                "name": f"{item_id}.wav",
                "status": "Terminé",
                "progress": 1.0,
                "partial": "Texte visible.",
                "comments": [],
                "result": {
                    "duration": 10.0,
                    "turns": [
                        {
                            "start": 0.0,
                            "end": 5.0,
                            "speaker_id": "SPEAKER_00",
                            "speaker": "Alice",
                            "text": "Bonjour client.",
                        },
                        {
                            "start": 5.0,
                            "end": 10.0,
                            "speaker_id": "SPEAKER_01",
                            "speaker": "Bob",
                            "text": "Réponse précise.",
                        },
                    ],
                },
            }
        return {"version": 4, "order": list(item_ids or ("audio-1",)), "items": items}

    def test_poll_reconnects_to_saved_partial_transcript(self) -> None:
        state = {
            "order": ["audio-1"],
            "items": {
                "audio-1": {
                    "id": "audio-1",
                    "path": "C:/managed/audio.wav",
                    "name": "Entretien.wav",
                    "status": "En cours",
                    "progress": 0.42,
                    "partial": "Texte durable déjà reconnu.",
                    "checkpoint_position": 42.0,
                    "stage": "Transcription de l'audio…",
                    "comments": [],
                }
            },
        }
        with (
            patch("transcription_locale.ui.load_workspace", return_value=state),
            patch("transcription_locale.ui.JOB_MANAGER.recover_orphaned", return_value=[]),
            patch(
                "transcription_locale.ui.JOB_MANAGER.active_item_ids",
                return_value={"audio-1"},
            ),
        ):
            outputs = poll_workspace_ui("audio-1", LIBRARY_SORTS[0])

        self.assertEqual(len(outputs), 28)
        self.assertEqual(outputs[5], "Texte durable déjà reconnu.")
        self.assertIn("42 s", outputs[13])

    def test_poll_never_recovers_jobs_during_periodic_refresh(self) -> None:
        state = self._discussion_state("audio-1")
        with (
            patch("transcription_locale.ui.load_workspace", return_value=state),
            patch(
                "transcription_locale.ui.JOB_MANAGER.recover_orphaned"
            ) as recover_orphaned,
        ):
            poll_workspace_ui("audio-1", LIBRARY_SORTS[0])

        recover_orphaned.assert_not_called()

    def test_running_fiche_refreshes_after_returning_to_the_desktop_app(self) -> None:
        header = _fiche_header_html(
            {
                "id": "audio-1",
                "name": "Entretien.wav",
                "status": "En cours",
                "progress": 0.57,
            }
        )

        self.assertIn('data-audio-status="En cours"', header)
        self.assertIn("visibilitychange", DESKTOP_VISIBILITY_SYNC_JS)
        self.assertIn("window.location.reload()", DESKTOP_VISIBILITY_SYNC_JS)

    def test_running_fiche_exposes_a_client_side_processing_clock(self) -> None:
        header = _fiche_header_html(
            {
                "id": "audio-1",
                "name": "Entretien.wav",
                "status": "En cours",
                "progress": 0.02,
                "processing_time": "0 s",
                "processing_seconds": 125.0,
                "processing_base_seconds": 120.0,
                "processing_started_at": "2026-07-23T10:00:00.000+00:00",
            }
        )

        self.assertIn("class=\"fiche-processing-time\"", header)
        self.assertIn("data-processing-live='true'", header)
        self.assertIn("data-processing-seconds='125.000'", header)
        self.assertIn("data-processing-base-seconds='120.000'", header)
        self.assertIn(
            "data-processing-started-at='2026-07-23T10:00:00.000+00:00'",
            header,
        )
        self.assertIn(">2 min 05 s</strong> traitement", header)
        self.assertIn("Date.now() - startedAt", PROCESSING_CLOCK_JS)
        self.assertIn("window.setInterval(refresh, 1000)", PROCESSING_CLOCK_JS)
        self.assertNotIn("window.location.reload()", PROCESSING_CLOCK_JS)

    def test_completed_fiche_has_a_frozen_processing_clock(self) -> None:
        header = _fiche_header_html(
            {
                "id": "audio-1",
                "name": "Entretien.wav",
                "status": "Terminé",
                "progress": 1.0,
                "processing_seconds": 125.0,
                "processing_base_seconds": 120.0,
                "processing_started_at": "2026-07-23T10:00:00.000+00:00",
            }
        )

        self.assertNotIn("data-processing-live='true'", header)
        self.assertIn(">2 min 05 s</strong> traitement", header)

    def test_poll_does_not_rerender_the_workspace_for_heartbeat_updates(self) -> None:
        state = {
            "order": ["audio-1"],
            "items": {
                "audio-1": {
                    "id": "audio-1",
                    "path": "C:/managed/audio.wav",
                    "name": "Entretien.wav",
                    "status": "En cours",
                    "progress": 0.42,
                    "processing_seconds": 42.0,
                    "partial": "Texte déjà reconnu.",
                    "checkpoint_position": 42.0,
                    "stage": "Transcription de l'audio…",
                    "heartbeat": "2026-07-23T10:00:00+00:00",
                    "updated_at": "2026-07-23T10:00:00+00:00",
                    "comments": [],
                }
            },
        }
        refreshed_state = copy.deepcopy(state)
        refreshed_item = refreshed_state["items"]["audio-1"]
        refreshed_item.update(
            {
                "progress": 0.43,
                "processing_seconds": 43.0,
                "checkpoint_position": 43.0,
                "heartbeat": "2026-07-23T10:00:01+00:00",
                "updated_at": "2026-07-23T10:00:01+00:00",
            }
        )
        with (
            patch(
                "transcription_locale.ui.load_workspace",
                side_effect=[state, refreshed_state],
            ),
            patch("transcription_locale.ui.JOB_MANAGER.recover_orphaned", return_value=[]),
            patch(
                "transcription_locale.ui.JOB_MANAGER.active_item_ids",
                return_value={"audio-1"},
            ),
        ):
            first_outputs = poll_workspace_ui("audio-1", LIBRARY_SORTS[0])
            refreshed_outputs = poll_workspace_ui(
                "audio-1",
                LIBRARY_SORTS[0],
                first_outputs[0],
            )

        self.assertEqual(refreshed_outputs[1], gr.skip())
        self.assertNotEqual(refreshed_outputs[4], gr.skip())
        self.assertEqual(refreshed_outputs[5], gr.skip())
        self.assertEqual(refreshed_outputs[6], gr.skip())
        self.assertEqual(refreshed_outputs[7], gr.skip())
        self.assertEqual(refreshed_outputs[12], gr.skip())
        self.assertNotEqual(refreshed_outputs[13], gr.skip())

    def test_poll_propagates_structural_workspace_changes(self) -> None:
        state = self._discussion_state("audio-1")
        with (
            patch("transcription_locale.ui.load_workspace", return_value=state),
            patch("transcription_locale.ui.JOB_MANAGER.recover_orphaned", return_value=[]),
            patch(
                "transcription_locale.ui.JOB_MANAGER.active_item_ids",
                return_value=set(),
            ),
        ):
            first_outputs = poll_workspace_ui("audio-1", LIBRARY_SORTS[0])

        updated_state = copy.deepcopy(state)
        updated_state["items"]["audio-1"]["name"] = "Entretien renommé.wav"
        with (
            patch("transcription_locale.ui.load_workspace", return_value=updated_state),
            patch("transcription_locale.ui.JOB_MANAGER.recover_orphaned", return_value=[]),
            patch(
                "transcription_locale.ui.JOB_MANAGER.active_item_ids",
                return_value=set(),
            ),
        ):
            refreshed_outputs = poll_workspace_ui(
                "audio-1",
                LIBRARY_SORTS[0],
                first_outputs[0],
            )

        self.assertEqual(refreshed_outputs[1], updated_state)
        self.assertIn("Entretien renommé.wav", refreshed_outputs[4])

    def test_poll_hydrates_roles_and_assistant_when_transcription_finishes(self) -> None:
        running_state = self._discussion_state("audio-1")
        running_item = running_state["items"]["audio-1"]
        running_item.update(
            {
                "status": "En cours",
                "progress": 0.65,
                "partial": "Texte en cours.",
                "result": None,
            }
        )
        completed_state = self._discussion_state("audio-1")

        with patch(
            "transcription_locale.ui.load_workspace",
            side_effect=[running_state, completed_state],
        ):
            first_outputs = poll_workspace_ui("audio-1", LIBRARY_SORTS[0])
            completed_outputs = poll_workspace_ui(
                "audio-1",
                LIBRARY_SORTS[0],
                first_outputs[0],
            )

        self.assertEqual(len(completed_outputs), 28)
        self.assertNotIn("Attribution en préparation", completed_outputs[21])
        self.assertIn("Modifier l'attribution", completed_outputs[21])
        self.assertTrue(completed_outputs[22].interactive)
        self.assertTrue(completed_outputs[23].interactive)
        self.assertNotIn(
            "Analyse disponible après la transcription",
            completed_outputs[26],
        )
        self.assertIn("Interrogez la consultation", completed_outputs[26])
        self.assertTrue(completed_outputs[27].interactive)

    @patch("transcription_locale.core.sys.platform", "win32")
    def test_empty_optional_vocabulary_and_partial_stream(self) -> None:
        result = TranscriptionResult(
            audio_name="test.wav",
            duration=2.0,
            processing_seconds=1.0,
            profile="Équilibré",
            model="small",
            language="fr",
            language_probability=1.0,
            requested_speakers=2,
            detected_speakers=2,
            turns=[
                TranscriptTurn(0.0, 1.0, "SPEAKER_00", "Interlocuteur 1", "Bonjour."),
                TranscriptTurn(1.0, 2.0, "SPEAKER_01", "Interlocuteur 2", "Bonjour !"),
            ],
        )

        def fake_pipeline(*args, **kwargs):
            self.assertEqual(kwargs["initial_prompt"], "")
            self.assertEqual(kwargs["language"], "fr")
            self.assertEqual(kwargs["asr_backend"], "openvino-arc-pilot")
            kwargs["progress"](0.2, "Transcription")
            kwargs["partial"]("Bonjour…", 1.0)
            return result

        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            with (
                patch("transcription_locale.ui.run_pipeline", side_effect=fake_pipeline),
                patch("transcription_locale.ui.write_exports", return_value=[Path("test.zip")]),
            ):
                updates = list(
                    transcribe_ui(
                        audio_file.name,
                        "Équilibré",
                        "Français",
                        2,
                        "Interlocuteur 1",
                        "Interlocuteur 2",
                        None,
                    )
                )

        partial_updates = [update for update in updates if update[4] == "Bonjour…"]
        self.assertEqual(len(partial_updates), 1)
        self.assertIn("File d'attente terminée", updates[-1][0])
        completed_updates = [
            update
            for update in updates
            if isinstance(update[4], str) and "Bonjour." in update[4]
        ]
        self.assertEqual(completed_updates[-1][4].splitlines()[1], "Bonjour.")

    @patch("transcription_locale.core.sys.platform", "win32")
    def test_openvino_pilot_is_limited_to_balanced_profile(self) -> None:
        self.assertEqual(
            _effective_engine_choice(OPENVINO_PILOT_ENGINE, "Équilibré"),
            OPENVINO_PILOT_ENGINE,
        )
        self.assertEqual(
            _effective_engine_choice(OPENVINO_PILOT_ENGINE, "Précis"),
            "faster-whisper",
        )

    @patch("transcription_locale.core.sys.platform", "win32")
    def test_engine_status_makes_fallback_and_detected_arc_explicit(self) -> None:
        unavailable = {
            "available": False,
            "model_ready": False,
            "device_name": "",
            "reason": "Modèle OpenVINO introuvable.",
        }
        ready = {
            "available": True,
            "model_ready": True,
            "device_name": "Intel(R) Arc(TM) 140V GPU",
            "reason": "",
        }
        with patch("transcription_locale.ui._read_openvino_status", return_value=unavailable):
            fallback_html = _engine_status_html(OPENVINO_PILOT_ENGINE, "Équilibré")
        with patch("transcription_locale.ui._read_openvino_status", return_value=ready):
            ready_html = _engine_status_html(OPENVINO_PILOT_ENGINE, "Équilibré")

        self.assertIn("Faster-Whisper prendra le relais", fallback_html)
        self.assertIn("Pilote non installé", fallback_html)
        self.assertIn("Pilote Intel Arc prêt", ready_html)
        self.assertIn("Arc(TM) 140V", ready_html)

    @patch("transcription_locale.core.sys.platform", "win32")
    def test_engine_choice_is_saved_in_workspace_preferences(self) -> None:
        state = {"order": [], "items": {}, "preferences": {}}

        def fake_mutate(mutator):
            mutator(state)
            return state

        with (
            patch("transcription_locale.ui.mutate_workspace", side_effect=fake_mutate),
            patch(
                "transcription_locale.ui._read_openvino_status",
                return_value={
                    "available": False,
                    "model_ready": False,
                    "reason": "Non installé",
                },
            ),
        ):
            saved, status_html = save_engine_preference_ui(
                OPENVINO_PILOT_ENGINE,
                "Équilibré",
            )

        self.assertEqual(
            saved["preferences"]["transcription_engine"],
            OPENVINO_PILOT_ENGINE,
        )
        self.assertIn("relais", status_html)

    @patch("transcription_locale.core.sys.platform", "win32")
    def test_engine_choice_is_restored_when_the_page_is_reloaded(self) -> None:
        workspace_state = {
            "order": [],
            "items": {},
            "preferences": {"transcription_engine": OPENVINO_PILOT_ENGINE},
        }
        with (
            patch("transcription_locale.ui.load_workspace", return_value=workspace_state),
            patch(
                "transcription_locale.ui._read_openvino_status",
                return_value={
                    "available": True,
                    "model_ready": True,
                    "device_name": "Intel(R) Arc(TM) 140V GPU",
                    "reason": "",
                },
            ),
        ):
            selected, status_html, summary_html = restore_engine_preference_ui()

        self.assertEqual(selected, OPENVINO_PILOT_ENGINE)
        self.assertIn("Pilote Intel Arc prêt", status_html)
        self.assertIn("Intel Arc · pilote", summary_html)

    @patch("transcription_locale.core.sys.platform", "win32")
    def test_existing_standard_choice_is_restored_without_switching_to_arc(self) -> None:
        workspace_state = {
            "order": [],
            "items": {},
            "preferences": {"transcription_engine": "faster-whisper"},
        }
        with (
            patch("transcription_locale.ui.load_workspace", return_value=workspace_state),
            patch(
                "transcription_locale.ui._read_openvino_status",
                return_value={
                    "available": True,
                    "model_ready": True,
                    "device_name": "Intel(R) Arc(TM) 140V GPU",
                    "reason": "",
                },
            ),
        ):
            selected, status_html, summary_html = restore_engine_preference_ui()

        self.assertEqual(selected, "faster-whisper")
        self.assertIn("Faster-Whisper Équilibré actif", status_html)
        self.assertIn("choix manuel", summary_html)

    @patch("transcription_locale.core.sys.platform", "darwin")
    def test_macos_hides_arc_and_never_mentions_the_windows_installer(self) -> None:
        choices = _engine_choices()
        status_html = _engine_status_html(OPENVINO_PILOT_ENGINE, "Équilibré")
        settings_copy = _engine_settings_copy()

        self.assertEqual(choices, [("Faster-Whisper · moteur local", "faster-whisper")])
        self.assertEqual(
            _effective_engine_choice(OPENVINO_PILOT_ENGINE, "Équilibré"),
            "faster-whisper",
        )
        self.assertIn("Faster-Whisper actif", status_html)
        self.assertNotIn("Intel Arc", status_html)
        self.assertNotIn("INSTALLER_OPENVINO", status_html)
        self.assertNotIn("Intel Arc", settings_copy)

    def test_two_audio_queue_keeps_order_and_creates_batch(self) -> None:
        def fake_pipeline(path, **kwargs):
            kwargs["progress"](0.5, "Transcription")
            name = Path(path).name
            return TranscriptionResult(
                audio_name=name,
                duration=3.0,
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
                        3.0,
                        "SPEAKER_00",
                        "Interlocuteur 1",
                        f"Texte de {name}",
                    )
                ],
            )

        with (
            tempfile.NamedTemporaryFile(suffix="-premier.wav") as first,
            tempfile.NamedTemporaryFile(suffix="-second.wav") as second,
            patch("transcription_locale.ui.run_pipeline", side_effect=fake_pipeline),
            patch(
                "transcription_locale.ui.write_exports",
                side_effect=lambda result: [Path(f"{result.audio_name}.zip")],
            ),
            patch(
                "transcription_locale.ui.write_batch_archive",
                return_value=Path("lot-transcriptions.zip"),
            ) as batch_writer,
        ):
            updates = list(
                transcribe_ui(
                    [first.name, second.name],
                    "Équilibré",
                    "Français",
                    2,
                    "Interlocuteur 1",
                    "Interlocuteur 2",
                    "",
                    {
                        first.name: "Entretien Dupont.wav",
                        second.name: "Entretien Martin.wav",
                    },
                )
            )

        final = updates[-1]
        self.assertIn("2/2 audio(s)", final[0])
        self.assertEqual([row[2] for row in final[1]], ["Terminé", "Terminé"])
        self.assertEqual(
            [row[1] for row in final[1]],
            ["Entretien Dupont.wav", "Entretien Martin.wav"],
        )
        self.assertEqual(len(final[6]["order"]), 2)
        self.assertEqual(
            final[6]["order"],
            ["01 · Entretien Dupont.wav", "02 · Entretien Martin.wav"],
        )
        self.assertEqual(final[6]["batch_zip"], "lot-transcriptions.zip")
        batch_writer.assert_called_once()

    def test_preview_queue_respects_dragged_order(self) -> None:
        preview, rows, hint = preview_queue_ui(["C:/audio/b.wav", "C:/audio/a.wav"])
        self.assertEqual(preview, "C:\\audio\\b.wav")
        self.assertEqual([row[1] for row in rows], ["b.wav", "a.wav"])
        self.assertIn("2 audios", hint)

    def test_audio_can_be_renamed_without_changing_the_source_file(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            synced = sync_queue_ui([audio_file.name], {})
            names = synced[3]
            renamed = rename_audio_ui(
                [audio_file.name],
                audio_file.name,
                "Entretien : client.wav",
                names,
            )

            self.assertTrue(Path(audio_file.name).is_file())
            self.assertEqual(renamed[0][audio_file.name], "Entretien client.wav")
            self.assertEqual(renamed[1][0][1], "Entretien client.wav")
            self.assertIn("Nom enregistré", renamed[2])

    def test_audio_can_be_renamed_from_a_sorted_library_title(self) -> None:
        with (
            tempfile.NamedTemporaryFile(suffix="-bravo.wav") as bravo,
            tempfile.NamedTemporaryFile(suffix="-alpha.wav") as alpha,
        ):
            paths = [bravo.name, alpha.name]
            names = {
                bravo.name: "Bravo.wav",
                alpha.name: "Alpha.wav",
            }
            state = {
                "order": ["bravo", "alpha"],
                "items": {
                    "bravo": {"path": bravo.name, "name": "Bravo.wav"},
                    "alpha": {"path": alpha.name, "name": "Alpha.wav"},
                },
            }

            renamed = rename_audio_by_target_ui(
                paths,
                "library:0",
                "Entretien Alpha",
                names,
                state,
                LIBRARY_SORTS[1],
            )

            self.assertEqual(renamed[0][alpha.name], "Entretien Alpha.wav")
            self.assertEqual(renamed[0][bravo.name], "Bravo.wav")

    def test_audio_can_be_deleted_from_a_sorted_library_row(self) -> None:
        state = {
            "order": ["bravo", "alpha"],
            "items": {
                "bravo": {
                    "id": "bravo",
                    "path": "C:/managed/bravo.wav",
                    "name": "Bravo.wav",
                    "status": "En attente",
                },
                "alpha": {
                    "id": "alpha",
                    "path": "C:/managed/alpha.wav",
                    "name": "Alpha.wav",
                    "status": "En attente",
                },
            },
        }
        deleted_state = {"order": ["bravo"], "items": {"bravo": state["items"]["bravo"]}}

        with (
            patch("transcription_locale.ui.load_workspace", return_value=state),
            patch(
                "transcription_locale.ui.delete_workspace_item",
                return_value=(deleted_state, "Alpha.wav"),
            ) as delete_item,
        ):
            feedback = delete_audio_by_target_ui(
                "library:0",
                LIBRARY_SORTS[1],
                state,
            )

        self.assertIn("Alpha.wav", feedback)
        self.assertEqual(delete_item.call_args.args[1], "alpha")

    def test_audio_library_can_be_sorted_by_name_and_date(self) -> None:
        state = {
            "order": ["b", "a"],
            "items": {
                "a": {
                    "name": "Alpha.wav",
                    "status": "Terminé",
                    "progress": 1.0,
                    "created_at": "2026-07-20T10:00:00",
                    "comments": [],
                },
                "b": {
                    "name": "Bravo.wav",
                    "status": "En attente",
                    "progress": 0.0,
                    "created_at": "2026-07-21T10:00:00",
                    "comments": [],
                },
            },
        }

        by_name = _workspace_library_rows(state, "Nom · A à Z")
        by_date = _workspace_library_rows(state, "Date d'ajout · plus récent")

        self.assertEqual([row[0] for row in by_name], ["Alpha.wav", "Bravo.wav"])
        self.assertEqual([row[0] for row in by_date], ["Bravo.wav", "Alpha.wav"])

    def test_completed_audio_leaves_the_active_queue_but_stays_in_the_library(self) -> None:
        state = {
            "order": ["done", "waiting"],
            "queue_order": ["done", "waiting"],
            "items": {
                "done": {
                    "name": "Consultation terminée.wav",
                    "status": "Terminé",
                    "progress": 1.0,
                    "result": {"turns": []},
                    "comments": [],
                },
                "waiting": {
                    "name": "Prochaine consultation.wav",
                    "status": "En attente",
                    "progress": 0.0,
                    "result": None,
                    "comments": [],
                },
            },
        }

        queue_rows = _workspace_queue_rows(state)
        library_rows = _workspace_library_rows(state, LIBRARY_SORTS[0])

        self.assertEqual([row[1] for row in queue_rows], ["Prochaine consultation.wav"])
        self.assertEqual(len(library_rows), 2)
        self.assertIn("1 audio", _workspace_status_markdown(state))

    def test_completed_only_workspace_has_an_explicit_empty_queue_state(self) -> None:
        state = self._discussion_state("audio-1", "audio-2")
        state["queue_order"] = ["audio-1", "audio-2"]

        self.assertEqual(_workspace_queue_rows(state), [])
        status = _workspace_status_markdown(state)
        self.assertIn("Aucun audio en attente", status)
        self.assertIn("Fiches", status)

    def test_queue_delete_targets_the_filtered_queue_identifier(self) -> None:
        state = {
            "order": ["done", "waiting"],
            "queue_order": ["waiting"],
            "items": {
                "done": {
                    "id": "done",
                    "name": "Terminé.wav",
                    "status": "Terminé",
                    "result": {"turns": []},
                },
                "waiting": {
                    "id": "waiting",
                    "name": "À traiter.wav",
                    "status": "En attente",
                    "result": None,
                },
            },
        }
        deleted_state = {
            "order": ["done"],
            "queue_order": [],
            "items": {"done": state["items"]["done"]},
        }

        with (
            patch("transcription_locale.ui.load_workspace", return_value=state),
            patch(
                "transcription_locale.ui.delete_workspace_item",
                return_value=(deleted_state, "À traiter.wav"),
            ) as delete_item,
        ):
            delete_audio_by_target_ui("queue:0", LIBRARY_SORTS[0], state)

        self.assertEqual(delete_item.call_args.args[1], "waiting")

    def test_selected_transcript_passage_gets_a_temporal_anchor(self) -> None:
        state = self._discussion_state("audio-1")
        transcript = (
            "[00:00:00] Alice\nBonjour client.\n\n"
            "[00:00:05] Bob\nRéponse précise.\n"
        )
        passage = "Réponse précise."
        selection_start = transcript.index(passage)
        event = SimpleNamespace(
            value=passage,
            index=(selection_start, selection_start + len(passage)),
        )

        selected, raw_anchor, context_html, draft_item = capture_selected_passage_ui(
            "audio-1",
            transcript,
            state,
            event,
        )
        anchor = json.loads(raw_anchor)

        self.assertEqual(selected, passage)
        self.assertEqual(draft_item, "audio-1")
        self.assertEqual(anchor["start_seconds"], 5.0)
        self.assertEqual(anchor["end_seconds"], 10.0)
        self.assertEqual(anchor["source"], "transcript")
        self.assertEqual(anchor["turn_ids"], ["turn-2"])
        self.assertEqual(anchor["speaker_ids"], ["SPEAKER_01"])
        self.assertEqual(anchor["speaker"], "Bob")
        self.assertIn("00:00:05", context_html)
        self.assertIn("00:00:10", context_html)

    def test_comment_full_lifecycle_keeps_anchor_and_cascades_replies_on_delete(self) -> None:
        state = self._discussion_state("audio-1")
        anchor = json.dumps(
            {
                "start_seconds": 5.0,
                "end_seconds": 10.0,
                "quote": "Réponse précise.",
                "source": "transcript",
                "turn_ids": ["SPEAKER_01"],
            },
            ensure_ascii=False,
        )

        def mutate(mutator):
            mutator(state)
            return state

        with (
            patch("transcription_locale.ui.load_workspace", return_value=state),
            patch("transcription_locale.ui.mutate_workspace", side_effect=mutate),
        ):
            created = save_comment_ui(
                "audio-1",
                "AuteurX",
                "Réponse précise.",
                "À vérifier avec Marie.",
                anchor,
                "audio-1",
                "",
                state,
            )
            comment = state["items"]["audio-1"]["comments"][0]
            comment_id = comment["id"]
            created_at = comment["created_at"]

            self.assertEqual(created[0], state)
            self.assertEqual(comment["anchor"]["start_seconds"], 5.0)
            self.assertEqual(comment["anchor"]["end_seconds"], 10.0)
            self.assertEqual(comment["status"], "À discuter")

            edited = save_comment_ui(
                "audio-1",
                "AuteurXEdition",
                "Réponse précise.",
                "Texte corrigé et confirmé.",
                anchor,
                "audio-1",
                comment_id,
                state,
            )
            self.assertEqual(edited[0], state)
            self.assertEqual(comment["id"], comment_id)
            self.assertEqual(comment["created_at"], created_at)
            self.assertEqual(comment["author"], "AuteurXEdition")
            self.assertEqual(comment["body"], "Texte corrigé et confirmé.")
            self.assertEqual(comment["anchor"]["start_seconds"], 5.0)

            replied = add_reply_ui(
                "audio-1",
                comment_id,
                "Marie",
                "Je confirme ce passage.",
                state,
            )
            self.assertEqual(replied[0], state)
            self.assertEqual(len(comment["replies"]), 1)
            self.assertEqual(comment["replies"][0]["author"], "Marie")
            self.assertEqual(comment["replies"][0]["body"], "Je confirme ce passage.")

            resolved = toggle_comment_ui("audio-1", comment_id, state)
            self.assertEqual(resolved[0], state)
            self.assertEqual(comment["status"], "Résolu")
            self.assertIsNotNone(comment["resolved_at"])

            deleted = handle_comment_feed_action_ui(
                "audio-1",
                state,
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                SimpleNamespace(action="delete", comment_id=comment_id),
            )

        self.assertEqual(deleted[0], state)
        self.assertEqual(state["items"]["audio-1"]["comments"], [])

    def test_comment_draft_cannot_be_saved_to_another_audio(self) -> None:
        state = self._discussion_state("audio-1", "audio-2")
        with (
            patch("transcription_locale.ui.load_workspace", return_value=state),
            patch("transcription_locale.ui.mutate_workspace") as mutate,
        ):
            with self.assertRaisesRegex(gr.Error, "autre fiche"):
                save_comment_ui(
                    "audio-2",
                    "AuteurX",
                    "Réponse précise.",
                    "Ce brouillon ne doit pas changer de fiche.",
                    json.dumps(
                        {
                            "start_seconds": 5.0,
                            "end_seconds": 10.0,
                            "source": "transcript",
                        }
                    ),
                    "audio-1",
                    "",
                    state,
                )

        mutate.assert_not_called()
        self.assertEqual(state["items"]["audio-1"]["comments"], [])
        self.assertEqual(state["items"]["audio-2"]["comments"], [])

    def test_discussion_feed_escapes_user_content_and_identifiers(self) -> None:
        item = {
            "comments": [
                {
                    "id": "discussion-' onclick='alert(1)",
                    "author": "<script>alert('auteur')</script>",
                    "passage": "<img src=x onerror=alert('passage')>",
                    "body": "<b onclick=alert('message')>Message</b>",
                    "status": "À discuter",
                    "anchor": {
                        "start_seconds": 1.0,
                        "end_seconds": 2.0,
                    },
                    "created_at": "<svg onload=alert('date')>",
                    "replies": [
                        {
                            "author": "<i>Marie</i>",
                            "body": "<script>alert('réponse')</script>",
                            "created_at": "maintenant",
                        }
                    ],
                }
            ]
        }

        rendered = _comments_feed_html(item)

        self.assertNotIn("<script>", rendered)
        self.assertNotIn("<img", rendered)
        self.assertNotIn("<svg", rendered)
        self.assertNotIn("<i>Marie</i>", rendered)
        self.assertNotIn("discussion-' onclick=", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("&lt;img src=x onerror=", rendered)
        self.assertIn("discussion-&#x27; onclick=&#x27;alert(1)", rendered)

    def test_discussion_can_be_added_then_resolved(self) -> None:
        state = {
            "order": ["audio-1"],
            "items": {
                "audio-1": {
                    "id": "audio-1",
                    "name": "Entretien.wav",
                    "status": "Terminé",
                    "progress": 1.0,
                    "partial": "Passage important",
                    "comments": [],
                    "result": None,
                }
            },
        }
        with (
            patch("transcription_locale.ui.load_workspace", return_value=state),
            patch(
                "transcription_locale.ui.mutate_workspace",
                side_effect=lambda mutator: (mutator(state), state)[1],
            ),
        ):
            added = add_comment_ui(
                "audio-1",
                "AuteurX",
                "Passage important",
                "À confirmer avec l'équipe",
                state,
            )

        saved = added[0]
        comment = saved["items"]["audio-1"]["comments"][0]
        self.assertEqual(comment["status"], "À discuter")

        with (
            patch("transcription_locale.ui.load_workspace", return_value=saved),
            patch(
                "transcription_locale.ui.mutate_workspace",
                side_effect=lambda mutator: (mutator(saved), saved)[1],
            ),
        ):
            toggled = toggle_comment_ui(
                "audio-1",
                comment["id"],
                saved,
            )
        self.assertEqual(
            toggled[0]["items"]["audio-1"]["comments"][0]["status"],
            "Résolu",
        )

    def test_consultation_roles_are_proposed_in_the_audio_sheet(self) -> None:
        item = self._discussion_state("audio-1")["items"]["audio-1"]
        item["result"]["turns"] = [
            {
                "start": 0.0,
                "end": 3.0,
                "speaker_id": "SPEAKER_00",
                "speaker": "Voix 1",
                "text": "Bonjour, pouvez-vous me donner votre date de naissance ?",
            },
            {
                "start": 3.0,
                "end": 8.0,
                "speaker_id": "SPEAKER_01",
                "speaker": "Voix 2",
                "text": "Je suis ClienteX et je suis née le 31 février 2099.",
            },
        ]

        proposal = _assistant_role_proposal(item)

        self.assertEqual(proposal["master_speaker_id"], "SPEAKER_00")
        self.assertEqual(proposal["client_speaker_id"], "SPEAKER_01")

    def test_role_summary_uses_a_global_confirmation_badge(self) -> None:
        item = self._discussion_state("audio-1")["items"]["audio-1"]
        item["speaker_roles"] = {
            "SPEAKER_00": {"role": "Master", "confirmed": True},
            "SPEAKER_01": {"role": "Client", "confirmed": True},
        }

        rendered = _role_summary_html(item)

        self.assertIn("Rôles des interlocuteurs", rendered)
        self.assertIn("Attribution confirmée", rendered)
        self.assertIn("Client", rendered)
        self.assertIn("Master", rendered)
        self.assertIn("data-role-editor-toggle", rendered)
        self.assertIn("Modifier l'attribution", rendered)
        self.assertNotIn("Corrigeable à tout moment", rendered)

    def test_role_summary_waits_for_diarization_before_enabling_editor(self) -> None:
        item = self._discussion_state("audio-1")["items"]["audio-1"]
        item["status"] = "En cours"
        item["progress"] = 0.65
        item["result"] = None

        rendered = _role_summary_html(item)

        self.assertIn("Attribution en préparation", rendered)
        self.assertIn("Disponible après séparation", rendered)
        self.assertIn("disabled", rendered)
        self.assertIn('aria-disabled="true"', rendered)
        self.assertNotIn("Modifier l'attribution", rendered)

    def test_role_voice_excerpt_explains_pending_diarization(self) -> None:
        item = self._discussion_state("audio-1")["items"]["audio-1"]
        item["result"] = None

        rendered = _role_voice_excerpt_html(item, None, "Client")

        self.assertIn("Disponible après la séparation des voix", rendered)
        self.assertIn("is-pending", rendered)

    def test_role_voice_excerpt_is_short_and_html_safe(self) -> None:
        item = self._discussion_state("audio-1")["items"]["audio-1"]
        item["result"]["turns"][0]["text"] = "<script>alert('x')</script> " + ("oracle " * 40)

        rendered = _role_voice_excerpt_html(item, "SPEAKER_00", "Master")

        self.assertIn("Extrait de la voix", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertNotIn("<script>", rendered)
        self.assertIn("…", rendered)

    def test_role_inversion_remains_a_draft_until_saved(self) -> None:
        state = self._discussion_state("audio-1")
        item = state["items"]["audio-1"]
        item["speaker_roles"] = {
            "SPEAKER_00": {"role": "Master", "confirmed": True},
            "SPEAKER_01": {"role": "Client", "confirmed": True},
        }
        before = json.loads(json.dumps(item["speaker_roles"]))

        updates = swap_roles_ui(
            "audio-1",
            "SPEAKER_01",
            "SPEAKER_00",
            state,
        )

        self.assertEqual(item["speaker_roles"], before)
        self.assertEqual(len(updates), 4)
        self.assertEqual(updates[0].value, "SPEAKER_00")
        self.assertEqual(updates[1].value, "SPEAKER_01")

    def test_role_inversion_is_blocked_until_two_voices_exist(self) -> None:
        state = self._discussion_state("audio-1")
        state["items"]["audio-1"]["result"] = None

        with self.assertRaises(gr.Error):
            swap_roles_ui("audio-1", None, None, state)

    def test_transcript_uses_client_master_labels_without_mutating_result(self) -> None:
        item = self._discussion_state("audio-1")["items"]["audio-1"]
        item["speaker_roles"] = {
            "SPEAKER_00": {"role": "Master", "confirmed": True},
            "SPEAKER_01": {"role": "Client", "confirmed": True},
        }

        rendered = _item_transcript(item)

        self.assertIn("[00:00:00] Master\nBonjour client.", rendered)
        self.assertIn("[00:00:05] Client\nRéponse précise.", rendered)
        self.assertNotIn("[00:00:00] Alice", rendered)
        self.assertEqual(item["result"]["turns"][0]["speaker"], "Alice")

    def test_transcript_role_labels_follow_an_inversion(self) -> None:
        item = self._discussion_state("audio-1")["items"]["audio-1"]
        item["speaker_roles"] = {
            "SPEAKER_00": {"role": "Client"},
            "SPEAKER_01": {"role": "Master"},
        }

        rendered = _item_transcript(item)

        self.assertIn("[00:00:00] Client", rendered)
        self.assertIn("[00:00:05] Master", rendered)

    def test_role_aware_transcript_selection_keeps_its_timecode(self) -> None:
        item = self._discussion_state("audio-1")["items"]["audio-1"]
        item["speaker_roles"] = {
            "SPEAKER_00": {"role": "Master", "confirmed": True},
            "SPEAKER_01": {"role": "Client", "confirmed": True},
        }
        transcript = _item_transcript(item)
        passage = "[00:00:05] Client\nRéponse précise."
        start = transcript.index(passage)

        anchor = _comment_anchor(
            item,
            transcript,
            passage,
            (start, start + len(passage)),
        )

        self.assertEqual(anchor["start_seconds"], 5.0)
        self.assertEqual(anchor["end_seconds"], 10.0)
        self.assertEqual(anchor["speaker"], "Client")
        self.assertEqual(anchor["speaker_ids"], ["SPEAKER_01"])

    def test_assistant_result_is_grounded_escaped_and_discussable(self) -> None:
        result = {
            "verdict": "Confirmé",
            "explanation": "Retrouvé dans la consultation.",
            "confidence": 0.92,
            "coverage": {
                "turns_analyzed": 2,
                "duration_seconds": 8.0,
                "role_mapping_confirmed": True,
            },
            "citations": [
                {
                    "quote": "<script>alert('x')</script> je suis née le 31 février",
                    "start_seconds": 3.0,
                    "end_seconds": 8.0,
                    "speaker_id": "SPEAKER_01",
                    "role": "Client",
                }
            ],
        }

        rendered = _assistant_result_html(result)

        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("data-assistant-action=\"toggle-excerpt\"", rendered)
        self.assertIn("data-assistant-action=\"stop-excerpt\"", rendered)
        self.assertIn("data-excerpt-range", rendered)
        self.assertIn("data-evidence-item open", rendered)
        self.assertIn("evidence-preview", rendered)
        self.assertIn("aria-valuetext=\"0:00 sur", rendered)
        self.assertIn("data-assistant-action=\"discuss\"", rendered)
        self.assertIn("Réponse détaillée", rendered)
        self.assertIn("Ce que montrent les passages", rendered)
        self.assertIn("Lecture prudente", rendered)
        self.assertIn("assistant-answer-head", rendered)

    def test_assistant_summary_has_its_own_grounded_presentation(self) -> None:
        result = {
            "intent": "open_summary",
            "verdict": "Synthèse",
            "answer": (
                "La consultation porte principalement sur l'avenir sentimental "
                "de la Cliente avec OrionX."
            ),
            "key_points": [
                {
                    "text": "La Cliente demande comment leur relation va évoluer.",
                    "citation_ids": ["turn-00001"],
                },
                "Le Master évoque ensuite un rapprochement possible.",
            ],
            "nuance": (
                "Cette synthèse décrit ce qui a été dit dans l'audio, "
                "pas ce qui se produira."
            ),
            "backend": "déterministe-local",
            "coverage": {
                "turns_analyzed": 18,
                "duration_seconds": 420.0,
                "role_mapping_confirmed": True,
            },
            "citations": [
                {
                    "quote": "Je voudrais savoir comment notre relation va évoluer.",
                    "start_seconds": 24.0,
                    "end_seconds": 35.0,
                    "speaker_id": "SPEAKER_01",
                    "role": "Client",
                }
            ],
        }

        rendered = _assistant_result_html(result)

        self.assertIn("verdict-summary", rendered)
        self.assertIn(">Synthèse<", rendered)
        self.assertIn("Synthèse locale · Sources citées", rendered)
        self.assertIn("Synthèse de la consultation", rendered)
        self.assertIn("Points clés", rendered)
        self.assertIn("Portée de la synthèse", rendered)
        self.assertIn("Passages sources · 1", rendered)
        self.assertIn('aria-label="Résultat de la synthèse"', rendered)
        self.assertIn("La Cliente demande comment leur relation va évoluer.", rendered)
        self.assertNotIn("Passage contraire explicite", rendered)
        self.assertNotIn("Réponse détaillée", rendered)

    def test_assistant_topic_statuses_are_not_binary_verdicts(self) -> None:
        cases = (
            ("Thème principal", "theme-primary", "Thème central"),
            ("Thème présent", "theme-present", "Présence confirmée"),
            ("Mention ponctuelle", "mention", "Présence limitée"),
        )
        for verdict, css_class, meta in cases:
            with self.subTest(verdict=verdict):
                rendered = _assistant_result_html(
                    {
                        "intent": "topic_presence",
                        "verdict": verdict,
                        "answer": "Oui, ce thème est abordé dans la consultation.",
                        "analysis_points": ["Un passage explicite a été retrouvé."],
                        "nuance": "Le niveau indique sa place dans la consultation.",
                        "coverage": {
                            "turns_analyzed": 12,
                            "duration_seconds": 180.0,
                            "role_mapping_confirmed": True,
                        },
                        "citations": [],
                    }
                )

                self.assertIn(f"verdict-{css_class}", rendered)
                self.assertIn(f">{verdict}<", rendered)
                self.assertIn(meta, rendered)
                self.assertIn("Réponse thématique", rendered)
                self.assertIn("Lecture du thème", rendered)
                self.assertIn("Passages sources · 0", rendered)
                self.assertNotIn("Passage contraire explicite", rendered)

    def test_assistant_legacy_verification_labels_remain_unchanged(self) -> None:
        rendered = _assistant_result_html(
            {
                "verdict": "Confirmé",
                "answer": "L'affirmation est présente dans l'audio.",
                "confidence": 0.91,
                "coverage": {
                    "turns_analyzed": 2,
                    "duration_seconds": 8.0,
                    "role_mapping_confirmed": True,
                },
                "citations": [],
            }
        )

        self.assertIn("verdict-confirmed", rendered)
        self.assertIn("Appuyé par l&#x27;audio", rendered)
        self.assertIn("Réponse détaillée", rendered)
        self.assertIn("Passages retenus · 0", rendered)
        self.assertNotIn("Synthèse de la consultation", rendered)

    def test_assistant_empty_state_invites_all_supported_question_types(self) -> None:
        item = {"result": {"turns": [{"text": "Un passage.", "start": 0, "end": 1}]}}

        rendered = _assistant_empty_html(item)

        self.assertIn("Interrogez la consultation", rendered)
        self.assertIn("Demandez une synthèse", rendered)
        self.assertIn("Est-ce que ça parle d'amour ?", rendered)
        self.assertIn("Le Master a-t-il annoncé un retour ?", rendered)

    def test_assistant_history_restores_answer_passages_roles_and_timecodes(self) -> None:
        item = {
            "assistant_queries": [
                {
                    "id": "query-one",
                    "question": "Est-ce que le Master évoque un départ ?",
                    "created_at": "2026-07-22T16:42:00+02:00",
                    "result": {
                        "verdict": "Partiel",
                        "answer": "Le départ est évoqué sans date ferme.",
                        "confidence": 0.71,
                        "coverage": {
                            "turns_analyzed": 12,
                            "duration_seconds": 75.5,
                            "role_mapping_confirmed": True,
                        },
                        "citations": [
                            {
                                "excerpt": "Un départ peut arriver.",
                                "text": "Un départ peut arriver au mois de septembre.",
                                "start_seconds": 62.0,
                                "end_seconds": 75.5,
                                "role": "Master",
                                "speaker_id": "SPEAKER_00",
                            }
                        ],
                    },
                }
            ]
        }

        rendered = _assistant_history_html(item, include_latest=True)

        self.assertIn("Historique des analyses", rendered)
        self.assertIn("Est-ce que le Master évoque un départ ?", rendered)
        self.assertIn("Le départ est évoqué sans date ferme.", rendered)
        self.assertIn("Réponse détaillée", rendered)
        self.assertIn("Ce que montrent les passages", rendered)
        self.assertIn("Lecture prudente", rendered)
        self.assertIn("Passages retenus · 1", rendered)
        self.assertIn("00:01:02–00:01:15", rendered)
        self.assertIn("Master", rendered)
        self.assertIn("Voir le passage complet", rendered)
        self.assertIn('data-assistant-action="toggle-excerpt"', rendered)
        self.assertIn('data-assistant-action="stop-excerpt"', rendered)
        self.assertIn("data-excerpt-current", rendered)
        self.assertIn('data-assistant-action="discuss"', rendered)
        self.assertNotIn('aria-live="polite"', rendered)
        self.assertNotIn("assistant-answer-head", rendered)
        self.assertEqual(rendered.count("verdict-chip"), 1)

    def test_assistant_history_preserves_summary_and_topic_labels(self) -> None:
        item = {
            "assistant_queries": [
                {
                    "id": "summary",
                    "question": "De quoi parle principalement la consultation ?",
                    "created_at": "2026-07-23T15:48:00+02:00",
                    "result": {
                        "intent": "open_summary",
                        "verdict": "Synthèse",
                        "answer": "La consultation porte sur une relation sentimentale.",
                        "coverage": {
                            "turns_analyzed": 64,
                            "duration_seconds": 1107.0,
                            "role_mapping_confirmed": True,
                        },
                        "citations": [],
                    },
                },
                {
                    "id": "topic",
                    "question": "Est-ce que ça parle d'amour ?",
                    "created_at": "2026-07-23T15:49:00+02:00",
                    "result": {
                        "intent": "topic_presence",
                        "verdict": "Thème principal",
                        "answer": "Oui, l'amour est le thème principal.",
                        "coverage": {
                            "turns_analyzed": 64,
                            "duration_seconds": 1107.0,
                            "role_mapping_confirmed": True,
                        },
                        "citations": [],
                    },
                },
            ]
        }

        rendered = _assistant_history_html(item, include_latest=True)

        self.assertIn("De quoi parle principalement la consultation ?", rendered)
        self.assertIn("Est-ce que ça parle d&#x27;amour ?", rendered)
        self.assertIn(">Synthèse<", rendered)
        self.assertIn(">Thème principal<", rendered)
        self.assertIn("Synthèse locale · Sources citées", rendered)
        self.assertIn("Analyse thématique locale · Thème central", rendered)
        self.assertNotIn("Passage contraire explicite", rendered)

    def test_assistant_history_excludes_current_result_and_escapes_saved_content(self) -> None:
        item = {
            "assistant_queries": [
                {
                    "id": "old",
                    "question": "Ancienne <script>alert(1)</script>",
                    "result": {
                        "verdict": "Confirmé",
                        "answer": "Réponse <b>stockée</b>",
                        "confidence": 0.9,
                        "coverage": {"role_mapping_confirmed": True},
                        "citations": [
                            {
                                "quote": 'Citation "sensible"',
                                "start_seconds": 3.0,
                                "end_seconds": 8.0,
                                "role": 'Master "A"',
                                "speaker_id": 'SPEAKER_00" onclick="alert(1)',
                            }
                        ],
                    },
                },
                {
                    "id": "current",
                    "question": "Analyse affichée au-dessus",
                    "result": {
                        "verdict": "Partiel",
                        "answer": "Résultat courant",
                        "coverage": {"role_mapping_confirmed": True},
                        "citations": [],
                    },
                },
            ]
        }

        rendered = _assistant_history_html(item)

        self.assertIn("Analyses précédentes · 1", rendered)
        self.assertIn("Ancienne &lt;script&gt;alert(1)&lt;/script&gt;", rendered)
        self.assertIn("Réponse &lt;b&gt;stockée&lt;/b&gt;", rendered)
        self.assertIn("&quot;sensible&quot;", rendered)
        self.assertNotIn("<script>", rendered)
        self.assertNotIn("Analyse affichée au-dessus", rendered)
        self.assertNotIn("Résultat courant", rendered)

    def test_assistant_history_can_show_latest_after_reset(self) -> None:
        item = {
            "assistant_queries": [
                {
                    "id": "only",
                    "question": "La seule analyse",
                    "result": {
                        "verdict": "Non retrouvé",
                        "answer": "Aucun passage concluant.",
                        "coverage": {"role_mapping_confirmed": True},
                        "citations": [],
                    },
                }
            ]
        }

        self.assertEqual(_assistant_history_html(item), "")
        restored = _assistant_history_html(item, include_latest=True)
        self.assertIn("La seule analyse", restored)
        self.assertIn("Aucun passage suffisamment probant", restored)

    def test_ambiguous_assistant_result_offers_relation_choices_and_short_excerpt(self) -> None:
        result = {
            "verdict": "À clarifier",
            "answer": "La question peut viser deux relations.",
            "confidence": 0.5,
            "backend": "déterministe-local",
            "coverage": {
                "turns_analyzed": 65,
                "duration_seconds": 1106.0,
                "role_mapping_confirmed": True,
            },
            "clarification_options": [
                {
                    "label": "OrionX et LyraX",
                    "question": "La relation entre OrionX et LyraX va-t-elle se terminer ?",
                },
                {
                    "label": "La cliente et OrionX",
                    "question": "La relation entre la cliente et OrionX va-t-elle se terminer ?",
                },
            ],
            "citations": [
                {
                    "excerpt": "J'ai des notions de fin de relation.",
                    "text": "J'ai des notions de fin de relation. Le passage complet continue ici.",
                    "start_seconds": 388.0,
                    "end_seconds": 431.0,
                    "speaker_id": "SPEAKER_00",
                    "role": "Master",
                }
            ],
        }

        rendered = _assistant_result_html(result)

        self.assertIn("À clarifier", rendered)
        self.assertIn("Quelle relation souhaitez-vous vérifier", rendered)
        self.assertIn("data-assistant-action='clarify'", rendered)
        self.assertIn("OrionX et LyraX", rendered)
        self.assertIn("Passages retenus", rendered)
        self.assertIn("Voir le passage complet", rendered)
        self.assertIn("J&#x27;ai des notions de fin de relation.", rendered)
        self.assertNotIn("Confiance élevée", rendered)

    def test_assistant_analysis_persists_roles_query_and_real_citations(self) -> None:
        state = self._discussion_state("audio-1")
        item = state["items"]["audio-1"]
        item["result"]["turns"] = [
            {
                "start": 0.0,
                "end": 3.0,
                "speaker_id": "SPEAKER_00",
                "speaker": "Voix 1",
                "text": "Bonjour, pouvez-vous me donner votre date de naissance ?",
            },
            {
                "start": 3.0,
                "end": 8.0,
                "speaker_id": "SPEAKER_01",
                "speaker": "Voix 2",
                "text": "Je suis ClienteX et je suis née le 31 février 2099.",
            },
        ]

        def mutate(mutator):
            mutator(state)
            return state

        with (
            patch("transcription_locale.ui.load_workspace", return_value=state),
            patch("transcription_locale.ui.mutate_workspace", side_effect=mutate),
            patch("transcription_locale.ui.get_default_reasoner", return_value=None),
        ):
            output = run_assistant_verification_ui(
                "audio-1",
                "La cliente dit qu'elle est née le 31 février 2099.",
                "SPEAKER_01",
                "SPEAKER_00",
                state,
            )

        saved_item = output[0]["items"]["audio-1"]
        self.assertTrue(saved_item["speaker_roles"]["SPEAKER_01"]["confirmed"])
        self.assertEqual(saved_item["speaker_roles"]["SPEAKER_00"]["role"], "Master")
        self.assertEqual(len(saved_item["assistant_queries"]), 1)
        citations = saved_item["assistant_queries"][0]["result"]["citations"]
        self.assertTrue(citations)
        self.assertEqual(citations[0]["speaker_id"], "SPEAKER_01")
        self.assertIn("Ouvrir une discussion", output[2])

    def test_assistant_citation_prefills_a_timecoded_discussion(self) -> None:
        event = SimpleNamespace(
            action="discuss",
            quote="Je suis née le 31 février 2099.",
            start_seconds=3.0,
            end_seconds=8.0,
            speaker_id="SPEAKER_01",
            role="Client",
        )

        output = prepare_discussion_from_assistant_ui("audio-1", event)
        anchor = json.loads(output[1])

        self.assertEqual(output[0], event.quote)
        self.assertEqual(output[3], "audio-1")
        self.assertEqual(anchor["source"], "assistant-citation")
        self.assertEqual(anchor["start_seconds"], 3.0)
        self.assertIn("Passage sélectionné", output[2])


if __name__ == "__main__":
    unittest.main()
