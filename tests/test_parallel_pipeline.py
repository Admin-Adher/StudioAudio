from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import transcription_locale.core as core


class ParallelPipelineTests(unittest.TestCase):
    def _run_with_source(self, callback):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "consultation.wav"
            source.write_bytes(b"audio")
            with patch(
                "faster_whisper.audio.decode_audio",
                return_value=np.zeros(16000, dtype=np.float32),
            ):
                return callback(source)

    @staticmethod
    def _completed_asr_checkpoint() -> dict[str, object]:
        return core._transcription_checkpoint(
            stage="transcription_complete",
            words=[core.WordStamp(0.0, 0.5, " Bonjour")],
            position=1.0,
            model_id="openvino/test",
            language="fr",
            language_probability=0.99,
        )

    def test_arc_starts_diarization_before_transcription_finishes(self) -> None:
        diarization_started = threading.Event()
        diarization_checkpoint_seen = threading.Event()
        checkpoints: list[dict[str, object]] = []
        progress_values: list[float] = []

        def save_checkpoint(payload: dict[str, object]) -> None:
            checkpoints.append(dict(payload))
            if payload.get("diarization_stage") == "complete":
                diarization_checkpoint_seen.set()

        def fake_diarize(*args, **kwargs):
            self.assertIsNone(kwargs["progress"])
            diarization_started.set()
            return [core.SpeakerSegment(0.0, 1.0, "SPEAKER_00")], []

        def fake_transcribe(*args, **kwargs):
            self.assertTrue(diarization_started.wait(2.0))
            # Le résultat de diarisation est durable alors que l'ASR n'a pas
            # encore publié son checkpoint final.
            self.assertTrue(diarization_checkpoint_seen.wait(2.0))
            kwargs["progress"](0.60, "ASR avancé")
            kwargs["progress"](0.30, "Repli simulé")
            kwargs["checkpoint"](self._completed_asr_checkpoint())
            return (
                [core.WordStamp(0.0, 0.5, " Bonjour")],
                "fr",
                0.99,
                "openvino/test",
            )

        def run(source: Path):
            with (
                patch.object(
                    core,
                    "platform_compatible_asr_backend",
                    return_value=core.OPENVINO_ARC_BACKEND,
                ),
                patch.object(core, "_diarize_audio", side_effect=fake_diarize),
                patch.object(core, "_transcribe_words", side_effect=fake_transcribe),
            ):
                return core.run_pipeline(
                    source,
                    asr_backend=core.OPENVINO_ARC_BACKEND,
                    requested_speakers=2,
                    checkpoint=save_checkpoint,
                    progress=lambda value, _message: progress_values.append(value),
                )

        result = self._run_with_source(run)

        self.assertEqual(result.turns[0].text, "Bonjour")
        self.assertEqual(progress_values, sorted(progress_values))
        self.assertTrue(checkpoints)
        self.assertEqual(checkpoints[-1]["stage"], "transcription_complete")
        self.assertEqual(checkpoints[-1]["diarization_stage"], "complete")
        self.assertTrue(checkpoints[-1]["speaker_segments"])

    def test_diarization_worker_can_propagate_an_error_for_retry(self) -> None:
        def fail_diarization(*args, **kwargs):
            raise RuntimeError("modèle indisponible")

        fake_module = SimpleNamespace(diarize=fail_diarization)
        with patch.dict(sys.modules, {"diarize": fake_module}):
            with self.assertRaisesRegex(RuntimeError, "modèle indisponible"):
                core._diarize_audio(
                    Path("audio.wav"),
                    duration=1.0,
                    requested_speakers=2,
                    progress=None,
                    raise_on_error=True,
                )
            segments, warnings = core._diarize_audio(
                Path("audio.wav"),
                duration=1.0,
                requested_speakers=2,
                progress=None,
            )

        self.assertEqual(segments[0].speaker, "SPEAKER_00")
        self.assertIn("RuntimeError", warnings[0])

    def test_cpu_backend_keeps_diarization_sequential(self) -> None:
        transcription_finished = threading.Event()
        order: list[str] = []

        def fake_transcribe(*args, **kwargs):
            order.append("transcription")
            self.assertFalse(transcription_finished.is_set())
            transcription_finished.set()
            return (
                [core.WordStamp(0.0, 0.5, " Bonjour")],
                "fr",
                0.99,
                "small",
            )

        def fake_diarize(*args, **kwargs):
            self.assertTrue(transcription_finished.is_set())
            order.append("diarisation")
            return [core.SpeakerSegment(0.0, 1.0, "SPEAKER_00")], []

        def run(source: Path):
            with (
                patch.object(core, "_diarize_audio", side_effect=fake_diarize),
                patch.object(core, "_transcribe_words", side_effect=fake_transcribe),
            ):
                return core.run_pipeline(
                    source,
                    asr_backend=core.FASTER_WHISPER_BACKEND,
                    requested_speakers=2,
                )

        self._run_with_source(run)
        self.assertEqual(order, ["transcription", "diarisation"])

    def test_parallel_diarization_error_retries_sequentially(self) -> None:
        diarization_modes: list[bool] = []

        def fake_transcribe(*args, **kwargs):
            kwargs["checkpoint"](self._completed_asr_checkpoint())
            return (
                [core.WordStamp(0.0, 0.5, " Bonjour")],
                "fr",
                0.99,
                "openvino/test",
            )

        def fake_diarize(*args, **kwargs):
            raise_on_error = bool(kwargs.get("raise_on_error"))
            diarization_modes.append(raise_on_error)
            if raise_on_error:
                raise RuntimeError("échec du worker")
            return (
                [core.SpeakerSegment(0.0, 1.0, "SPEAKER_00")],
                ["repli séquentiel utilisé"],
            )

        def run(source: Path):
            with (
                patch.object(
                    core,
                    "platform_compatible_asr_backend",
                    return_value=core.OPENVINO_ARC_BACKEND,
                ),
                patch.object(core, "_diarize_audio", side_effect=fake_diarize),
                patch.object(core, "_transcribe_words", side_effect=fake_transcribe),
            ):
                return core.run_pipeline(
                    source,
                    asr_backend=core.OPENVINO_ARC_BACKEND,
                    requested_speakers=2,
                    checkpoint=lambda _payload: None,
                )

        result = self._run_with_source(run)
        self.assertEqual(diarization_modes, [True, False])
        self.assertIn("repli séquentiel utilisé", result.warnings)

    def test_arc_resume_on_cpu_checkpoint_avoids_parallel_contention(self) -> None:
        transcription_finished = threading.Event()
        resume = self._completed_asr_checkpoint()
        resume["model"] = "small"
        resume["stage"] = "transcription"
        order: list[str] = []

        def fake_transcribe(*args, **kwargs):
            order.append("transcription")
            transcription_finished.set()
            return (
                [core.WordStamp(0.0, 0.5, " Bonjour")],
                "fr",
                0.99,
                "small",
            )

        def fake_diarize(*args, **kwargs):
            self.assertTrue(transcription_finished.is_set())
            order.append("diarisation")
            return [core.SpeakerSegment(0.0, 1.0, "SPEAKER_00")], []

        def run(source: Path):
            with (
                patch.object(
                    core,
                    "platform_compatible_asr_backend",
                    return_value=core.OPENVINO_ARC_BACKEND,
                ),
                patch.object(core, "_diarize_audio", side_effect=fake_diarize),
                patch.object(core, "_transcribe_words", side_effect=fake_transcribe),
            ):
                return core.run_pipeline(
                    source,
                    asr_backend=core.OPENVINO_ARC_BACKEND,
                    requested_speakers=2,
                    resume_checkpoint=resume,
                )

        self._run_with_source(run)
        self.assertEqual(order, ["transcription", "diarisation"])

    def test_completed_diarization_checkpoint_is_reused(self) -> None:
        resume = self._completed_asr_checkpoint()
        resume.update(
            {
                "diarization_stage": "complete",
                "speaker_segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "speaker": "SPEAKER_00",
                    }
                ],
                "diarization_warnings": [],
            }
        )

        def fake_transcribe(*args, **kwargs):
            self.assertIs(kwargs["resume_checkpoint"], resume)
            return (
                [core.WordStamp(0.0, 0.5, " Bonjour")],
                "fr",
                0.99,
                "openvino/test",
            )

        def run(source: Path):
            with (
                patch.object(
                    core,
                    "platform_compatible_asr_backend",
                    return_value=core.OPENVINO_ARC_BACKEND,
                ),
                patch.object(
                    core,
                    "_diarize_audio",
                    side_effect=AssertionError("La diarisation ne doit pas être relancée"),
                ),
                patch.object(core, "_transcribe_words", side_effect=fake_transcribe),
            ):
                return core.run_pipeline(
                    source,
                    asr_backend=core.OPENVINO_ARC_BACKEND,
                    requested_speakers=2,
                    resume_checkpoint=resume,
                )

        result = self._run_with_source(run)
        self.assertEqual(result.turns[0].speaker_id, "SPEAKER_00")


if __name__ == "__main__":
    unittest.main()
