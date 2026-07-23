from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts import download_assistant_model


def _write_fake_model(destination: Path, weights: bytes) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in download_assistant_model.REQUIRED_MODEL_FILES:
        path = destination / name
        if name == download_assistant_model.WEIGHTS_FILENAME:
            path.write_bytes(weights)
        else:
            path.write_text("test", encoding="utf-8")


class DownloadAssistantModelTests(unittest.TestCase):
    def _run_main(
        self,
        destination: Path,
        *,
        expected_hash: str,
        minimum_size: int = 1,
        downloader=None,
    ) -> int:
        argv = [
            "download_assistant_model.py",
            "--destination",
            str(destination),
            "--repository",
            "test/model",
            "--revision",
            "test-revision",
        ]
        with (
            patch.object(
                download_assistant_model,
                "EXPECTED_WEIGHTS_SHA256",
                expected_hash,
            ),
            patch.object(
                download_assistant_model,
                "MINIMUM_WEIGHTS_BYTES",
                minimum_size,
            ),
            patch.object(
                download_assistant_model,
                "snapshot_download",
                side_effect=downloader,
            ) as snapshot,
            patch.object(sys, "argv", argv),
        ):
            result = download_assistant_model.main()
        self.snapshot = snapshot
        return result

    def test_reuses_only_a_complete_model_with_the_exact_weight_hash(self) -> None:
        weights = b"valid-openvino-weights"
        expected_hash = hashlib.sha256(weights).hexdigest()
        with TemporaryDirectory() as temporary:
            destination = Path(temporary) / "model"
            _write_fake_model(destination, weights)

            result = self._run_main(
                destination,
                expected_hash=expected_hash,
                downloader=AssertionError("aucun téléchargement attendu"),
            )
            manifest = json.loads(
                (destination / "studio-audio-model.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(result, 0)
        self.snapshot.assert_not_called()
        self.assertEqual(manifest["weights_sha256"], expected_hash)

    def test_corrupt_weights_are_replaced_and_validated_before_manifest(self) -> None:
        valid_weights = b"new-valid-openvino-weights"
        expected_hash = hashlib.sha256(valid_weights).hexdigest()
        with TemporaryDirectory() as temporary:
            destination = Path(temporary) / "model"
            _write_fake_model(destination, b"corrupt-weights")
            (destination / "studio-audio-model.json").write_text(
                '{"stale": true}',
                encoding="utf-8",
            )

            def downloader(**kwargs) -> None:
                self.assertFalse(
                    (
                        destination
                        / download_assistant_model.WEIGHTS_FILENAME
                    ).exists()
                )
                _write_fake_model(destination, valid_weights)

            result = self._run_main(
                destination,
                expected_hash=expected_hash,
                downloader=downloader,
            )
            manifest = json.loads(
                (destination / "studio-audio-model.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(result, 0)
        self.snapshot.assert_called_once()
        self.assertEqual(manifest["weights_sha256"], expected_hash)

    def test_failed_post_download_hash_leaves_no_manifest(self) -> None:
        expected_hash = hashlib.sha256(b"expected").hexdigest()
        with TemporaryDirectory() as temporary:
            destination = Path(temporary) / "model"

            def downloader(**kwargs) -> None:
                _write_fake_model(destination, b"still-corrupt")

            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                self._run_main(
                    destination,
                    expected_hash=expected_hash,
                    downloader=downloader,
                )

            self.assertFalse(
                (destination / "studio-audio-model.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
