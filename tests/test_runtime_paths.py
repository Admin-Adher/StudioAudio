from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from transcription_locale.runtime_paths import (
    resolve_app_data_root,
    resolve_resource_root,
)


class RuntimePathsTests(unittest.TestCase):
    def test_source_mode_keeps_the_existing_project_workspace(self) -> None:
        project = Path("C:/projet-transcription")
        self.assertEqual(
            resolve_app_data_root(
                frozen=False,
                environ={},
                project_root=project,
            ),
            project.resolve(),
        )

    def test_environment_override_has_priority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            expected = Path(temporary).resolve()
            self.assertEqual(
                resolve_app_data_root(
                    frozen=True,
                    platform_name="win32",
                    environ={"TRANSCRIPTION_APP_DATA_DIR": temporary},
                ),
                expected,
            )

    def test_frozen_windows_uses_local_app_data(self) -> None:
        self.assertEqual(
            resolve_app_data_root(
                frozen=True,
                platform_name="win32",
                environ={"LOCALAPPDATA": "C:/Users/Test/AppData/Local"},
                home=Path("C:/Users/Test"),
            ),
            Path("C:/Users/Test/AppData/Local/TranscriptionLocale"),
        )

    def test_frozen_macos_uses_application_support(self) -> None:
        self.assertEqual(
            resolve_app_data_root(
                frozen=True,
                platform_name="darwin",
                environ={},
                home=Path("/Users/test"),
            ),
            Path("/Users/test/Library/Application Support/TranscriptionLocale"),
        )

    def test_resource_override_is_supported(self) -> None:
        self.assertEqual(
            resolve_resource_root(
                environ={"TRANSCRIPTION_APP_RESOURCES_DIR": "C:/bundle"}
            ),
            Path("C:/bundle").resolve(),
        )


if __name__ == "__main__":
    unittest.main()
