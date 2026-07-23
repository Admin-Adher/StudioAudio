from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "build-windows.yml"
INNO_SETUP_PATH = (
    PROJECT_ROOT / "packaging" / "windows" / "TranscriptionLocale.iss"
)
BUILD_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_desktop_windows.ps1"


class WindowsPackagingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        cls.inno_setup = INNO_SETUP_PATH.read_text(encoding="utf-8")
        cls.build_script = BUILD_SCRIPT_PATH.read_text(encoding="utf-8")

    def test_workflow_builds_windows_x64_with_inno_setup(self) -> None:
        self.assertIn("workflow_dispatch:", self.workflow)
        self.assertIn('default: "0.3.1"', self.workflow)
        self.assertIn('- "v*.*.*"', self.workflow)
        self.assertIn("runs-on: windows-2025", self.workflow)
        self.assertIn("architecture: x64", self.workflow)
        self.assertIn("choco install innosetup", self.workflow)
        self.assertIn("scripts/build_desktop_windows.ps1", self.workflow)
        self.assertIn("-SkipPortable", self.workflow)

    def test_workflow_smokes_the_executable_installed_by_the_setup(self) -> None:
        install_position = self.workflow.index(
            "Installer silencieusement le véritable Setup"
        )
        installed_executable_position = self.workflow.index(
            '$installedExecutable = Join-Path $installDirectory "TranscriptionLocale.exe"'
        )
        smoke_position = self.workflow.index(
            '-FilePath $env:INSTALLED_EXECUTABLE'
        )
        publish_position = self.workflow.index(
            "Publier le Setup validé"
        )
        self.assertLess(install_position, installed_executable_position)
        self.assertLess(installed_executable_position, smoke_position)
        self.assertLess(smoke_position, publish_position)
        self.assertIn('"/VERYSILENT"', self.workflow)
        self.assertIn('"/DIR=`\"$installDirectory`\""', self.workflow)
        self.assertIn('-ArgumentList "--smoke-test"', self.workflow)
        self.assertIn("WaitForExit", self.workflow)
        self.assertIn(
            "smoke test bureau : ok",
            self.workflow,
        )
        self.assertIn("Get-CimInstance Win32_Process", self.workflow)
        self.assertIn("Un processus du binaire installé est resté actif", self.workflow)

    def test_workflow_publishes_the_validated_setup(self) -> None:
        release_step = self.workflow.split(
            "- name: Publier le Setup validé",
            maxsplit=1,
        )[1].split("- name: Résumé de livraison", maxsplit=1)[0]
        self.assertIn("${{ steps.package.outputs.setup }}", release_step)
        self.assertNotIn("steps.package.outputs.portable", release_step)
        self.assertIn("if-no-files-found: error", release_step)
        self.assertIn("compression-level: 0", release_step)

    def test_build_script_smokes_pyinstaller_output_before_packaging(self) -> None:
        smoke_position = self.build_script.index(
            '-ArgumentList "--smoke-test"'
        )
        archive_position = self.build_script.index("Compress-Archive")
        iscc_position = self.build_script.index("Get-Command ISCC.exe")
        self.assertLess(smoke_position, archive_position)
        self.assertLess(smoke_position, iscc_position)
        self.assertIn("WaitForExit(240000)", self.build_script)
        self.assertIn(
            "smoke test bureau : ok",
            self.build_script,
        )
        self.assertIn("Stop-Process -Id $SmokeProcess.Id -Force", self.build_script)

    def test_inno_setup_creates_a_per_user_launcher_and_shortcuts(self) -> None:
        self.assertIn("AppName=Studio Audio", self.inno_setup)
        self.assertIn("PrivilegesRequired=lowest", self.inno_setup)
        self.assertIn(
            r"DefaultDirName={localappdata}\Programs\Studio Audio",
            self.inno_setup,
        )
        self.assertIn(
            "OutputBaseFilename=StudioAudio-Windows-x64-Setup-{#AppVersion}",
            self.inno_setup,
        )
        self.assertIn(
            r'Name: "{autoprograms}\Studio Audio"; Filename: "{app}\TranscriptionLocale.exe"',
            self.inno_setup,
        )
        self.assertIn("skipifsilent", self.inno_setup)


if __name__ == "__main__":
    unittest.main()
