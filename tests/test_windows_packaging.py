from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "build-windows.yml"
INNO_SETUP_PATH = (
    PROJECT_ROOT / "packaging" / "windows" / "TranscriptionLocale.iss"
)
BUILD_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_desktop_windows.ps1"
PYINSTALLER_SPEC_PATH = PROJECT_ROOT / "packaging" / "transcription_locale.spec"


class WindowsPackagingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        cls.inno_setup = INNO_SETUP_PATH.read_text(encoding="utf-8")
        cls.build_script = BUILD_SCRIPT_PATH.read_text(encoding="utf-8")
        cls.pyinstaller_spec = PYINSTALLER_SPEC_PATH.read_text(encoding="utf-8")

    def test_workflow_builds_windows_x64_with_inno_setup(self) -> None:
        self.assertIn("workflow_dispatch:", self.workflow)
        self.assertIn('default: "0.3.8"', self.workflow)
        self.assertIn('- "v*.*.*"', self.workflow)
        self.assertIn("runs-on: windows-2025", self.workflow)
        self.assertIn("architecture: x64", self.workflow)
        self.assertIn("choco install innosetup", self.workflow)
        self.assertIn("scripts/build_desktop_windows.ps1", self.workflow)
        self.assertIn("-SkipPortable", self.workflow)
        self.assertIn('[string]$Version = "0.3.8"', self.build_script)
        self.assertIn('#define AppVersion "0.3.8"', self.inno_setup)
        self.assertIn(
            'TRANSCRIPTION_DESKTOP_VERSION", "0.3.8"',
            self.pyinstaller_spec,
        )
        self.assertIn(
            '$AssistantModelRepository = "OpenVINO/Qwen3-8B-int4-ov"',
            self.build_script,
        )
        self.assertIn(
            "5c47abf4b8e12ebe8e99745bb0c1ec17e0c0abcc",
            self.build_script,
        )
        self.assertIn("[switch]$SkipBundledAssistantModel", self.build_script)
        self.assertIn("download_assistant_model.py", self.build_script)
        for filename in (
            "chat_template.jinja",
            "openvino_config.json",
            "openvino_detokenizer.xml",
            "openvino_detokenizer.bin",
            "openvino_tokenizer.xml",
            "openvino_tokenizer.bin",
        ):
            self.assertIn(filename, self.build_script)
        self.assertIn(
            "f2e6960fde61d24a83c477c46f9ee7158748762089f2018fc2bb6a6d6c7f2a7e",
            self.build_script,
        )
        self.assertIn(
            "Get-FileHash -LiteralPath $WeightsPath -Algorithm SHA256",
            self.build_script,
        )
        self.assertNotIn("-SkipBundledAssistantModel", self.workflow)
        self.assertIn(
            "modeles/assistant/qwen3-8b-int4-ov",
            self.pyinstaller_spec,
        )

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
        self.assertIn(
            "packages/StudioAudio-Windows-x64-Setup-${{ steps.package.outputs.version }}*",
            release_step,
        )
        self.assertNotIn("steps.package.outputs.portable", release_step)
        self.assertIn("if-no-files-found: error", release_step)
        self.assertIn("compression-level: 0", release_step)
        self.assertIn("volume_count", self.workflow)
        self.assertIn("openvino_model.bin", self.workflow)
        self.assertIn("studio-audio-model.json", self.workflow)

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

    def test_build_script_runs_real_offline_qwen_inference(self) -> None:
        server_smoke_position = self.build_script.index(
            '-ArgumentList "--smoke-test"'
        )
        assistant_smoke_position = self.build_script.index(
            '-ArgumentList "--assistant-smoke-test"'
        )
        iscc_position = self.build_script.index("Get-Command ISCC.exe")
        self.assertLess(server_smoke_position, assistant_smoke_position)
        self.assertLess(assistant_smoke_position, iscc_position)
        self.assertIn("WaitForExit(600000)", self.build_script)
        self.assertIn("$AssistantSmokeProcess.WaitForExit()", self.build_script)
        self.assertIn(
            "$AssistantSmokeDetail = [string](",
            self.build_script,
        )
        self.assertIn('$env:HF_HUB_OFFLINE = "1"', self.build_script)
        self.assertIn('$env:TRANSFORMERS_OFFLINE = "1"', self.build_script)
        self.assertIn('"assistant-smoke.json"', self.build_script)
        self.assertIn(
            '$AssistantSmokeReport.status -ne "ok"',
            self.build_script,
        )
        self.assertIn(
            "$AssistantSmokeReport.response.ready -ne $true",
            self.build_script,
        )

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

    def test_large_offline_installer_uses_inno_volumes(self) -> None:
        self.assertIn("#ifdef BundledAssistantModel", self.inno_setup)
        self.assertIn("DiskSpanning=yes", self.inno_setup)
        self.assertIn("DiskSliceSize=1900000000", self.inno_setup)
        self.assertIn("SlicesPerDisk=1", self.inno_setup)
        self.assertIn('"/DBundledAssistantModel=1"', self.build_script)
        self.assertIn('"$InstallerBaseName-*.bin"', self.build_script)
        self.assertIn("Les volumes Inno Setup", self.build_script)
        self.assertIn("allowZip64=True", self.build_script)


if __name__ == "__main__":
    unittest.main()
