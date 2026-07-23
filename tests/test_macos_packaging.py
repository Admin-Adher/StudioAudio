from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "build-macos.yml"
BUILD_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_desktop_macos.sh"
REQUIREMENTS_PATH = PROJECT_ROOT / "packaging" / "requirements-macos.txt"
PYINSTALLER_SPEC_PATH = PROJECT_ROOT / "packaging" / "transcription_locale.spec"


class MacOSPackagingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        cls.build_script = BUILD_SCRIPT_PATH.read_text(encoding="utf-8")
        cls.requirements = REQUIREMENTS_PATH.read_text(encoding="utf-8")
        cls.pyinstaller_spec = PYINSTALLER_SPEC_PATH.read_text(encoding="utf-8")

    def test_workflow_builds_only_apple_silicon(self) -> None:
        self.assertIn('default: "0.3.8"', self.workflow)
        self.assertIn("runs-on: macos-15", self.workflow)
        self.assertIn('- "macos-v*.*.*"', self.workflow)
        self.assertIn('VERSION="${RELEASE_VERSION#macos-v}"', self.workflow)
        self.assertIn('VERSION="${VERSION#v}"', self.workflow)
        self.assertIn('build_desktop_macos.sh "$VERSION" arm64', self.workflow)
        self.assertIn(
            "TRANSCRIPTION_DESKTOP_VERSION:-0.3.8",
            self.build_script,
        )
        self.assertIn('EXECUTABLE_ARCHS" != "arm64', self.workflow)
        self.assertNotIn("strategy:", self.workflow)
        self.assertNotIn("macos-15-intel", self.workflow)
        self.assertNotIn("x86_64", self.workflow)
        self.assertIn('MACOSX_DEPLOYMENT_TARGET:-13.0', self.build_script)
        self.assertIn('"LSMinimumSystemVersion": "13.0"', self.pyinstaller_spec)

    def test_offline_dmg_contains_pinned_qwen3_mlx_model(self) -> None:
        self.assertIn(
            'ASSISTANT_MODEL_REPOSITORY="mlx-community/Qwen3-8B-4bit"',
            self.build_script,
        )
        self.assertIn(
            'ASSISTANT_MODEL_REVISION="545dc4251c05440727734bcd94334791f6ab0192"',
            self.build_script,
        )
        self.assertIn(
            'BUNDLE_ASSISTANT_MODEL="${3:-${BUNDLE_ASSISTANT_MODEL:-1}}"',
            self.build_script,
        )
        self.assertIn(
            "modeles/assistant/qwen3-8b-mlx-4bit",
            self.pyinstaller_spec,
        )
        self.assertIn("mlx==0.25.2", self.requirements)
        self.assertIn("mlx-lm==0.25.3", self.requirements)
        self.assertIn("model_is_reusable", self.build_script)
        self.assertIn(
            'ASSISTANT_MODEL_WEIGHTS_SIZE="4607835174"',
            self.build_script,
        )
        self.assertIn(
            'ASSISTANT_MODEL_WEIGHTS_SHA256="'
            'f2d29621aab300336ad645567ff38c42'
            'aac755513006ef4e8a579cf7ef5256d8"',
            self.build_script,
        )
        self.assertIn(
            "weights_path.stat().st_size == expected_size",
            self.build_script,
        )
        self.assertIn("file_sha256(weights_path)", self.build_script)
        self.assertIn('"weights_size": expected_size', self.build_script)
        self.assertIn('"weights_sha256": expected_sha256', self.build_script)
        self.assertLess(
            self.build_script.index("if actual_sha256 != expected_sha256:"),
            self.build_script.index(
                '(model_dir / "studio-audio-model.json").write_text'
            ),
        )
        self.assertIn("--platform macosx_13_0_arm64", self.build_script)
        self.assertIn("model.safetensors", self.workflow)
        self.assertIn(
            "545dc4251c05440727734bcd94334791f6ab0192",
            self.workflow,
        )
        self.assertIn("STUDIO_AUDIO_ASSISTANT_MODEL", self.workflow)
        self.assertIn("expected_size = 4_607_835_174", self.workflow)
        self.assertIn(
            '"f2d29621aab300336ad645567ff38c42"',
            self.workflow,
        )
        self.assertIn(
            'manifest.get("weights_sha256") != expected_sha256',
            self.workflow,
        )
        self.assertIn(
            'manifest.get("weights_size") != expected_size',
            self.workflow,
        )

    def test_workflow_tests_the_app_mounted_from_the_final_dmg(self) -> None:
        attach_position = self.workflow.index("hdiutil attach -nobrowse -readonly")
        smoke_position = self.workflow.index('[str(executable), "--smoke-test"]')
        assistant_smoke_position = self.workflow.index(
            '[str(executable), "--assistant-smoke-test"]'
        )
        publish_position = self.workflow.index("Publier uniquement le DMG arm64 validé")
        self.assertLess(attach_position, smoke_position)
        self.assertLess(smoke_position, assistant_smoke_position)
        self.assertLess(assistant_smoke_position, publish_position)
        self.assertIn(
            'STUDIO_AUDIO_APP="$MOUNT_POINT/Studio Audio.app"',
            self.workflow,
        )
        self.assertIn('report.get("status") != "ok"', self.workflow)
        self.assertIn('report.get("clean_shutdown") is not True', self.workflow)
        self.assertIn("200 <= http_status < 400", self.workflow)
        self.assertIn("lingering_after_exit", self.workflow)
        self.assertIn('hdiutil detach -force "$MOUNT_POINT"', self.workflow)

    def test_workflow_loads_bundled_qwen_and_validates_structured_json(self) -> None:
        self.assertIn('ASSISTANT_SMOKE_TIMEOUT_SECONDS: "600"', self.workflow)
        self.assertIn("timeout-minutes: 18", self.workflow)
        self.assertIn("export HF_HUB_OFFLINE=1", self.workflow)
        self.assertIn("export TRANSFORMERS_OFFLINE=1", self.workflow)
        self.assertIn('"assistant-smoke.json"', self.workflow)
        self.assertIn('report.get("mode") != "assistant-smoke"', self.workflow)
        self.assertIn('report.get("status") != "ok"', self.workflow)
        self.assertIn('response.get("ready") is not True', self.workflow)
        self.assertIn(
            "Path(str(model_value)).resolve() != expected_model",
            self.workflow,
        )
        self.assertIn("assistant-verification.json", self.workflow)

    def test_successful_workflow_publishes_only_the_dmg(self) -> None:
        release_step = self.workflow.split(
            "- name: Publier uniquement le DMG arm64 validé",
            maxsplit=1,
        )[1].split("- name: Résumé de livraison", maxsplit=1)[0]
        self.assertIn("path: ${{ steps.package.outputs.dmg }}", release_step)
        self.assertNotIn(".app.zip", self.workflow)
        self.assertNotIn("SHA256", self.workflow)
        self.assertNotIn("APP_ZIP", self.build_script)
        self.assertNotIn("CHECKSUM_PATH", self.build_script)
        self.assertIn('DMG_PATH="$PACKAGE_DIR/$PACKAGE_BASE.dmg"', self.build_script)

    def test_build_limits_disk_peak_without_removing_the_real_app(self) -> None:
        codesign_position = self.build_script.index(
            'codesign --verify --deep --strict --verbose=2 "$APP_PATH"'
        )
        workdir_cleanup_position = self.build_script.index(
            'case "$WORK_DIR" in',
            codesign_position,
        )
        model_cleanup_position = self.build_script.index(
            'rm -rf -- "$BUNDLED_ASSISTANT_MODEL_DIR"',
            workdir_cleanup_position,
        )
        venv_cleanup_position = self.build_script.index(
            'rm -rf -- "$DESKTOP_VENV"',
            model_cleanup_position,
        )
        staging_position = self.build_script.index(
            'STAGED_APP_PATH="$DMG_STAGING/Studio Audio.app"'
        )
        clone_position = self.build_script.index(
            'cp -cR "$APP_PATH" "$STAGED_APP_PATH"'
        )
        fallback_cleanup_position = self.build_script.index(
            'rm -rf -- "$STAGED_APP_PATH"',
            clone_position,
        )
        fallback_position = self.build_script.index(
            'ditto "$APP_PATH" "$STAGED_APP_PATH"',
            fallback_cleanup_position,
        )
        output_position = self.build_script.index(
            'echo "app=$APP_PATH"',
            fallback_position,
        )

        self.assertLess(codesign_position, workdir_cleanup_position)
        self.assertLess(workdir_cleanup_position, model_cleanup_position)
        self.assertLess(model_cleanup_position, venv_cleanup_position)
        self.assertLess(venv_cleanup_position, staging_position)
        self.assertLess(staging_position, clone_position)
        self.assertLess(clone_position, fallback_cleanup_position)
        self.assertLess(fallback_cleanup_position, fallback_position)
        self.assertLess(fallback_position, output_position)
        self.assertIn(
            "MACOS_PRUNE_MODEL_SOURCE_AFTER_BUILD",
            self.build_script,
        )
        self.assertIn(
            'MACOS_PRUNE_MODEL_SOURCE_AFTER_BUILD:-${GITHUB_ACTIONS:-false}',
            self.build_script,
        )
        self.assertIn(
            'MACOS_PRUNE_VENV_AFTER_BUILD:-${GITHUB_ACTIONS:-false}',
            self.build_script,
        )
        self.assertIn(
            '"$PROJECT_ROOT"/.venv-desktop-macos-"$PACKAGE_ARCH")',
            self.build_script,
        )
        self.assertIn(
            'stat -f \'%z\' "$STAGED_MODEL_WEIGHTS"',
            self.build_script,
        )
        staging_validation = self.build_script.split(
            "staged_app_is_complete() {",
            maxsplit=1,
        )[1].split(
            "# Sur APFS",
            maxsplit=1,
        )[0]
        self.assertNotIn("sha256", staging_validation.lower())
        self.assertNotIn('rm -rf -- "$APP_PATH"', self.build_script)
        self.assertIn('echo "dmg=$DMG_PATH"', self.build_script)


if __name__ == "__main__":
    unittest.main()
