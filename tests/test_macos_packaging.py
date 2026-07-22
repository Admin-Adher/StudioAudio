from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "build-macos.yml"
BUILD_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_desktop_macos.sh"


class MacOSPackagingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        cls.build_script = BUILD_SCRIPT_PATH.read_text(encoding="utf-8")

    def test_workflow_builds_only_apple_silicon(self) -> None:
        self.assertIn("runs-on: macos-15", self.workflow)
        self.assertIn('build_desktop_macos.sh "$VERSION" arm64', self.workflow)
        self.assertIn('EXECUTABLE_ARCHS" != "arm64', self.workflow)
        self.assertNotIn("strategy:", self.workflow)
        self.assertNotIn("macos-15-intel", self.workflow)
        self.assertNotIn("x86_64", self.workflow)

    def test_workflow_tests_the_app_mounted_from_the_final_dmg(self) -> None:
        attach_position = self.workflow.index("hdiutil attach -nobrowse -readonly")
        smoke_position = self.workflow.index('[str(executable), "--smoke-test"]')
        publish_position = self.workflow.index("Publier uniquement le DMG arm64 validé")
        self.assertLess(attach_position, smoke_position)
        self.assertLess(smoke_position, publish_position)
        self.assertIn(
            'STUDIO_AUDIO_APP="$MOUNT_POINT/Studio Audio.app"',
            self.workflow,
        )
        self.assertIn('report.get("status") != "ok"', self.workflow)
        self.assertIn('report.get("clean_shutdown") is not True', self.workflow)
        self.assertIn("200 <= http_status < 400", self.workflow)
        self.assertIn("lingering_after_exit", self.workflow)
        self.assertIn('hdiutil detach -force "$MOUNT_POINT"', self.workflow)

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


if __name__ == "__main__":
    unittest.main()
