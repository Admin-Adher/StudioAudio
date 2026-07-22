from __future__ import annotations

import unittest

from transcription_locale.ui import (
    AUDIO_TIMELINE_HTML,
    CUSTOM_CSS,
    ROLE_EDITOR_JS,
    TIMELINE_INIT_JS,
)


class DesktopResponsiveCssTests(unittest.TestCase):
    def test_mobile_layout_does_not_expand_past_the_viewport(self) -> None:
        self.assertNotIn("width: calc(100% + 68px)", CUSTOM_CSS)
        self.assertIn("overflow-x: clip", CUSTOM_CSS)

    def test_role_editor_can_reflow_without_horizontal_scrolling(self) -> None:
        self.assertIn(".role-selector-grid", CUSTOM_CSS)
        self.assertIn("grid-template-columns: minmax(0,1fr) 150px minmax(0,1fr)", CUSTOM_CSS)
        self.assertIn("grid-template-columns: minmax(0,1fr) auto auto", CUSTOM_CSS)
        self.assertIn("overflow: visible !important", CUSTOM_CSS)
        self.assertIn("@media (max-width: 900px)", CUSTOM_CSS)
        self.assertIn(".role-editor:not(:has(.role-editor-heading))", CUSTOM_CSS)
        compact = CUSTOM_CSS.split("@media (max-width: 900px)", 1)[1].split(
            "@media (max-width: 720px)", 1
        )[0]
        self.assertIn(".role-selector-grid { grid-template-columns: 1fr", compact)

    def test_role_editor_actions_keep_cancel_and_save_on_the_same_responsive_row(self) -> None:
        self.assertIn('grid-template-areas: "detect cancel save"', CUSTOM_CSS)
        responsive = CUSTOM_CSS.split("@media (max-width: 900px)", 1)[1].split(
            "@media (max-width: 720px)", 1
        )[0]
        self.assertIn('grid-template-areas: "detect detect" "cancel save"', responsive)
        self.assertIn("#assistant-confirm-roles { grid-area: save;", CUSTOM_CSS)

    def test_role_editor_disclosure_exposes_its_accessible_state(self) -> None:
        self.assertIn('setAttribute("aria-expanded"', ROLE_EDITOR_JS)
        self.assertIn('setAttribute("aria-controls"', ROLE_EDITOR_JS)
        self.assertIn('setAttribute("role", "region")', ROLE_EDITOR_JS)
        self.assertIn('setAttribute("aria-labelledby"', ROLE_EDITOR_JS)

    def test_audio_waveform_keeps_pan_but_hides_native_scrollbar(self) -> None:
        self.assertIn("#audio-zoom-waveform > div::part(scroll)", CUSTOM_CSS)
        self.assertIn("scrollbar-width: none", CUSTOM_CSS)

    def test_desktop_window_stacks_the_workbench_at_1024_pixels(self) -> None:
        self.assertIn("@media (max-width: 1120px)", CUSTOM_CSS)
        responsive = CUSTOM_CSS.split("@media (max-width: 1120px)", 1)[1].split(
            "@media (max-width: 720px)", 1
        )[0]
        self.assertIn('[data-context-mode="assistant"] .transcript-pane', responsive)
        self.assertIn('[data-context-mode="discussions"] .transcript-pane', responsive)

    def test_primary_navigation_never_uses_a_native_horizontal_scrollbar(self) -> None:
        navigation = CUSTOM_CSS.split("#saas-nav {", 1)[1].split(
            ".page-shell", 1
        )[0]
        self.assertNotIn("overflow-x: auto", navigation)
        self.assertIn("overflow: hidden !important", navigation)
        self.assertIn("flex: 1 1 0 !important", navigation)

    def test_audio_tables_drop_secondary_columns_at_compact_widths(self) -> None:
        responsive = CUSTOM_CSS.split("@media (max-width: 1120px)", 1)[1].split(
            "@media (max-width: 720px)", 1
        )[0]
        self.assertIn('#queue-table .header-cell[data-heading="0"]', responsive)
        self.assertIn('#queue-table .header-cell[data-heading="4"]', responsive)
        self.assertIn('#fiche-library-table .header-cell[data-heading="1"]', responsive)
        self.assertIn("display: none !important", responsive)

    def test_queue_status_is_a_flat_note_instead_of_a_nested_card(self) -> None:
        self.assertIn("#queue-job-status", CUSTOM_CSS)
        self.assertIn("border-left: 2px solid", CUSTOM_CSS)

    def test_play_pause_icon_is_optically_centered_inside_its_circle(self) -> None:
        player = CUSTOM_CSS.split("#fiche-audio-player .play-pause-button {", 1)[1].split(
            "@media (max-width: 760px)", 1
        )[0]
        self.assertIn("width: 15px !important", player)
        self.assertIn(":has(polygon) svg", player)
        self.assertIn("transform: translateX(.75px)", player)

    def test_audio_timeline_opens_in_complete_view(self) -> None:
        self.assertIn('value="0"', AUDIO_TIMELINE_HTML)
        self.assertIn('aria-valuetext="Vue complète"', AUDIO_TIMELINE_HTML)
        self.assertIn("Enregistrement complet", AUDIO_TIMELINE_HTML)
        self.assertIn("zoom: 0", TIMELINE_INIT_JS)


if __name__ == "__main__":
    unittest.main()
