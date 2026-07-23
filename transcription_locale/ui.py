"""Interface Gradio en français avec file d'attente multi-audios."""

from __future__ import annotations

import copy
import hashlib
import html
import json
import os
import queue
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import gradio as gr

from .assistant_engine import (
    build_answer_sections,
    get_default_reasoner,
    suggest_speaker_roles,
    verify_claim,
)
from .audio_dynamics import (
    analyze_audio_dynamics,
    question_needs_audio_analysis,
)
from .core import (
    PROFILE_CONFIG,
    TranscriptTurn,
    openvino_arc_supported,
    platform_compatible_asr_backend,
    run_pipeline,
)
from .exports import (
    RESULTS_DIR,
    format_duration,
    format_timestamp,
    plain_transcript,
    result_from_rows,
    turns_to_rows,
    write_batch_archive,
    write_exports,
)
from .model_setup import (
    EXPECTED_DOWNLOAD_BYTES,
    downloaded_bytes,
    models_ready,
    prepare_all_models,
    setup_components_status,
)
from .jobs import JOB_MANAGER, JobSettings
from .workspace import (
    DATA_DIR,
    DEFAULT_TRANSCRIPTION_ENGINE,
    STANDARD_TRANSCRIPTION_ENGINE,
    blank_workspace,
    delete_workspace_item,
    import_audio,
    load_workspace,
    mutate_workspace,
    normalise_workspace,
    rename_library_audio,
    save_workspace,
    update_workspace_item,
)


ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
LANGUAGES = {
    "Français": "fr",
    "Détection automatique": None,
    "Anglais": "en",
    "Espagnol": "es",
    "Allemand": "de",
    "Italien": "it",
    "Portugais": "pt",
}

QUEUE_HEADERS = ["Ordre", "Audio", "Statut", "Progression", "Traitement", ""]
LIBRARY_HEADERS = ["Audio", "Ajouté le", "Statut", "Progression", "Discussions", ""]
LIBRARY_SORTS = [
    "Date d'ajout · plus récent",
    "Nom · A à Z",
    "Statut",
    "Progression",
]

OPENVINO_PILOT_ENGINE = "openvino-arc-pilot"


def _engine_choices() -> list[tuple[str, str]]:
    if openvino_arc_supported():
        return [
            ("Intel Arc · pilote recommandé", OPENVINO_PILOT_ENGINE),
            ("Standard local · compatibilité", STANDARD_TRANSCRIPTION_ENGINE),
        ]
    return [("Faster-Whisper · moteur local", STANDARD_TRANSCRIPTION_ENGINE)]


ENGINE_CHOICES = _engine_choices()


def _engine_settings_copy() -> str:
    if openvino_arc_supported():
        return (
            "<p class='step-title'>Accélération matérielle</p>"
            "<p class='step-copy'>Intel Arc est sélectionné par défaut pour le profil "
            "Équilibré. Faster-Whisper reprend automatiquement le traitement si "
            "l'Arc, OpenVINO ou le modèle ne sont pas disponibles.</p>"
        )
    return (
        "<p class='step-title'>Moteur local</p>"
        "<p class='step-copy'>Faster-Whisper est sélectionné automatiquement sur "
        "cette plateforme pour garantir un traitement local stable.</p>"
    )


CUSTOM_CSS = """
:root {
  --canvas: #f3eee6;
  --paper: rgba(255, 255, 255, .72);
  --paper-strong: rgba(255, 255, 255, .92);
  --ink: #302b26;
  --muted: #756d64;
  --line: rgba(112, 96, 78, .15);
  --accent: #65584c;
  --accent-hover: #4f453c;
  --sand: #e7dccd;
}
html, body {
  width: 100%;
  min-width: 0;
  overflow-x: hidden;
  overflow-x: clip;
  background: var(--canvas) !important;
}
.gradio-container {
  width: 100% !important;
  max-width: none !important;
  min-width: 0 !important;
  min-height: 100vh !important;
  margin: 0 !important;
  padding: clamp(16px, 2.5vw, 38px) !important;
  color: var(--ink) !important;
  background: var(--canvas) !important;
  font-family: Inter, Aptos, "Segoe UI", sans-serif !important;
  color-scheme: light !important;
  --background-fill-primary: #fffdfb;
  --background-fill-secondary: #f7f2ec;
  --body-background-fill: var(--canvas);
  --block-background-fill: rgba(255,255,255,.74);
  --block-label-background-fill: rgba(255,255,255,.88);
  --input-background-fill: rgba(255,255,255,.90);
  --button-secondary-background-fill: rgba(255,255,255,.84);
  --button-secondary-background-fill-hover: #f0e7dc;
  --button-secondary-text-color: var(--ink);
  --body-text-color: var(--ink);
  --body-text-color-subdued: var(--muted);
  --border-color-primary: rgba(112,96,78,.18);
  --border-color-accent: #8a715a;
  --color-accent: #8a715a;
  --color-accent-soft: #eadfd2;
  --table-even-background-fill: #fffdfb;
  --table-odd-background-fill: #fbf7f1;
  --table-border-color: rgba(112,96,78,.15);
  --table-row-focus: #f1e8dc;
}
#app-screen {
  width: 100%;
  max-width: 1580px;
  min-width: 0;
  margin: 0 auto;
}
#app-screen,
#app-screen * { box-sizing: border-box; }
.glass-card {
  width: 100%;
  max-width: 100%;
  min-width: 0;
  padding: 18px !important;
  border: 1px solid rgba(255,255,255,.90) !important;
  border-radius: 24px !important;
  background: var(--paper) !important;
  box-shadow: 0 18px 48px rgba(91, 76, 59, .09) !important;
  /* A backdrop filter turns position:fixed dropdown menus into card-relative
     elements in Chromium. The translucent fill keeps the glass look without
     moving Gradio's language menu off-screen. */
  backdrop-filter: none !important;
  -webkit-backdrop-filter: none !important;
  overflow: visible !important;
}
.glass-card > .glass-card, .setup-card > .setup-card {
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  backdrop-filter: none !important;
}
.glass-card > .styler, .setup-card > .styler {
  background: transparent !important;
  border: 0 !important;
}
.glass-card > .styler,
.glass-card .form {
  overflow: visible !important;
}
.app-header {
  padding: 22px 24px;
  margin-bottom: 16px;
  border: 1px solid rgba(255,255,255,.92);
  border-radius: 24px;
  background: rgba(255,255,255,.62);
  box-shadow: 0 16px 42px rgba(91,76,59,.08);
  backdrop-filter: blur(20px);
}
.app-header-top { display: flex; justify-content: space-between; align-items: center; gap: 24px; }
.app-header-copy { min-width: 0; }
.header-pills { display: flex; align-items: center; flex-wrap: nowrap; justify-content: flex-end; gap: 8px; }
.app-header h1 {
  margin: 5px 0 7px;
  color: var(--ink) !important;
  font-size: clamp(1.75rem, 3vw, 2.35rem);
  letter-spacing: -.035em;
}
.app-header p { margin: 0; max-width: 760px; color: var(--muted) !important; line-height: 1.5; }
.app-brand-line { display: flex; align-items: center; gap: 11px; margin-bottom: 7px; }
.app-brand-mark {
  display: grid;
  place-items: center;
  width: 34px;
  height: 34px;
  border: 1px solid rgba(112,96,78,.20);
  border-radius: 11px;
  color: #fff;
  background: var(--accent);
  box-shadow: 0 8px 18px rgba(77,64,52,.16);
  font-size: .78rem;
  font-weight: 800;
  letter-spacing: -.02em;
}
.app-brand-copy strong { display: block; color: var(--ink); font-size: .96rem; }
.app-brand-copy span { display: block; margin-top: 1px; color: var(--muted); font-size: .72rem; }
.eyebrow { color: #8a765f; font-size: .74rem; font-weight: 750; letter-spacing: .13em; text-transform: uppercase; }
.privacy-pill { flex: none; padding: 8px 12px; border: 1px solid var(--line); border-radius: 999px;
                color: #5f554b; background: rgba(255,255,255,.66); font-size: .84rem; }
.step-title { margin: 0 0 4px; color: var(--ink); font-size: 1.05rem; font-weight: 720; }
.step-copy { margin: 0 0 14px; color: var(--muted); font-size: .89rem; }
.primary-btn {
  min-height: 50px !important;
  color: #fff !important;
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  border-radius: 15px !important;
  box-shadow: 0 10px 24px rgba(77,64,52,.18) !important;
}
.primary-btn:hover { background: var(--accent-hover) !important; }
#app-screen button:hover:not(.primary-btn),
.setup-card button:hover {
  color: var(--ink) !important;
  border-color: #d8c8b6 !important;
  background: #ebe1d5 !important;
}
#app-screen .quality-selector .wrap { display: grid !important; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px !important; }
#app-screen .quality-selector label {
  min-height: 46px !important;
  justify-content: center !important;
  margin: 0 !important;
  padding: 10px 9px !important;
  border: 1px solid rgba(112,96,78,.16) !important;
  border-radius: 13px !important;
  color: #5f564d !important;
  background: rgba(255,255,255,.72) !important;
  box-shadow: 0 4px 12px rgba(91,76,59,.04) !important;
  transition: background .18s ease, color .18s ease, border-color .18s ease, transform .18s ease !important;
}
#app-screen .quality-selector label:hover {
  border-color: rgba(101,88,76,.40) !important;
  background: #f5eee5 !important;
  transform: translateY(-1px);
}
#app-screen .quality-selector label:has(input:checked) {
  color: #fff !important;
  border-color: var(--accent) !important;
  background: var(--accent) !important;
  box-shadow: 0 8px 18px rgba(77,64,52,.18) !important;
}
#app-screen .quality-selector label:has(input:checked) span { color: #fff !important; }
#app-screen .quality-selector input[type="radio"] { position: absolute !important; opacity: 0 !important; pointer-events: none !important; }
#app-screen .engine-selector .wrap {
  display: grid !important;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px !important;
}
#app-screen .engine-selector label {
  min-height: 54px !important;
  align-items: center !important;
  justify-content: center !important;
  margin: 0 !important;
  padding: 11px 14px !important;
  border: 1px solid rgba(112,96,78,.18) !important;
  border-radius: 15px !important;
  color: #5f564d !important;
  background: rgba(255,255,255,.78) !important;
  box-shadow: 0 5px 16px rgba(91,76,59,.05) !important;
  transition: background .18s ease, border-color .18s ease, transform .18s ease !important;
}
#app-screen .engine-selector label:hover {
  color: var(--ink) !important;
  border-color: rgba(101,88,76,.42) !important;
  background: #f6efe7 !important;
  transform: translateY(-1px);
}
#app-screen .engine-selector label:has(input:checked) {
  color: #fff !important;
  border-color: var(--accent) !important;
  background: linear-gradient(145deg, #756657, #5d5146) !important;
  box-shadow: 0 10px 22px rgba(77,64,52,.18) !important;
}
#app-screen .engine-selector label:has(input:checked) span { color: #fff !important; }
#app-screen .engine-selector input[type="radio"] { position: absolute !important; opacity: 0 !important; pointer-events: none !important; }
.engine-status {
  display: grid;
  gap: 7px;
  margin-top: 10px;
  padding: 14px 16px;
  border: 1px solid var(--line);
  border-radius: 16px;
  background: rgba(248,244,239,.86);
}
.engine-status-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.engine-status strong { color: var(--ink); font-size: .91rem; }
.engine-status p { margin: 0; color: var(--muted); font-size: .81rem; line-height: 1.5; }
.engine-status-badge {
  flex: 0 0 auto;
  padding: 5px 9px;
  border: 1px solid rgba(112,96,78,.16);
  border-radius: 999px;
  color: #695f55;
  background: rgba(255,255,255,.76);
  font-size: .7rem;
  font-weight: 700;
  letter-spacing: .015em;
}
.engine-status.is-ready { border-color: rgba(74,117,89,.24); background: rgba(239,247,241,.78); }
.engine-status.is-ready .engine-status-badge { color: #315e41; border-color: rgba(74,117,89,.22); background: #eef8f1; }
.engine-status.is-fallback { border-color: rgba(154,111,61,.25); background: rgba(251,245,235,.86); }
.engine-refresh { width: fit-content !important; min-width: 0 !important; }
#language-choice,
#rename-audio-select {
  position: relative !important;
  z-index: 10000 !important;
  overflow: visible !important;
  isolation: isolate;
}
.glass-card:has(#language-choice [aria-expanded="true"]),
.glass-card:has(#rename-audio-select [aria-expanded="true"]) {
  position: relative !important;
  z-index: 9999 !important;
  overflow: visible !important;
}
#language-choice .wrap,
#rename-audio-select .wrap {
  position: relative !important;
  z-index: 10001 !important;
  overflow: visible !important;
  color: var(--ink) !important;
  border: 1px solid rgba(112,96,78,.22) !important;
  background: rgba(255,255,255,.94) !important;
  box-shadow: none !important;
}
#language-choice .wrap:focus-within,
#rename-audio-select .wrap:focus-within {
  border-color: #8a715a !important;
  box-shadow: 0 0 0 3px rgba(138,113,90,.13) !important;
}
#language-choice .wrap-inner,
#language-choice .secondary-wrap,
#language-choice input,
#rename-audio-select .wrap-inner,
#rename-audio-select .secondary-wrap,
#rename-audio-select input {
  color: var(--ink) !important;
  background: transparent !important;
}
#language-choice .options,
#language-choice [role="listbox"],
#rename-audio-select .options,
#rename-audio-select [role="listbox"] {
  z-index: 10002 !important;
  opacity: 1 !important;
  color: var(--ink) !important;
  background: #fffdfb !important;
  border-color: var(--line) !important;
  box-shadow: 0 16px 34px rgba(72,58,45,.16) !important;
}
#language-choice [role="option"],
#rename-audio-select [role="option"] {
  color: #3f3933 !important;
  background: #fffdfb !important;
  opacity: 1 !important;
}
#language-choice [role="option"]:hover,
#language-choice [role="option"].active,
#rename-audio-select [role="option"]:hover,
#rename-audio-select [role="option"].active {
  color: #302b26 !important;
  background: #f1e8dc !important;
}
#language-choice [role="option"][aria-selected="true"],
#language-choice [role="option"].selected,
#rename-audio-select [role="option"][aria-selected="true"],
#rename-audio-select [role="option"].selected {
  color: #302b26 !important;
  background: #e7d9ca !important;
  font-weight: 650 !important;
}
.hint { color: var(--muted); font-size: .89rem; line-height: 1.45; }
.queue-count { margin-top: 4px; padding: 11px 13px; border: 1px solid var(--line); border-radius: 13px;
               color: #62594f; background: rgba(247,243,237,.74); }
.rename-confirmation { color: #69533f; font-weight: 650; }
.rename-panel { margin-top: 8px !important; }
.rename-row { align-items: end !important; }
.rename-row button { min-height: 44px !important; }
#audio-queue { min-height: 230px; }
#queue-table,
#fiche-library-table {
  min-width: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
#queue-table > .styler,
#fiche-library-table > .styler {
  min-width: 0 !important;
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
#queue-table .table-wrap,
#fiche-library-table .table-wrap {
  min-width: 0 !important;
  overflow-x: auto !important;
  border: 1px solid rgba(112,96,78,.14) !important;
  border-radius: 13px !important;
  background: rgba(255,253,251,.70) !important;
}
#queue-table .table-container,
#queue-table .table-wrap,
#queue-table .virtual-table-viewport,
#queue-table .header-table,
#queue-table .virtual-body,
#queue-table .virtual-row {
  width: 100% !important;
  min-width: 100% !important;
}
#queue-table .virtual-table-viewport { background: #fffdfb !important; }
#queue-table .header-table { table-layout: fixed !important; }
#queue-table table { font-size: .91rem; }
#queue-table .header-cell,
#fiche-library-table .header-cell,
#queue-table .header-cell *,
#fiche-library-table .header-cell * {
  white-space: nowrap !important;
  word-break: normal !important;
  overflow-wrap: normal !important;
}
#queue-table th { background: #eee6db !important; color: #544b43 !important; }
#queue-table .virtual-row,
#queue-table .body-cell,
#queue-table .cell-wrap,
#queue-table [data-testid^="cell-"] {
  color: #514940 !important;
  background: #fffdfb !important;
}
#queue-table .virtual-row:hover,
#queue-table .virtual-row:hover .body-cell,
#queue-table .virtual-row:hover .cell-wrap,
#queue-table .virtual-row:hover [data-testid^="cell-"] {
  color: #302b26 !important;
  background: #f5eee5 !important;
}
#queue-table [data-editable="false"] {
  color: inherit !important;
  background: transparent !important;
  opacity: 1 !important;
}
#queue-table .header-cell button { cursor: default !important; pointer-events: none !important; }
@media (min-width: 1121px) {
  #queue-table .header-cell[data-heading="0"],
  #queue-table .body-cell[data-col="0"] {
    width: 7% !important;
    min-width: 0 !important;
    max-width: none !important;
    flex: 0 0 7% !important;
    text-align: center !important;
  }
  #queue-table .header-cell[data-heading="1"],
  #queue-table .body-cell[data-col="1"] {
    width: 34% !important;
    min-width: 0 !important;
    max-width: none !important;
    flex: 0 0 34% !important;
  }
  #queue-table .header-cell[data-heading="2"],
  #queue-table .body-cell[data-col="2"] {
    width: 16% !important;
    min-width: 0 !important;
    max-width: none !important;
    flex: 0 0 16% !important;
  }
  #queue-table .header-cell[data-heading="3"],
  #queue-table .body-cell[data-col="3"] {
    width: 14% !important;
    min-width: 0 !important;
    max-width: none !important;
    flex: 0 0 14% !important;
  }
  #queue-table .header-cell[data-heading="4"],
  #queue-table .body-cell[data-col="4"] {
    width: calc(29% - 48px) !important;
    min-width: 0 !important;
    max-width: none !important;
    flex: 0 0 calc(29% - 48px) !important;
  }
}
#fiche-library-table table { font-size: .9rem; }
#fiche-library-table th { color: #544b43 !important; background: #eee6db !important; }
#fiche-library-table .virtual-row,
#fiche-library-table .body-cell,
#fiche-library-table .cell-wrap,
#fiche-library-table [data-testid^="cell-"] {
  color: #514940 !important;
  background: #fffdfb !important;
}
#fiche-library-table .virtual-row:hover,
#fiche-library-table .virtual-row:hover .body-cell,
#fiche-library-table .virtual-row:hover .cell-wrap,
#fiche-library-table .virtual-row:hover [data-testid^="cell-"] {
  color: #302b26 !important;
  background: #f1e8dc !important;
}
#fiche-library-table [data-editable="false"] {
  color: inherit !important;
  background: transparent !important;
  opacity: 1 !important;
}
#fiche-library-table .table-container,
#fiche-library-table .table-wrap,
#fiche-library-table .virtual-table-viewport,
#fiche-library-table .header-table,
#fiche-library-table .virtual-body,
#fiche-library-table .virtual-row {
  width: 100% !important;
  min-width: 100% !important;
}
#fiche-library-table .virtual-table-viewport {
  background: #fffdfb !important;
}
#fiche-library-table .header-table {
  table-layout: fixed !important;
}
@media (min-width: 1121px) {
  #fiche-library-table .header-cell[data-heading="0"],
  #fiche-library-table .body-cell[data-col="0"] {
    width: 31% !important;
    min-width: 0 !important;
    max-width: none !important;
    flex: 0 0 31% !important;
  }
  #fiche-library-table .header-cell[data-heading="1"],
  #fiche-library-table .body-cell[data-col="1"] {
    width: 18% !important;
    min-width: 0 !important;
    max-width: none !important;
    flex: 0 0 18% !important;
  }
  #fiche-library-table .header-cell[data-heading="2"],
  #fiche-library-table .body-cell[data-col="2"] {
    width: 14% !important;
    min-width: 0 !important;
    max-width: none !important;
    flex: 0 0 14% !important;
  }
  #fiche-library-table .header-cell[data-heading="3"],
  #fiche-library-table .body-cell[data-col="3"] {
    width: 13% !important;
    min-width: 0 !important;
    max-width: none !important;
    flex: 0 0 13% !important;
  }
  #fiche-library-table .header-cell[data-heading="4"],
  #fiche-library-table .body-cell[data-col="4"] {
    width: calc(24% - 48px) !important;
    min-width: 0 !important;
    max-width: none !important;
    flex: 0 0 calc(24% - 48px) !important;
  }
}
.library-toolbar { align-items: end !important; margin-bottom: 8px !important; }
.library-toolbar-copy { padding: 2px 2px 8px; }
.library-toolbar-copy h3 { margin: 3px 0 4px; color: var(--ink); font-size: 1.08rem; }
.library-toolbar-copy p { margin: 0; color: var(--muted); font-size: .82rem; }
.audio-player-card {
  margin: 0 0 14px !important;
  padding: 18px 20px 14px !important;
  overflow: visible !important;
  border: 1px solid rgba(112,96,78,.14) !important;
  border-radius: 18px !important;
  background: rgba(255,253,251,.82) !important;
  box-shadow: 0 8px 22px rgba(91,76,59,.045) !important;
}
.audio-player-card > .styler { border: 0 !important; background: transparent !important; }
.audio-player-card > .audio-player-card {
  margin: 0 !important;
  padding: 0 !important;
  overflow: visible !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
.audio-player-card > .audio-player-card > .styler {
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
.audio-timeline-panel {
  margin: 0;
  padding: 0;
  overflow: visible;
  border: 0;
  border-radius: 0;
  color: var(--ink);
  background: transparent;
}
.audio-timeline-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  margin-bottom: 10px;
}
.audio-timeline-heading {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 150px;
}
.audio-timeline-kicker {
  color: #8b7866;
  font-size: .67rem;
  font-weight: 760;
  letter-spacing: .11em;
  text-transform: uppercase;
}
#audio-timeline-zoom-status {
  color: #443c35;
  font-size: .86rem;
  font-weight: 680;
}
.audio-timeline-actions {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 7px;
  min-width: 0;
}
.audio-timeline-actions button {
  min-height: 32px;
  padding: 6px 10px;
  border: 1px solid rgba(103,87,72,.18);
  border-radius: 9px;
  color: #554a40;
  background: rgba(255,255,255,.76);
  font-size: .75rem;
  font-weight: 650;
  white-space: nowrap;
  box-shadow: 0 3px 9px rgba(84,69,54,.05);
  transition: border-color .16s ease, background .16s ease, color .16s ease, transform .16s ease;
}
.audio-timeline-actions button:hover:not(:disabled) {
  border-color: rgba(103,87,72,.34);
  color: #342e29;
  background: #f1e8dc;
  transform: translateY(-1px);
}
.audio-timeline-actions button:disabled {
  color: #a69a8e;
  background: rgba(244,239,233,.72);
  box-shadow: none;
  cursor: default;
}
#audio-timeline-range {
  width: clamp(110px, 17vw, 220px);
  height: 4px;
  margin: 0 2px;
  border: 0 !important;
  border-radius: 999px;
  appearance: none;
  -webkit-appearance: none;
  background: linear-gradient(90deg, #746455 var(--timeline-zoom, 0%), #ded5ca var(--timeline-zoom, 0%));
  cursor: pointer;
}
#audio-timeline-range::-webkit-slider-thumb {
  width: 17px;
  height: 17px;
  border: 3px solid #fffdfb;
  border-radius: 50%;
  appearance: none;
  -webkit-appearance: none;
  background: #6d5e50;
  box-shadow: 0 2px 8px rgba(66,53,42,.25);
}
#audio-timeline-range::-moz-range-thumb {
  width: 12px;
  height: 12px;
  border: 3px solid #fffdfb;
  border-radius: 50%;
  background: #6d5e50;
  box-shadow: 0 2px 8px rgba(66,53,42,.25);
}
#audio-zoom-waveform {
  position: relative;
  width: 100%;
  max-width: 100%;
  min-width: 0;
  min-height: 70px;
  padding: 8px 0 11px;
  overflow: hidden;
  border: 0;
  border-radius: 0;
  background: transparent;
}
#audio-zoom-waveform > div::part(scroll) {
  scrollbar-width: none;
  -ms-overflow-style: none;
}
#audio-zoom-waveform > div::part(scroll)::-webkit-scrollbar {
  width: 0;
  height: 0;
  display: none;
}
.audio-timeline-loading {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 54px;
  color: #887b6f;
  font-size: .78rem;
}
.audio-timeline-times {
  display: grid;
  grid-template-columns: minmax(54px, auto) 1fr minmax(54px, auto);
  align-items: center;
  gap: 12px;
  margin-top: 4px;
  padding: 8px 3px 0;
  border-top: 1px solid rgba(112,96,78,.1);
  color: #5b5148;
  font-size: .75rem;
  font-variant-numeric: tabular-nums;
}
.audio-timeline-times span { color: #8a7d71; text-align: center; }
.audio-timeline-times time:last-child { text-align: right; }
#fiche-audio-player {
  margin-top: 10px !important;
  padding: 10px 0 0 !important;
  overflow: visible !important;
  border: 0 !important;
  border-top: 1px solid rgba(112,96,78,.13) !important;
  border-radius: 0 !important;
  color: var(--ink) !important;
  background: transparent !important;
  box-shadow: none !important;
}
#fiche-audio-player > label,
#fiche-audio-player [data-testid="block-label"] {
  color: #51483f !important;
  background: transparent !important;
  border-color: transparent !important;
  opacity: 1 !important;
}
#fiche-audio-player .icon-button-wrapper,
#fiche-audio-player .icon-button,
#fiche-audio-player .component-wrapper,
#fiche-audio-player .waveform-container,
#fiche-audio-player .controls {
  border-color: rgba(112,96,78,.14) !important;
  background: transparent !important;
}
#fiche-audio-player .component-wrapper { padding: 6px 0 0 !important; }
#fiche-audio-player .waveform-container,
#fiche-audio-player .timestamps {
  display: none !important;
}
#fiche-audio-player .controls {
  margin-top: 0 !important;
  padding-top: 6px !important;
  border-top: 0 !important;
}
#fiche-audio-player button,
#fiche-audio-player button *,
#fiche-audio-player time,
#fiche-audio-player .timestamps,
#fiche-audio-player svg {
  color: #5b5046 !important;
  stroke: currentColor !important;
  opacity: 1 !important;
}
#fiche-audio-player .play-pause-button {
  display: grid !important;
  width: 32px !important;
  min-width: 32px !important;
  max-width: 32px !important;
  height: 32px !important;
  min-height: 32px !important;
  max-height: 32px !important;
  padding: 0 !important;
  place-items: center !important;
  overflow: hidden !important;
  color: #fff !important;
  background: var(--accent) !important;
  border-radius: 999px !important;
  box-shadow: 0 5px 13px rgba(77,64,52,.16) !important;
  line-height: 0 !important;
  transition: background .16s ease, box-shadow .16s ease, transform .16s ease !important;
}
#fiche-audio-player .play-pause-button svg,
#fiche-audio-player .play-pause-button svg * {
  color: #fff !important;
  fill: currentColor !important;
  stroke: currentColor !important;
}
#fiche-audio-player .play-pause-button svg {
  display: block !important;
  width: 15px !important;
  min-width: 15px !important;
  max-width: 15px !important;
  height: 15px !important;
  min-height: 15px !important;
  max-height: 15px !important;
  margin: 0 !important;
  flex: none !important;
  transform: none;
}
#fiche-audio-player .play-pause-button:has(polygon) svg {
  transform: translateX(.75px);
}
#fiche-audio-player .rewind svg,
#fiche-audio-player .skip svg {
  width: 19px !important;
  height: 19px !important;
}
#fiche-audio-player .play-pause-button:focus-visible {
  outline: 2px solid rgba(101,88,76,.42) !important;
  outline-offset: 3px !important;
}
#fiche-audio-player .play-pause-button:active {
  transform: scale(.94);
}
#fiche-audio-player button:hover { color: #3d352f !important; background: #eee4d8 !important; }
#fiche-audio-player .play-pause-button:hover {
  color: #fff !important;
  background: var(--accent-hover) !important;
  box-shadow: 0 6px 16px rgba(77,64,52,.21) !important;
}
@media (max-width: 760px) {
  .audio-timeline-toolbar { align-items: stretch; flex-direction: column; }
  .audio-timeline-actions { justify-content: flex-start; flex-wrap: wrap; }
  #audio-timeline-range { flex: 1 1 130px; width: auto; }
}
@media (max-width: 470px) {
  .audio-player-card { padding: 14px 12px 11px !important; }
  .audio-timeline-actions button { flex: 1 1 auto; }
  #audio-timeline-range { order: 4; flex-basis: 100%; margin: 7px 4px 3px; }
}
#app-screen .block, #app-screen .form, #app-screen .panel {
  border-color: var(--line) !important;
}
#app-screen input, #app-screen textarea, #app-screen select {
  background: rgba(255,255,255,.78) !important;
}
#app-screen .block-title,
#app-screen [data-testid="block-title"],
#app-screen .label-wrap,
#app-screen .label-wrap span,
#app-screen label > span,
#app-screen .accordion > button,
#app-screen .tab-nav button,
#app-screen th,
#app-screen td {
  color: #4e473f !important;
  opacity: 1 !important;
}
#app-screen .block-info,
#app-screen [data-testid="block-info"],
#app-screen .info,
#app-screen .caption,
#app-screen .secondary-wrap,
#app-screen .upload-container,
#app-screen .upload-container *,
#app-screen .file-preview-holder,
#app-screen .file-preview-holder * {
  color: #625b53 !important;
  opacity: 1 !important;
}
#audio-queue button,
#audio-queue button .wrap,
#audio-queue button .wrap * {
  color: #5d554d !important;
  opacity: 1 !important;
}
#audio-queue table,
#audio-queue tbody,
#audio-queue tr.file,
#audio-queue tr.file td {
  color: #514940 !important;
  background: #fffdfb !important;
  opacity: 1 !important;
}
#audio-queue tr.file:hover,
#audio-queue tr.file:hover td {
  color: #302b26 !important;
  background: #f1e8dc !important;
}
#audio-queue tr.file a,
#audio-queue tr.file span {
  color: inherit !important;
  opacity: 1 !important;
}
.inline-rename-hint {
  margin: 9px 0 0;
  color: var(--muted);
  font-size: .78rem;
}
.renameable-audio-title,
#fiche-library-table .body-cell[data-col="0"] [role="button"],
#queue-table .body-cell[data-col="1"] [role="button"] {
  cursor: text !important;
  border-radius: 6px;
  transition: color .15s ease, background .15s ease, box-shadow .15s ease;
}
.renameable-audio-title:hover .audio-title-text,
#fiche-library-table .body-cell[data-col="0"] [role="button"]:hover,
#queue-table .body-cell[data-col="1"] [role="button"]:hover {
  color: #3d342d !important;
  background: rgba(226,214,200,.48) !important;
  box-shadow: 0 0 0 4px rgba(226,214,200,.48);
}
.inline-name-editor {
  box-sizing: border-box;
  display: block;
  width: 100%;
  min-width: 0;
  border: 1px solid #8a715a !important;
  border-radius: 8px !important;
  color: var(--ink) !important;
  background: #fffdfb !important;
  box-shadow: 0 0 0 3px rgba(138,113,90,.14) !important;
  font: inherit !important;
  outline: 0 !important;
}
#fiche-library-table .inline-name-editor,
#queue-table .inline-name-editor {
  height: 28px;
  margin: -1px 0;
  padding: 3px 7px;
  border-color: #b9a58f !important;
  box-shadow: 0 0 0 2px rgba(138,113,90,.11) !important;
}
#fiche-library-table * {
  scrollbar-width: none;
}
#queue-table * {
  scrollbar-width: none;
}
#fiche-library-table *::-webkit-scrollbar,
#queue-table *::-webkit-scrollbar {
  display: none !important;
  width: 0 !important;
  height: 0 !important;
}
#fiche-library-table *::-webkit-scrollbar-button,
#queue-table *::-webkit-scrollbar-button {
  display: none !important;
  width: 0 !important;
  height: 0 !important;
}
#fiche-library-table .delete-action-cell,
#queue-table .delete-action-cell {
  width: 48px !important;
  min-width: 48px !important;
  max-width: 48px !important;
  flex: 0 0 48px !important;
  padding: 0 !important;
  overflow: visible !important;
  text-align: center !important;
}
#fiche-library-table .delete-action-cell .cell-wrap,
#queue-table .delete-action-cell .cell-wrap {
  position: relative !important;
  display: grid !important;
  width: 34px !important;
  height: 34px !important;
  margin: 0 auto !important;
  padding: 0 !important;
  place-items: center;
  overflow: hidden !important;
  border: 1px solid transparent;
  border-radius: 9px;
  color: transparent !important;
  cursor: pointer !important;
  transition: background .16s ease, border-color .16s ease, transform .16s ease;
}
#fiche-library-table .header-cell[data-heading="5"],
#queue-table .header-cell[data-heading="5"] {
  width: 48px !important;
  min-width: 48px !important;
  max-width: 48px !important;
  flex: 0 0 48px !important;
}
#fiche-library-table .delete-action-cell [role="button"],
#queue-table .delete-action-cell [role="button"] {
  position: absolute !important;
  inset: 0 !important;
  z-index: 2;
  overflow: hidden !important;
  color: transparent !important;
  font-size: 0 !important;
}
#fiche-library-table .delete-action-cell:hover .cell-wrap,
#queue-table .delete-action-cell:hover .cell-wrap {
  border-color: rgba(151,71,62,.16);
  background: rgba(151,71,62,.08) !important;
  transform: translateY(-1px);
}
#fiche-library-table .delete-action-cell.is-disabled .cell-wrap,
#queue-table .delete-action-cell.is-disabled .cell-wrap {
  border-color: transparent !important;
  background: transparent !important;
  cursor: not-allowed !important;
  opacity: .36;
  transform: none;
}
.delete-action-icon {
  position: absolute;
  top: 50%;
  left: 50%;
  z-index: 1;
  display: block;
  width: 17px;
  height: 17px;
  pointer-events: none;
  transform: translate(-50%, -50%);
}
.delete-action-icon svg {
  display: block;
  width: 100%;
  height: 100%;
}
.delete-confirm-backdrop {
  position: fixed;
  inset: 0;
  z-index: 10000;
  display: grid;
  padding: 24px;
  place-items: center;
  background: rgba(48,43,38,.28);
  backdrop-filter: blur(10px);
  opacity: 0;
  visibility: hidden;
  transition: opacity .18s ease, visibility .18s ease;
}
.delete-confirm-backdrop.is-open {
  opacity: 1;
  visibility: visible;
}
.delete-confirm-dialog {
  width: min(440px, 100%);
  padding: 26px;
  border: 1px solid rgba(112,96,78,.16);
  border-radius: 22px;
  background: rgba(255,253,250,.96);
  box-shadow: 0 24px 70px rgba(77,63,48,.22);
  color: var(--ink);
  transform: translateY(8px) scale(.985);
  transition: transform .18s ease;
}
.delete-confirm-backdrop.is-open .delete-confirm-dialog { transform: none; }
.delete-confirm-icon {
  display: grid;
  width: 42px;
  height: 42px;
  margin-bottom: 18px;
  place-items: center;
  border-radius: 13px;
  background: rgba(151,71,62,.09);
}
.delete-confirm-icon svg { width: 19px; height: 19px; }
.delete-confirm-dialog h3 { margin: 0 0 8px; font-size: 1.17rem; }
.delete-confirm-dialog p { margin: 0; color: var(--muted); line-height: 1.55; }
.delete-confirm-dialog strong { color: var(--ink); }
.delete-confirm-actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: 24px;
}
.delete-confirm-actions button {
  min-height: 40px;
  padding: 0 17px;
  border: 1px solid var(--line);
  border-radius: 11px;
  color: var(--ink);
  background: #fffdfb;
  font-weight: 700;
  cursor: pointer;
}
.delete-confirm-actions button:hover { background: #f5eee5 !important; color: var(--ink) !important; }
.delete-confirm-actions .delete-confirm-primary {
  border-color: #8b4b44;
  color: #fff !important;
  background: #8b4b44;
}
.delete-confirm-actions .delete-confirm-primary:hover {
  color: #fff !important;
  background: #713b36 !important;
}
.delete-confirm-actions button:disabled { cursor: wait; opacity: .58; }
.fiche-head .inline-name-editor {
  max-width: min(560px, 72vw);
  min-height: 40px;
  padding: 5px 9px;
  font-size: inherit !important;
  font-weight: inherit !important;
}
.recent-audio .inline-name-editor {
  height: 32px;
  padding: 4px 7px;
  font-size: .88rem !important;
  font-weight: 700 !important;
}
#audio-queue tbody td:first-child { position: relative !important; cursor: text !important; }
#audio-queue tbody tr:has(.inline-name-editor),
#audio-queue tbody tr:has(.inline-name-editor) td {
  min-height: 48px !important;
  height: auto !important;
  overflow: visible !important;
}
#audio-queue .inline-name-editor {
  position: relative;
  z-index: 20;
  display: block;
  width: 100%;
  min-height: 36px;
  margin: 2px 0;
  padding: 6px 9px;
  border: 1px solid #8a715a;
  border-radius: 8px;
  color: var(--ink);
  background: #fffdfb;
  box-shadow: 0 0 0 3px rgba(138,113,90,.14);
  font: inherit;
  outline: 0;
}
.rename-bridge,
#studio-audio-navigation-state { display: none !important; }
#app-screen:has(#studio-audio-navigation-state [data-ready="true"][data-fiche-view="library"]) #fiche-detail-hero,
#app-screen:has(#studio-audio-navigation-state [data-ready="true"][data-fiche-view="library"]) #fiche-detail-view,
#app-screen:has(#studio-audio-navigation-state [data-ready="true"]:not([data-context-mode="assistant"])) #assistant-panel,
#app-screen:has(#studio-audio-navigation-state [data-ready="true"]:not([data-context-mode="discussions"])) #discussion-panel {
  display: none !important;
}
#app-screen:has(#studio-audio-navigation-state [data-ready="true"][data-fiche-view="detail"]) #fiche-library-hero,
#app-screen:has(#studio-audio-navigation-state [data-ready="true"][data-fiche-view="detail"]) #fiche-library-view {
  display: none !important;
}
#audio-queue .block-label,
#audio-queue .block-label *,
#audio-queue [class*="block-label"],
#audio-queue [class*="block-label"] *,
#audio-queue [data-testid="block-label"],
#audio-queue [data-testid="block-label"] *,
#audio-queue [data-testid="block-info"],
#audio-queue [data-testid="block-title"],
#queue-table p,
#queue-table [data-testid="block-info"],
#queue-table [data-testid="block-title"] {
  color: #554d45 !important;
  opacity: 1 !important;
}
#app-screen input,
#app-screen textarea,
#app-screen select,
#app-screen [role="listbox"],
#app-screen [role="option"] {
  color: #302b26 !important;
  opacity: 1 !important;
}
#app-screen .options,
#app-screen [role="listbox"] {
  z-index: 12040 !important;
  border: 1px solid rgba(112,96,78,.18) !important;
  border-radius: 12px !important;
  color: var(--ink) !important;
  background: #fffdfb !important;
  box-shadow: 0 16px 34px rgba(72,58,45,.16) !important;
}
#app-screen [role="option"] {
  color: #3f3933 !important;
  background: #fffdfb !important;
}
#app-screen [role="option"]:hover,
#app-screen [role="option"].active,
#app-screen [role="option"][data-highlighted="true"] {
  color: var(--ink) !important;
  background: #f1e8dc !important;
}
#app-screen [role="option"][aria-selected="true"],
#app-screen [role="option"].selected {
  color: var(--ink) !important;
  background: #e7d9ca !important;
  font-weight: 650 !important;
}
#app-screen [role="option"][aria-disabled="true"] {
  color: #9b9289 !important;
  background: #fffdfb !important;
  opacity: .7 !important;
}
#queue-table .cell-menu-button,
#fiche-library-table .cell-menu-button {
  display: none !important;
}
#queue-table .selection-button,
#fiche-library-table .selection-button {
  display: none !important;
}
#queue-table .header-cell > .cell-wrap,
#fiche-library-table .header-cell > .cell-wrap {
  padding-right: 10px !important;
}
#app-screen .cell-menu[role="menu"] {
  z-index: 12020 !important;
  min-width: 190px !important;
  padding: 6px !important;
  border: 1px solid rgba(112,96,78,.18) !important;
  border-radius: 12px !important;
  color: var(--ink) !important;
  background: #fffdfb !important;
  box-shadow: 0 18px 38px rgba(72,58,45,.18) !important;
}
#app-screen .cell-menu[role="menu"] [role="menuitem"] {
  width: 100% !important;
  min-height: 36px !important;
  padding: 8px 10px !important;
  border: 0 !important;
  border-radius: 8px !important;
  color: #4e473f !important;
  background: transparent !important;
  opacity: 1 !important;
}
#app-screen .cell-menu[role="menu"] [role="menuitem"]:hover,
#app-screen .cell-menu[role="menu"] [role="menuitem"]:focus-visible {
  color: var(--ink) !important;
  background: #f1e8dc !important;
}
#app-screen .cell-menu[role="menu"] [role="menuitem"]:disabled {
  color: #9b9289 !important;
  background: transparent !important;
  opacity: .62 !important;
}
#app-screen .cell-menu[role="menu"] svg { color: currentColor !important; }
#app-screen .filter-menu,
#app-screen .dropdown-filter-options {
  z-index: 12031 !important;
  color: var(--ink) !important;
  border: 1px solid rgba(112,96,78,.18) !important;
  border-radius: 12px !important;
  background: #fffdfb !important;
  box-shadow: 0 18px 38px rgba(72,58,45,.18) !important;
}
#app-screen .dropdown-filter-options { z-index: 12032 !important; }
#app-screen .background:has(+ .filter-menu) {
  z-index: 12030 !important;
  background: rgba(48,43,38,.24) !important;
}
#app-screen .filter-menu input,
#app-screen .filter-menu select,
#app-screen .dropdown-filter-options input,
#app-screen .dropdown-filter-options select {
  color: var(--ink) !important;
  border-color: rgba(112,96,78,.2) !important;
  background: #fffdfb !important;
  color-scheme: light !important;
}
#app-screen .filter-menu button,
#app-screen .dropdown-filter-options button {
  color: #4e473f !important;
  background: transparent !important;
}
#app-screen .filter-menu button:hover,
#app-screen .dropdown-filter-options button:hover {
  color: var(--ink) !important;
  background: #f1e8dc !important;
}
#app-screen .filter-menu .check-button {
  color: #fff !important;
  border-color: var(--accent) !important;
  background: var(--accent) !important;
}
#app-screen .filter-menu .check-button:hover {
  color: #fff !important;
  background: var(--accent-hover) !important;
}
#app-screen select option,
#app-screen select optgroup {
  color: var(--ink) !important;
  background: #fffdfb !important;
}
#app-screen .quality-selector label:has(input:checked),
#app-screen .quality-selector label:has(input:checked) * ,
#app-screen .primary-btn,
#app-screen .primary-btn * {
  color: #fff !important;
}
.status-copy {
  padding: 12px 14px !important;
  overflow: visible !important;
}
.status-copy h3 {
  margin-left: 0 !important;
  padding-left: 2px !important;
  color: var(--ink) !important;
  line-height: 1.35 !important;
}
.status-banner { margin: 14px 0 !important; }
#queue-job-status {
  min-width: 0 !important;
  margin: 2px 0 0 !important;
  padding: 12px 14px !important;
  overflow: hidden !important;
  border: 1px solid rgba(112,96,78,.14) !important;
  border-radius: 14px !important;
  background: rgba(247,243,237,.72) !important;
  box-shadow: none !important;
}
#queue-job-status > .styler,
#queue-job-status [data-testid="markdown-wrapper"],
#queue-job-status .prose {
  margin: 0 !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
#queue-job-status [data-testid="markdown"].prose,
#queue-job-status .md.prose {
  padding: 0 !important;
  border-left: 0 !important;
}
#queue-job-status h3 {
  margin: 0 0 2px !important;
  color: var(--ink) !important;
  font-size: .92rem !important;
  line-height: 1.35 !important;
}
#queue-job-status p {
  margin: 0 !important;
  color: var(--muted) !important;
  font-size: .79rem !important;
  line-height: 1.45 !important;
}
.table-card {
  padding: 16px !important;
  box-shadow: 0 13px 36px rgba(91,76,59,.07) !important;
}
.queue-toolbar { align-items: end !important; margin-bottom: 8px !important; }
.queue-flow-note {
  min-width: 250px;
  padding: 12px 14px;
  border: 1px solid rgba(112,96,78,.14);
  border-radius: 14px;
  background: rgba(247,243,237,.68);
}
.queue-flow-note span,
.queue-flow-note strong { display: block; }
.queue-flow-note span {
  margin-bottom: 3px;
  color: var(--muted);
  font-size: .72rem;
}
.queue-flow-note strong { color: var(--ink); font-size: .86rem; }
#queue-empty-state {
  display: flex;
  align-items: center;
  min-height: 112px;
  padding: 22px;
  border: 1px dashed rgba(112,96,78,.22);
  border-radius: 16px;
  background: linear-gradient(145deg, rgba(255,255,255,.72), rgba(247,243,237,.68));
}
.queue-empty-copy { min-width: 0; }
.queue-empty-copy strong,
.queue-empty-copy span { display: block; }
.queue-empty-copy strong { margin-bottom: 4px; color: var(--ink); font-size: .96rem; }
.queue-empty-copy span { color: var(--muted); font-size: .82rem; line-height: 1.5; }
#queue-active-card:has(#queue-table .virtual-row) #queue-empty-state { display: none !important; }
#queue-active-card:not(:has(#queue-table .virtual-row)) #queue-table { display: none !important; }
.section-heading {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 20px;
  margin-bottom: 14px;
}
.section-heading h2 { margin: 3px 0 0; color: var(--ink); font-size: 1.18rem; letter-spacing: -.02em; }
.section-heading p { max-width: 460px; margin: 0; color: var(--muted); font-size: .86rem; text-align: right; }
.workspace-shell { padding: 24px !important; }
.workspace-heading { margin-bottom: 18px; }
[id="fiche-detail-view"] {
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
[id="fiche-detail-view"] > .styler {
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
.fiche-back-button {
  width: fit-content !important;
  min-width: 150px !important;
  margin-bottom: 10px !important;
  border-color: rgba(112,96,78,.18) !important;
  background: rgba(255,255,255,.72) !important;
}
#fiche-selector,
#comment-selector,
#library-sort { position: relative !important; z-index: 5000 !important; overflow: visible !important; }
#fiche-selector {
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
#fiche-selector .container > .wrap,
#comment-selector .container > .wrap,
#library-sort .container > .wrap {
  color: var(--ink) !important;
  border: 1px solid rgba(112,96,78,.22) !important;
  background: #fffdfb !important;
  box-shadow: none !important;
}
#fiche-selector [role="listbox"],
#comment-selector [role="listbox"],
#library-sort [role="listbox"] {
  z-index: 5002 !important;
  color: var(--ink) !important;
  background: #fffdfb !important;
  box-shadow: 0 16px 34px rgba(72,58,45,.16) !important;
}
#fiche-selector [role="option"],
#comment-selector [role="option"],
#library-sort [role="option"] { color: #3f3933 !important; background: #fffdfb !important; }
#fiche-selector [role="option"]:hover,
#fiche-selector [role="option"].active,
#comment-selector [role="option"]:hover,
#comment-selector [role="option"].active,
#library-sort [role="option"]:hover,
#library-sort [role="option"].active { color: #302b26 !important; background: #f1e8dc !important; }
.fiche-empty {
  display: flex;
  flex-direction: column;
  gap: 5px;
  margin: 14px 0 18px;
  padding: 22px;
  border: 1px dashed rgba(112,96,78,.24);
  border-radius: 18px;
  color: var(--muted);
  background: rgba(250,247,242,.62);
}
.fiche-empty strong { color: var(--ink); }
.fiche-head {
  margin: 12px 0 14px;
  padding: 20px;
  border: 1px solid rgba(112,96,78,.14);
  border-radius: 20px;
  background: linear-gradient(135deg, rgba(255,255,255,.92), rgba(244,237,228,.76));
  box-shadow: 0 8px 22px rgba(91,76,59,.045);
}
.fiche-title-row { display: flex; align-items: flex-start; justify-content: space-between; gap: 20px; }
.fiche-title-row h2 { margin: 4px 0 4px; color: var(--ink); font-size: 1.35rem; letter-spacing: -.025em; }
.fiche-title-row p { margin: 0; color: var(--muted); font-size: .82rem; }
.status-chip {
  flex: none;
  padding: 7px 11px;
  border: 1px solid transparent;
  border-radius: 999px;
  font-size: .78rem;
  font-weight: 750;
}
.status-waiting { color: #6d5946; border-color: #dac8b3; background: #f2e8dc; }
.status-running { color: #72501f; border-color: #e2bd7d; background: #fff0cf; }
.status-done { color: #365c49; border-color: #a9c9b8; background: #e5f2eb; }
.status-failed { color: #88473f; border-color: #dfb0aa; background: #f9e6e3; }
.fiche-progress { display: grid; grid-template-columns: 1fr auto; align-items: center; gap: 12px; margin-top: 18px; }
.fiche-progress progress { width: 100%; height: 7px; overflow: hidden; border: 0; border-radius: 999px; background: #ddd3c7; }
.fiche-progress progress::-webkit-progress-bar { border-radius: 999px; background: #ddd3c7; }
.fiche-progress progress::-webkit-progress-value { border-radius: 999px; background: var(--accent); }
.fiche-progress progress::-moz-progress-bar { border-radius: 999px; background: var(--accent); }
.fiche-progress span { color: var(--muted); font-size: .8rem; font-variant-numeric: tabular-nums; }
.fiche-live-stage { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 6px 16px; margin-top: 7px; color: var(--muted); font-size: .78rem; }
.fiche-live-stage strong { color: var(--ink); font-weight: 650; }
.fiche-metrics { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 8px; margin-top: 14px; }
.fiche-metrics span {
  padding: 5px 12px;
  border-left: 1px solid rgba(112,96,78,.12);
  border-radius: 0;
  color: var(--muted);
  background: transparent;
  font-size: .76rem;
}
.fiche-metrics span:first-child { padding-left: 0; border-left: 0; }
.fiche-metrics strong { display: block; margin-bottom: 2px; color: var(--ink); font-size: .88rem; }
.audio-detail-layout { align-items: stretch !important; gap: 0 !important; }
.audio-detail-layout > * { min-width: 0 !important; }
.transcript-pane,
.discussion-pane {
  padding: 22px !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
.discussion-pane { border-left: 1px solid var(--line) !important; }
.pane-heading { margin-bottom: 12px; }
.pane-heading h3 { margin: 3px 0 4px; color: var(--ink); font-size: 1.08rem; }
.pane-heading p { margin: 0; color: var(--muted); font-size: .82rem; line-height: 1.45; }
#fiche-transcript {
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
#fiche-transcript > .block,
#fiche-transcript > .container,
#fiche-transcript .container {
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
#fiche-transcript textarea {
  min-height: 500px !important;
  padding: 20px !important;
  color: #342e29 !important;
  border: 1px solid rgba(112,96,78,.14) !important;
  border-radius: 16px !important;
  background: rgba(255,253,251,.76) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.9) !important;
  line-height: 1.72 !important;
  cursor: text !important;
  resize: none !important;
}
#fiche-transcript textarea:focus-visible {
  outline: 2px solid #8a715a !important;
  outline-offset: 3px !important;
}
#comments-feed { padding: 0 !important; border: 0 !important; background: transparent !important; }
.discussion-feed { display: grid; gap: 12px; }
.discussion-empty {
  display: grid;
  gap: 7px;
  padding: 22px 4px;
  color: #625b53;
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
}
.discussion-empty strong { color: var(--ink); font-size: .94rem; }
.discussion-empty p { margin: 0; font-size: .82rem; line-height: 1.5; }
.discussion-thread {
  padding: 16px 0;
  border-top: 1px solid var(--line);
  background: transparent;
}
.discussion-thread:first-child { border-top: 0; padding-top: 4px; }
.discussion-thread.is-resolved { border-color: rgba(82,99,84,.18); }
.thread-head,
.thread-head > div,
.thread-actions,
.thread-delete-confirm > span + button { display: flex; align-items: center; }
.thread-head { justify-content: space-between; gap: 10px; flex-wrap: wrap; }
.thread-head > div { flex-wrap: wrap; gap: 7px; }
.thread-head strong { color: var(--ink); font-size: .86rem; }
.thread-head time { color: #625b53; font-size: .74rem; }
.thread-status {
  flex: none;
  padding: 5px 9px;
  color: #694e31;
  border: 1px solid rgba(157,113,65,.22);
  border-radius: 999px;
  background: #fbf1e4;
  font-size: .69rem;
  font-weight: 750;
}
.is-resolved .thread-status { color: #526354; border-color: rgba(82,99,84,.18); background: #edf3ed; }
.thread-quote {
  display: grid !important;
  gap: 6px !important;
  width: 100%;
  min-height: 0 !important;
  margin: 12px 0 10px !important;
  padding: 11px 13px !important;
  text-align: left !important;
  color: #4e473f !important;
  border: 0 !important;
  border-left: 3px solid #b69c7f !important;
  border-radius: 0 11px 11px 0 !important;
  background: #f7f2ec !important;
  box-shadow: none !important;
}
.thread-quote:hover { color: var(--ink) !important; background: #f1e8dc !important; }
.thread-quote span { color: #625b53; font-size: .7rem; font-weight: 750; letter-spacing: .02em; }
.thread-quote q {
  display: -webkit-box;
  overflow: hidden;
  overflow-wrap: anywhere;
  font-size: .8rem;
  line-height: 1.5;
  text-overflow: ellipsis;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 3;
}
.thread-message { margin: 0; color: #403a34; font-size: .86rem; line-height: 1.55; white-space: pre-wrap; overflow-wrap: anywhere; }
.thread-replies-label { margin: 12px 0 5px; color: #625b53; font-size: .7rem; font-weight: 700; }
.thread-replies { display: grid; gap: 7px; }
.thread-reply { padding: 9px 0 9px 13px; border-left: 2px solid rgba(112,96,78,.18); }
.thread-reply > div { display: flex; flex-wrap: wrap; gap: 7px; align-items: baseline; }
.thread-reply strong { color: var(--ink); font-size: .76rem; }
.thread-reply time { color: #625b53; font-size: .69rem; }
.thread-reply p { margin: 4px 0 0; color: #514940; font-size: .79rem; line-height: 1.45; white-space: pre-wrap; overflow-wrap: anywhere; }
.thread-actions { flex-wrap: wrap; gap: 5px; margin-top: 11px; }
.thread-actions button,
.thread-delete-confirm button {
  min-height: 42px !important;
  padding: 7px 10px !important;
  color: #514940 !important;
  border: 1px solid transparent !important;
  border-radius: 9px !important;
  background: transparent !important;
  box-shadow: none !important;
  font-size: .73rem !important;
  font-weight: 700 !important;
}
.thread-actions button:hover,
.thread-actions button:focus-visible { color: var(--ink) !important; border-color: rgba(112,96,78,.16) !important; background: #f1e8dc !important; }
.thread-actions .thread-delete:hover { color: #8b3f38 !important; background: #f8ebe8 !important; }
.thread-quote:focus-visible,
.thread-actions button:focus-visible,
.thread-delete-confirm button:focus-visible { outline: 2px solid #8a715a !important; outline-offset: 2px !important; }
.thread-delete-confirm {
  display: none;
  align-items: center;
  gap: 7px;
  margin-top: 10px;
  padding: 9px 11px;
  color: #6b3934;
  border-radius: 11px;
  background: #f8ebe8;
  font-size: .75rem;
}
.discussion-thread.is-confirming-delete .thread-actions { display: none; }
.discussion-thread.is-confirming-delete .thread-delete-confirm { display: flex; }
.thread-delete-confirm span { flex: 1; }
.thread-delete-confirm .confirm-delete { color: #fff !important; background: #8b3f38 !important; }
.selection-context {
  display: grid;
  gap: 8px;
  padding: 13px 0;
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
}
#comment-selection-context {
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
.selection-context > div,
.selection-context > header { display: flex; justify-content: space-between; gap: 10px; align-items: center; }
.selection-kicker { color: #625b53; font-size: .69rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
.selection-time { color: #625b53; font-size: .72rem; font-weight: 700; }
.selection-state {
  display: inline-flex;
  align-items: center;
  min-height: 22px;
  padding: 3px 8px;
  color: #6c6258;
  border: 1px solid rgba(112,96,78,.14);
  border-radius: 999px;
  background: rgba(255,255,255,.62);
  font-size: .65rem;
  font-weight: 740;
}
.selection-context strong { color: var(--ink); font-size: .87rem; }
.selection-context p { margin: 0; color: #625b53; font-size: .78rem; line-height: 1.45; }
.selection-context blockquote {
  max-height: 96px;
  margin: 0;
  overflow: auto;
  color: #403a34;
  font-size: .81rem;
  line-height: 1.5;
}
.reply-context {
  display: grid;
  gap: 7px;
  color: #625b53;
}
.reply-context > div { display: flex; justify-content: space-between; gap: 10px; align-items: center; }
.reply-context p { margin: 0; font-size: .78rem; line-height: 1.45; }
.reply-context q {
  overflow: hidden;
  color: #403a34;
  font-size: .78rem;
  line-height: 1.45;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.comment-composer {
  margin-top: 16px !important;
  padding: 0 !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
#comment-composer > .styler {
  background: transparent !important;
  box-shadow: none !important;
}
.comment-primary { min-height: 44px !important; color: #fff !important; background: var(--accent) !important; border-color: var(--accent) !important; }
.comment-primary:hover { color: #fff !important; background: var(--accent-hover) !important; }
.comment-primary:focus-visible,
.comment-composer-actions button:focus-visible,
.comment-reply-composer button:focus-visible,
#comment-composer .accordion > button:focus-visible {
  outline: 2px solid #8a715a !important;
  outline-offset: 2px !important;
}
.comment-composer-actions { align-items: center !important; gap: 8px !important; }
.comment-composer-actions button { min-height: 44px !important; }
.comment-reply-composer {
  margin-top: 16px !important;
  padding-top: 14px !important;
  border-top: 1px solid var(--line) !important;
  background: transparent !important;
}
.comment-reply-composer button { min-height: 44px !important; }
.comment-reply-composer:has(.reply-context.is-empty) { display: none !important; }
#comment-composer .accordion {
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
#comment-composer .accordion > button {
  padding-inline: 0 !important;
  color: #625b53 !important;
  background: transparent !important;
}
#comment-author input,
#comment-body textarea,
#comment-reply textarea { color: #342e29 !important; background: rgba(255,253,251,.84) !important; }
#comment-author input:focus-visible,
#comment-body textarea:focus-visible,
#comment-reply textarea:focus-visible { outline: 2px solid #8a715a !important; outline-offset: 2px !important; }
.discussion-live-status {
  position: absolute !important;
  width: 1px !important;
  height: 1px !important;
  padding: 0 !important;
  margin: -1px !important;
  overflow: hidden !important;
  clip: rect(0, 0, 0, 0) !important;
  white-space: nowrap !important;
  border: 0 !important;
}
#saas-nav {
  overflow: visible !important;
  border: 0 !important;
  background: transparent !important;
}
#saas-nav > .tab-wrapper {
  position: sticky;
  top: 10px;
  z-index: 3000;
  width: min(100%, 720px);
  max-width: 100%;
  margin: 0 auto 18px !important;
}
#saas-nav .tab-container[role="tablist"] {
  display: flex !important;
  gap: 4px !important;
  width: 100% !important;
  max-width: 100%;
  margin: 0 auto !important;
  padding: 5px !important;
  overflow: hidden !important;
  border: 1px solid rgba(255,255,255,.92) !important;
  border-radius: 16px !important;
  background: rgba(255,253,250,.90) !important;
  box-shadow: 0 12px 34px rgba(91,76,59,.10) !important;
  backdrop-filter: blur(18px) saturate(125%);
}
#saas-nav .tab-container[role="tablist"] button {
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  flex: 1 1 0 !important;
  width: auto !important;
  min-width: 0 !important;
  min-height: 40px !important;
  padding: 8px 10px !important;
  overflow: hidden !important;
  border: 1px solid transparent !important;
  border-radius: 11px !important;
  color: #665d54 !important;
  background: transparent !important;
  font-size: .86rem !important;
  font-weight: 650 !important;
  line-height: 1.15 !important;
  text-align: center !important;
  text-overflow: clip !important;
  white-space: nowrap !important;
  transition: color .18s ease, background .18s ease, box-shadow .18s ease !important;
}
#saas-nav .tab-container[role="tablist"]::-webkit-scrollbar {
  display: none !important;
  width: 0 !important;
  height: 0 !important;
}
#saas-nav .tab-container[role="tablist"] button:hover {
  color: var(--ink) !important;
  background: #f2e9de !important;
}
#saas-nav .tab-container[role="tablist"] button.selected,
#saas-nav .tab-container[role="tablist"] button[aria-selected="true"] {
  color: #fff !important;
  background: var(--accent) !important;
  box-shadow: 0 7px 16px rgba(77,64,52,.17) !important;
}
#saas-nav .overflow-menu > button {
  color: #665d54 !important;
  border: 1px solid var(--line) !important;
  border-radius: 10px !important;
  background: rgba(255,253,250,.94) !important;
}
#saas-nav .overflow-dropdown {
  z-index: 3500 !important;
  min-width: 190px !important;
  padding: 6px !important;
  border: 1px solid rgba(112,96,78,.17) !important;
  border-radius: 13px !important;
  background: #fffdfb !important;
  box-shadow: 0 18px 38px rgba(72,58,45,.18) !important;
}
#saas-nav .overflow-dropdown button {
  width: 100% !important;
  padding: 9px 11px !important;
  border: 0 !important;
  border-radius: 8px !important;
  color: #4e473f !important;
  background: transparent !important;
  text-align: left !important;
}
#saas-nav .overflow-dropdown button:hover,
#saas-nav .overflow-dropdown button:focus-visible {
  color: var(--ink) !important;
  background: #f1e8dc !important;
}
#saas-nav .overflow-dropdown button.selected {
  color: #4f4338 !important;
  background: #e7d9ca !important;
  font-weight: 700 !important;
}
#saas-nav .tabitem {
  overflow: visible !important;
  border: 0 !important;
  background: transparent !important;
}
.page-shell { gap: 14px !important; }
.page-hero {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 24px;
  margin-bottom: 2px;
  padding: 6px 4px 10px;
}
.page-hero h2 { margin: 4px 0 5px; color: var(--ink); font-size: 1.52rem; letter-spacing: -.03em; }
.page-hero p { max-width: 680px; margin: 0; color: var(--muted); line-height: 1.5; }
.page-hero-note {
  flex: none;
  padding: 8px 11px;
  border: 1px solid var(--line);
  border-radius: 999px;
  color: #665d54;
  background: rgba(255,255,255,.65);
  font-size: .78rem;
}
.dashboard-content { display: flex; flex-direction: column; gap: 16px; }
.dashboard-intro {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 24px;
  padding: 24px;
  border: 1px solid rgba(255,255,255,.92);
  border-radius: 24px;
  background: linear-gradient(135deg, rgba(255,255,255,.86), rgba(239,229,216,.74));
  box-shadow: 0 18px 48px rgba(91,76,59,.08);
}
.dashboard-intro h2 { margin: 4px 0 0; color: var(--ink); font-size: 1.55rem; letter-spacing: -.03em; }
.dashboard-intro p { max-width: 460px; margin: 0; color: var(--muted); text-align: right; line-height: 1.5; }
.kpi-grid { display: grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap: 12px; }
.kpi-grid article {
  min-height: 138px;
  padding: 18px;
  border: 1px solid rgba(255,255,255,.92);
  border-radius: 20px;
  background: rgba(255,255,255,.72);
  box-shadow: 0 12px 32px rgba(91,76,59,.07);
}
.kpi-grid span { display: block; color: var(--muted); font-size: .82rem; }
.kpi-grid strong { display: block; margin: 13px 0 6px; color: var(--ink); font-size: 2rem; line-height: 1; letter-spacing: -.04em; }
.kpi-grid small { color: #8a8178; font-size: .75rem; }
.kpi-grid .attention-kpi { background: linear-gradient(145deg, #6e6053, #574b41); }
.kpi-grid .attention-kpi span,
.kpi-grid .attention-kpi small { color: rgba(255,255,255,.72); }
.kpi-grid .attention-kpi strong { color: #fff; }
.dashboard-recent {
  padding: 22px;
  border: 1px solid rgba(255,255,255,.92);
  border-radius: 24px;
  background: rgba(255,255,255,.72);
  box-shadow: 0 16px 42px rgba(91,76,59,.08);
}
.dashboard-subhead { display: flex; justify-content: space-between; align-items: center; gap: 16px; margin-bottom: 12px; }
.dashboard-subhead h3 { margin: 0; color: var(--ink); font-size: 1.05rem; }
.dashboard-subhead span { color: var(--muted); font-size: .79rem; }
.recent-audio {
  display: grid;
  grid-template-columns: minmax(180px,1.4fr) minmax(160px,2fr) 50px;
  align-items: center;
  gap: 16px;
  padding: 13px 4px;
  border-top: 1px solid var(--line);
}
.recent-audio > div:first-child { min-width: 0; }
.recent-audio strong { display: block; overflow: hidden; color: var(--ink); font-size: .88rem; text-overflow: ellipsis; white-space: nowrap; }
.recent-audio span { display: block; margin-top: 2px; color: var(--muted); font-size: .74rem; }
.recent-audio small { color: var(--muted); text-align: right; font-variant-numeric: tabular-nums; }
.recent-progress { height: 6px; overflow: hidden; border-radius: 999px; background: #e5ddd3; }
.recent-progress i { display: block; height: 100%; border-radius: inherit; background: var(--accent); }
.dashboard-empty { display: flex; flex-direction: column; gap: 4px; padding: 28px 4px 8px; border-top: 1px solid var(--line); }
.dashboard-empty strong { color: var(--ink); }
.dashboard-empty span { color: var(--muted); font-size: .84rem; }
.launch-card { display: grid !important; gap: 10px !important; }
.launch-summary {
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: 18px;
  background: linear-gradient(145deg, rgba(255,255,255,.86), rgba(239,231,221,.74));
}
.launch-summary strong { display: block; margin: 9px 0 4px; color: var(--ink); font-size: 1.2rem; }
.launch-summary p { margin: 0 0 12px; color: #625950; }
.launch-summary small { color: var(--muted); line-height: 1.45; }
.settings-card { padding: 22px !important; }
.settings-note {
  padding: 14px 16px;
  border: 1px solid var(--line);
  border-radius: 15px;
  color: var(--muted);
  background: rgba(247,243,237,.78);
  font-size: .84rem;
  line-height: 1.5;
}
.settings-note strong { color: var(--ink); }
.report-summary { margin-bottom: 2px; }
.report-downloads { min-height: 180px; }
.report-editor-intro {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  margin: 4px 0 10px;
  padding: 13px 14px;
  border: 1px solid rgba(112,96,78,.14);
  border-radius: 14px;
  background: rgba(248,244,238,.72);
}
.report-editor-intro strong,
.report-editor-intro span { display: block; }
.report-editor-intro strong { color: var(--ink); font-size: .91rem; }
.report-editor-intro div > span { margin-top: 3px; color: var(--muted); font-size: .78rem; line-height: 1.4; }
.report-editor-status {
  flex: none;
  padding: 6px 9px;
  border: 1px solid rgba(112,96,78,.14);
  border-radius: 999px;
  color: #665b50;
  background: rgba(255,255,255,.78);
  font-size: .71rem;
  font-weight: 650;
}
.report-editor-launch {
  min-height: 43px !important;
  border-color: rgba(112,96,78,.18) !important;
  color: #4f463e !important;
  background: rgba(255,255,255,.82) !important;
}
.report-editor-launch:hover {
  background: #f1e8dc !important;
}
#setup-screen {
  min-height: calc(100vh - 44px);
  max-width: 690px;
  margin: 0 auto;
  justify-content: center;
}
.setup-card {
  padding: 38px 42px !important;
  text-align: center;
  border: 1px solid rgba(255,255,255,.94) !important;
  border-radius: 30px !important;
  background: rgba(255,255,255,.74) !important;
  box-shadow: 0 26px 70px rgba(91,76,59,.12) !important;
  backdrop-filter: blur(24px) saturate(125%);
}
.setup-copy h1 { margin: 10px 0 8px; color: var(--ink) !important; font-size: 2rem; letter-spacing: -.035em; }
.setup-copy p { max-width: 510px; margin: 0 auto 24px; color: var(--muted) !important; line-height: 1.55; }
.setup-progress { display: grid; grid-template-columns: 1fr auto; align-items: center; gap: 12px; margin: 4px 0 18px; }
.setup-progress progress { width: 100%; height: 8px; border: 0; border-radius: 999px; overflow: hidden; background: #ded6cd; }
.setup-progress progress::-webkit-progress-bar { background: #ded6cd; border-radius: 999px; }
.setup-progress progress::-webkit-progress-value { background: #8a715a; border-radius: 999px; }
.setup-progress progress::-moz-progress-bar { background: #8a715a; border-radius: 999px; }
.setup-progress span { min-width: 42px; color: var(--muted); font-size: .84rem; font-variant-numeric: tabular-nums; }
.model-grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 9px; margin-top: 18px; text-align: left; }
.model-item { display: flex; justify-content: space-between; gap: 12px; padding: 11px 13px;
              border: 1px solid var(--line); border-radius: 13px; background: rgba(248,245,240,.76); }
.model-item span { color: var(--muted); font-size: .84rem; }
.model-item strong { color: var(--ink); font-size: .84rem; font-weight: 650; }
.setup-footnote { margin-top: 16px; color: #8a8178; font-size: .80rem; text-align: center; }
.fiche-role-bar {
  display: grid;
  grid-template-columns: minmax(180px, .72fr) minmax(250px, 1.28fr) auto;
  gap: 12px 22px;
  align-items: center;
  width: 100%;
  max-width: 100%;
  min-width: 0;
  margin: 2px 0 0;
  padding: 14px 16px;
  overflow: visible;
  border-top: 1px solid rgba(112,96,78,.13);
  border-bottom: 1px solid rgba(112,96,78,.13);
  background: rgba(251,248,244,.48);
}
#assistant-role-summary,
#assistant-role-summary > .html-container,
#assistant-role-summary .prose {
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: visible !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
.fiche-role-heading { display: grid; gap: 6px; align-content: center; }
.fiche-role-heading h3 { margin: 0; color: var(--ink); font-size: .96rem; line-height: 1.2; }
.fiche-role-people { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 10px; min-width: 0; }
.fiche-role-person { display: grid; gap: 3px; min-width: 0; padding-left: 12px; border-left: 2px solid rgba(146,113,78,.24); }
.fiche-role-person small { color: #766d64; font-size: .66rem; font-weight: 760; letter-spacing: .07em; text-transform: uppercase; }
.fiche-role-person strong { overflow: hidden; color: var(--ink); font-size: .86rem; text-overflow: ellipsis; white-space: nowrap; }
.fiche-role-confidence {
  display: inline-flex !important;
  align-items: center;
  justify-self: start;
  min-height: 28px;
  padding: 5px 9px;
  color: #386b58;
  border: 1px solid rgba(69,124,101,.22);
  border-radius: 999px;
  background: #eef8f2;
  font-size: .70rem !important;
  font-weight: 720;
}
.fiche-role-confidence.needs-confirmation { color: #735223; border-color: #dfbd87; background: #fff3dc; }
.fiche-role-confidence.is-pending { color: #675f56; border-color: rgba(112,96,78,.18); background: #f4efe8; }
.role-edit-button {
  min-width: 176px;
  min-height: 42px;
  padding: 9px 14px;
  color: #514940;
  border: 1px solid rgba(112,96,78,.19);
  border-radius: 12px;
  background: rgba(255,255,255,.8);
  font: inherit;
  font-size: .78rem;
  font-weight: 720;
  cursor: pointer;
}
.role-edit-button:hover { color: var(--ink); background: #eee4d8; }
.role-edit-button:focus-visible { outline: 3px solid rgba(138,113,90,.24); outline-offset: 2px; }
.role-edit-button:disabled,
.role-edit-button[aria-disabled="true"] {
  color: #8a8178;
  border-color: rgba(112,96,78,.12);
  background: rgba(242,238,232,.72);
  cursor: not-allowed;
  opacity: 1;
}
.role-edit-button:disabled:hover,
.role-edit-button[aria-disabled="true"]:hover { color: #8a8178; background: rgba(242,238,232,.72); }
.role-edit-bridge {
  display: none !important;
  position: absolute !important;
  width: 1px !important;
  height: 1px !important;
  min-width: 1px !important;
  min-height: 1px !important;
  margin: -1px !important;
  padding: 0 !important;
  overflow: hidden !important;
  clip-path: inset(50%) !important;
  white-space: nowrap !important;
}
.role-editor {
  width: calc(100% - 32px) !important;
  max-width: none !important;
  min-width: 0 !important;
  margin: 12px 16px 20px !important;
  padding: 0 !important;
  overflow: visible !important;
  border: 1px solid rgba(112,96,78,.16) !important;
  border-radius: 18px !important;
  background: rgba(255,253,250,.84) !important;
  box-shadow: 0 14px 34px rgba(82,68,53,.07) !important;
}
.role-editor > .form,
.role-editor > .styler,
.role-editor > div {
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
  margin: 0 !important;
  background: transparent !important;
}
.role-editor:not(:has(.role-editor-heading)) {
  display: none !important;
  width: 0 !important;
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  border: 0 !important;
}
.role-editor-heading {
  display: grid;
  grid-template-columns: minmax(0,1fr) minmax(260px,.85fr);
  gap: 12px 28px;
  align-items: end;
  margin: 0;
  padding: 20px 22px 16px;
  border-bottom: 1px solid rgba(112,96,78,.12);
}
.role-editor-heading > div { display: grid; gap: 4px; min-width: 0; }
.role-editor-kicker { color: #897866; font-size: .66rem; font-weight: 780; letter-spacing: .08em; text-transform: uppercase; }
.role-editor-heading h3 { margin: 0; color: var(--ink); font-size: 1.04rem; line-height: 1.25; }
.role-editor-heading p { margin: 0; color: var(--muted); font-size: .78rem; line-height: 1.5; }
.role-selector-grid {
  display: grid !important;
  grid-template-columns: minmax(0,1fr) 150px minmax(0,1fr) !important;
  gap: 14px !important;
  align-items: center !important;
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
  margin: 0 !important;
  padding: 18px 22px !important;
  overflow: visible !important;
  background: transparent !important;
}
.role-selector-column {
  gap: 12px !important;
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
  padding: 16px !important;
  overflow: visible !important;
  border: 1px solid rgba(112,96,78,.14) !important;
  border-radius: 14px !important;
  background: rgba(255,255,255,.72) !important;
  box-shadow: none !important;
}
.role-swap-column {
  align-self: center !important;
  justify-content: center !important;
  width: 150px !important;
  max-width: 150px !important;
  min-width: 0 !important;
  padding: 0 !important;
  background: transparent !important;
}
.role-swap-column #assistant-swap-roles { width: 100% !important; min-width: 0 !important; }
#assistant-client-excerpt,
#assistant-master-excerpt,
#assistant-client-excerpt > .html-container,
#assistant-master-excerpt > .html-container,
#assistant-client-excerpt .prose,
#assistant-master-excerpt .prose {
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
.role-voice-sample {
  display: grid;
  gap: 5px;
  min-height: 86px;
  padding: 13px 0 0;
  border-top: 1px solid rgba(112,96,78,.12);
}
.role-voice-sample small { color: #766d64; font-size: .66rem; font-weight: 760; letter-spacing: .06em; text-transform: uppercase; }
.role-voice-sample p { margin: 0; color: #504941; font-size: .76rem; line-height: 1.48; }
.role-voice-sample span { color: #8b7c6c; font-size: .68rem; font-variant-numeric: tabular-nums; }
.role-editor-actions,
.role-editor-actions > .form,
.role-editor-actions > .styler > .form {
  display: grid !important;
  grid-template-columns: minmax(0,1fr) auto auto !important;
  grid-template-areas: "detect cancel save" !important;
  align-items: center !important;
  gap: 10px !important;
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
  margin: 0 !important;
  padding: 14px 22px 18px !important;
  border-top: 1px solid rgba(112,96,78,.13) !important;
  background: transparent !important;
}
.role-editor-actions #assistant-auto-roles { grid-area: detect; justify-self: start !important; width: auto !important; min-width: 180px !important; margin: 0 !important; }
.role-editor-actions #assistant-cancel-roles,
.role-editor-actions #assistant-confirm-roles { justify-self: end !important; width: auto !important; min-width: 150px !important; margin: 0 !important; }
.role-editor-actions #assistant-cancel-roles { grid-area: cancel; }
.role-editor-actions #assistant-confirm-roles { grid-area: save; }
.role-editor-actions > * { flex: none !important; max-width: 100% !important; }
.fiche-detail-toolbar { align-items: center !important; flex-wrap: nowrap !important; gap: 10px !important; margin: 0 0 12px !important; }
.fiche-detail-toolbar .fiche-back-button { flex: 0 0 auto !important; min-width: 154px !important; }
.fiche-detail-toolbar #fiche-selector { flex: 1 1 auto !important; width: auto !important; min-width: 240px !important; max-width: none !important; }
#assistant-auto-roles,
#assistant-swap-roles,
#assistant-cancel-roles { color: #514940 !important; border-color: rgba(112,96,78,.16) !important; background: rgba(255,255,255,.72) !important; }
#assistant-auto-roles:hover,
#assistant-swap-roles:hover,
#assistant-cancel-roles:hover { color: var(--ink) !important; background: #eee4d8 !important; }
#assistant-confirm-roles { color: #fff !important; border-color: var(--accent) !important; background: var(--accent) !important; }
#assistant-confirm-roles:hover { color: #fff !important; background: var(--accent-hover) !important; }
#assistant-client-role,
#assistant-master-role { position: relative !important; z-index: 4200 !important; }
#assistant-client-role .container > .wrap,
#assistant-master-role .container > .wrap { background: #fffdfb !important; }
#assistant-client-role .options,
#assistant-client-role [role="listbox"],
#assistant-master-role .options,
#assistant-master-role [role="listbox"] {
  z-index: 12060 !important;
  color: var(--ink) !important;
  background: #fffdfb !important;
  border: 1px solid rgba(112,96,78,.18) !important;
  box-shadow: 0 16px 34px rgba(72,58,45,.16) !important;
}
#assistant-client-role [role="option"],
#assistant-master-role [role="option"] { color: #3f3933 !important; background: #fffdfb !important; }
#assistant-client-role [role="option"]:hover,
#assistant-client-role [role="option"].active,
#assistant-master-role [role="option"]:hover,
#assistant-master-role [role="option"].active { color: #302b26 !important; background: #f1e8dc !important; }
.role-action-button { min-height: 42px !important; white-space: nowrap !important; }
.fiche-context-nav {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  margin: 18px 0 0;
  padding: 0 4px 12px;
  border-bottom: 1px solid rgba(112,96,78,.14);
}
.fiche-context-nav-copy { min-width: 0; }
.fiche-context-nav-copy strong { display: block; color: var(--ink); font-size: .88rem; }
.fiche-context-nav-copy span { display: block; margin-top: 2px; color: var(--muted); font-size: .74rem; }
#fiche-context-mode { flex: none; width: auto !important; min-width: min(100%, 350px); }
#fiche-context-mode > .wrap,
#fiche-context-mode .wrap {
  display: flex !important;
  gap: 4px !important;
  padding: 4px !important;
  border: 1px solid rgba(112,96,78,.15) !important;
  border-radius: 13px !important;
  background: rgba(244,238,230,.76) !important;
}
#fiche-context-mode label {
  flex: 1 1 auto !important;
  min-height: 38px !important;
  margin: 0 !important;
  padding: 7px 10px !important;
  justify-content: center !important;
  border: 0 !important;
  border-radius: 9px !important;
  color: #665d54 !important;
  background: transparent !important;
  font-size: .76rem !important;
  font-weight: 720 !important;
}
#fiche-context-mode label:has(input:checked),
#fiche-context-mode label.selected {
  color: #fff !important;
  background: var(--accent) !important;
  box-shadow: 0 5px 12px rgba(77,64,52,.15) !important;
}
#fiche-context-mode label:has(input:checked) *,
#fiche-context-mode label.selected * { color: #fff !important; }
#fiche-context-mode input { position: absolute !important; opacity: 0 !important; pointer-events: none !important; }
.fiche-workbench { align-items: stretch !important; gap: 0 !important; }
.fiche-workbench > * { min-width: 0 !important; }
.fiche-workbench .transcript-pane:focus-visible,
.fiche-workbench .context-rail:focus-visible {
  outline: 2px solid #8f765a;
  outline-offset: -3px;
}
.fiche-workbench[data-context-mode="transcript"] .context-rail { display: none !important; }
.fiche-workbench[data-context-mode="transcript"] .transcript-pane {
  flex-grow: 1 !important;
  width: 100% !important;
  max-width: none !important;
}
.fiche-workbench[data-context-mode="assistant"] .shared-context-shell:has(.selection-context.is-empty) { display: none !important; }
.fiche-workbench[data-context-mode="assistant"],
.fiche-workbench[data-context-mode="discussions"] {
  display: block !important;
  height: auto !important;
  min-height: 0 !important;
  overflow: visible !important;
}
.fiche-workbench[data-context-mode="assistant"] .transcript-pane,
.fiche-workbench[data-context-mode="discussions"] .transcript-pane {
  display: none !important;
}
.fiche-workbench[data-context-mode="assistant"] .context-rail,
.fiche-workbench[data-context-mode="discussions"] .context-rail {
  display: block !important;
  width: 100% !important;
  min-width: 0 !important;
  max-width: none !important;
  height: auto !important;
  overflow: visible !important;
  border-top: 0 !important;
  border-left: 0 !important;
}
.fiche-workbench[data-context-mode="assistant"] .context-rail > .styler,
.fiche-workbench[data-context-mode="discussions"] .context-rail > .styler {
  width: min(100%, 940px) !important;
  margin-inline: auto !important;
}
.context-rail {
  position: relative;
  padding: 22px !important;
  border-left: 1px solid var(--line) !important;
  background: linear-gradient(180deg, rgba(248,243,237,.44), rgba(255,255,255,.05)) !important;
}
.context-rail > .styler,
.assistant-panel > .styler,
.discussion-panel > .styler {
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
.shared-context-shell { margin: 0 0 14px; padding: 0 !important; border: 0 !important; background: transparent !important; box-shadow: none !important; }
.shared-context-shell > .styler { padding: 0 !important; border: 0 !important; background: transparent !important; box-shadow: none !important; }
.shared-context-shell .selection-context {
  gap: 5px;
  padding: 11px 13px;
  border: 1px solid rgba(112,96,78,.13);
  border-radius: 12px;
  background: rgba(255,253,250,.58);
}
.shared-context-shell .selection-context.is-empty {
  grid-template-columns: minmax(0,1fr);
  border-style: dashed;
  background: rgba(248,244,239,.5);
}
.shared-context-shell .selection-context strong { font-size: .80rem; }
.shared-context-shell .selection-context p { font-size: .73rem; line-height: 1.4; }
.shared-context-shell .selection-context blockquote {
  display: -webkit-box;
  max-height: none;
  overflow: hidden;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 3;
}
.assistant-panel,
.discussion-panel { padding: 0 !important; border: 0 !important; background: transparent !important; box-shadow: none !important; }
.assistant-intro {
  display: grid;
  gap: 5px;
  margin-bottom: 11px;
}
.assistant-intro > div { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-start; gap: 8px; }
.assistant-intro-title { display: grid; gap: 2px; min-width: 0; }
.assistant-step { color: #8a7764; font-size: .64rem; font-weight: 820; letter-spacing: .09em; text-transform: uppercase; }
.assistant-intro strong,
.assistant-intro h3 { margin: 0; color: var(--ink); font-size: .96rem; line-height: 1.3; }
.assistant-intro p { margin: 0; color: #625b53; font-size: .78rem; line-height: 1.48; }
.assistant-local-badge {
  padding: 5px 8px;
  color: #4f604f;
  border: 1px solid rgba(79,96,79,.18);
  border-radius: 999px;
  background: #edf3ed;
  font-size: .67rem;
  font-weight: 760;
}
#assistant-panel > .styler > .form,
#assistant-question {
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
#assistant-question { overflow: visible !important; }
#assistant-question [data-testid="block-info"] {
  margin-bottom: 6px !important;
  color: #4f4841 !important;
  font-size: .74rem !important;
  font-weight: 740 !important;
}
#assistant-question textarea {
  min-height: 88px !important;
  padding: 12px 14px !important;
  color: #342e29 !important;
  border: 1px solid rgba(112,96,78,.18) !important;
  border-radius: 13px !important;
  background: rgba(255,253,251,.9) !important;
  box-shadow: 0 1px 0 rgba(82,67,52,.03) !important;
  line-height: 1.5 !important;
  transition: border-color .18s ease, box-shadow .18s ease, background .18s ease !important;
}
#assistant-question textarea:hover { border-color: rgba(112,96,78,.3) !important; }
#assistant-question textarea:focus {
  border-color: #8a715a !important;
  background: #fffdfb !important;
  box-shadow: 0 0 0 3px rgba(138,113,90,.12) !important;
}
.assistant-form-hint {
  margin: 7px 1px 0;
  color: #746b62;
  font-size: .70rem;
  line-height: 1.42;
}
.assistant-actions {
  justify-content: flex-end !important;
  align-items: center !important;
  gap: 8px !important;
  margin: 8px 0 14px !important;
}
.assistant-actions button {
  flex: 0 0 auto !important;
  width: auto !important;
  min-height: 44px !important;
  border-radius: 11px !important;
  font-size: .76rem !important;
  font-weight: 760 !important;
}
.assistant-actions #assistant-submit { min-width: 184px !important; }
.assistant-actions button.secondary {
  min-width: 76px !important;
  color: #625b53 !important;
  border-color: transparent !important;
  background: transparent !important;
  box-shadow: none !important;
}
.assistant-actions button.secondary:hover,
.assistant-actions button.secondary:focus-visible {
  color: var(--ink) !important;
  border-color: rgba(112,96,78,.12) !important;
  background: #f1e8dc !important;
}
.assistant-actions #assistant-reset.is-idle { display: none !important; }
.assistant-answer {
  display: grid;
  gap: 13px;
  padding: 17px 0 3px;
  border-top: 1px solid var(--line);
}
.assistant-answer-head { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.verdict-chip {
  display: inline-flex;
  align-items: center;
  min-height: 29px;
  padding: 5px 10px;
  border: 1px solid transparent;
  border-radius: 999px;
  font-size: .70rem;
  font-weight: 820;
  letter-spacing: .035em;
  text-transform: uppercase;
}
.verdict-confirmed { color: #365c49; border-color: #a9c9b8; background: #e5f2eb; }
.verdict-partial { color: #72501f; border-color: #e2bd7d; background: #fff0cf; }
.verdict-contradicted { color: #88473f; border-color: #dfb0aa; background: #f9e6e3; }
.verdict-not-found { color: #625b53; border-color: #d9cec2; background: #f0e9e0; }
.verdict-clarify { color: #664b24; border-color: #dfc18f; background: #fff4dc; }
.verdict-summary { color: #4e5362; border-color: #bdc3d2; background: #edf0f6; }
.verdict-answered { color: #365c49; border-color: #a9c9b8; background: #e5f2eb; }
.verdict-theme-primary { color: #365c49; border-color: #9fc3af; background: #e1f0e8; }
.verdict-theme-present { color: #466150; border-color: #bad0c2; background: #edf5f0; }
.verdict-mention { color: #72501f; border-color: #e2bd7d; background: #fff0cf; }
.assistant-confidence { color: #625b53; font-size: .72rem; font-weight: 700; }
.assistant-result-label {
  color: #746b62;
  font-size: .67rem;
  font-weight: 800;
  letter-spacing: .075em;
  text-transform: uppercase;
}
.assistant-response-copy p {
  margin: 0;
  color: var(--ink);
  font-size: .98rem;
  font-weight: 680;
  line-height: 1.55;
}
.assistant-analysis-points { display: grid; gap: 7px; }
.assistant-analysis-points > strong,
.assistant-analysis-nuance > strong {
  color: #625b53;
  font-size: .68rem;
  font-weight: 820;
  letter-spacing: .07em;
  text-transform: uppercase;
}
.assistant-analysis-points ol {
  display: grid;
  gap: 7px;
  margin: 0;
  padding-left: 20px;
}
.assistant-analysis-points li {
  padding-left: 3px;
  color: #49423b;
  font-size: .80rem;
  line-height: 1.5;
}
.assistant-analysis-nuance {
  display: grid;
  gap: 4px;
  padding-left: 12px;
  border-left: 2px solid #c9b79f;
}
.assistant-analysis-nuance p {
  margin: 0;
  color: #625b53;
  font-size: .76rem;
  line-height: 1.52;
}
.assistant-answer > p { margin: 0; color: #49423b; font-size: .84rem; line-height: 1.58; }
.assistant-coverage { color: #746b62 !important; font-size: .72rem !important; }
.assistant-clarify { display: grid; gap: 8px; padding: 12px; border: 1px solid #e4cfaa; border-radius: 12px; background: #fff8e9; }
.assistant-clarify > span { color: #5f5040; font-size: .74rem; font-weight: 760; }
.assistant-clarify-options { display: flex; flex-wrap: wrap; gap: 7px; }
.assistant-clarify button {
  min-height: 38px;
  padding: 7px 10px;
  color: #4d4339;
  border: 1px solid rgba(112,96,78,.18);
  border-radius: 9px;
  background: rgba(255,255,255,.82);
  font-size: .72rem;
  font-weight: 740;
}
.assistant-clarify button:hover { color: var(--ink); background: #f0e4d3; }
.assistant-citations { display: grid; gap: 8px; }
.assistant-citations > strong { color: #625b53; font-size: .69rem; letter-spacing: .075em; text-transform: uppercase; }
.assistant-no-evidence { margin: 0; color: #746b62; font-size: .78rem; line-height: 1.5; }
.evidence-citation {
  border-left: 3px solid #b69c7f;
  border-radius: 0 12px 12px 0;
  background: #f7f2ec;
}
.evidence-citation > summary {
  display: grid;
  grid-template-columns: minmax(0,1fr) auto;
  align-items: center;
  gap: 12px;
  min-height: 64px;
  padding: 10px 13px;
  color: #403a34;
  cursor: pointer;
  list-style: none;
}
.evidence-citation > summary::-webkit-details-marker { display: none; }
.evidence-citation > summary:hover { background: rgba(255,255,255,.30); }
.evidence-citation > summary:focus-visible {
  outline: 2px solid #8f765a;
  outline-offset: 2px;
  border-radius: 0 10px 10px 0;
}
.evidence-summary-copy { display: grid; min-width: 0; gap: 5px; }
.evidence-preview {
  display: -webkit-box;
  overflow: hidden;
  color: #514940;
  font-size: .76rem;
  line-height: 1.42;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
}
.evidence-toggle {
  min-width: 52px;
  color: #6a5e53;
  font-size: .70rem;
  font-weight: 780;
  text-align: right;
}
.evidence-toggle .when-open { display: none; }
.evidence-citation[open] .evidence-toggle .when-closed { display: none; }
.evidence-citation[open] .evidence-toggle .when-open { display: inline; }
.evidence-citation[open] .evidence-preview { display: none; }
.evidence-citation[open] > summary { min-height: 48px; border-bottom: 1px solid rgba(112,96,78,.10); }
.evidence-citation-body {
  display: grid;
  gap: 9px;
  padding: 11px 13px 13px;
}
.evidence-meta { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px; color: #665d54; font-size: .69rem; font-weight: 760; }
.evidence-citation q {
  display: block;
  color: #403a34;
  font-size: .80rem;
  line-height: 1.5;
  overflow-wrap: anywhere;
}
.evidence-full { color: #665d54; font-size: .72rem; }
.evidence-full summary { cursor: pointer; font-weight: 730; }
.evidence-full p { margin: 8px 0 0; color: #514940; line-height: 1.5; }
.excerpt-player {
  display: grid;
  gap: 7px;
  padding-top: 10px;
  border-top: 1px solid rgba(112,96,78,.12);
}
.excerpt-player-head,
.excerpt-player-times {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}
.excerpt-player-head > span:first-child {
  color: #514940;
  font-size: .70rem;
  font-weight: 780;
}
.excerpt-player-head [data-excerpt-status] {
  color: #655d55;
  font-size: .75rem;
  font-weight: 700;
}
.excerpt-player.is-active .excerpt-player-head [data-excerpt-status] { color: #765936; }
.excerpt-player-controls {
  display: grid;
  grid-template-columns: minmax(58px,max-content) minmax(100px,1fr) minmax(58px,max-content);
  align-items: center;
  gap: 9px;
}
.excerpt-player-controls button {
  min-width: 58px;
  min-height: 44px;
  padding: 6px 10px;
  color: #514940;
  border: 1px solid rgba(112,96,78,.16);
  border-radius: 9px;
  background: rgba(255,255,255,.70);
  font-size: .70rem;
  font-weight: 760;
}
.excerpt-player-controls button:hover,
.excerpt-player-controls button:focus-visible {
  color: var(--ink);
  border-color: #b49b7e;
  background: #fffaf4;
}
.excerpt-player.is-active [data-assistant-action="toggle-excerpt"] {
  color: #fff;
  border-color: #756452;
  background: #756452;
}
.excerpt-player-timeline {
  display: grid;
  min-width: 0;
  gap: 1px;
}
.excerpt-player-controls input[type="range"] {
  --excerpt-progress: 0%;
  appearance: none;
  -webkit-appearance: none;
  width: 100%;
  height: 44px;
  margin: 0;
  padding: 0;
  border: 0;
  background: transparent;
  cursor: pointer;
}
.excerpt-player-controls input[type="range"]::-webkit-slider-runnable-track {
  height: 6px;
  border-radius: 999px;
  background: linear-gradient(to right, #8f765a 0 var(--excerpt-progress), #dfd6cb var(--excerpt-progress) 100%);
}
.excerpt-player-controls input[type="range"]::-webkit-slider-thumb {
  appearance: none;
  -webkit-appearance: none;
  width: 18px;
  height: 18px;
  margin-top: -6px;
  border: 2px solid #fffaf4;
  border-radius: 50%;
  background: #765f47;
  box-shadow: 0 1px 4px rgba(72,58,45,.24);
}
.excerpt-player-controls input[type="range"]::-moz-range-track {
  height: 6px;
  border-radius: 999px;
  background: #dfd6cb;
}
.excerpt-player-controls input[type="range"]::-moz-range-progress {
  height: 6px;
  border-radius: 999px;
  background: #8f765a;
}
.excerpt-player-controls input[type="range"]::-moz-range-thumb {
  width: 18px;
  height: 18px;
  border: 2px solid #fffaf4;
  border-radius: 50%;
  background: #765f47;
  box-shadow: 0 1px 4px rgba(72,58,45,.24);
}
.excerpt-player-controls input[type="range"]:focus-visible {
  outline: 2px solid #9d8162;
  outline-offset: 3px;
  border-radius: 999px;
}
.excerpt-player-times {
  color: #655d55;
  font-size: .72rem;
  font-variant-numeric: tabular-nums;
}
.evidence-actions { display: flex; flex-wrap: wrap; gap: 5px; }
.evidence-actions button {
  min-height: 38px;
  padding: 6px 9px;
  color: #514940;
  border: 1px solid rgba(112,96,78,.14);
  border-radius: 8px;
  background: rgba(255,255,255,.68);
  font-size: .69rem;
  font-weight: 730;
}
.evidence-actions button:hover { color: var(--ink); background: #eee4d8; }
.assistant-empty {
  display: grid;
  gap: 7px;
  padding: 18px 0;
  color: #625b53;
  border-top: 1px solid var(--line);
  font-size: .80rem;
  line-height: 1.5;
}
.assistant-empty strong { color: var(--ink); font-size: .90rem; }
.assistant-history { margin-top: 16px; padding-top: 14px; border-top: 1px solid var(--line); }
.assistant-history-overview > summary {
  display: flex;
  align-items: center;
  min-height: 44px;
  color: #625b53;
  cursor: pointer;
  font-size: .72rem;
  font-weight: 820;
  letter-spacing: .07em;
  text-transform: uppercase;
}
.assistant-history-overview > summary:focus-visible,
.assistant-history-item > summary:focus-visible {
  outline: 2px solid #8f765a;
  outline-offset: 3px;
  border-radius: 8px;
}
.assistant-history-list { border-top: 1px solid rgba(112,96,78,.12); }
.assistant-history-item { border-bottom: 1px solid rgba(112,96,78,.12); }
.assistant-history-item > summary {
  position: relative;
  display: grid;
  grid-template-columns: auto minmax(0,1fr) minmax(150px,auto) 16px;
  align-items: center;
  gap: 11px;
  min-height: 76px;
  padding: 12px 2px;
  color: var(--ink);
  cursor: pointer;
  list-style: none;
}
.assistant-history-item > summary::-webkit-details-marker { display: none; }
.assistant-history-item > summary:hover { background: rgba(255,255,255,.34); }
.assistant-history-item[open] > summary { border-bottom: 1px solid rgba(112,96,78,.10); }
.assistant-history-item[open] .assistant-history-copy > span { display: none; }
.assistant-history-item .verdict-chip {
  min-height: 25px;
  padding: 4px 8px;
  font-size: .62rem;
}
.assistant-history-copy { display: grid; gap: 3px; min-width: 0; }
.assistant-history-copy strong {
  overflow: hidden;
  color: #403a34;
  font-size: .80rem;
  line-height: 1.35;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.assistant-history-copy > span {
  overflow: hidden;
  color: #746b62;
  font-size: .72rem;
  line-height: 1.4;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.assistant-history-meta {
  color: #746b62;
  font-size: .66rem;
  line-height: 1.45;
  text-align: right;
}
.assistant-history-chevron {
  width: 8px;
  height: 8px;
  border-right: 2px solid #7c7167;
  border-bottom: 2px solid #7c7167;
  transform: rotate(45deg) translateY(-2px);
  transition: transform .16s ease;
}
.assistant-history-item[open] .assistant-history-chevron { transform: rotate(225deg) translate(-1px,-1px); }
.assistant-history-detail { padding: 0 0 18px 36px; }
.assistant-history-detail .assistant-answer { padding-top: 15px; border-top: 0; }
.discussion-toolbar { display: flex; justify-content: space-between; align-items: center; gap: 10px; margin: 0 0 12px; }
.discussion-toolbar > div { display: grid; gap: 3px; min-width: 0; }
.discussion-toolbar .eyebrow,
.discussion-toolbar strong { display: block; }
.discussion-toolbar strong { color: var(--ink); font-size: .88rem; }
.discussion-toolbar span { color: #625b53; font-size: .72rem; }
.comment-composer.is-compact { margin-top: 12px !important; padding-top: 14px !important; border-top: 1px solid var(--line) !important; }
#fiche-detail-view:has(.selection-context.is-empty) #comment-composer { display: none !important; }
#fiche-detail-view:has(.selection-context.is-ready) #comment-composer { display: block !important; }
.comment-identity { margin-top: 4px !important; }
.comment-identity .accordion > button { min-height: 40px !important; font-size: .74rem !important; }
footer { display: none !important; }
@media (max-width: 1120px) {
  .glass-card {
    padding: 16px !important;
    border-radius: 20px !important;
  }
  .workspace-shell { padding: 18px !important; }
  .section-heading {
    align-items: flex-start;
    flex-direction: column;
    gap: 7px;
  }
  .section-heading p {
    max-width: none;
    text-align: left;
  }
  .queue-toolbar { align-items: stretch !important; flex-direction: column !important; }
  .queue-flow-note { width: 100%; min-width: 0; }
  #queue-table .header-cell[data-heading="0"],
  #queue-table .body-cell[data-col="0"],
  #queue-table .header-cell[data-heading="4"],
  #queue-table .body-cell[data-col="4"],
  #fiche-library-table .header-cell[data-heading="1"],
  #fiche-library-table .body-cell[data-col="1"],
  #fiche-library-table .header-cell[data-heading="4"],
  #fiche-library-table .body-cell[data-col="4"] {
    display: none !important;
  }
  #fiche-library-table .header-table thead,
  #fiche-library-table .header-table thead tr {
    width: 100% !important;
  }
  #fiche-library-table .header-table thead {
    display: block !important;
  }
  #fiche-library-table .header-table thead tr {
    display: flex !important;
  }
  #fiche-library-table .header-cell[data-heading="0"] {
    width: auto !important;
    min-width: 0 !important;
    max-width: none !important;
    flex: 1 1 0 !important;
  }
  #fiche-library-table .body-cell[data-col="0"] {
    width: auto !important;
    min-width: 0 !important;
    max-width: none !important;
    flex: 1 1 0 !important;
  }
  #fiche-library-table .header-cell[data-heading="2"],
  #fiche-library-table .body-cell[data-col="2"] {
    width: 140px !important;
    min-width: 140px !important;
    max-width: 140px !important;
    flex: 0 0 140px !important;
  }
  #fiche-library-table .header-cell[data-heading="3"],
  #fiche-library-table .body-cell[data-col="3"] {
    width: 130px !important;
    min-width: 130px !important;
    max-width: 130px !important;
    flex: 0 0 130px !important;
  }
  .audio-detail-layout,
  .fiche-workbench { flex-direction: column !important; }
  .audio-detail-layout > div,
  .fiche-workbench > div { width: 100% !important; min-width: 0 !important; }
  .discussion-pane,
  .context-rail {
    padding-top: 26px !important;
    border-top: 1px solid var(--line) !important;
    border-left: 0 !important;
  }
  .fiche-workbench[data-context-mode="assistant"] .context-rail {
    max-width: 760px !important;
    margin: 0 auto !important;
    padding-top: 18px !important;
    border-top: 0 !important;
    background: transparent !important;
  }
  #fiche-transcript textarea { min-height: 360px !important; }
  .fiche-workbench[data-context-mode="assistant"] .transcript-pane,
  .fiche-workbench[data-context-mode="discussions"] .transcript-pane { display: none !important; }
  .fiche-workbench[data-context-mode="transcript"] .context-rail { display: none !important; }
}
@media (max-width: 900px) {
  .fiche-role-bar { grid-template-columns: minmax(0,1fr) auto; }
  .fiche-role-heading { grid-column: 1; grid-row: 1; }
  .fiche-role-people { grid-column: 1 / -1; grid-row: 2; }
  .role-edit-button { grid-column: 2; grid-row: 1; }
  .role-selector-grid { grid-template-columns: 1fr !important; }
  .role-editor-heading { grid-template-columns: 1fr; align-items: start; }
  .role-swap-column { width: 100% !important; max-width: 100% !important; min-width: 0 !important; }
  .role-editor-actions,
  .role-editor-actions > .form,
  .role-editor-actions > .styler > .form {
    display: grid !important;
    grid-template-columns: minmax(120px,.72fr) minmax(190px,1.28fr) !important;
    grid-template-areas: "detect detect" "cancel save" !important;
    align-items: stretch !important;
  }
  .role-editor-actions #assistant-auto-roles {
    grid-column: auto;
    width: 100% !important;
    min-width: 0 !important;
    margin-right: 0 !important;
  }
  .role-editor-actions #assistant-cancel-roles,
  .role-editor-actions #assistant-confirm-roles,
  .role-editor-actions > * { width: 100% !important; min-width: 0 !important; }
}
@media (max-width: 720px) {
  .gradio-container { padding: 0 !important; }
  .gradio-container > .main.fillable { padding: 10px 14px !important; }
  #app-screen {
    width: 100% !important;
    max-width: none !important;
    margin-right: 0 !important;
    margin-left: 0 !important;
  }
  .fiche-detail-toolbar { flex-wrap: wrap !important; }
  .fiche-detail-toolbar .fiche-back-button,
  .fiche-detail-toolbar #fiche-selector { width: 100% !important; min-width: 0 !important; }
  .app-header-top { align-items: flex-start; flex-direction: column; }
  .header-pills { width: 100%; flex-wrap: wrap; justify-content: flex-start; }
  .section-heading { align-items: flex-start; flex-direction: column; }
  .section-heading p { text-align: left; }
  .fiche-title-row { flex-direction: column; }
  .fiche-metrics { grid-template-columns: repeat(2, minmax(0,1fr)); }
  .fiche-metrics span:nth-child(3) { padding-left: 0; border-left: 0; }
  .audio-detail-layout { flex-wrap: wrap !important; }
  .audio-detail-layout > div { min-width: 100% !important; }
  .transcript-pane, .discussion-pane, .context-rail { padding: 18px 4px !important; }
  .fiche-role-bar { grid-template-columns: 1fr; }
  .fiche-role-heading { grid-column: 1; grid-row: 1; }
  .fiche-role-people { grid-column: 1; grid-row: 2; }
  .role-edit-button { grid-column: 1; grid-row: 3; width: 100%; min-width: 0; }
  .role-editor { width: calc(100% - 16px) !important; margin: 8px 8px 16px !important; border-radius: 14px !important; }
  .role-editor-heading,
  .role-selector-grid,
  .role-editor-actions { padding-right: 14px !important; padding-left: 14px !important; }
  .role-selector-grid { grid-template-columns: 1fr !important; }
  .role-swap-column { width: 100% !important; max-width: 100% !important; min-width: 0 !important; }
  .role-editor-actions,
  .role-editor-actions > .form,
  .role-editor-actions > .styler > .form {
    grid-template-columns: minmax(104px,.7fr) minmax(160px,1.3fr) !important;
    grid-template-areas: "detect detect" "cancel save" !important;
    align-items: stretch !important;
  }
  .role-editor-actions #assistant-auto-roles { grid-column: auto; }
  .role-editor-actions > * { width: 100% !important; min-width: 0 !important; }
  .role-action-button { white-space: normal !important; }
  .fiche-context-nav { align-items: stretch; flex-direction: column; }
  #fiche-context-mode { width: 100% !important; min-width: 0; }
  #fiche-context-mode > .wrap,
  #fiche-context-mode .wrap { flex-wrap: nowrap !important; }
  #fiche-context-mode label { padding-inline: 6px !important; font-size: .69rem !important; }
  .selection-context > div { align-items: flex-start; flex-direction: column; }
  .selection-context > header { align-items: flex-start; flex-direction: column; }
  .assistant-intro > div { align-items: flex-start; }
  .assistant-actions {
    display: grid !important;
    grid-template-columns: 1fr !important;
    gap: 4px !important;
  }
  .assistant-actions #assistant-submit {
    width: 100% !important;
    min-height: 44px !important;
    min-width: 0 !important;
  }
  .assistant-actions #assistant-reset {
    width: 100% !important;
    min-height: 40px !important;
  }
  .assistant-history-item > summary {
    grid-template-columns: auto minmax(0,1fr) 16px;
    gap: 9px;
    padding-block: 13px;
  }
  .assistant-history-meta {
    grid-column: 2;
    text-align: left;
  }
  .assistant-history-chevron { grid-column: 3; grid-row: 1 / span 2; }
  .assistant-history-detail { padding-left: 0; }
  .report-editor-intro { align-items: flex-start; flex-direction: column; }
  .reply-context > div { align-items: flex-start; flex-direction: column; }
  .thread-actions button { min-height: 44px !important; }
  .thread-delete-confirm { align-items: stretch; flex-wrap: wrap; }
  .thread-delete-confirm span { flex-basis: 100%; }
  .setup-card { padding: 28px 20px !important; }
  .model-grid { grid-template-columns: 1fr; }
  #saas-nav > .tab-wrapper {
    top: 6px;
    width: 100%;
    margin-bottom: 14px !important;
  }
  #saas-nav .tab-container[role="tablist"] {
    justify-content: stretch !important;
    width: 100% !important;
  }
  #saas-nav .tab-container[role="tablist"] button {
    padding-inline: 6px !important;
    font-size: .76rem !important;
  }
  #saas-nav .overflow-menu { display: none !important; }
  #queue-table .header-cell[data-heading="3"],
  #queue-table .body-cell[data-col="3"],
  #fiche-library-table .header-cell[data-heading="3"],
  #fiche-library-table .body-cell[data-col="3"] {
    display: none !important;
  }
  #queue-table .header-table,
  #fiche-library-table .header-table {
    width: 100% !important;
    min-width: 0 !important;
    table-layout: fixed !important;
  }
  #queue-table .header-cell[data-heading="1"],
  #fiche-library-table .header-cell[data-heading="0"] {
    width: calc(100% - 160px) !important;
    min-width: calc(100% - 160px) !important;
    max-width: calc(100% - 160px) !important;
  }
  #queue-table .body-cell[data-col="1"],
  #fiche-library-table .body-cell[data-col="0"] {
    width: auto !important;
    min-width: 0 !important;
    flex: 1 1 0 !important;
  }
  #queue-table .header-cell[data-heading="2"],
  #queue-table .body-cell[data-col="2"],
  #fiche-library-table .header-cell[data-heading="2"],
  #fiche-library-table .body-cell[data-col="2"] {
    width: 112px !important;
    min-width: 112px !important;
    max-width: 112px !important;
    flex: 0 0 112px !important;
  }
  #queue-table .header-cell[data-heading="5"],
  #fiche-library-table .header-cell[data-heading="5"] {
    width: 48px !important;
  }
  #queue-table .body-cell[data-col="1"] [role="button"],
  #fiche-library-table .body-cell[data-col="0"] [role="button"] {
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
  }
  .page-hero, .dashboard-intro, .dashboard-subhead { align-items: flex-start; flex-direction: column; }
  .page-hero-note { align-self: flex-start; }
  .dashboard-intro p { text-align: left; }
  .kpi-grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
  .kpi-grid article { min-height: 118px; }
  .recent-audio { grid-template-columns: minmax(130px,1.3fr) minmax(90px,1fr) 44px; gap: 10px; }
}
@media (max-width: 430px) {
  .role-editor-actions,
  .role-editor-actions > .form,
  .role-editor-actions > .styler > .form {
    grid-template-columns: 1fr !important;
    grid-template-areas: "detect" "cancel" "save" !important;
  }
  #saas-nav .tab-container[role="tablist"] {
    gap: 2px !important;
    padding: 4px !important;
  }
  #saas-nav .tab-container[role="tablist"] button {
    min-height: 42px !important;
    padding-inline: 3px !important;
    font-size: .68rem !important;
  }
  #queue-table .header-cell[data-heading="2"],
  #queue-table .body-cell[data-col="2"],
  #fiche-library-table .header-cell[data-heading="2"],
  #fiche-library-table .body-cell[data-col="2"] {
    display: none !important;
  }
  #queue-table .header-cell[data-heading="1"],
  #fiche-library-table .header-cell[data-heading="0"] {
    width: calc(100% - 48px) !important;
    min-width: calc(100% - 48px) !important;
    max-width: calc(100% - 48px) !important;
  }
  .assistant-history-item > summary {
    grid-template-columns: minmax(0,1fr) 16px;
  }
  .assistant-history-item .verdict-chip {
    grid-column: 1;
    justify-self: start;
  }
  .assistant-history-copy,
  .assistant-history-meta { grid-column: 1; }
  .assistant-history-chevron { grid-column: 2; grid-row: 1 / span 3; }
  .assistant-history-detail .evidence-actions { display: grid; grid-template-columns: 1fr; }
  .assistant-history-detail .evidence-actions button { width: 100%; min-height: 44px; }
  .excerpt-player-controls { grid-template-columns: 1fr 1fr; }
  .excerpt-player-controls button { width: 100%; }
  .excerpt-player-timeline { grid-column: 1 / -1; grid-row: 2; }
}
@media (min-width: 721px) and (max-width: 1080px) {
  .kpi-grid { grid-template-columns: repeat(3, minmax(0,1fr)); }
}
/* La bibliothèque garde ses informations essentielles alignées à toute largeur. */
@media (min-width: 721px) and (max-width: 1120px) {
  #fiche-library-table .header-table {
    display: block !important;
  }
  #fiche-library-table .header-cell[data-heading="1"],
  #fiche-library-table .body-cell[data-col="1"] {
    display: block !important;
  }
  #fiche-library-table .header-cell[data-heading="4"],
  #fiche-library-table .body-cell[data-col="4"] {
    display: none !important;
  }
  #fiche-library-table .header-table thead,
  #fiche-library-table .header-table thead tr {
    display: block !important;
    width: 100% !important;
  }
  #fiche-library-table .header-table thead tr {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) 180px 100px 125px 48px;
  }
  #fiche-library-table .header-cell[data-heading="0"],
  #fiche-library-table .header-cell[data-heading="1"],
  #fiche-library-table .header-cell[data-heading="2"],
  #fiche-library-table .header-cell[data-heading="3"],
  #fiche-library-table .header-cell[data-heading="5"] {
    width: auto !important;
    min-width: 0 !important;
    max-width: none !important;
    flex: none !important;
  }
  #fiche-library-table .body-cell[data-col="0"] {
    width: auto !important;
    min-width: 0 !important;
    max-width: none !important;
    flex: 1 1 0 !important;
  }
  #fiche-library-table .body-cell[data-col="1"] {
    width: 180px !important;
    min-width: 180px !important;
    max-width: 180px !important;
    flex: 0 0 180px !important;
  }
  #fiche-library-table .body-cell[data-col="2"] {
    width: 100px !important;
    min-width: 100px !important;
    max-width: 100px !important;
    flex: 0 0 100px !important;
  }
  #fiche-library-table .body-cell[data-col="3"] {
    width: 125px !important;
    min-width: 125px !important;
    max-width: 125px !important;
    flex: 0 0 125px !important;
  }
}
@media (min-width: 561px) and (max-width: 720px) {
  #fiche-library-table .header-table {
    display: block !important;
  }
  #fiche-library-table .header-cell[data-heading="1"],
  #fiche-library-table .body-cell[data-col="1"] {
    display: block !important;
  }
  #fiche-library-table .header-cell[data-heading="3"],
  #fiche-library-table .body-cell[data-col="3"],
  #fiche-library-table .header-cell[data-heading="4"],
  #fiche-library-table .body-cell[data-col="4"] {
    display: none !important;
  }
  #fiche-library-table .header-table thead,
  #fiche-library-table .header-table thead tr {
    display: block !important;
    width: 100% !important;
  }
  #fiche-library-table .header-table thead tr {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) 145px 100px 48px;
  }
  #fiche-library-table .header-cell[data-heading="0"],
  #fiche-library-table .header-cell[data-heading="1"],
  #fiche-library-table .header-cell[data-heading="2"],
  #fiche-library-table .header-cell[data-heading="5"] {
    width: auto !important;
    min-width: 0 !important;
    max-width: none !important;
    flex: none !important;
  }
  #fiche-library-table .body-cell[data-col="0"] {
    width: auto !important;
    min-width: 0 !important;
    max-width: none !important;
    flex: 1 1 0 !important;
  }
  #fiche-library-table .body-cell[data-col="1"] {
    width: 145px !important;
    min-width: 145px !important;
    max-width: 145px !important;
    flex: 0 0 145px !important;
  }
  #fiche-library-table .body-cell[data-col="2"] {
    width: 100px !important;
    min-width: 100px !important;
    max-width: 100px !important;
    flex: 0 0 100px !important;
  }
}
@media (min-width: 431px) and (max-width: 560px) {
  #fiche-library-table .header-table {
    display: block !important;
  }
  #fiche-library-table .header-cell[data-heading="1"],
  #fiche-library-table .body-cell[data-col="1"],
  #fiche-library-table .header-cell[data-heading="3"],
  #fiche-library-table .body-cell[data-col="3"],
  #fiche-library-table .header-cell[data-heading="4"],
  #fiche-library-table .body-cell[data-col="4"] {
    display: none !important;
  }
  #fiche-library-table .header-table thead,
  #fiche-library-table .header-table thead tr {
    display: block !important;
    width: 100% !important;
  }
  #fiche-library-table .header-table thead tr {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) 100px 48px;
  }
  #fiche-library-table .header-cell[data-heading="0"],
  #fiche-library-table .header-cell[data-heading="2"],
  #fiche-library-table .header-cell[data-heading="5"] {
    width: auto !important;
    min-width: 0 !important;
    max-width: none !important;
    flex: none !important;
  }
  #fiche-library-table .body-cell[data-col="0"] {
    width: auto !important;
    min-width: 0 !important;
    max-width: none !important;
    flex: 1 1 0 !important;
  }
  #fiche-library-table .body-cell[data-col="2"] {
    width: 100px !important;
    min-width: 100px !important;
    max-width: 100px !important;
    flex: 0 0 100px !important;
  }
}
@media (max-width: 430px) {
  #fiche-library-table .header-table {
    display: block !important;
  }
  #fiche-library-table .header-table thead,
  #fiche-library-table .header-table thead tr {
    display: block !important;
    width: 100% !important;
  }
  #fiche-library-table .header-table thead tr {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) 48px;
  }
  #fiche-library-table .header-cell[data-heading="0"],
  #fiche-library-table .header-cell[data-heading="5"] {
    width: auto !important;
    min-width: 0 !important;
    max-width: none !important;
    flex: none !important;
  }
  #fiche-library-table .body-cell[data-col="0"] {
    width: auto !important;
    min-width: 0 !important;
    max-width: none !important;
    flex: 1 1 0 !important;
  }
}
.cell-overflow-tooltip {
  position: fixed;
  z-index: 15050;
  max-width: min(360px, calc(100vw - 24px));
  padding: 9px 11px;
  border: 1px solid rgba(112, 96, 78, .18);
  border-radius: 10px;
  color: #fffdfb;
  background: #4f463e;
  box-shadow: 0 12px 30px rgba(54, 44, 35, .20);
  font-size: .78rem;
  font-weight: 620;
  line-height: 1.35;
  letter-spacing: 0;
  overflow-wrap: anywhere;
  pointer-events: none;
  opacity: 0;
  transform: translateY(3px);
  transition: opacity .14s ease, transform .14s ease;
}
.cell-overflow-tooltip.is-visible {
  opacity: 1;
  transform: translateY(0);
}
.cell-overflow-tooltip[hidden] { display: none !important; }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    scroll-behavior: auto !important;
    transition-duration: .01ms !important;
    animation-duration: .01ms !important;
    animation-iteration-count: 1 !important;
  }
}
"""


OVERFLOW_TOOLTIP_JS = r"""
() => {
  if (window.__studioOverflowTooltipInstalled) return [];
  window.__studioOverflowTooltipInstalled = true;

  const tooltip = document.createElement("div");
  tooltip.id = "studio-cell-overflow-tooltip";
  tooltip.className = "cell-overflow-tooltip";
  tooltip.setAttribute("role", "tooltip");
  tooltip.hidden = true;
  document.body.appendChild(tooltip);

  let activeCell = null;
  let activeLabel = null;

  const resolveCell = (target) => {
    if (!(target instanceof Element)) return null;
    return target.closest(
      "#fiche-library-table .body-cell:not([data-col='5']), " +
      "#queue-table .body-cell:not([data-col='5'])"
    );
  };

  const hideTooltip = () => {
    if (activeLabel) activeLabel.removeAttribute("aria-describedby");
    activeCell = null;
    activeLabel = null;
    tooltip.classList.remove("is-visible");
    tooltip.hidden = true;
  };

  const positionTooltip = () => {
    if (!activeCell || tooltip.hidden) return;
    const anchor = activeCell.getBoundingClientRect();
    const bubble = tooltip.getBoundingClientRect();
    const margin = 8;
    const left = Math.min(
      window.innerWidth - bubble.width - 12,
      Math.max(12, anchor.left + (anchor.width - bubble.width) / 2)
    );
    const above = anchor.top - bubble.height - margin;
    const top = above >= 12 ? above : anchor.bottom + margin;
    tooltip.style.left = `${Math.round(left)}px`;
    tooltip.style.top = `${Math.round(top)}px`;
  };

  const showTooltip = (cell) => {
    const label = cell?.querySelector("[role='button']") || cell?.querySelector(".cell-wrap");
    const value = (label?.textContent || "").trim();
    const truncated = Boolean(
      label && value && label.scrollWidth > label.clientWidth + 1
    );
    if (!truncated) {
      hideTooltip();
      return;
    }
    activeCell = cell;
    activeLabel = label;
    tooltip.textContent = value;
    tooltip.hidden = false;
    label.setAttribute("aria-describedby", tooltip.id);
    positionTooltip();
    requestAnimationFrame(() => tooltip.classList.add("is-visible"));
  };

  document.addEventListener("pointerover", (event) => {
    const cell = resolveCell(event.target);
    if (cell && cell !== activeCell) showTooltip(cell);
  });
  document.addEventListener("pointerout", (event) => {
    if (activeCell && !activeCell.contains(event.relatedTarget)) hideTooltip();
  });
  document.addEventListener("focusin", (event) => {
    const cell = resolveCell(event.target);
    if (cell) showTooltip(cell);
  });
  document.addEventListener("focusout", (event) => {
    if (activeCell && !activeCell.contains(event.relatedTarget)) hideTooltip();
  });
  window.addEventListener("resize", hideTooltip, { passive: true });
  window.addEventListener("scroll", hideTooltip, { passive: true, capture: true });
  return [];
}
"""


INLINE_RENAME_JS = r"""
() => {
  if (window.__transcriptionInlineRenameInstalled) return [];
  window.__transcriptionInlineRenameInstalled = true;
  const delayedLibraryOpens = new WeakMap();

  const setNativeValue = (element, value) => {
    if (!element) return;
    const prototype = element.tagName === "TEXTAREA"
      ? window.HTMLTextAreaElement.prototype
      : window.HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(prototype, "value")?.set;
    if (setter) setter.call(element, value);
    else element.value = value;
    element.dispatchEvent(new Event("input", { bubbles: true }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
  };

  const resolveTitle = (target) => {
    if (!(target instanceof Element) || target.closest(".inline-name-editor")) return null;

    const libraryCell = target.closest("#fiche-library-table .body-cell[data-col='0']");
    if (libraryCell) {
      const title = libraryCell.querySelector("[role='button']");
      return {
        container: libraryCell.querySelector(".cell-wrap") || libraryCell,
        hidden: title ? [title] : [],
        fullName: (title?.textContent || libraryCell.textContent || "Audio").trim(),
        target: `library:${libraryCell.dataset.row || "0"}`,
        deferLibraryOpen: true,
        title,
      };
    }

    const queueCell = target.closest("#queue-table .body-cell[data-col='1']");
    if (queueCell) {
      const title = queueCell.querySelector("[role='button']");
      return {
        container: queueCell.querySelector(".cell-wrap") || queueCell,
        hidden: title ? [title] : [],
        fullName: (title?.textContent || queueCell.textContent || "Audio").trim(),
        target: `queue:${queueCell.dataset.row || "0"}`,
        deferLibraryOpen: false,
        title,
      };
    }

    const uploaderRow = target.closest("#audio-queue tbody tr");
    if (uploaderRow) {
      const table = uploaderRow.closest("table");
      const rows = Array.from(table?.querySelectorAll("tbody tr") || []);
      const index = rows.indexOf(uploaderRow);
      const cell = uploaderRow.querySelector("td:first-child");
      if (index < 0 || !cell) return null;
      return {
        container: cell,
        hidden: Array.from(cell.children),
        fullName: (cell.getAttribute("aria-label") || cell.innerText || "Audio").trim(),
        target: `queue:${index}`,
        deferLibraryOpen: false,
        title: cell,
      };
    }

    const title = target.closest(".renameable-audio-title[data-audio-id]");
    if (title) {
      const text = title.querySelector(".audio-title-text");
      const itemId = title.getAttribute("data-audio-id") || "";
      if (!itemId) return null;
      return {
        container: title,
        hidden: text ? [text] : [],
        fullName: (text?.textContent || title.textContent || "Audio").trim(),
        target: `item:${itemId}`,
        deferLibraryOpen: false,
        title,
      };
    }
    return null;
  };

  const beginRename = (info) => {
    if (!info?.container || info.container.querySelector(".inline-name-editor")) return;
    document.querySelector(".inline-name-editor")?.blur();

    const fullName = info.fullName;
    const extensionMatch = fullName.match(/(\.[^.]+)$/);
    const extension = extensionMatch ? extensionMatch[1] : "";
    const initialStem = extension ? fullName.slice(0, -extension.length) : fullName;
    const hidden = info.hidden.filter((element) => element instanceof HTMLElement);
    hidden.forEach((element) => { element.style.display = "none"; });

    const editor = document.createElement("input");
    editor.className = "inline-name-editor";
    editor.value = initialStem;
    editor.setAttribute("aria-label", "Nouveau nom de l'audio");
    editor.setAttribute("autocomplete", "off");
    info.container.classList.add("is-renaming-title");
    info.container.appendChild(editor);

    let finished = false;
    const close = (commit) => {
      if (finished) return;
      finished = true;
      const name = editor.value.trim();
      hidden.forEach((element) => { element.style.display = ""; });
      info.container.classList.remove("is-renaming-title");
      editor.remove();
      if (!commit || !name || name === initialStem) return;

      const indexInput = document.querySelector("#inline-rename-index input, #inline-rename-index textarea");
      const nameInput = document.querySelector("#inline-rename-name input, #inline-rename-name textarea");
      const submit = document.querySelector("button#inline-rename-submit, #inline-rename-submit button");
      setNativeValue(indexInput, info.target);
      setNativeValue(nameInput, name);
      window.__renameBridgeSubmit = { target: info.target, name, scheduled: true, fired: false };
      window.setTimeout(() => {
        window.__renameBridgeSubmit.fired = true;
        submit?.click();
      }, 350);
    };

    editor.addEventListener("keydown", (keyboardEvent) => {
      if (keyboardEvent.key === "Enter") {
        keyboardEvent.preventDefault();
        close(true);
      } else if (keyboardEvent.key === "Escape") {
        keyboardEvent.preventDefault();
        close(false);
      }
    });
    ["click", "dblclick", "pointerdown"].forEach((eventName) => {
      editor.addEventListener(eventName, (editorEvent) => editorEvent.stopPropagation());
    });
    editor.addEventListener("blur", () => close(true));
    window.requestAnimationFrame(() => {
      editor.focus();
      editor.select();
    });
  };

  const stopLibraryTitleEvent = (event) => {
    const info = resolveTitle(event.target);
    if (!info?.deferLibraryOpen) return;
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();
  };

  ["pointerdown", "mousedown", "pointerup", "mouseup"].forEach((eventName) => {
    document.addEventListener(eventName, stopLibraryTitleEvent, true);
  });

  document.addEventListener("click", (event) => {
    const info = resolveTitle(event.target);
    if (!info?.deferLibraryOpen) return;
    stopLibraryTitleEvent(event);

    const previous = delayedLibraryOpens.get(info.container);
    if (previous) window.clearTimeout(previous);
    if (event.detail > 1) {
      delayedLibraryOpens.delete(info.container);
      return;
    }

    const timer = window.setTimeout(() => {
      delayedLibraryOpens.delete(info.container);
      if (!info.container.isConnected) return;
      const indexInput = document.querySelector("#library-open-index input, #library-open-index textarea");
      const submit = document.querySelector("button#library-open-submit, #library-open-submit button");
      setNativeValue(indexInput, info.target.slice("library:".length));
      window.setTimeout(() => submit?.click(), 350);
    }, 500);
    delayedLibraryOpens.set(info.container, timer);
  }, true);

  document.addEventListener("dblclick", (event) => {
    const info = resolveTitle(event.target);
    if (!info) return;
    const pending = delayedLibraryOpens.get(info.container);
    if (pending) window.clearTimeout(pending);
    delayedLibraryOpens.delete(info.container);
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();
    beginRename(info);
  }, true);

  document.addEventListener("mouseover", (event) => {
    const info = resolveTitle(event.target);
    if (info?.title && !info.title.getAttribute("title")) {
      info.title.setAttribute("title", "Double-cliquez pour renommer");
    }
  }, true);
  return [];
}
"""


DELETE_ACTION_JS = r"""
() => {
  if (window.__transcriptionDeleteActionsInstalled) return [];
  window.__transcriptionDeleteActionsInstalled = true;
  const iconMarkup = `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"
         aria-hidden="true">
      <path d="M3 6h18"></path>
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"></path>
      <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
      <line x1="10" x2="10" y1="11" y2="17"></line>
      <line x1="14" x2="14" y1="11" y2="17"></line>
    </svg>`;
  let activeAction = null;

  const setNativeValue = (element, value) => {
    if (!element) return;
    const prototype = element.tagName === "TEXTAREA"
      ? window.HTMLTextAreaElement.prototype
      : window.HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(prototype, "value")?.set;
    if (setter) setter.call(element, value);
    else element.value = value;
    element.dispatchEvent(new Event("input", { bubbles: true }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
  };

  const actionInfo = (target) => {
    if (!(target instanceof Element)) return null;
    const cell = target.closest(
      "#queue-table .body-cell[data-col='5'], #fiche-library-table .body-cell[data-col='5']"
    );
    if (!cell) return null;
    const row = cell.dataset.row || "0";
    const isLibrary = Boolean(cell.closest("#fiche-library-table"));
    const table = cell.closest("#fiche-library-table, #queue-table");
    const nameColumn = isLibrary ? "0" : "1";
    const nameCell = table?.querySelector(
      `.body-cell[data-row="${CSS.escape(row)}"][data-col="${nameColumn}"]`
    );
    const statusCell = table?.querySelector(
      `.body-cell[data-row="${CSS.escape(row)}"][data-col="2"]`
    );
    return {
      cell,
      target: `${isLibrary ? "library" : "queue"}:${row}`,
      name: (nameCell?.textContent || "cet audio").trim(),
      running: (statusCell?.textContent || "").trim() === "En cours",
    };
  };

  const backdrop = document.createElement("div");
  backdrop.className = "delete-confirm-backdrop";
  backdrop.setAttribute("role", "presentation");
  backdrop.innerHTML = `
    <section class="delete-confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="delete-confirm-title" aria-describedby="delete-confirm-copy">
      <div class="delete-confirm-icon">${iconMarkup}</div>
      <h3 id="delete-confirm-title">Supprimer cet audio ?</h3>
      <p id="delete-confirm-copy">La fiche <strong data-delete-name></strong>, ses commentaires, ses exports et sa copie audio locale seront supprimés. Le fichier source d'origine ne sera pas touché.</p>
      <div class="delete-confirm-actions">
        <button type="button" data-delete-cancel>Annuler</button>
        <button type="button" class="delete-confirm-primary" data-delete-confirm>Supprimer</button>
      </div>
    </section>
  `;
  document.body.appendChild(backdrop);
  const cancelButton = backdrop.querySelector("[data-delete-cancel]");
  const confirmButton = backdrop.querySelector("[data-delete-confirm]");
  const nameTarget = backdrop.querySelector("[data-delete-name]");

  const closeDialog = () => {
    backdrop.classList.remove("is-open");
    activeAction = null;
  };
  const openDialog = (info) => {
    if (info.running) return;
    activeAction = info;
    if (nameTarget) nameTarget.textContent = `« ${info.name} »`;
    backdrop.classList.add("is-open");
    window.setTimeout(() => cancelButton?.focus(), 30);
  };

  cancelButton?.addEventListener("click", closeDialog);
  backdrop.addEventListener("click", (event) => {
    if (event.target === backdrop) closeDialog();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && backdrop.classList.contains("is-open")) closeDialog();
  });
  confirmButton?.addEventListener("click", () => {
    if (!activeAction) return;
    const targetInput = document.querySelector("#delete-audio-target input, #delete-audio-target textarea");
    const submit = document.querySelector("button#delete-audio-submit, #delete-audio-submit button");
    setNativeValue(targetInput, activeAction.target);
    confirmButton.disabled = true;
    window.setTimeout(() => {
      submit?.click();
      confirmButton.disabled = false;
      closeDialog();
    }, 350);
  });

  const enhanceActions = () => {
    document.querySelectorAll(
      "#queue-table .body-cell[data-col='5'], #fiche-library-table .body-cell[data-col='5']"
    ).forEach((cell) => {
      if (!(cell instanceof HTMLElement)) return;
      const info = actionInfo(cell);
      const wrap = cell.querySelector(".cell-wrap") || cell;
      cell.classList.add("delete-action-cell");
      cell.setAttribute("aria-label", info?.running ? "Suppression indisponible pendant la transcription" : "Supprimer cet audio");
      cell.setAttribute("title", info?.running ? "Transcription en cours" : "Supprimer cet audio");
      cell.classList.toggle("is-disabled", Boolean(info?.running));
      if (!wrap.querySelector(".delete-action-icon")) {
        const icon = document.createElement("span");
        icon.className = "delete-action-icon";
        icon.setAttribute("aria-hidden", "true");
        icon.innerHTML = iconMarkup;
        wrap.appendChild(icon);
      }
    });
  };

  const stopActionEvent = (event) => {
    if (!actionInfo(event.target)) return;
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();
  };
  ["pointerdown", "mousedown", "pointerup", "mouseup", "dblclick"].forEach((eventName) => {
    document.addEventListener(eventName, stopActionEvent, true);
  });
  document.addEventListener("click", (event) => {
    const info = actionInfo(event.target);
    if (!info) return;
    stopActionEvent(event);
    openDialog(info);
  }, true);

  const observer = new MutationObserver(() => window.requestAnimationFrame(enhanceActions));
  observer.observe(document.body, { childList: true, subtree: true });
  enhanceActions();
  return [];
}
"""


COMMENTS_FEED_JS = r"""
if (!element.dataset.discussionFeedReady) {
  element.dataset.discussionFeedReady = "true";
  const resolvePlayerAudio = () => {
    const player = document.querySelector("#fiche-audio-player");
    const waveformHost = player?.querySelector("#waveform > div");
    return waveformHost?.shadowRoot?.querySelector("audio")
      || player?.querySelector("audio[src]")
      || null;
  };
  element.addEventListener("click", (event) => {
    const button = event.target.closest("[data-comment-action]");
    if (!button || !element.contains(button)) return;
    const action = button.dataset.commentAction || "";
    const card = button.closest(".discussion-thread");
    if (action === "seek") {
      const audio = resolvePlayerAudio();
      const start = Number.parseFloat(button.dataset.start || "0");
      if (audio && Number.isFinite(start)) {
        audio.currentTime = Math.max(0, start);
        audio.play().catch(() => {});
      }
      return;
    }
    if (action === "request-delete") {
      card?.classList.add("is-confirming-delete");
      card?.querySelector(".thread-delete-confirm .confirm-delete")?.focus();
      return;
    }
    if (action === "cancel-delete") {
      card?.classList.remove("is-confirming-delete");
      card?.querySelector("[data-comment-action='request-delete']")?.focus();
      return;
    }
    document.body.dataset.discussionPendingAction = action;
    document.body.dataset.discussionPendingId = button.dataset.commentId || card?.dataset.commentId || "";
    trigger("click", {
      action,
      comment_id: button.dataset.commentId || card?.dataset.commentId || "",
    });
  });
}
"""


ASSISTANT_RESULT_JS = r"""
if (!element.dataset.assistantResultReady) {
  element.dataset.assistantResultReady = "true";
  const resolvePlayerAudio = () => {
    const player = document.querySelector("#fiche-audio-player");
    const waveformHost = player?.querySelector("#waveform > div");
    return waveformHost?.shadowRoot?.querySelector("audio")
      || player?.querySelector("audio[src]")
      || null;
  };
  let activeExcerpt = null;
  const excerptBounds = (player) => ({
    start: Number.parseFloat(player?.dataset.start || "0"),
    end: Number.parseFloat(player?.dataset.end || "0"),
  });
  const excerptTime = (seconds) => {
    const safe = Math.max(0, Math.round(Number(seconds) || 0));
    const hours = Math.floor(safe / 3600);
    const minutes = Math.floor((safe % 3600) / 60);
    const rest = safe % 60;
    return hours > 0
      ? `${hours}:${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`
      : `${minutes}:${String(rest).padStart(2, "0")}`;
  };
  const renderExcerpt = (player, absoluteTime) => {
    const { start, end } = excerptBounds(player);
    const duration = Math.max(0.001, end - start);
    const elapsed = Math.max(0, Math.min(duration, absoluteTime - start));
    const range = player.querySelector("[data-excerpt-range]");
    const current = player.querySelector("[data-excerpt-current]");
    const progress = (elapsed / duration) * 100;
    if (range) {
      range.value = String(progress);
      range.style.setProperty("--excerpt-progress", `${progress}%`);
      range.setAttribute(
        "aria-valuetext",
        `${excerptTime(elapsed)} sur ${excerptTime(duration)}`
      );
    }
    if (current) current.textContent = excerptTime(elapsed);
  };
  const setExcerptState = (player, status, actionLabel, active = false) => {
    player?.classList.toggle("is-active", active);
    const statusNode = player?.querySelector("[data-excerpt-status]");
    const playButton = player?.querySelector("[data-assistant-action='toggle-excerpt']");
    if (statusNode) statusNode.textContent = status;
    if (playButton) playButton.textContent = actionLabel;
  };
  const clearExcerptListeners = (state) => {
    if (!state) return;
    Object.entries(state.listeners || {}).forEach(([eventName, handler]) => {
      state.audio.removeEventListener(eventName, handler);
    });
  };
  const releaseExcerpt = ({
    reset = true,
    status = "Prêt",
    actionLabel = "Lire",
    pauseAudio = true,
  } = {}) => {
    const state = activeExcerpt;
    if (!state) return;
    activeExcerpt = null;
    clearExcerptListeners(state);
    const { player, audio, start } = state;
    if (pauseAudio) audio.pause();
    if (reset) {
      audio.currentTime = start;
      renderExcerpt(player, start);
    }
    setExcerptState(player, status, actionLabel, false);
  };
  const prepareExcerpt = (player) => {
    if (!player) return null;
    if (activeExcerpt?.player === player) return activeExcerpt;
    releaseExcerpt();
    const audio = resolvePlayerAudio();
    const { start, end } = excerptBounds(player);
    if (!audio || !Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
      setExcerptState(player, "Audio indisponible", "Réessayer", false);
      return null;
    }
    const state = { player, audio, start, end, listeners: {} };
    const isCurrent = () => activeExcerpt === state;
    const finishExcerpt = () => {
      if (!isCurrent()) return;
      activeExcerpt = null;
      clearExcerptListeners(state);
      audio.pause();
      audio.currentTime = end;
      renderExcerpt(player, end);
      setExcerptState(player, "Terminé", "Relire", false);
    };
    state.listeners.timeupdate = () => {
      if (!isCurrent()) return;
      if (audio.currentTime < start - 0.35 || audio.currentTime > end + 0.35) {
        releaseExcerpt({ reset: false, status: "Interrompu", pauseAudio: false });
        return;
      }
      if (audio.currentTime >= end) {
        finishExcerpt();
        return;
      }
      renderExcerpt(player, audio.currentTime);
    };
    state.listeners.play = () => {
      if (!isCurrent()) return;
      if (audio.currentTime < start - 0.35 || audio.currentTime >= end) {
        releaseExcerpt({ reset: false, status: "Interrompu", pauseAudio: false });
        return;
      }
      setExcerptState(player, "Lecture", "Pause", true);
    };
    state.listeners.pause = () => {
      if (!isCurrent() || audio.currentTime >= end) return;
      renderExcerpt(player, audio.currentTime);
      setExcerptState(player, "En pause", "Reprendre", true);
    };
    state.listeners.seeking = () => {
      if (!isCurrent()) return;
      if (audio.currentTime < start - 0.35 || audio.currentTime > end + 0.35) {
        releaseExcerpt({ reset: false, status: "Interrompu", pauseAudio: false });
        return;
      }
      renderExcerpt(player, audio.currentTime);
    };
    state.listeners.ended = finishExcerpt;
    activeExcerpt = state;
    Object.entries(state.listeners).forEach(([eventName, handler]) => {
      audio.addEventListener(eventName, handler);
    });
    return state;
  };
  element.addEventListener("click", (event) => {
    const evidenceSummary = event.target.closest(".evidence-citation > summary");
    if (evidenceSummary && element.contains(evidenceSummary)) {
      releaseExcerpt();
      const currentEvidence = evidenceSummary.parentElement;
      if (currentEvidence && !currentEvidence.open) {
        element.querySelectorAll(".evidence-citation[open]").forEach((item) => {
          if (item !== currentEvidence) item.removeAttribute("open");
        });
      }
    }
    const historySummary = event.target.closest(".assistant-history-item > summary");
    if (historySummary && element.contains(historySummary)) {
      releaseExcerpt();
      const current = historySummary.parentElement;
      if (current && !current.open) {
        element.querySelectorAll(".assistant-history-item[open]").forEach((item) => {
          if (item !== current) item.removeAttribute("open");
        });
      }
    }
    const button = event.target.closest("[data-assistant-action]");
    if (!button || !element.contains(button)) return;
    event.preventDefault();
    event.stopPropagation();
    const action = button.dataset.assistantAction || "";
    const start = Number.parseFloat(button.dataset.start || "0");
    const end = Number.parseFloat(button.dataset.end || "0");
    if (action === "clarify") {
      const textarea = document.querySelector("#assistant-question textarea");
      const question = button.dataset.question || "";
      if (textarea && question) {
        const setter = Object.getOwnPropertyDescriptor(
          window.HTMLTextAreaElement.prototype,
          "value"
        )?.set;
        if (setter) setter.call(textarea, question);
        else textarea.value = question;
        textarea.dispatchEvent(new Event("input", { bubbles: true }));
        textarea.dispatchEvent(new Event("change", { bubbles: true }));
        window.requestAnimationFrame(() => {
          document.querySelector("#assistant-submit button, button#assistant-submit")?.click();
        });
      }
      return;
    }
    if (action === "toggle-excerpt") {
      const player = button.closest("[data-excerpt-player]");
      const state = prepareExcerpt(player);
      if (!state) return;
      if (!state.audio.paused) {
        state.audio.pause();
        renderExcerpt(player, state.audio.currentTime);
        setExcerptState(player, "En pause", "Reprendre", true);
        return;
      }
      if (state.audio.currentTime < state.start || state.audio.currentTime >= state.end) {
        state.audio.currentTime = state.start;
      }
      renderExcerpt(player, state.audio.currentTime);
      setExcerptState(player, "Lecture", "Pause", true);
      state.audio.play().catch(() => {
        if (activeExcerpt === state) {
          releaseExcerpt({
            reset: false,
            status: "Lecture impossible",
            actionLabel: "Réessayer",
          });
        }
      });
      return;
    }
    if (action === "stop-excerpt") {
      const player = button.closest("[data-excerpt-player]");
      if (activeExcerpt?.player === player) {
        releaseExcerpt({ reset: true, status: "Arrêté" });
      } else if (player) {
        const audio = resolvePlayerAudio();
        const excerptStart = Number.parseFloat(player.dataset.start || "0");
        if (!activeExcerpt && audio && Number.isFinite(excerptStart)) {
          audio.pause();
          audio.currentTime = excerptStart;
        }
        renderExcerpt(player, excerptStart);
        setExcerptState(player, "Arrêté", "Lire", false);
      }
      return;
    }
    if (action === "discuss") {
      releaseExcerpt();
      window.__setFicheContextMode?.("Discussions");
      trigger("click", {
        action,
        start_seconds: Number.isFinite(start) ? start : 0,
        end_seconds: Number.parseFloat(button.dataset.end || button.dataset.start || "0"),
        speaker_id: button.dataset.speakerId || "",
        role: button.dataset.role || "",
        quote: button.dataset.quote || "",
      });
    }
  });
  element.addEventListener("input", (event) => {
    const range = event.target.closest("[data-excerpt-range]");
    if (!range || !element.contains(range)) return;
    const player = range.closest("[data-excerpt-player]");
    const state = prepareExcerpt(player);
    if (!state) return;
    state.audio.pause();
    const ratio = Math.max(0, Math.min(100, Number.parseFloat(range.value || "0"))) / 100;
    state.audio.currentTime = state.start + (state.end - state.start) * ratio;
    renderExcerpt(player, state.audio.currentTime);
    setExcerptState(player, "Position choisie", "Lire", true);
  });
  document.addEventListener("click", (event) => {
    if (!activeExcerpt) return;
    const destination = event.target.closest(
      "[role='tab'], #workspace-nav button, #fiche-context-mode label"
    );
    if (destination && !element.contains(destination)) releaseExcerpt();
  }, true);
  window.addEventListener("pagehide", () => releaseExcerpt());
}
"""


WORKSPACE_NAV_JS = r"""
() => {
  if (window.__studioAudioNavigationInstalled) {
    window.__restoreStudioAudioNavigation?.();
    return [];
  }
  window.__studioAudioNavigationInstalled = true;

  const keys = {
    activeTab: "studioAudio.activeTab",
    ficheView: "studioAudio.ficheView",
    ficheItemId: "studioAudio.ficheItemId",
    ficheItemName: "studioAudio.ficheItemName",
  };
  const seedAttributes = {
    [keys.activeTab]: "data-active-tab",
    [keys.ficheView]: "data-fiche-view",
    [keys.ficheItemId]: "data-fiche-item-id",
    [keys.ficheItemName]: "data-fiche-item-name",
  };
  const read = (key) => {
    try {
      const stored = window.localStorage?.getItem(key);
      if (stored) return stored;
    } catch (_) {}
    const seed = document.querySelector("#studio-audio-navigation-state [data-navigation-state][data-ready='true']");
    return seed?.getAttribute(seedAttributes[key] || "") || "";
  };
  const write = (key, value) => {
    try {
      if (value) window.localStorage.setItem(key, value);
      else window.localStorage.removeItem(key);
    } catch (_) {}
  };
  const setNativeValue = (element, value) => {
    if (!element) return;
    const prototype = element.tagName === "TEXTAREA"
      ? window.HTMLTextAreaElement.prototype
      : window.HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(
      prototype,
      "value"
    )?.set;
    if (setter) setter.call(element, value);
    else element.value = value;
    element.dispatchEvent(new Event("input", { bubbles: true }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
  };

  let topTabRestored = false;
  let detailRestoreStarted = false;

  const topTabs = () => Array.from(
    document.querySelectorAll("#saas-nav .tab-container[role='tablist'] [role='tab'][data-tab-id]")
  );

  const restoreTopTab = () => {
    const tabs = topTabs();
    if (!tabs.length || topTabRestored) return;
    const saved = read(keys.activeTab);
    const target = tabs.find((tab) => tab.dataset.tabId === saved);
    if (!saved || !target) return;
    topTabRestored = true;
    if (target.getAttribute("aria-selected") !== "true") target.click();
  };

  const restoreFicheDetail = () => {
    if (detailRestoreStarted || read(keys.ficheView) !== "detail") return;
    const savedName = read(keys.ficheItemName);
    if (!savedName) return;
    const cells = Array.from(
      document.querySelectorAll("#fiche-library-table .body-cell[data-col='0'] [role='button']")
    );
    if (!cells.length) return;
    const target = cells.find((cell) => cell.textContent.trim() === savedName);
    if (!target) {
      write(keys.ficheView, "library");
      write(keys.ficheItemId, "");
      write(keys.ficheItemName, "");
      return;
    }
    detailRestoreStarted = true;
    target.click();
  };

  const restore = () => {
    restoreTopTab();
    window.requestAnimationFrame(restoreFicheDetail);
  };
  window.__restoreStudioAudioNavigation = restore;

  document.addEventListener("click", (event) => {
    if (!(event.target instanceof Element)) return;
    const tab = event.target.closest("#saas-nav [role='tab'][data-tab-id]");
    if (tab && event.isTrusted) {
      const tabId = tab.dataset.tabId || "dashboard";
      write(keys.activeTab, tabId);
      const bridge = document.querySelector("#navigation-tab-bridge input, #navigation-tab-bridge textarea");
      const submit = document.querySelector("button#navigation-tab-submit, #navigation-tab-submit button");
      setNativeValue(bridge, tabId);
      window.setTimeout(() => submit?.click(), 30);
    }
    const back = event.target.closest(".fiche-back-button");
    if (back && event.isTrusted) {
      const submit = document.querySelector(
        "button#navigation-fiche-close-submit, #navigation-fiche-close-submit button"
      );
      window.setTimeout(() => submit?.click(), 30);
    }
  }, true);

  const observer = new MutationObserver(restore);
  observer.observe(document.body, { childList: true, subtree: true });
  restore();
  window.setTimeout(restore, 120);
  window.setTimeout(restore, 500);
  return [];
}
"""


DESKTOP_VISIBILITY_SYNC_JS = r"""
() => {
  if (window.__studioAudioVisibilitySyncInstalled) return [];
  window.__studioAudioVisibilitySyncInstalled = true;
  let hiddenAt = 0;

  const rememberHidden = () => {
    if (document.hidden && !hiddenAt) hiddenAt = Date.now();
  };
  const refreshStaleRunningView = () => {
    if (document.hidden) {
      rememberHidden();
      return;
    }
    const elapsed = hiddenAt ? Date.now() - hiddenAt : 0;
    hiddenAt = 0;
    if (elapsed < 2000) return;
    const running = document.querySelector(
      ".fiche-head[data-audio-status='En cours'],"
      + ".fiche-head[data-audio-status='À reprendre']"
    );
    if (running) window.location.reload();
  };

  document.addEventListener("visibilitychange", refreshStaleRunningView);
  window.addEventListener("focus", refreshStaleRunningView);
  rememberHidden();
  return [];
}
"""


PROCESSING_CLOCK_JS = r"""
() => {
  if (window.__studioAudioProcessingClockInstalled) return [];
  window.__studioAudioProcessingClockInstalled = true;

  const formatDuration = (rawSeconds) => {
    const total = Math.max(0, Math.round(Number(rawSeconds) || 0));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const seconds = total % 60;
    if (hours) {
      return `${hours} h ${String(minutes).padStart(2, "0")} min `
        + `${String(seconds).padStart(2, "0")} s`;
    }
    if (minutes) return `${minutes} min ${String(seconds).padStart(2, "0")} s`;
    return `${seconds} s`;
  };

  const refresh = () => {
    document.querySelectorAll(
      ".fiche-processing-time[data-processing-live='true']"
    ).forEach((node) => {
      const saved = Math.max(0, Number(node.dataset.processingSeconds) || 0);
      const base = Math.max(0, Number(node.dataset.processingBaseSeconds) || 0);
      const startedAt = Date.parse(node.dataset.processingStartedAt || "");
      if (!Number.isFinite(startedAt)) return;
      const activeSeconds = base + Math.max(0, (Date.now() - startedAt) / 1000);
      const nextText = formatDuration(Math.max(saved, activeSeconds));
      if (node.textContent !== nextText) node.textContent = nextText;
    });
  };

  let scheduled = false;
  const schedule = () => {
    if (scheduled) return;
    scheduled = true;
    window.requestAnimationFrame(() => {
      scheduled = false;
      refresh();
    });
  };

  const observer = new MutationObserver(schedule);
  observer.observe(document.body, { childList: true, subtree: true });
  window.setInterval(refresh, 1000);
  document.addEventListener("visibilitychange", schedule);
  window.addEventListener("focus", schedule);
  schedule();
  return [];
}
"""


FICHE_CONTEXT_JS = r"""
() => {
  if (window.__ficheContextInstalled) return [];
  window.__ficheContextInstalled = true;

  const normalise = (value) => {
    const folded = String(value || "").toLowerCase();
    if (folded.includes("discussion")) return "discussions";
    if (folded.includes("assistant")) return "assistant";
    return "transcript";
  };
  const storageKey = "studioAudio.ficheContextMode";
  const readStoredMode = () => {
    let stored = "";
    try { stored = window.localStorage?.getItem(storageKey) || ""; }
    catch (_) {}
    if (!stored) {
      const seed = document.querySelector("#studio-audio-navigation-state [data-navigation-state][data-ready='true']");
      stored = seed?.getAttribute("data-context-mode") || "";
    }
    return stored ? normalise(stored) : "assistant";
  };
  const storeMode = (mode) => {
    try { window.localStorage.setItem(storageKey, mode); }
    catch (_) {}
  };
  let restoredShell = null;

  const activeLabel = () => {
    const root = document.querySelector("#fiche-context-mode");
    const checked = root?.querySelector("input:checked");
    if (checked?.value) return checked.value;
    const selected = root?.querySelector("label.selected, label:has(input:checked)");
    return selected?.textContent?.trim() || "Assistant IA";
  };

  const applyMode = () => {
    const shell = document.querySelector(".fiche-workbench");
    if (!shell) return;
    const transcriptPane = shell.querySelector(".transcript-pane");
    const contextRail = shell.querySelector(".context-rail");
    transcriptPane?.setAttribute("role", "region");
    transcriptPane?.setAttribute("aria-label", "Transcription de la consultation");
    transcriptPane?.setAttribute("tabindex", "0");
    contextRail?.setAttribute("role", "region");
    contextRail?.setAttribute("aria-label", "Outils de la fiche audio");
    contextRail?.setAttribute("tabindex", "0");
    const readySeed = document.querySelector(
      "#studio-audio-navigation-state [data-navigation-state][data-ready='true']"
    );
    let localMode = "";
    try { localMode = window.localStorage?.getItem(storageKey) || ""; }
    catch (_) {}
    if (shell !== restoredShell && (readySeed || localMode)) {
      restoredShell = shell;
      const saved = readStoredMode();
      const current = normalise(activeLabel());
      if (saved !== current) {
        const labels = [...(document.querySelectorAll("#fiche-context-mode label") || [])];
        const target = labels.find((item) => normalise(item.textContent) === saved);
        const input = target?.querySelector("input");
        if (input && !input.checked) {
          input.click();
          window.requestAnimationFrame(applyMode);
          return;
        }
      }
    }
    const mode = normalise(activeLabel());
    shell.dataset.contextMode = mode;
    document.body.dataset.ficheContextMode = mode;
    storeMode(mode);
    const reset = document.querySelector("#assistant-reset");
    const question = document.querySelector("#assistant-question textarea");
    const hasQuestion = Boolean(question?.value?.trim());
    const hasResult = Boolean(document.querySelector("#assistant-result .assistant-answer"));
    reset?.classList.toggle("is-idle", !hasQuestion && !hasResult);
  };

  window.__setFicheContextMode = (label) => {
    const root = document.querySelector("#fiche-context-mode");
    const wanted = normalise(label);
    const labels = [...(root?.querySelectorAll("label") || [])];
    const target = labels.find((item) => normalise(item.textContent) === wanted);
    const input = target?.querySelector("input");
    if (input && !input.checked) input.click();
    window.requestAnimationFrame(applyMode);
  };

  document.addEventListener("change", (event) => {
    if (event.target instanceof Element && event.target.closest("#fiche-context-mode")) {
      const seed = document.querySelector(
        "#studio-audio-navigation-state [data-navigation-state][data-ready='true']"
      );
      seed?.setAttribute("data-context-mode", normalise(activeLabel()));
      window.requestAnimationFrame(applyMode);
    }
  });
  document.addEventListener("input", (event) => {
    if (event.target instanceof Element && event.target.matches("#assistant-question textarea")) {
      window.requestAnimationFrame(applyMode);
    }
  });
  document.addEventListener("click", (event) => {
    if (event.target instanceof Element && event.target.closest("#assistant-reset")) {
      window.setTimeout(() => {
        document.querySelector("#assistant-question textarea")?.focus();
        applyMode();
      }, 200);
    }
  });
  const observer = new MutationObserver(() => window.requestAnimationFrame(applyMode));
  observer.observe(document.body, { childList: true, subtree: true });
  applyMode();
  return [];
}
"""


ROLE_EDITOR_JS = r"""
() => {
  if (window.__roleEditorDisclosureInstalled) {
    window.__syncRoleEditorDisclosure?.();
    return [];
  }
  window.__roleEditorDisclosureInstalled = true;
  let scheduled = false;
  let focusEditorWhenOpen = false;
  let restoreTriggerWhenClosed = false;

  const isVisible = (element) => Boolean(
    element
    && !element.hidden
    && getComputedStyle(element).display !== "none"
    && getComputedStyle(element).visibility !== "hidden"
  );

  const sync = () => {
    scheduled = false;
    const trigger = document.querySelector("[data-role-editor-toggle]");
    const bridgeRoot = document.querySelector("#assistant-role-edit-bridge");
    const bridge = bridgeRoot?.querySelector("button") || bridgeRoot;
    const panel = document.querySelector("#assistant-role-editor");
    if (!(trigger instanceof HTMLElement)) return;
    const expanded = Boolean(
      panel?.querySelector(".role-editor-heading") && isVisible(panel)
    );
    if (!trigger.id) trigger.id = "assistant-role-editor-toggle";
    trigger.setAttribute("aria-expanded", String(expanded));
    trigger.setAttribute("aria-controls", "assistant-role-editor");
    if (bridge instanceof HTMLElement) {
      bridge.tabIndex = -1;
      bridge.setAttribute("aria-hidden", "true");
    }
    if (panel) {
      panel.setAttribute("role", "region");
      panel.setAttribute("aria-labelledby", trigger.id);
    }
    if (expanded && focusEditorWhenOpen) {
      focusEditorWhenOpen = false;
      window.setTimeout(() => {
        document.querySelector(
          "#assistant-client-role [role='combobox'], #assistant-client-role input"
        )?.focus();
      }, 30);
    }
    if (!expanded && restoreTriggerWhenClosed) {
      restoreTriggerWhenClosed = false;
      window.setTimeout(() => trigger.focus(), 30);
    }
  };

  const schedule = () => {
    if (scheduled) return;
    scheduled = true;
    window.requestAnimationFrame(sync);
  };

  document.addEventListener("click", (event) => {
    const target = event.target instanceof Element ? event.target : null;
    const trigger = target?.closest("[data-role-editor-toggle]");
    if (trigger) {
      event.preventDefault();
      if (trigger.matches(":disabled, [aria-disabled='true']")) return;
      focusEditorWhenOpen = true;
      const bridgeRoot = document.querySelector("#assistant-role-edit-bridge");
      const bridge = bridgeRoot?.querySelector("button") || bridgeRoot;
      if (bridge instanceof HTMLElement) bridge.click();
    }
    if (target?.closest("#assistant-cancel-roles, #assistant-confirm-roles")) {
      restoreTriggerWhenClosed = true;
    }
    schedule();
  });

  const observer = new MutationObserver(schedule);
  observer.observe(document.querySelector("gradio-app") || document.body, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["class", "style", "hidden"],
  });
  window.__syncRoleEditorDisclosure = sync;
  sync();
  return [];
}
"""


TRANSCRIPT_DISCUSSION_JS = r"""
() => {
  if (window.__transcriptionDiscussionUxInstalled) return [];
  window.__transcriptionDiscussionUxInstalled = true;

  const scrollBehavior = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
  let lastReadySelection = "";
  let lastReadyReply = "";
  let lastFeedContent = "";

  const liveStatus = () => {
    let node = document.querySelector("#discussion-action-status");
    if (node) return node;
    node = document.createElement("div");
    node.id = "discussion-action-status";
    node.className = "discussion-live-status";
    node.setAttribute("role", "status");
    node.setAttribute("aria-live", "polite");
    document.querySelector(".context-rail")?.appendChild(node);
    return node;
  };

  const announce = (message) => {
    const node = liveStatus();
    if (!node) return;
    node.textContent = "";
    window.requestAnimationFrame(() => { node.textContent = message; });
  };

  const enhanceTranscript = () => {
    document.querySelectorAll("#fiche-transcript textarea").forEach((textarea) => {
      textarea.readOnly = true;
      textarea.setAttribute("aria-readonly", "true");
      textarea.setAttribute("aria-describedby", "transcript-selection-help");
    });
  };

  const restoreDiscussionFocus = () => {
    const pendingAction = document.body.dataset.discussionPendingAction || "";
    const pendingId = document.body.dataset.discussionPendingId || "";
    const readyReply = document.querySelector(".reply-context.is-ready");
    const replyKey = readyReply?.textContent?.trim() || "";
    if (replyKey && replyKey !== lastReadyReply) {
      lastReadyReply = replyKey;
      document.querySelector("#comment-reply textarea")?.focus();
      document.querySelector("#comment-reply")?.scrollIntoView({ behavior: scrollBehavior, block: "center" });
      announce("Zone de réponse ouverte.");
      delete document.body.dataset.discussionPendingAction;
      delete document.body.dataset.discussionPendingId;
      return;
    }
    if (!replyKey) lastReadyReply = "";

    const readySelection = document.querySelector(".selection-context.is-ready");
    const selectionKey = readySelection?.textContent?.trim() || "";
    if (selectionKey && selectionKey !== lastReadySelection) {
      lastReadySelection = selectionKey;
      document.querySelector("#comment-body textarea")?.focus();
      if (window.innerWidth < 1180 || pendingAction === "edit") {
        document.querySelector("#comment-composer")?.scrollIntoView({ behavior: scrollBehavior, block: "center" });
      }
      announce(pendingAction === "edit" ? "Discussion prête à être modifiée." : "Passage prêt à commenter.");
      delete document.body.dataset.discussionPendingAction;
      delete document.body.dataset.discussionPendingId;
      return;
    }
    if (!selectionKey) lastReadySelection = "";

    if (pendingAction === "resolve") {
      const target = document.querySelector(`.discussion-thread[data-comment-id="${CSS.escape(pendingId)}"] [data-comment-action="resolve"]`);
      if (target) {
        target.focus();
        announce("Statut de la discussion mis à jour.");
        delete document.body.dataset.discussionPendingAction;
        delete document.body.dataset.discussionPendingId;
      }
    } else if (pendingAction === "delete") {
      const target = document.querySelector(".discussion-thread [data-comment-action='reply'], #comment-body textarea");
      target?.focus();
      announce("Discussion supprimée.");
      delete document.body.dataset.discussionPendingAction;
      delete document.body.dataset.discussionPendingId;
    }

    const feedContent = document.querySelector("#comments-feed")?.textContent?.trim() || "";
    if (!lastFeedContent) {
      lastFeedContent = feedContent;
    } else if (feedContent && feedContent !== lastFeedContent) {
      lastFeedContent = feedContent;
      if (!document.body.dataset.discussionPendingAction) announce("Discussion mise à jour.");
    }
  };

  document.addEventListener("keydown", (event) => {
    if (!(event.ctrlKey || event.metaKey) || event.key !== "Enter") return;
    const target = event.target;
    if (!(target instanceof Element)) return;
    if (target.closest("#comment-body")) {
      event.preventDefault();
      document.querySelector("button#comment-submit, #comment-submit button")?.click();
    } else if (target.closest("#comment-reply")) {
      event.preventDefault();
      document.querySelector("button#comment-reply-submit, #comment-reply-submit button")?.click();
    }
  });

  const observer = new MutationObserver(() => window.requestAnimationFrame(() => {
    enhanceTranscript();
    restoreDiscussionFocus();
  }));
  observer.observe(document.body, { childList: true, subtree: true });
  enhanceTranscript();
  restoreDiscussionFocus();
  return [];
}
"""


AUDIO_TIMELINE_HTML = """
<section id="audio-timeline-panel" class="audio-timeline-panel" aria-label="Navigation dans l'audio">
  <div class="audio-timeline-toolbar">
    <div class="audio-timeline-heading">
      <span class="audio-timeline-kicker">Timeline</span>
      <strong id="audio-timeline-zoom-status">Préparation de la timeline…</strong>
    </div>
    <div class="audio-timeline-actions" role="group" aria-label="Contrôles du zoom temporel">
      <button type="button" id="audio-timeline-fit" disabled>Vue complète</button>
      <button type="button" id="audio-timeline-out" disabled>Dézoomer</button>
      <input
        id="audio-timeline-range"
        type="range"
        min="0"
        max="100"
        step="1"
        value="0"
        aria-label="Niveau de zoom de la timeline"
        aria-valuetext="Vue complète"
        disabled
      >
      <button type="button" id="audio-timeline-in" disabled>Zoomer</button>
    </div>
  </div>
  <div id="audio-zoom-waveform" aria-label="Forme d'onde interactive">
    <div class="audio-timeline-loading">Chargement de la forme d'onde…</div>
  </div>
  <div class="audio-timeline-times" aria-live="polite">
    <time id="audio-timeline-current">0:00</time>
    <span id="audio-timeline-scale">Enregistrement complet</span>
    <time id="audio-timeline-duration">0:00</time>
  </div>
</section>
"""


TIMELINE_INIT_JS = r"""
() => {
  if (window.__transcriptionTimeline?.installed) {
    window.__transcriptionTimeline.refreshSoon(0);
    return [];
  }

  const state = {
    installed: true,
    waveform: null,
    media: null,
    source: "",
    detachMedia: null,
    refreshTimer: null,
    createToken: 0,
    zoom: 0,
  };

  const byId = (id) => document.getElementById(id);
  const formatTime = (seconds) => {
    const safe = Number.isFinite(seconds) && seconds > 0 ? Math.floor(seconds) : 0;
    const hours = Math.floor(safe / 3600);
    const minutes = Math.floor((safe % 3600) / 60);
    const remaining = safe % 60;
    if (hours > 0) {
      return `${hours}:${String(minutes).padStart(2, "0")}:${String(remaining).padStart(2, "0")}`;
    }
    return `${minutes}:${String(remaining).padStart(2, "0")}`;
  };

  const setLoading = (message) => {
    const container = byId("audio-zoom-waveform");
    if (!container) return;
    container.replaceChildren();
    const loading = document.createElement("div");
    loading.className = "audio-timeline-loading";
    loading.textContent = message;
    container.appendChild(loading);
  };

  const setControlsDisabled = (disabled) => {
    ["audio-timeline-fit", "audio-timeline-out", "audio-timeline-range", "audio-timeline-in"]
      .forEach((id) => {
        const control = byId(id);
        if (control) control.disabled = disabled;
      });
  };

  const updateTimes = () => {
    const current = byId("audio-timeline-current");
    const duration = byId("audio-timeline-duration");
    if (current) current.textContent = formatTime(state.media?.currentTime || 0);
    if (duration) duration.textContent = formatTime(state.media?.duration || state.waveform?.getDuration?.() || 0);
  };

  const updateZoomCopy = (value) => {
    const status = byId("audio-timeline-zoom-status");
    const scale = byId("audio-timeline-scale");
    const range = byId("audio-timeline-range");
    const rounded = Math.round(value);
    let label = `Zoom ${rounded} %`;
    let scaleLabel = `Niveau de détail ${Math.max(1, Math.round(rounded / 10))} / 10`;
    if (rounded <= 0) {
      label = "Vue complète";
      scaleLabel = "Enregistrement complet";
    } else if (rounded >= 100) {
      label = "Zoom maximal";
      scaleLabel = "Détail maximal";
    }
    if (status) status.textContent = label;
    if (scale) scale.textContent = scaleLabel;
    if (range) {
      range.value = String(rounded);
      range.style.setProperty("--timeline-zoom", `${rounded}%`);
      range.setAttribute("aria-valuetext", label);
    }
    const zoomOut = byId("audio-timeline-out");
    const zoomIn = byId("audio-timeline-in");
    const fit = byId("audio-timeline-fit");
    if (zoomOut) zoomOut.disabled = rounded <= 0 || !state.waveform;
    if (zoomIn) zoomIn.disabled = rounded >= 100 || !state.waveform;
    if (fit) fit.disabled = rounded <= 0 || !state.waveform;
  };

  const applyZoom = (value) => {
    const next = Math.max(0, Math.min(100, Number(value) || 0));
    state.zoom = next;
    if (state.waveform?.getDecodedData?.()) {
      try {
        state.waveform.zoom(20 * next / 100);
        if (next > 0 && state.media?.currentTime > 0) {
          window.setTimeout(() => state.waveform?.setScrollTime(state.media.currentTime), 0);
        }
      } catch (_error) {
        updateZoomCopy(next);
        return;
      }
    }
    updateZoomCopy(next);
  };

  const findPlayerMedia = () => {
    const waveformNodes = document.querySelectorAll("#fiche-audio-player #waveform");
    for (const waveformNode of waveformNodes) {
      const host = waveformNode.firstElementChild;
      const media = host?.shadowRoot?.querySelector("audio");
      if (media) return media;
    }
    return null;
  };

  const findWaveSurferModule = () => performance
    .getEntriesByType("resource")
    .map((entry) => entry.name)
    .find((name) => /MinimalAudioPlayer-.*\.js(?:\?.*)?$/.test(name));

  const cleanupWaveform = () => {
    state.createToken += 1;
    if (state.detachMedia) state.detachMedia();
    state.detachMedia = null;
    if (state.waveform) state.waveform.destroy();
    state.waveform = null;
    state.media = null;
    state.source = "";
  };

  const createWaveform = async (media, source) => {
    const container = byId("audio-zoom-waveform");
    if (!container) return;
    cleanupWaveform();
    const token = state.createToken;
    state.media = media;
    state.source = source;
    setControlsDisabled(true);
    setLoading("Chargement de la forme d'onde…");
    const status = byId("audio-timeline-zoom-status");
    if (status) status.textContent = "Préparation de la timeline…";

    try {
      const moduleUrl = findWaveSurferModule();
      if (!moduleUrl) throw new Error("Module de forme d'onde indisponible");
      const module = await import(moduleUrl);
      if (token !== state.createToken || !container.isConnected) return;
      const WaveSurfer = module.u || Object.values(module).find((candidate) => (
        typeof candidate === "function"
        && typeof candidate.create === "function"
        && typeof candidate.prototype?.zoom === "function"
      ));
      if (!WaveSurfer) throw new Error("Moteur de forme d'onde indisponible");

      container.replaceChildren();
      const waveform = WaveSurfer.create({
        container,
        media,
        height: 58,
        waveColor: "#b9a58f",
        progressColor: "#65584c",
        cursorColor: "#65584c",
        cursorWidth: 2,
        barWidth: 2,
        barGap: 2,
        barRadius: 2,
        dragToSeek: true,
        interact: true,
        minPxPerSec: 20 * state.zoom / 100,
        fillParent: true,
        sampleRate: 8000,
      });
      state.waveform = waveform;

      const sync = () => updateTimes();
      media.addEventListener("timeupdate", sync);
      media.addEventListener("durationchange", sync);
      media.addEventListener("loadedmetadata", sync);
      state.detachMedia = () => {
        media.removeEventListener("timeupdate", sync);
        media.removeEventListener("durationchange", sync);
        media.removeEventListener("loadedmetadata", sync);
      };

      const ready = () => {
        if (token !== state.createToken) return;
        setControlsDisabled(false);
        applyZoom(state.zoom);
        updateTimes();
      };
      waveform.on("ready", ready);
      waveform.on("decode", ready);
      waveform.on("error", () => {
        if (token !== state.createToken) return;
        setControlsDisabled(true);
        setLoading("La timeline n'a pas pu être affichée.");
        if (status) status.textContent = "Timeline indisponible";
      });
      if (waveform.getDecodedData?.()) ready();
    } catch (error) {
      if (token !== state.createToken) return;
      console.error("Timeline audio:", error);
      setControlsDisabled(true);
      setLoading("La timeline n'a pas pu être affichée.");
      if (status) status.textContent = "Timeline indisponible";
    }
  };

  state.refresh = () => {
    bindControls();
    const panel = byId("audio-timeline-panel");
    if (!panel) return;
    const media = findPlayerMedia();
    const source = media?.currentSrc || media?.src || "";
    if (!media || !source) return;
    if (state.media === media && state.source === source && state.waveform) {
      updateTimes();
      return;
    }
    createWaveform(media, source);
  };
  state.refreshSoon = (delay = 90) => {
    window.clearTimeout(state.refreshTimer);
    state.refreshTimer = window.setTimeout(state.refresh, delay);
  };
  state.pause = () => state.media?.pause();

  const bind = (id, event, handler) => {
    const element = byId(id);
    if (!element || element.dataset.timelineBound === "true") return;
    element.dataset.timelineBound = "true";
    element.addEventListener(event, handler);
  };
  const bindControls = () => {
    bind("audio-timeline-fit", "click", () => applyZoom(0));
    bind("audio-timeline-out", "click", () => applyZoom(state.zoom - 10));
    bind("audio-timeline-in", "click", () => applyZoom(state.zoom + 10));
    bind("audio-timeline-range", "input", (event) => applyZoom(event.currentTarget.value));
  };
  bindControls();

  window.__transcriptionTimeline = state;
  state.refreshSoon(0);
  window.setInterval(() => state.refreshSoon(0), 700);
  return [];
}
"""


OPEN_FICHE_JS = r"""
(...args) => {
  const show = (id, visible) => {
    document.querySelectorAll(`[id="${id}"]`).forEach((element) => {
      element.style.setProperty("display", visible ? "flex" : "none", "important");
    });
  };
  show("fiche-library-hero", false);
  show("fiche-library-view", false);
  show("fiche-detail-hero", true);
  show("fiche-detail-view", true);
  try { window.localStorage.setItem("studioAudio.ficheView", "detail"); }
  catch (_) {}
  window.__transcriptionTimeline?.refreshSoon(120);
  window.setTimeout(() => {
    const title = document.querySelector("#fiche-detail-view .renameable-audio-title[data-audio-id]");
    try {
      if (title?.dataset.audioId) {
        window.localStorage.setItem("studioAudio.ficheItemId", title.dataset.audioId);
        window.localStorage.setItem("studioAudio.ficheItemName", title.textContent.trim());
      }
    } catch (_) {}
    document.getElementById("fiche-detail-hero")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, 120);
  return args;
}
"""


CLOSE_FICHE_JS = r"""
() => {
  const show = (id, visible) => {
    document.querySelectorAll(`[id="${id}"]`).forEach((element) => {
      element.style.setProperty("display", visible ? "flex" : "none", "important");
    });
  };
  show("fiche-library-hero", true);
  show("fiche-library-view", true);
  show("fiche-detail-hero", false);
  show("fiche-detail-view", false);
  try {
    window.localStorage.setItem("studioAudio.ficheView", "library");
    window.localStorage.removeItem("studioAudio.ficheItemId");
    window.localStorage.removeItem("studioAudio.ficheItemName");
  } catch (_) {}
  window.__transcriptionTimeline?.pause();
  window.setTimeout(() => {
    document.getElementById("fiche-library-hero")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, 40);
  return [];
}
"""


def _setup_details_html(active_stage: str = "") -> str:
    status = setup_components_status()
    items = [
        ("Mode Très rapide", status["base"]),
        ("Mode Équilibré", status["small"]),
        ("Mode Précis", status["turbo"]),
    ]
    if openvino_arc_supported():
        items.append(("Intel Arc · pilote", status["openvino"]))
        items.append(("Assistant Premium · Qwen3 8B", status["assistant"]))
    items.append(("Séparation des voix", status["speakers"]))
    blocks = []
    for label, installed in items:
        if installed:
            state = "Installé"
        elif active_stage == label:
            state = "Téléchargement"
        else:
            state = "En attente"
        blocks.append(
            f"<div class='model-item'><strong>{label}</strong><span>{state}</span></div>"
        )
    return "<div class='model-grid'>" + "".join(blocks) + "</div>"


def _setup_progress_html(percent: int) -> str:
    value = max(0, min(100, int(percent)))
    return (
        "<div class='setup-progress'>"
        f"<progress max='100' value='{value}' aria-label='Progression : {value} %'></progress>"
        f"<span>{value} %</span>"
        "</div>"
    )


def prepare_models_ui():
    if models_ready():
        yield (
            _setup_progress_html(100),
            "### Application prête\n\nTous les modèles sont installés sur cet ordinateur.",
            _setup_details_html(),
            gr.Column(visible=False),
            gr.Column(visible=True),
            gr.Button(visible=False),
        )
        return

    events: queue.Queue[tuple[str, object]] = queue.Queue()

    def report(label: str, index: int, total: int) -> None:
        events.put(("stage", (label, index, total)))

    def worker() -> None:
        try:
            prepare_all_models(report)
            events.put(("done", None))
        except Exception as exc:
            events.put(("error", exc))

    threading.Thread(
        target=worker,
        daemon=True,
        name="preparation-modeles",
    ).start()

    active_stage = ""
    last_percent = -1
    while True:
        try:
            event_type, payload = events.get(timeout=0.65)
        except queue.Empty:
            percent = min(
                94,
                max(1, int(downloaded_bytes() / EXPECTED_DOWNLOAD_BYTES * 94)),
            )
            if percent == last_percent:
                continue
            last_percent = percent
            yield (
                _setup_progress_html(percent),
                f"### Préparation en cours\n\n{active_stage or 'Vérification des modèles'}… **{percent} %**",
                _setup_details_html(active_stage),
                gr.skip(),
                gr.skip(),
                gr.skip(),
            )
            continue

        if event_type == "stage":
            active_stage, index, total = payload
            percent = min(
                94,
                max(int((int(index) - 1) / int(total) * 94), int(downloaded_bytes() / EXPECTED_DOWNLOAD_BYTES * 94)),
            )
            last_percent = percent
            yield (
                _setup_progress_html(percent),
                f"### Préparation en cours\n\n{active_stage}… **{percent} %**",
                _setup_details_html(active_stage),
                gr.skip(),
                gr.skip(),
                gr.skip(),
            )
        elif event_type == "done":
            yield (
                _setup_progress_html(100),
                "### Application prête\n\nLes moteurs sont installés. Ouverture de votre espace de transcription…",
                _setup_details_html(),
                gr.Column(visible=False),
                gr.Column(visible=True),
                gr.Button(visible=False),
            )
            return
        elif event_type == "error":
            exc = payload
            yield (
                _setup_progress_html(max(0, last_percent)),
                "### La préparation n'a pas pu se terminer\n\n"
                f"Vérifiez la connexion Internet puis réessayez. Détail : `{type(exc).__name__}: {exc}`",
                _setup_details_html(active_stage),
                gr.Column(visible=True),
                gr.Column(visible=False),
                gr.Button(visible=True),
            )
            return


def _status_markdown(result) -> str:
    if result.duration > 0 and result.processing_seconds > 0:
        ratio = result.duration / result.processing_seconds
        speed = (
            f"{ratio:.1f}× le temps réel"
            if ratio >= 1
            else f"{1 / ratio:.1f}× plus lent que le temps réel"
        )
    else:
        speed = "vitesse non calculée"

    probability = ""
    if result.language_probability is not None:
        probability = f" ({result.language_probability:.0%})"

    status = (
        f"**{len(result.turns)} tours de parole** · "
        f"audio de **{format_duration(result.duration)}** · "
        f"traité en **{format_duration(result.processing_seconds)}** ({speed}) · "
        f"langue **{result.language}{probability}** · "
        f"**{result.detected_speakers} voix étiquetée(s)**"
    )
    if result.warnings:
        status += "\n\n> Attention : " + "  \n> ".join(result.warnings)
    return status


def _normalise_audio_paths(value: object) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, os.PathLike)):
        raw_items: Iterable[object] = [value]
    elif isinstance(value, (list, tuple)):
        raw_items = value
    else:
        raw_items = [value]

    paths: list[Path] = []
    for item in raw_items:
        if item is None:
            continue
        if isinstance(item, dict):
            item = item.get("path") or item.get("name")
        elif not isinstance(item, (str, os.PathLike)) and hasattr(item, "name"):
            item = getattr(item, "name")
        if item:
            paths.append(Path(str(item)))
    return paths


def _audio_key(path: Path) -> str:
    return str(path)


def _audio_path_identity(value: str | os.PathLike[str] | Path) -> str:
    """Construit une identité de chemin stable pour les événements du File.

    Gradio peut présenter le chemin temporaire reçu lors de l'upload ou le
    chemin de la copie gérée restaurée par le serveur. Le workspace conserve
    les deux ; cette normalisation permet de les comparer sans dépendre de la
    casse utilisée par Windows.
    """

    try:
        resolved = Path(value).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        resolved = Path(os.path.abspath(os.fspath(value)))
    return os.path.normcase(str(resolved))


def _workspace_audio_id_for_path(
    state: dict[str, object] | None,
    path: str | os.PathLike[str] | Path,
) -> str | None:
    """Résout un chemin visible vers l'identifiant durable de son audio."""

    identity = _audio_path_identity(path)
    items = _workspace_items(state)
    for item_id in _workspace_order(state):
        item = items.get(item_id)
        if not isinstance(item, dict):
            continue
        candidates = (item.get("path"), item.get("source_path"))
        if any(
            candidate
            and _audio_path_identity(str(candidate)) == identity
            for candidate in candidates
        ):
            return item_id
    return None


def _display_audio_name(path: Path, audio_names: dict[str, str] | None = None) -> str:
    custom_name = (audio_names or {}).get(_audio_key(path), "").strip()
    return custom_name or path.name


def _initial_queue_rows(
    paths: list[Path],
    audio_names: dict[str, str] | None = None,
) -> list[list[object]]:
    return [
        [index, _display_audio_name(path, audio_names), "En attente", "0 %", "—", "Supprimer"]
        for index, path in enumerate(paths, start=1)
    ]


def _queue_hint(paths: list[Path], confirmation: str = "") -> str:
    if paths:
        count = len(paths)
        noun = "audio" if count == 1 else "audios"
        lead = (
            f"<strong>{count} {noun} à traiter.</strong> "
            "Faites glisser les fichiers pour définir l'ordre de traitement, "
            "puis lancez la file."
        )
    else:
        lead = (
            "<strong>Aucun audio en attente.</strong> "
            "Ajoutez un enregistrement ici ; les transcriptions terminées restent dans Fiches."
        )
    suffix = f" <span class='rename-confirmation'>{html.escape(confirmation)}</span>" if confirmation else ""
    return f"<div class='queue-count'>{lead}{suffix}</div>"


def _rename_choices(
    paths: list[Path],
    audio_names: dict[str, str],
) -> list[tuple[str, str]]:
    return [
        (f"{index}. {_display_audio_name(path, audio_names)}", _audio_key(path))
        for index, path in enumerate(paths, start=1)
    ]


def _rename_stem(display_name: str) -> str:
    return Path(display_name).stem


def _clean_audio_stem(value: str | None, source: Path) -> str:
    candidate = (value or "").strip()
    if source.suffix and candidate.lower().endswith(source.suffix.lower()):
        candidate = candidate[: -len(source.suffix)].rstrip()
    candidate = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip(" .")
    if not candidate:
        raise gr.Error("Saisissez un nom pour cet audio.")
    return candidate[:120].rstrip()


def preview_queue_ui(audio_files: object):
    paths = _normalise_audio_paths(audio_files)
    first = str(paths[0]) if paths else None
    rows = _initial_queue_rows(paths)
    return first, rows, _queue_hint(paths)


def sync_queue_ui(
    audio_files: object,
    audio_names: dict[str, str] | None,
):
    """Synchronise l'ordre visible et conserve les noms déjà personnalisés."""

    paths = _normalise_audio_paths(audio_files)
    previous_names = audio_names or {}
    names = {
        _audio_key(path): _display_audio_name(path, previous_names)
        for path in paths
    }
    selected = _audio_key(paths[0]) if paths else None
    current_stem = _rename_stem(names[selected]) if selected else ""
    enabled = bool(paths)
    return (
        str(paths[0]) if paths else None,
        _initial_queue_rows(paths, names),
        _queue_hint(paths),
        names,
        gr.Dropdown(
            choices=_rename_choices(paths, names),
            value=selected,
            interactive=enabled,
        ),
        gr.Textbox(value=current_stem, interactive=enabled),
        gr.Button(interactive=enabled),
        gr.Button(value="Lancer la file d'attente", interactive=enabled),
    )


def load_rename_name_ui(
    selected_path: str | None,
    audio_names: dict[str, str] | None,
):
    display_name = (audio_names or {}).get(selected_path or "", "")
    enabled = bool(selected_path and display_name)
    return (
        gr.Textbox(
            value=_rename_stem(display_name) if enabled else "",
            interactive=enabled,
        ),
        gr.Button(interactive=enabled),
    )


def rename_audio_ui(
    audio_files: object,
    selected_path: str | None,
    proposed_name: str | None,
    audio_names: dict[str, str] | None,
):
    paths = _normalise_audio_paths(audio_files)
    by_key = {_audio_key(path): path for path in paths}
    if not selected_path or selected_path not in by_key:
        raise gr.Error("Sélectionnez d'abord un audio à renommer.")

    source = by_key[selected_path]
    clean_stem = _clean_audio_stem(proposed_name, source)
    display_name = f"{clean_stem}{source.suffix}"
    names = {
        _audio_key(path): _display_audio_name(path, audio_names)
        for path in paths
    }
    names[selected_path] = display_name
    return (
        names,
        _initial_queue_rows(paths, names),
        _queue_hint(paths, f"Nom enregistré : {display_name}"),
        gr.Dropdown(
            choices=_rename_choices(paths, names),
            value=selected_path,
            interactive=True,
        ),
        gr.Textbox(value=clean_stem, interactive=True),
        gr.Button(interactive=True),
    )


def rename_audio_by_index_ui(
    audio_files: object,
    selected_index: str | int | float | None,
    proposed_name: str | None,
    audio_names: dict[str, str] | None,
):
    paths = _normalise_audio_paths(audio_files)
    try:
        index = int(float(selected_index or 0))
        selected_path = _audio_key(paths[index])
    except (IndexError, TypeError, ValueError):
        raise gr.Error("Ce fichier n'est plus présent dans la file d'attente.") from None
    return rename_audio_ui(audio_files, selected_path, proposed_name, audio_names)


def rename_audio_by_target_ui(
    audio_files: object,
    rename_target: str | int | float | None,
    proposed_name: str | None,
    audio_names: dict[str, str] | None,
    workspace_state: dict[str, object] | None,
    library_sort: str | None,
):
    """Renomme par identifiant stable, sans confondre bibliothèque et file active."""

    target = str(rename_target or "").strip()
    state = normalise_workspace(workspace_state or load_workspace())
    order = _workspace_order(state)
    item_id: str | None = None

    if target.startswith("item:"):
        candidate = target.removeprefix("item:")
        item_id = candidate if candidate in order else None
    elif target.startswith("library:"):
        try:
            library_index = int(target.removeprefix("library:"))
            item_id = _workspace_library_ids(state, library_sort)[library_index]
        except (IndexError, TypeError, ValueError):
            item_id = None
    elif target.startswith("queue:"):
        try:
            index = int(target.removeprefix("queue:"))
        except ValueError:
            index = -1
        return rename_audio_by_index_ui(
            audio_files,
            index,
            proposed_name,
            audio_names,
        )
    else:
        try:
            index = int(float(target))
        except (TypeError, ValueError):
            index = -1
        return rename_audio_by_index_ui(
            audio_files,
            index,
            proposed_name,
            audio_names,
        )

    item = _workspace_items(state).get(item_id or "")
    if not isinstance(item, dict):
        raise gr.Error("Cet audio n'est plus disponible pour le renommage.")
    source = Path(str(item.get("path") or ""))
    if not source.is_file():
        raise gr.Error("Le fichier audio associé à cette fiche est introuvable.")

    clean_stem = _clean_audio_stem(proposed_name, source)
    display_name = f"{clean_stem}{source.suffix}"
    paths = _normalise_audio_paths(audio_files)
    names = {
        _audio_key(path): _display_audio_name(path, audio_names)
        for path in paths
    }
    for current_id in order:
        current = _workspace_items(state).get(current_id)
        if not isinstance(current, dict):
            continue
        current_path = Path(str(current.get("path") or ""))
        if current_path:
            names[str(current_path)] = str(current.get("name") or current_path.name)
    names[str(source)] = display_name
    selected = str(paths[0]) if paths else None
    return (
        names,
        _initial_queue_rows(paths, names),
        _queue_hint(paths, f"Nom enregistré : {display_name}"),
        gr.Dropdown(
            choices=_rename_choices(paths, names),
            value=selected,
            interactive=bool(paths),
        ),
        gr.Textbox(
            value=_rename_stem(names[selected]) if selected else "",
            interactive=bool(paths),
        ),
        gr.Button(interactive=bool(paths)),
    )


def delete_audio_by_target_ui(
    delete_target: str | None,
    library_sort: str | None,
    workspace_state: dict[str, object] | None,
) -> str:
    """Supprime l'élément confirmé dans l'une des deux listes visibles."""

    target = str(delete_target or "").strip()
    state = workspace_state if isinstance(workspace_state, dict) else load_workspace()
    order = _workspace_order(state)
    item_id: str | None = None

    if target.startswith("library:"):
        try:
            item_id = _workspace_library_ids(
                state,
                library_sort,
            )[int(target.removeprefix("library:"))]
        except (IndexError, TypeError, ValueError):
            item_id = None
    elif target.startswith("queue:"):
        try:
            item_id = _workspace_queue_ids(state)[int(target.removeprefix("queue:"))]
        except (IndexError, TypeError, ValueError):
            item_id = None
    elif target.startswith("item:"):
        candidate = target.removeprefix("item:")
        item_id = candidate if candidate in order else None

    if not item_id:
        raise gr.Error("Cet audio n'est plus disponible.")

    current_item = _workspace_items(state).get(item_id, {})
    if str(current_item.get("status") or "") == "En cours" or JOB_MANAGER.is_active(item_id):
        raise gr.Error("Attendez la fin de la transcription avant de supprimer cet audio.")

    try:
        disk_state = load_workspace()
        _, deleted_name = delete_workspace_item(disk_state, item_id)
    except KeyError:
        raise gr.Error("Cet audio a déjà été supprimé.") from None
    except ValueError as exc:
        raise gr.Error(str(exc)) from None
    except OSError as exc:
        raise gr.Error(f"La suppression a échoué : {exc}") from None

    return (
        "### Audio supprimé\n\n"
        f"**{deleted_name}** a été retiré du workspace. "
        "Le fichier source d'origine n'a pas été touché."
    )


def _turns_from_result_data(data: dict[str, object]) -> list[TranscriptTurn]:
    turns: list[TranscriptTurn] = []
    for raw in data.get("turns", []) or []:
        if not isinstance(raw, dict):
            continue
        turns.append(
            TranscriptTurn(
                start=float(raw.get("start") or 0.0),
                end=float(raw.get("end") or 0.0),
                speaker_id=str(raw.get("speaker_id") or "SPEAKER"),
                speaker=str(raw.get("speaker") or "Interlocuteur"),
                text=str(raw.get("text") or ""),
            )
        )
    return turns


def _metadata_from_result_data(data: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in data.items() if key != "turns"}


def _workspace_items(state: dict[str, object] | None) -> dict[str, dict[str, object]]:
    items = (state or {}).get("items", {})
    return items if isinstance(items, dict) else {}


def _workspace_order(state: dict[str, object] | None) -> list[str]:
    order = (state or {}).get("order", [])
    return [str(value) for value in order] if isinstance(order, list) else []


def _workspace_queue_ids(state: dict[str, object] | None) -> list[str]:
    """Retourne uniquement l'ordre de la file active, jamais l'historique terminé."""

    items = _workspace_items(state)
    has_explicit_queue = isinstance(state, dict) and "queue_order" in state
    raw_queue = (state or {}).get("queue_order", [])
    queue_order = (
        [str(value) for value in raw_queue]
        if has_explicit_queue and isinstance(raw_queue, list)
        else _workspace_order(state)
    )
    active_ids = JOB_MANAGER.active_item_ids()
    ids: list[str] = []
    for item_id in queue_order:
        item = items.get(item_id)
        if not isinstance(item, dict) or item_id in ids:
            continue
        completed = (
            str(item.get("status") or "") == "Terminé"
            and isinstance(item.get("result"), dict)
        )
        if not completed:
            ids.append(item_id)
    for item_id in _workspace_order(state):
        item = items.get(item_id)
        if (
            isinstance(item, dict)
            and item_id in active_ids
            and item_id not in ids
        ):
            ids.append(item_id)
    return ids


def _workspace_queue_assets(
    state: dict[str, object] | None,
) -> list[tuple[str, Path]]:
    items = _workspace_items(state)
    return [
        (item_id, path)
        for item_id in _workspace_queue_ids(state)
        if (path := Path(str(items.get(item_id, {}).get("path") or ""))).is_file()
    ]


def _workspace_queue_rows(state: dict[str, object] | None) -> list[list[object]]:
    items = _workspace_items(state)
    rows: list[list[object]] = []
    for index, item_id in enumerate(_workspace_queue_ids(state), start=1):
        item = items.get(item_id, {})
        progress = max(0.0, min(1.0, float(item.get("progress") or 0.0)))
        rows.append(
            [
                index,
                str(item.get("name") or "Audio"),
                str(item.get("status") or "En attente"),
                f"{progress:.0%}",
                str(item.get("processing_time") or "—"),
                "Supprimer",
            ]
        )
    return rows


def _item_created_at(item: dict[str, object]) -> datetime:
    raw = str(item.get("created_at") or "")
    if raw:
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(Path(str(item.get("path") or "")).stat().st_mtime)
    except OSError:
        return datetime.fromtimestamp(0)


def _workspace_library_ids(
    state: dict[str, object] | None,
    sort_by: str | None,
) -> list[str]:
    items = _workspace_items(state)
    ids = [item_id for item_id in _workspace_order(state) if item_id in items]
    selected_sort = sort_by or LIBRARY_SORTS[0]
    if selected_sort.startswith("Nom"):
        ids.sort(key=lambda item_id: str(items[item_id].get("name") or "").casefold())
    elif selected_sort == "Statut":
        status_rank = {"En cours": 0, "À reprendre": 1, "En attente": 2, "Échec": 3, "Terminé": 4}
        ids.sort(
            key=lambda item_id: (
                status_rank.get(str(items[item_id].get("status") or ""), 9),
                str(items[item_id].get("name") or "").casefold(),
            )
        )
    elif selected_sort == "Progression":
        ids.sort(
            key=lambda item_id: float(items[item_id].get("progress") or 0.0),
            reverse=True,
        )
    else:
        ids.sort(key=lambda item_id: _item_created_at(items[item_id]), reverse=True)
    return ids


def _workspace_library_rows(
    state: dict[str, object] | None,
    sort_by: str | None = None,
) -> list[list[object]]:
    items = _workspace_items(state)
    rows: list[list[object]] = []
    for item_id in _workspace_library_ids(state, sort_by):
        item = items[item_id]
        comments = item.get("comments") if isinstance(item.get("comments"), list) else []
        progress = max(0.0, min(1.0, float(item.get("progress") or 0.0)))
        rows.append(
            [
                str(item.get("name") or "Audio"),
                _item_created_at(item).strftime("%d/%m/%Y · %H:%M"),
                str(item.get("status") or "En attente"),
                f"{progress:.0%}",
                len(comments),
                "Supprimer",
            ]
        )
    return rows


def sync_library_ui(
    workspace_state: dict[str, object] | None,
    sort_by: str | None,
):
    return _workspace_library_rows(workspace_state, sort_by)


def _workspace_choices(state: dict[str, object] | None) -> list[tuple[str, str]]:
    items = _workspace_items(state)
    choices: list[tuple[str, str]] = []
    for index, item_id in enumerate(_workspace_order(state), start=1):
        item = items.get(item_id, {})
        name = str(item.get("name") or "Audio")
        status = str(item.get("status") or "En attente")
        choices.append((f"{index:02d} · {name} — {status}", item_id))
    return choices


def _workspace_item(
    item_id: str | None,
    state: dict[str, object] | None,
) -> dict[str, object] | None:
    if not item_id:
        return None
    item = _workspace_items(state).get(item_id)
    return item if isinstance(item, dict) else None


def _speaker_role_labels(item: dict[str, object] | None) -> dict[str, str]:
    """Retourne les libellés métier associés aux identifiants de voix.

    Les identifiants ``SPEAKER_*`` restent la source de vérité afin de ne pas
    altérer la diarisation. Seuls les libellés présentés à l'utilisateur sont
    remplacés, ce qui permet à une inversion Client/Master de s'appliquer sans
    réécrire le résultat de transcription enregistré.
    """

    raw_roles = item.get("speaker_roles") if isinstance(item, dict) else None
    if not isinstance(raw_roles, dict):
        return {}
    labels: dict[str, str] = {}
    for raw_speaker_id, payload in raw_roles.items():
        if isinstance(payload, dict):
            role = str(payload.get("role") or "").strip()
        else:
            role = str(payload or "").strip()
        if role in {"Client", "Master"}:
            labels[str(raw_speaker_id)] = role
    return labels


def _display_turns(item: dict[str, object] | None) -> list[TranscriptTurn]:
    """Construit les tours visibles sans modifier les tours persistés."""

    if not item:
        return []
    result = item.get("result")
    if not isinstance(result, dict):
        return []
    role_labels = _speaker_role_labels(item)
    turns = _turns_from_result_data(result)
    return [
        TranscriptTurn(
            start=turn.start,
            end=turn.end,
            speaker_id=turn.speaker_id,
            speaker=role_labels.get(turn.speaker_id, turn.speaker),
            text=turn.text,
        )
        for turn in turns
    ]


def _item_transcript(item: dict[str, object] | None) -> str:
    if not item:
        return ""
    result = item.get("result")
    if isinstance(result, dict):
        return plain_transcript(_display_turns(item))
    return str(item.get("partial") or "")


def _commenting_enabled(item: dict[str, object] | None) -> bool:
    """Autorise une discussion dès qu'un texte réellement visible existe."""

    return bool(item and _item_transcript(item).strip())


def _comment_anchor(
    item: dict[str, object] | None,
    transcript: str,
    passage: str,
    selection_index: object = None,
) -> dict[str, object]:
    """Relie une sélection textuelle aux tours et timecodes disponibles."""

    clean_passage = (passage or "").strip()
    anchor: dict[str, object] = {
        "start_seconds": None,
        "end_seconds": None,
        "quote": clean_passage,
        "source": "partial-text",
        "selection_start": None,
        "selection_end": None,
        "turn_ids": [],
    }
    if not clean_passage:
        return anchor

    selection_start: int | None = None
    selection_end: int | None = None
    if isinstance(selection_index, (list, tuple)) and len(selection_index) == 2:
        try:
            selection_start = max(0, int(selection_index[0]))
            selection_end = max(selection_start, int(selection_index[1]))
        except (TypeError, ValueError):
            selection_start = selection_end = None
    if selection_start is None:
        found = (transcript or "").find(clean_passage)
        if found >= 0:
            selection_start = found
            selection_end = found + len(clean_passage)
    anchor["selection_start"] = selection_start
    anchor["selection_end"] = selection_end

    result = item.get("result") if isinstance(item, dict) else None
    turns = _display_turns(item) if isinstance(result, dict) else []
    if not turns or selection_start is None or selection_end is None:
        return anchor

    blocks = [
        f"[{format_timestamp(turn.start)}] {turn.speaker}\n{turn.text}"
        for turn in turns
    ]
    cursor = 0
    matched: list[tuple[int, TranscriptTurn]] = []
    for turn_index, (turn, block) in enumerate(zip(turns, blocks), start=1):
        block_start = cursor
        block_end = block_start + len(block)
        if selection_end > block_start and selection_start < block_end:
            matched.append((turn_index, turn))
        cursor = block_end + 2
    if not matched:
        return anchor

    matched_turns = [turn for _, turn in matched]
    anchor.update(
        {
            "start_seconds": float(matched_turns[0].start),
            "end_seconds": float(matched_turns[-1].end),
            "source": "transcript",
            "turn_ids": [f"turn-{turn_index}" for turn_index, _ in matched],
            "speaker_ids": list(dict.fromkeys(turn.speaker_id for turn in matched_turns)),
        }
    )
    speakers = {turn.speaker for turn in matched_turns if turn.speaker}
    if len(speakers) == 1:
        anchor["speaker"] = next(iter(speakers))
    return anchor


def _anchor_label(anchor: object) -> str:
    if not isinstance(anchor, dict):
        return "Repère temporel indisponible"
    try:
        start = float(anchor.get("start_seconds"))
    except (TypeError, ValueError):
        return "Repère temporel indisponible"
    try:
        end = max(start, float(anchor.get("end_seconds")))
    except (TypeError, ValueError):
        end = start
    return f"{format_timestamp(start)} → {format_timestamp(end)}"


def _selection_context_html(
    passage: str | None = None,
    anchor: object = None,
    *,
    editing: bool = False,
) -> str:
    clean_passage = (passage or "").strip()
    if not clean_passage:
        return (
            "<section class='selection-context is-empty' aria-live='polite'>"
            "<header><span class='selection-kicker'>Passage de travail</span>"
            "<span class='selection-state'>Aucun extrait</span></header>"
            "<strong>Aucun passage sélectionné</strong>"
            "<p>Sélectionnez une phrase dans l'onglet Transcription pour ouvrir une discussion à ce timecode.</p>"
            "</section>"
        )
    safe_passage = html.escape(clean_passage)
    safe_label = html.escape(_anchor_label(anchor))
    title = "Discussion en cours de modification" if editing else "Passage sélectionné"
    return (
        "<section class='selection-context is-ready' aria-live='polite'>"
        f"<header><span class='selection-kicker'>{title}</span><span class='selection-time'>{safe_label}</span></header>"
        f"<blockquote>{safe_passage}</blockquote>"
        "</section>"
    )


def _discussion_datetime_label(raw: object) -> str:
    """Affiche une date courte et locale, plus lisible qu'un timestamp ISO."""

    value = str(raw or "").strip()
    if not value:
        return "À l'instant"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone()
    except (TypeError, ValueError):
        return value
    day_delta = (datetime.now().astimezone().date() - parsed.date()).days
    if day_delta == 0:
        return f"Aujourd'hui · {parsed:%H:%M}"
    if day_delta == 1:
        return f"Hier · {parsed:%H:%M}"
    return parsed.strftime("%d/%m/%Y · %H:%M")


def capture_selected_passage_ui(
    item_id: str | None,
    text: str | None,
    workspace_state: dict[str, object] | None,
    evt: gr.SelectData,
):
    passage = capture_selected_passage(text, evt)
    state = normalise_workspace(workspace_state or load_workspace())
    item = _workspace_item(item_id, state)
    anchor = _comment_anchor(item, text or "", passage, evt.index)
    return (
        passage,
        json.dumps(anchor, ensure_ascii=False),
        _selection_context_html(passage, anchor),
        str(item_id or ""),
    )


def _comments_feed_html(item: dict[str, object] | None) -> str:
    comments = item.get("comments") if isinstance(item, dict) else None
    if not isinstance(comments, list) or not comments:
        return (
            "<section class='discussion-empty'>"
            "<span class='selection-kicker'>Aucune discussion</span>"
            "<strong>Créez le premier échange sur un passage important.</strong>"
            "<p>Après la sélection, écrivez votre message ci-dessous : le timecode restera attaché à la fiche.</p>"
            "</section>"
        )

    cards: list[str] = []
    for comment in reversed(comments):
        if not isinstance(comment, dict):
            continue
        comment_id = html.escape(str(comment.get("id") or ""), quote=True)
        author_text = str(comment.get("author") or "Moi")
        author = html.escape(author_text)
        created_at_raw = html.escape(str(comment.get("created_at") or ""), quote=True)
        created_at = html.escape(_discussion_datetime_label(comment.get("created_at")))
        body = html.escape(str(comment.get("body") or ""))
        raw_anchor = comment.get("anchor") if isinstance(comment.get("anchor"), dict) else {}
        passage = html.escape(str(comment.get("passage") or raw_anchor.get("quote") or ""))
        status = "Résolu" if str(comment.get("status") or "") == "Résolu" else "À discuter"
        status_class = "resolved" if status == "Résolu" else "open"
        anchor = raw_anchor
        start = anchor.get("start_seconds") if isinstance(anchor, dict) else None
        try:
            start_value = float(start)
        except (TypeError, ValueError):
            start_value = None
        anchor_label = _anchor_label(anchor)
        action_context = html.escape(
            f"la discussion de {author_text}, passage {anchor_label}",
            quote=True,
        )
        listen_button = (
            f"<button type='button' data-comment-action='seek' data-start='{start_value:.3f}' "
            f"aria-label='Écouter {action_context}'>Écouter</button>"
            if start_value is not None
            else ""
        )
        replies_html: list[str] = []
        replies = comment.get("replies") if isinstance(comment.get("replies"), list) else []
        for reply in replies:
            if not isinstance(reply, dict):
                continue
            reply_created_at = html.escape(str(reply.get("created_at") or ""), quote=True)
            replies_html.append(
                "<article class='thread-reply'>"
                f"<div><strong>{html.escape(str(reply.get('author') or 'Moi'))}</strong>"
                f"<time datetime='{reply_created_at}'>"
                f"{html.escape(_discussion_datetime_label(reply.get('created_at')))}</time></div>"
                f"<p>{html.escape(str(reply.get('body') or ''))}</p>"
                "</article>"
            )
        reply_count = len(replies_html)
        reply_label = f"{reply_count} réponse{'s' if reply_count != 1 else ''}"
        replies_section = (
            f"<div class='thread-replies-label'>{reply_label}</div>"
            f"<div class='thread-replies'>{''.join(replies_html)}</div>"
            if replies_html
            else ""
        )
        action_label = "Rouvrir" if status == "Résolu" else "Résoudre"
        cards.append(
            f"<article class='discussion-thread is-{status_class}' data-comment-id='{comment_id}'>"
            "<header class='thread-head'>"
            f"<div><strong>{author}</strong><time datetime='{created_at_raw}'>{created_at}</time></div>"
            f"<span class='thread-status'>{status}</span>"
            "</header>"
            f"<button type='button' class='thread-quote' data-comment-action='reply' data-comment-id='{comment_id}' "
            f"aria-label='Répondre à {action_context}'>"
            f"<span>{html.escape(anchor_label)}</span><q>{passage}</q></button>"
            f"<p class='thread-message'>{body}</p>"
            f"{replies_section}"
            "<div class='thread-actions'>"
            f"{listen_button}"
            f"<button type='button' data-comment-action='reply' data-comment-id='{comment_id}' aria-label='Répondre à {action_context}'>Répondre</button>"
            f"<button type='button' data-comment-action='edit' data-comment-id='{comment_id}' aria-label='Modifier {action_context}'>Modifier</button>"
            f"<button type='button' data-comment-action='resolve' data-comment-id='{comment_id}' aria-label='{action_label} {action_context}'>{action_label}</button>"
            f"<button type='button' class='thread-delete' data-comment-action='request-delete' data-comment-id='{comment_id}' aria-label='Supprimer {action_context}'>Supprimer</button>"
            "</div>"
            "<div class='thread-delete-confirm' role='alert'>"
            "<span>Supprimer cette discussion et ses réponses ?</span>"
            f"<button type='button' data-comment-action='cancel-delete' data-comment-id='{comment_id}'>Annuler</button>"
            f"<button type='button' class='confirm-delete' data-comment-action='delete' data-comment-id='{comment_id}'>Confirmer</button>"
            "</div>"
            "</article>"
        )
    return "<section class='discussion-feed'>" + "".join(cards) + "</section>"


def _comment_rows(item: dict[str, object] | None) -> list[list[str]]:
    if not item:
        return []
    comments = item.get("comments")
    if not isinstance(comments, list):
        return []
    return [
        [
            str(comment.get("author") or "—"),
            str(comment.get("passage") or "—"),
            str(comment.get("body") or ""),
            str(comment.get("status") or "À discuter"),
            str(comment.get("created_at") or ""),
        ]
        for comment in comments
        if isinstance(comment, dict)
    ]


def _comment_choices(item: dict[str, object] | None) -> list[tuple[str, str]]:
    if not item or not isinstance(item.get("comments"), list):
        return []
    choices: list[tuple[str, str]] = []
    indexed_comments = list(enumerate(item["comments"], start=1))
    for index, comment in reversed(indexed_comments):
        if not isinstance(comment, dict):
            continue
        passage = str(comment.get("passage") or "Passage").replace("\n", " ")
        label = (
            f"{index:02d} · {comment.get('status') or 'À discuter'} · "
            f"{passage[:54]}{'…' if len(passage) > 54 else ''}"
        )
        choices.append((label, str(comment.get("id") or index)))
    return choices


def _comment_action_label(
    item: dict[str, object] | None,
    comment_id: str | None,
) -> str:
    comments = item.get("comments") if item else None
    if isinstance(comments, list):
        for comment in comments:
            if isinstance(comment, dict) and str(comment.get("id")) == str(comment_id):
                return "Rouvrir la discussion" if comment.get("status") == "Résolu" else "Marquer comme résolu"
    return "Marquer comme résolu"


def _fiche_header_html(item: dict[str, object] | None) -> str:
    if not item:
        return (
            "<section class='fiche-empty'><strong>Aucune fiche ouverte</strong>"
            "<span>Ajoutez un audio ou choisissez une ligne de la file d'attente.</span></section>"
        )

    name = html.escape(str(item.get("name") or "Audio"))
    item_id = html.escape(str(item.get("id") or ""), quote=True)
    original = html.escape(str(item.get("original_name") or ""))
    status = str(item.get("status") or "En attente")
    status_class = {
        "En attente": "waiting",
        "À reprendre": "waiting",
        "En cours": "running",
        "Terminé": "done",
        "Échec": "failed",
    }.get(status, "waiting")
    progress = max(0.0, min(1.0, float(item.get("progress") or 0.0)))
    comments = item.get("comments") if isinstance(item.get("comments"), list) else []
    unresolved = sum(
        1
        for comment in comments
        if isinstance(comment, dict) and comment.get("status") != "Résolu"
    )
    result = item.get("result") if isinstance(item.get("result"), dict) else {}
    duration = format_duration(float(result.get("duration") or 0.0)) if result else "À analyser"
    processing_seconds = max(0.0, float(item.get("processing_seconds") or 0.0))
    raw_processing_base = item.get("processing_base_seconds")
    processing_base_seconds = (
        processing_seconds
        if raw_processing_base is None
        else max(0.0, float(raw_processing_base or 0.0))
    )
    processing_started_at = str(item.get("processing_started_at") or "")
    processing_clock_attributes = ""
    if status == "En cours" and processing_started_at:
        processing_clock_attributes = (
            " data-processing-live='true'"
            f" data-processing-seconds='{processing_seconds:.3f}'"
            f" data-processing-base-seconds='{processing_base_seconds:.3f}'"
            " data-processing-started-at='"
            f"{html.escape(processing_started_at, quote=True)}'"
        )
    processing = html.escape(
        format_duration(processing_seconds)
        if processing_seconds > 0 or status == "En cours"
        else str(item.get("processing_time") or "—")
    )
    stage = html.escape(str(item.get("stage") or ""))
    checkpoint_position = max(0.0, float(item.get("checkpoint_position") or 0.0))
    resume_note = (
        f"<span>Point de reprise enregistré à {format_duration(checkpoint_position)}</span>"
        if checkpoint_position > 0 and status in {"En cours", "À reprendre", "Échec"}
        else ""
    )
    return f"""
      <section class="fiche-head" data-audio-status="{html.escape(status, quote=True)}">
        <div class="fiche-title-row">
          <div>
            <span class="eyebrow">Fiche audio</span>
            <h2 class="renameable-audio-title" data-audio-id="{item_id}" title="Double-cliquez pour renommer"><span class="audio-title-text">{name}</span></h2>
            <p>Fichier source : {original}</p>
          </div>
          <span class="status-chip status-{status_class}">{html.escape(status)}</span>
        </div>
        <div class="fiche-progress">
          <progress max="100" value="{int(progress * 100)}"></progress>
          <span>{progress:.0%}</span>
        </div>
        <div class="fiche-live-stage"><strong>{stage or status}</strong>{resume_note}</div>
        <div class="fiche-metrics">
          <span><strong>{duration}</strong> durée</span>
          <span><strong class="fiche-processing-time"{processing_clock_attributes}>{processing}</strong> traitement</span>
          <span><strong>{len(comments)}</strong> commentaire(s)</span>
          <span><strong>{unresolved}</strong> à discuter</span>
        </div>
      </section>
    """


def _dashboard_html(state: dict[str, object] | None) -> str:
    items = _workspace_items(state)
    order = _workspace_order(state)
    statuses = {"En attente": 0, "À reprendre": 0, "En cours": 0, "Terminé": 0, "Échec": 0}
    unresolved = 0
    for item_id in order:
        item = items.get(item_id, {})
        status = str(item.get("status") or "En attente")
        statuses[status] = statuses.get(status, 0) + 1
        comments = item.get("comments") if isinstance(item.get("comments"), list) else []
        unresolved += sum(
            1
            for comment in comments
            if isinstance(comment, dict) and comment.get("status") != "Résolu"
        )

    recent_cards: list[str] = []
    for item_id in order[:5]:
        item = items.get(item_id, {})
        name = html.escape(str(item.get("name") or "Audio"))
        safe_item_id = html.escape(str(item_id), quote=True)
        status = html.escape(str(item.get("status") or "En attente"))
        progress = max(0.0, min(1.0, float(item.get("progress") or 0.0)))
        recent_cards.append(
            "<article class='recent-audio'>"
            f"<div><strong class='renameable-audio-title' data-audio-id='{safe_item_id}' title='Double-cliquez pour renommer'><span class='audio-title-text'>{name}</span></strong><span>{status}</span></div>"
            f"<div class='recent-progress'><i style='width:{progress:.0%}'></i></div>"
            f"<small>{progress:.0%}</small>"
            "</article>"
        )
    if not recent_cards:
        recent_cards.append(
            "<div class='dashboard-empty'><strong>Aucun audio pour le moment</strong>"
            "<span>Ajoutez votre premier fichier depuis l'onglet Audios.</span></div>"
        )

    return f"""
      <section class="dashboard-content">
        <div class="dashboard-intro">
          <div><span class="eyebrow">Vue d'ensemble</span><h2>Pilotage du workspace</h2></div>
          <p>Suivez le traitement et les sujets qui nécessitent encore une décision.</p>
        </div>
        <div class="kpi-grid">
          <article><span>Total</span><strong>{len(order)}</strong><small>fiches audio</small></article>
          <article><span>En attente</span><strong>{statuses.get('En attente', 0) + statuses.get('À reprendre', 0)}</strong><small>à lancer ou reprendre</small></article>
          <article><span>En cours</span><strong>{statuses.get('En cours', 0)}</strong><small>transcription active</small></article>
          <article><span>Terminés</span><strong>{statuses.get('Terminé', 0)}</strong><small>prêts à relire</small></article>
          <article class="attention-kpi"><span>À discuter</span><strong>{unresolved}</strong><small>commentaires ouverts</small></article>
        </div>
        <div class="dashboard-recent">
          <div class="dashboard-subhead"><h3>Audios récents</h3><span>Les cinq premières fiches de la file</span></div>
          {''.join(recent_cards)}
        </div>
      </section>
    """


def _normalise_engine_choice(engine: object) -> str:
    selected = platform_compatible_asr_backend(engine or DEFAULT_TRANSCRIPTION_ENGINE)
    if selected == OPENVINO_PILOT_ENGINE and openvino_arc_supported():
        return OPENVINO_PILOT_ENGINE
    return STANDARD_TRANSCRIPTION_ENGINE


def _workspace_engine_preference(state: dict[str, object] | None) -> str:
    preferences = state.get("preferences") if isinstance(state, dict) else None
    engine = preferences.get("transcription_engine") if isinstance(preferences, dict) else None
    return _normalise_engine_choice(engine)


_UI_TOP_TABS = {"dashboard", "audios", "fiche", "reports", "settings"}
_UI_CONTEXT_MODES = {"transcript", "assistant", "discussions"}


def _workspace_ui_navigation(state: dict[str, object] | None) -> dict[str, str]:
    preferences = state.get("preferences") if isinstance(state, dict) else None
    raw_navigation = preferences.get("ui_navigation") if isinstance(preferences, dict) else None
    navigation = raw_navigation if isinstance(raw_navigation, dict) else {}
    active_tab = str(navigation.get("active_tab") or "dashboard")
    context_mode = str(navigation.get("context_mode") or "assistant")
    fiche_view = str(navigation.get("fiche_view") or "library")
    return {
        "active_tab": active_tab if active_tab in _UI_TOP_TABS else "dashboard",
        "context_mode": context_mode if context_mode in _UI_CONTEXT_MODES else "assistant",
        "fiche_view": "detail" if fiche_view == "detail" else "library",
        "fiche_item_id": str(navigation.get("fiche_item_id") or ""),
    }


def _ui_navigation_state_html(state: dict[str, object] | None = None) -> str:
    workspace = normalise_workspace(state or load_workspace())
    navigation = _workspace_ui_navigation(workspace)
    item = _workspace_item(navigation["fiche_item_id"], workspace)
    item_name = str(item.get("name") or "") if item else ""
    item_id = str(item.get("id") or "") if item else ""
    fiche_view = "detail" if navigation["fiche_view"] == "detail" and item else "library"
    return (
        "<span data-navigation-state='true' data-ready='true'"
        f" data-active-tab='{html.escape(navigation['active_tab'], quote=True)}'"
        f" data-fiche-view='{html.escape(fiche_view, quote=True)}'"
        f" data-fiche-item-id='{html.escape(item_id, quote=True)}'"
        f" data-fiche-item-name='{html.escape(item_name, quote=True)}'"
        f" data-context-mode='{html.escape(navigation['context_mode'], quote=True)}'></span>"
    )


def restore_ui_navigation_state_ui() -> str:
    return _ui_navigation_state_html(load_workspace())


def _persist_ui_navigation(**updates: str) -> dict[str, object]:
    def apply(state: dict[str, object]) -> None:
        preferences = state.setdefault("preferences", {})
        if not isinstance(preferences, dict):
            preferences = {}
            state["preferences"] = preferences
        raw_navigation = preferences.get("ui_navigation")
        navigation = dict(raw_navigation) if isinstance(raw_navigation, dict) else {}
        navigation.update({key: str(value) for key, value in updates.items()})
        preferences["ui_navigation"] = navigation

    return mutate_workspace(apply)


def _save_workspace_from_ui(state: dict[str, object]) -> dict[str, object]:
    """Évite qu'un ancien gr.State n'efface la navigation enregistrée entre-temps."""

    latest_navigation = _workspace_ui_navigation(load_workspace())
    current = normalise_workspace(state)
    preferences = current.setdefault("preferences", {})
    if not isinstance(preferences, dict):
        preferences = {}
        current["preferences"] = preferences
    preferences["ui_navigation"] = latest_navigation
    return save_workspace(current)


def persist_active_tab_ui(tab_id: str) -> None:
    selected = str(tab_id or "dashboard")
    if selected in _UI_TOP_TABS:
        _persist_ui_navigation(active_tab=selected)


def persist_active_tab_when_ready_ui(tab_id: str, ready: bool) -> None:
    if ready:
        persist_active_tab_ui(tab_id)


def restore_navigation_components_ui():
    """Restaure la vue exacte depuis le workspace à chaque chargement de page."""

    state = load_workspace()
    navigation = _workspace_ui_navigation(state)
    item = _workspace_item(navigation["fiche_item_id"], state)
    detail_visible = navigation["fiche_view"] == "detail" and item is not None
    context_label = {
        "transcript": "Transcription",
        "assistant": "Assistant IA",
        "discussions": "Discussions",
    }[navigation["context_mode"]]
    return (
        gr.Tabs(selected=navigation["active_tab"]),
        gr.update(visible=not detail_visible),
        gr.update(visible=not detail_visible),
        gr.update(visible=detail_visible),
        gr.update(visible=detail_visible),
        gr.Radio(value=context_label),
        gr.update(visible=context_label == "Assistant IA"),
        gr.update(visible=context_label == "Discussions"),
    )


def _read_openvino_status() -> dict[str, object]:
    """Lit l'état du pilote sans installation ni chargement du modèle."""

    if not openvino_arc_supported():
        return {
            "available": False,
            "model_ready": False,
            "device": "",
            "device_name": "",
            "model_path": "",
            "reason": "Le pilote Intel Arc est réservé à Windows.",
        }

    try:
        from .openvino_backend import get_openvino_status

        raw_status = get_openvino_status()
    except Exception as exc:
        return {
            "available": False,
            "model_ready": False,
            "device": "GPU",
            "device_name": "",
            "model_path": "",
            "reason": f"Composant pilote non installé ({exc}).",
        }

    def value(name: str, default: object = "") -> object:
        if isinstance(raw_status, dict):
            return raw_status.get(name, default)
        return getattr(raw_status, name, default)

    model_path = str(value("model_path") or "")
    explicit_model_ready = value("model_ready", None)
    model_ready = (
        bool(explicit_model_ready)
        if explicit_model_ready is not None
        else bool(model_path and Path(model_path).exists())
    )
    return {
        "available": bool(value("available", False)),
        "model_ready": model_ready,
        "device": str(value("device") or "GPU"),
        "device_name": str(value("device_name") or ""),
        "model_path": model_path,
        "reason": str(value("reason") or ""),
    }


def _pilot_unavailable_label(status: dict[str, object]) -> str:
    reason = str(status.get("reason") or "").casefold()
    if not bool(status.get("model_ready")) or any(
        marker in reason for marker in ("modèle", "model", "install", "introuvable", "missing")
    ):
        return "Pilote non installé"
    if any(marker in reason for marker in ("arc", "gpu", "device", "périphérique")):
        return "Intel Arc indisponible"
    return "Pilote indisponible"


def _engine_status_html(engine: str, profile: str = "Équilibré") -> str:
    selected = _normalise_engine_choice(engine)
    if not openvino_arc_supported():
        return """
          <section class="engine-status is-ready">
            <div class="engine-status-head"><strong>Faster-Whisper actif</strong><span class="engine-status-badge">Compatible avec ce système</span></div>
            <p>Le moteur local compatible avec cette plateforme est sélectionné. Le profil métier Voyance et le traitement hors ligne restent actifs.</p>
          </section>
        """
    status = _read_openvino_status()
    available = bool(status.get("available"))
    device_name = html.escape(str(status.get("device_name") or "Intel Arc"))
    reason = html.escape(
        str(status.get("reason") or "Le pilote local n'est pas prêt sur cet ordinateur.")
    )

    if selected == OPENVINO_PILOT_ENGINE and profile != "Équilibré":
        return f"""
          <section class="engine-status is-fallback">
            <div class="engine-status-head"><strong>Repli automatique actif</strong><span class="engine-status-badge">Profil incompatible</span></div>
            <p>Le pilote Intel Arc est réservé au profil Équilibré. Le profil {html.escape(profile or 'sélectionné')} utilisera Faster-Whisper, sans interrompre la file.</p>
          </section>
        """

    if selected == OPENVINO_PILOT_ENGINE and available:
        return f"""
          <section class="engine-status is-ready">
            <div class="engine-status-head"><strong>Pilote Intel Arc prêt</strong><span class="engine-status-badge">Profil Voyance actif</span></div>
            <p>{device_name} est détecté. Le vocabulaire métier guide les termes spécialisés et les dates de naissance. En cas d'anomalie, Faster-Whisper Équilibré reprendra automatiquement le traitement.</p>
          </section>
        """

    if selected == OPENVINO_PILOT_ENGINE:
        install_help = (
            " Fermez l'application puis lancez <code>INSTALLER_OPENVINO.bat</code> "
            "pour préparer explicitement ce pilote."
            if not bool(status.get("model_ready"))
            else ""
        )
        return f"""
          <section class="engine-status is-fallback">
            <div class="engine-status-head"><strong>Faster-Whisper prendra le relais</strong><span class="engine-status-badge">{_pilot_unavailable_label(status)}</span></div>
            <p>{reason}{install_help} Le profil métier Voyance reste actif pendant le repli. Aucun téléchargement n'est déclenché depuis cet écran.</p>
          </section>
        """

    option_note = (
        f"Le pilote Intel Arc est également prêt sur {device_name}."
        if available
        else f"{_pilot_unavailable_label(status)} : {reason}"
    )
    return f"""
      <section class="engine-status{' is-ready' if available else ''}">
        <div class="engine-status-head"><strong>Faster-Whisper Équilibré actif</strong><span class="engine-status-badge">Choix manuel</span></div>
        <p>{option_note} Le profil métier Voyance reste actif. Le moteur standard reste sélectionné sur cet espace jusqu'à ce que vous le changiez.</p>
      </section>
    """


def save_engine_preference_ui(engine: str, profile: str):
    selected = _normalise_engine_choice(engine)

    def apply(state: dict[str, object]) -> None:
        preferences = state.setdefault("preferences", {})
        if not isinstance(preferences, dict):
            preferences = {}
            state["preferences"] = preferences
        preferences["transcription_engine"] = selected

    state = mutate_workspace(apply)
    return state, _engine_status_html(selected, profile)


def restore_engine_preference_ui(
    profile: str = "Équilibré",
    language: str = "Français",
    requested_speakers: int = 2,
):
    """Recharge le moteur mémorisé à chaque ouverture de la page.

    Les valeurs initiales des composants Gradio sont figées au démarrage du
    serveur. Cette restauration explicite évite donc qu'un simple rafraîchissement
    visuel réaffiche le moteur standard alors que le pilote a bien été enregistré.
    """

    selected = _workspace_engine_preference(load_workspace())
    return (
        selected,
        _engine_status_html(selected, profile),
        _run_summary_html(profile, language, requested_speakers, selected),
    )


def _effective_engine_choice(engine: str, profile: str) -> str:
    selected = _normalise_engine_choice(engine)
    if (
        openvino_arc_supported()
        and selected == OPENVINO_PILOT_ENGINE
        and profile == "Équilibré"
    ):
        return selected
    return STANDARD_TRANSCRIPTION_ENGINE


def _run_summary_html(
    profile: str,
    language: str,
    speakers: int,
    engine: str = DEFAULT_TRANSCRIPTION_ENGINE,
) -> str:
    safe_profile = html.escape(profile or "Équilibré")
    safe_language = html.escape(language or "Français")
    speaker_count = max(1, int(speakers or 1))
    speaker_label = "interlocuteur" if speaker_count == 1 else "interlocuteurs"
    selected_engine = _normalise_engine_choice(engine)
    if (
        openvino_arc_supported()
        and selected_engine == OPENVINO_PILOT_ENGINE
        and profile == "Équilibré"
    ):
        engine_label = "Intel Arc · pilote avec repli automatique"
    elif openvino_arc_supported() and selected_engine == OPENVINO_PILOT_ENGINE:
        engine_label = "Faster-Whisper · repli automatique"
    else:
        engine_label = (
            "Faster-Whisper · choix manuel"
            if openvino_arc_supported()
            else "Faster-Whisper · moteur local compatible"
        )
    return f"""
      <section class="launch-summary">
        <span class="eyebrow">Réglages actifs</span>
        <strong>{safe_profile}</strong>
        <p>{safe_language} · {speaker_count} {speaker_label}</p>
        <p>{engine_label}</p>
        <small>Modifiables à tout moment depuis l'onglet Paramètres.</small>
      </section>
    """


def _item_turns(item: dict[str, object] | None) -> list[TranscriptTurn]:
    result = item.get("result") if isinstance(item, dict) else None
    if not isinstance(result, dict):
        return []
    return _turns_from_result_data(result)


def _speaker_choices(item: dict[str, object] | None) -> list[tuple[str, str]]:
    labels: dict[str, str] = {}
    for turn in _item_turns(item):
        labels.setdefault(turn.speaker_id, turn.speaker or turn.speaker_id)
    return [
        (f"{label} · {speaker_id.replace('SPEAKER_', 'Voix ')}", speaker_id)
        for speaker_id, label in labels.items()
    ]


def _role_assignment_ready(item: dict[str, object] | None) -> bool:
    """Les rôles ne sont éditables qu'une fois deux voix réellement disponibles."""

    return len(_speaker_choices(item)) >= 2


def _existing_role_map(item: dict[str, object] | None) -> dict[str, str]:
    return _speaker_role_labels(item)


def _assistant_role_proposal(item: dict[str, object] | None) -> dict[str, object]:
    turns = _item_turns(item)
    if not turns:
        return {
            "assignments": [],
            "roles": {},
            "client_speaker_id": None,
            "master_speaker_id": None,
            "confidence": 0.0,
            "needs_confirmation": True,
            "method": "indisponible",
        }
    existing = _existing_role_map(item)
    return suggest_speaker_roles(turns, existing or None)


def _confidence_label(value: object) -> str:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return "À confirmer"
    if confidence >= 0.78:
        return "Confiance élevée"
    if confidence >= 0.52:
        return "Confiance moyenne"
    return "Confiance faible"


def _speaker_display(item: dict[str, object] | None, speaker_id: object) -> str:
    target = str(speaker_id or "")
    for label, value in _speaker_choices(item):
        if value == target:
            return label.split(" · ", 1)[0]
    return target or "À confirmer"


def _role_voice_excerpt_html(
    item: dict[str, object] | None,
    speaker_id: object,
    role_label: str,
) -> str:
    target = str(speaker_id or "")
    if not _item_turns(item):
        return (
            f'<article class="role-voice-sample is-pending" aria-label="Extrait {html.escape(role_label)}">'
            "<small>Extrait de la voix</small>"
            "<p>Disponible après la séparation des voix.</p>"
            "</article>"
        )
    candidates = [
        turn
        for turn in _item_turns(item)
        if turn.speaker_id == target and str(turn.text or "").strip()
    ]
    if not candidates:
        return (
            f'<article class="role-voice-sample" aria-label="Extrait {html.escape(role_label)}">'
            "<small>Extrait de la voix</small>"
            "<p>Aucun extrait disponible pour cette voix.</p>"
            "</article>"
        )

    representative = max(candidates[:8], key=lambda turn: len(str(turn.text or "")))
    excerpt = " ".join(str(representative.text or "").split())
    if len(excerpt) > 180:
        excerpt = excerpt[:177].rstrip() + "…"
    return (
        f'<article class="role-voice-sample" aria-label="Extrait {html.escape(role_label)}">'
        "<small>Extrait de la voix</small>"
        f"<p>« {html.escape(excerpt)} »</p>"
        f"<span>À {html.escape(format_duration(float(representative.start or 0.0)))}</span>"
        "</article>"
    )


def _role_assignment_confirmed(
    item: dict[str, object] | None,
    proposal: dict[str, object] | None = None,
) -> bool:
    proposal = proposal or _assistant_role_proposal(item)
    existing = item.get("speaker_roles") if isinstance(item, dict) else None
    client_id = proposal.get("client_speaker_id")
    master_id = proposal.get("master_speaker_id")
    if not isinstance(existing, dict) or not client_id or not master_id:
        return False
    return all(
        isinstance(existing.get(str(speaker_id)), dict)
        and bool(existing[str(speaker_id)].get("confirmed"))
        for speaker_id in (client_id, master_id)
    )


def role_voice_excerpts_ui(
    item_id: str | None,
    client_speaker_id: str | None,
    master_speaker_id: str | None,
    workspace_state: dict[str, object] | None,
):
    state = normalise_workspace(workspace_state or load_workspace())
    item = _workspace_item(item_id, state)
    return (
        _role_voice_excerpt_html(item, client_speaker_id, "Client"),
        _role_voice_excerpt_html(item, master_speaker_id, "Master"),
    )


def _role_summary_html(
    item: dict[str, object] | None,
    proposal: dict[str, object] | None = None,
) -> str:
    proposal = proposal or _assistant_role_proposal(item)
    client_id = proposal.get("client_speaker_id")
    master_id = proposal.get("master_speaker_id")
    ready = _role_assignment_ready(item)
    confirmed = _role_assignment_confirmed(item, proposal)
    needs_confirmation = bool(proposal.get("needs_confirmation")) and not confirmed
    if not ready:
        status = "Attribution en préparation"
        status_class = " is-pending"
        client_label = master_label = "Disponible après séparation"
        button_label = "Disponible après séparation"
        button_state = ' disabled aria-disabled="true"'
    elif confirmed:
        status = "Attribution confirmée"
        status_class = ""
        client_label = _speaker_display(item, client_id)
        master_label = _speaker_display(item, master_id)
        button_label = "Modifier l'attribution"
        button_state = ""
    elif needs_confirmation:
        status = "Attribution à vérifier"
        status_class = " needs-confirmation"
        client_label = _speaker_display(item, client_id)
        master_label = _speaker_display(item, master_id)
        button_label = "Modifier l'attribution"
        button_state = ""
    else:
        status = "Attribution proposée"
        status_class = ""
        client_label = _speaker_display(item, client_id)
        master_label = _speaker_display(item, master_id)
        button_label = "Modifier l'attribution"
        button_state = ""
    return f"""
      <section class="fiche-role-bar" aria-label="Rôles des interlocuteurs">
        <div class="fiche-role-heading">
          <h3>Rôles des interlocuteurs</h3>
          <span class="fiche-role-confidence{status_class}" role="status" aria-live="polite">{html.escape(status)}</span>
        </div>
        <div class="fiche-role-people">
          <span class="fiche-role-person"><small>Client</small><strong>{html.escape(client_label)}</strong></span>
          <span class="fiche-role-person"><small>Master</small><strong>{html.escape(master_label)}</strong></span>
        </div>
        <button type="button" class="role-edit-button" data-role-editor-toggle{button_state}
                aria-expanded="false" aria-controls="assistant-role-editor">{html.escape(button_label, quote=False)}</button>
      </section>
    """


def _assistant_empty_html(item: dict[str, object] | None) -> str:
    if not _item_turns(item):
        return (
            "<section class='assistant-empty'><strong>Analyse disponible après la transcription</strong>"
            "<span>L'assistant utilisera uniquement les passages réellement transcrits et leurs timecodes.</span></section>"
        )
    return (
        "<section class='assistant-empty'><strong>Interrogez la consultation</strong>"
        "<span>Demandez une synthèse, recherchez un thème ou vérifiez précisément ce qui a été dit.</span>"
        "<span>Exemples : « De quoi parle principalement la consultation ? », "
        "« Est-ce que ça parle d'amour ? » ou « Le Master a-t-il annoncé un retour ? »</span>"
        "<span>Les réponses portent sur ce qui est dit pendant la consultation, pas sur la vérité extérieure.</span></section>"
    )


def _assistant_result_presentation(
    result: dict[str, object],
) -> dict[str, str | bool]:
    """Décrit le rendu d'un résultat sans casser le schéma historique.

    Les versions initiales de l'assistant exposaient uniquement un ``verdict``
    de vérification. Le moteur plus récent distingue les synthèses et les
    recherches de thème avec ``intent`` et, selon la version persistée, un
    ``verdict`` ou un ``status`` dédié. Cette couche accepte les deux formes :
    les anciens libellés restent strictement inchangés dans le rendu.
    """

    intent = str(
        result.get("intent")
        or result.get("query_intent")
        or result.get("question_intent")
        or ""
    ).strip().casefold().replace("-", "_").replace(" ", "_")
    outcome = str(
        result.get("verdict")
        or result.get("status")
        or "Non retrouvé"
    ).strip()
    folded_outcome = outcome.casefold()

    summary_intents = {
        "open_summary",
        "summary",
        "synthesis",
        "synthese",
        "resume",
    }
    topic_intents = {
        "topic",
        "topic_presence",
        "theme",
        "theme_presence",
        "presence",
    }
    general_intents = {
        "consultation_qa",
        "general_question",
        "question_answering",
        "qa",
    }
    summary_outcomes = {"synthèse", "synthese", "résumé", "resume"}
    topic_presentations = {
        "thème principal": (
            "theme-primary",
            "Thème principal",
            "Analyse thématique locale · Thème central",
        ),
        "theme principal": (
            "theme-primary",
            "Thème principal",
            "Analyse thématique locale · Thème central",
        ),
        "thème présent": (
            "theme-present",
            "Thème présent",
            "Analyse thématique locale · Présence confirmée",
        ),
        "theme present": (
            "theme-present",
            "Thème présent",
            "Analyse thématique locale · Présence confirmée",
        ),
        "mention ponctuelle": (
            "mention",
            "Mention ponctuelle",
            "Analyse thématique locale · Présence limitée",
        ),
    }

    if intent in summary_intents or folded_outcome in summary_outcomes:
        return {
            "is_new": True,
            "kind": "summary",
            "verdict": "Synthèse",
            "class": "summary",
            "label": "Synthèse",
            "confidence": "Synthèse locale · Sources citées",
            "result_label": "Synthèse de la consultation",
            "points_label": "Points clés",
            "nuance_label": "Portée de la synthèse",
            "evidence_label": "Passages sources",
            "region_label": "Résultat de la synthèse",
        }

    topic_presentation = topic_presentations.get(folded_outcome)
    if topic_presentation is not None:
        css_class, label, confidence = topic_presentation
        return {
            "is_new": True,
            "kind": "topic",
            "verdict": label,
            "class": css_class,
            "label": label,
            "confidence": confidence,
            "result_label": "Réponse thématique",
            "points_label": "Ce que montrent les passages",
            "nuance_label": "Lecture du thème",
            "evidence_label": "Passages sources",
            "region_label": "Résultat de l'analyse thématique",
        }

    # Une ancienne ou future persistance peut contenir l'intention mais omettre
    # le statut spécialisé. On conserve alors le verdict lisible, tout en
    # présentant correctement la nature de la réponse.
    if intent in topic_intents:
        return {
            "is_new": True,
            "kind": "topic",
            "verdict": outcome,
            "class": "theme-present",
            "label": outcome,
            "confidence": "Analyse thématique locale · Sources citées",
            "result_label": "Réponse thématique",
            "points_label": "Ce que montrent les passages",
            "nuance_label": "Lecture du thème",
            "evidence_label": "Passages sources",
            "region_label": "Résultat de l'analyse thématique",
        }

    if intent in general_intents:
        label = {
            "Réponse": "Réponse fondée sur l'audio",
            "Partiel": "Réponse partielle",
            "Non retrouvé": "Éléments insuffisants",
            "À clarifier": "À clarifier",
        }.get(outcome, outcome)
        return {
            "is_new": True,
            "kind": "question",
            "verdict": outcome,
            "class": "answered" if outcome == "Réponse" else {
                "Partiel": "partial",
                "Non retrouvé": "not-found",
                "À clarifier": "clarify",
            }.get(outcome, "answered"),
            "label": label,
            "confidence": "Assistant local · Sources citées",
            "result_label": "Réponse à votre question",
            "points_label": "Éléments de réponse",
            "nuance_label": "Limites de l'analyse",
            "evidence_label": "Passages sources",
            "region_label": "Réponse de l'assistant sur la consultation",
        }

    legacy_class = {
        "Confirmé": "confirmed",
        "Partiel": "partial",
        "Contredit": "contradicted",
        "Non retrouvé": "not-found",
        "À clarifier": "clarify",
    }.get(outcome, "not-found")
    legacy_label = {
        "Confirmé": "Appuyé par l'audio",
        "Partiel": "Appuyé avec nuance",
        "Contredit": "Passage contraire explicite",
        "Non retrouvé": "Éléments insuffisants",
        "À clarifier": "À clarifier",
    }.get(outcome, outcome)
    return {
        "is_new": False,
        "kind": "verification",
        "verdict": outcome,
        "class": legacy_class,
        "label": legacy_label,
        "confidence": "",
        "result_label": "Réponse détaillée",
        "points_label": "Ce que montrent les passages",
        "nuance_label": "Lecture prudente",
        "evidence_label": "Passages retenus",
        "region_label": "Résultat de la vérification",
    }


def _assistant_result_html(
    result: dict[str, object] | None,
    item: dict[str, object] | None = None,
    *,
    live: bool = True,
    show_header: bool = True,
) -> str:
    if not isinstance(result, dict):
        return _assistant_empty_html(item)

    presentation = _assistant_result_presentation(result)
    verdict = str(presentation["verdict"])
    verdict_class = str(presentation["class"])
    verdict_label = str(presentation["label"])
    is_new_result = bool(presentation["is_new"])
    stored_answer = str(result.get("answer") or result.get("explanation") or "")
    backend = str(result.get("backend") or "")
    if is_new_result:
        confidence = str(presentation["confidence"])
    elif verdict == "À clarifier":
        confidence = "Référent ambigu · Choix nécessaire"
    elif backend.startswith("déterministe"):
        confidence = "Analyse lexicale locale · Relecture conseillée"
    else:
        confidence = _confidence_label(result.get("confidence"))
    coverage = result.get("coverage") if isinstance(result.get("coverage"), dict) else {}
    turns_analyzed = int(coverage.get("turns_analyzed") or 0)
    duration = float(coverage.get("duration_seconds") or 0.0)
    mapping_confirmed = bool(coverage.get("role_mapping_confirmed"))
    provisional = " · Attribution à confirmer" if not mapping_confirmed else ""

    clarification_html = ""
    raw_options = result.get("clarification_options")
    option_buttons: list[str] = []
    if isinstance(raw_options, list):
        for option in raw_options[:3]:
            if not isinstance(option, dict):
                continue
            label = str(option.get("label") or "").strip()
            refined_question = str(option.get("question") or "").strip()
            if not label or not refined_question:
                continue
            option_buttons.append(
                "<button type='button' data-assistant-action='clarify' "
                f"data-question='{html.escape(refined_question, quote=True)}'>"
                f"{html.escape(label)}</button>"
            )
    if option_buttons:
        clarification_html = (
            "<section class='assistant-clarify'><span>Quelle relation souhaitez-vous vérifier ?</span>"
            "<div class='assistant-clarify-options'>"
            + "".join(option_buttons)
            + "</div></section>"
        )

    citations_html: list[str] = []
    raw_citations = result.get("citations")
    if isinstance(raw_citations, list):
        for citation_index, citation in enumerate(raw_citations[:4]):
            if not isinstance(citation, dict):
                continue
            quote = str(
                citation.get("excerpt")
                or citation.get("quote")
                or citation.get("text")
                or ""
            ).strip()
            if not quote:
                continue
            full_quote = str(citation.get("text") or quote).strip()
            try:
                start = float(citation.get("start_seconds", citation.get("start", 0.0)) or 0.0)
                end = float(citation.get("end_seconds", citation.get("end", start)) or start)
            except (TypeError, ValueError):
                start = end = 0.0
            role = str(citation.get("role") or "Interlocuteur")
            speaker_id = str(citation.get("speaker_id") or "")
            start_label = format_timestamp(start)
            end_label = format_timestamp(end)
            clip_seconds = max(0, int(round(end - start)))
            clip_hours, clip_remainder = divmod(clip_seconds, 3600)
            clip_minutes, clip_seconds_part = divmod(clip_remainder, 60)
            clip_duration_label = (
                f"{clip_hours:d}:{clip_minutes:02d}:{clip_seconds_part:02d}"
                if clip_hours
                else f"{clip_minutes:d}:{clip_seconds_part:02d}"
            )
            safe_quote_attr = html.escape(quote, quote=True)
            full_quote_html = ""
            if full_quote and full_quote != quote:
                full_quote_html = (
                    "<details class='evidence-full'><summary>Voir le passage complet</summary>"
                    f"<p>{html.escape(full_quote)}</p></details>"
                )
            open_attribute = " open" if live and citation_index == 0 else ""
            citations_html.append(
                f"""
                <details class="evidence-citation" data-evidence-item{open_attribute}>
                  <summary>
                    <span class="evidence-summary-copy">
                      <span class="evidence-meta"><span>{html.escape(start_label)}–{html.escape(end_label)}</span><span>{html.escape(role)}</span></span>
                      <span class="evidence-preview">«{html.escape(quote)}»</span>
                    </span>
                    <span class="evidence-toggle" aria-hidden="true"><span class="when-closed">Voir</span><span class="when-open">Réduire</span></span>
                  </summary>
                  <div class="evidence-citation-body">
                    <q>{html.escape(quote)}</q>
                    {full_quote_html}
                    <div class="excerpt-player" data-excerpt-player data-start="{start:.3f}" data-end="{end:.3f}" role="group" aria-label="Extrait du {html.escape(role, quote=True)} de {html.escape(start_label, quote=True)} à {html.escape(end_label, quote=True)}">
                      <div class="excerpt-player-head">
                        <span>Extrait audio</span>
                        <span data-excerpt-status>Prêt</span>
                      </div>
                      <div class="excerpt-player-controls">
                        <button type="button" data-assistant-action="toggle-excerpt">Lire</button>
                        <div class="excerpt-player-timeline">
                          <input type="range" min="0" max="100" step="0.1" value="0" data-excerpt-range aria-label="Position dans l'extrait" aria-valuetext="0:00 sur {html.escape(clip_duration_label, quote=True)}">
                          <div class="excerpt-player-times" aria-hidden="true">
                            <time data-excerpt-current>0:00</time>
                            <time data-excerpt-duration>{html.escape(clip_duration_label)}</time>
                          </div>
                        </div>
                        <button type="button" data-assistant-action="stop-excerpt" aria-label="Arrêter l'extrait et revenir au début">Stop</button>
                      </div>
                    </div>
                    <div class="evidence-actions">
                      <button type="button" data-assistant-action="discuss" data-start="{start:.3f}" data-end="{end:.3f}" data-speaker-id="{html.escape(speaker_id, quote=True)}" data-role="{html.escape(role, quote=True)}" data-quote="{safe_quote_attr}">Ouvrir une discussion</button>
                    </div>
                  </div>
                </details>
                """
            )

    evidence_count = len(citations_html)
    evidence_label = str(presentation["evidence_label"])
    evidence = (
        "<section class='assistant-citations'>"
        f"<strong>{html.escape(evidence_label)} · {evidence_count}</strong>"
        + (
            "".join(citations_html)
            if citations_html
            else "<p class='assistant-no-evidence'>Aucun passage suffisamment probant n'a été retenu.</p>"
        )
        + "</section>"
    )
    raw_points = result.get("analysis_points")
    if not isinstance(raw_points, (list, tuple)):
        raw_points = result.get("key_points")

    def point_text(point: object) -> str:
        if isinstance(point, dict):
            return str(
                point.get("text")
                or point.get("label")
                or point.get("summary")
                or ""
            ).strip()
        return str(point or "").strip()

    if isinstance(raw_points, (list, tuple)) and any(
        point_text(point) for point in raw_points
    ):
        answer = stored_answer
        analysis_points = tuple(
            point_text(point) for point in raw_points if point_text(point)
        )
        nuance = str(result.get("nuance") or "").strip()
    elif is_new_result:
        # Les synthèses et recherches de thème ne doivent jamais repasser par
        # ``build_answer_sections`` : cette fonction historique interprète
        # toute entrée comme une affirmation binaire et recréerait précisément
        # les faux « Contredit » que le nouveau routage évite.
        answer = stored_answer
        analysis_points = ()
        nuance = str(result.get("nuance") or "").strip()
    else:
        generated_answer, analysis_points, nuance = build_answer_sections(
            verdict,
            str(result.get("claim") or result.get("question") or ""),
            str(result.get("target_role") or "") or None,
            raw_citations if isinstance(raw_citations, list) else [],
            clarification_answer=(stored_answer if verdict == "À clarifier" else ""),
            role_mapping_confirmed=mapping_confirmed,
        )
        generic_prefixes = (
            "L'affirmation est appuyée",
            "Une partie de l'affirmation est retrouvée",
            "Un passage pertinent",
            "Aucun passage suffisamment proche",
            "La formulation peut désigner plusieurs",
        )
        answer = (
            generated_answer
            if not stored_answer or stored_answer.startswith(generic_prefixes)
            else stored_answer
        )
    points_html = ""
    if analysis_points:
        points_label = str(presentation["points_label"])
        points_html = (
            "<section class='assistant-analysis-points'>"
            f"<strong>{html.escape(points_label)}</strong><ol>"
            + "".join(f"<li>{html.escape(point)}</li>" for point in analysis_points)
            + "</ol></section>"
        )
    nuance_label = str(presentation["nuance_label"])
    nuance_html = (
        f"<section class='assistant-analysis-nuance'><strong>{html.escape(nuance_label)}</strong>"
        f"<p>{html.escape(nuance)}</p></section>"
        if nuance
        else ""
    )
    header_html = (
        "<div class='assistant-answer-head'>"
        f"<span class='verdict-chip verdict-{verdict_class}'>{html.escape(verdict_label)}</span>"
        f"<span class='assistant-confidence'>{html.escape(confidence)}{html.escape(provisional)}</span>"
        "</div>"
        if show_header
        else ""
    )
    live_attribute = ' aria-live="polite"' if live else ""
    result_label = str(presentation["result_label"])
    region_label = str(presentation["region_label"])
    return f"""
      <article class="assistant-answer" role="region" aria-label="{html.escape(region_label, quote=True)}"{live_attribute} tabindex="-1">
        {header_html}
        <span class="assistant-result-label">{html.escape(result_label)}</span>
        <div class="assistant-response-copy">
          <p>{html.escape(answer or "Aucun passage suffisamment probant n'a été identifié.")}</p>
        </div>
        {points_html}
        {nuance_html}
        {clarification_html}
        <p class="assistant-coverage">Périmètre analysé : {turns_analyzed} tours · {html.escape(format_duration(duration))}</p>
        {evidence}
      </article>
    """


def _assistant_history_timestamp(value: object) -> tuple[str, str]:
    raw = str(value or "").strip()
    if not raw:
        return "", ""
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return raw, raw
    return raw, parsed.strftime("%d/%m/%Y · %H:%M")


def _assistant_history_html(
    item: dict[str, object] | None,
    *,
    include_latest: bool = False,
) -> str:
    queries = item.get("assistant_queries") if isinstance(item, dict) else None
    if not isinstance(queries, list) or not queries:
        return ""
    completed = [
        query
        for query in queries
        if isinstance(query, dict) and isinstance(query.get("result"), dict)
    ]
    if not include_latest and completed:
        completed = completed[:-1]
    if not completed:
        return ""
    rows: list[str] = []
    for index, query in enumerate(reversed(completed)):
        question = str(query.get("question") or "")
        result = query.get("result") if isinstance(query.get("result"), dict) else {}
        rendered_result = dict(result)
        rendered_result.setdefault("question", question)
        presentation = _assistant_result_presentation(rendered_result)
        verdict = str(presentation["verdict"])
        verdict_class = str(presentation["class"])
        preview = str(result.get("answer") or result.get("explanation") or "").strip()
        confidence = (
            str(presentation["confidence"])
            if bool(presentation["is_new"])
            else _confidence_label(result.get("confidence"))
        )
        citations = result.get("citations") if isinstance(result.get("citations"), list) else []
        created_raw, created_label = _assistant_history_timestamp(query.get("created_at"))
        query_id = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "-",
            str(query.get("id") or f"history-{index}"),
        ).strip("-") or f"history-{index}"
        detail_id = f"assistant-history-detail-{query_id}-{index}"
        date_html = (
            f"<time datetime='{html.escape(created_raw, quote=True)}'>{html.escape(created_label)}</time>"
            if created_label
            else ""
        )
        meta_parts = [
            part
            for part in (
                date_html,
                html.escape(confidence),
                f"{len(citations)} passage{'s' if len(citations) != 1 else ''}",
            )
            if part
        ]
        rows.append(
            "<details class='assistant-history-item'>"
            f"<summary aria-controls='{html.escape(detail_id, quote=True)}'>"
            f"<span class='verdict-chip verdict-{verdict_class}'>{html.escape(verdict)}</span>"
            "<span class='assistant-history-copy'>"
            f"<strong>{html.escape(question or 'Analyse sans question enregistrée')}</strong>"
            f"<span>{html.escape(preview or 'Détail indisponible pour cette ancienne analyse.')}</span>"
            "</span>"
            f"<span class='assistant-history-meta'>{' · '.join(meta_parts)}</span>"
            "<span class='assistant-history-chevron' aria-hidden='true'></span>"
            "</summary>"
            f"<div class='assistant-history-detail' id='{html.escape(detail_id, quote=True)}'>"
            f"{_assistant_result_html(rendered_result, item, live=False, show_header=False)}"
            "</div></details>"
        )
    if not rows:
        return ""
    return (
        "<section class='assistant-history' aria-label='Historique des analyses'>"
        "<details class='assistant-history-overview' open>"
        f"<summary>Analyses précédentes · {len(completed)}</summary>"
        "<div class='assistant-history-list'>"
        + "".join(rows)
        + "</div></details></section>"
    )


def _assistant_latest_result(item: dict[str, object] | None) -> dict[str, object] | None:
    queries = item.get("assistant_queries") if isinstance(item, dict) else None
    if not isinstance(queries, list):
        return None
    for query in reversed(queries):
        if isinstance(query, dict) and isinstance(query.get("result"), dict):
            return query["result"]
    return None


def _assistant_panel_payload(item: dict[str, object] | None):
    choices = _speaker_choices(item)
    proposal = _assistant_role_proposal(item)
    latest = _assistant_latest_result(item)
    client_speaker_id = proposal.get("client_speaker_id")
    master_speaker_id = proposal.get("master_speaker_id")
    return (
        _role_summary_html(item, proposal),
        gr.Dropdown(
            choices=choices,
            value=client_speaker_id,
            interactive=bool(choices),
        ),
        gr.Dropdown(
            choices=choices,
            value=master_speaker_id,
            interactive=bool(choices),
        ),
        _role_voice_excerpt_html(item, client_speaker_id, "Client"),
        _role_voice_excerpt_html(item, master_speaker_id, "Master"),
        _assistant_result_html(latest, item) + _assistant_history_html(item),
        gr.Button(interactive=bool(_item_turns(item))),
    )


def _fiche_payload(
    item_id: str | None,
    state: dict[str, object] | None,
):
    item = _workspace_item(item_id, state)
    comment_choices = _comment_choices(item)
    selected_comment = comment_choices[0][1] if comment_choices else None
    enabled = _commenting_enabled(item)
    return (
        _fiche_header_html(item),
        _item_transcript(item),
        _comments_feed_html(item),
        gr.Dropdown(
            choices=comment_choices,
            value=selected_comment,
            interactive=bool(comment_choices),
        ),
        gr.Button(interactive=enabled),
        gr.Button(
            value=_comment_action_label(item, selected_comment),
            interactive=bool(comment_choices),
        ),
    )


def sync_workspace_ui(
    audio_files: object,
    audio_names: dict[str, str] | None,
    workspace_state: dict[str, object] | None,
):
    paths = _normalise_audio_paths(audio_files)
    # Le composant File peut émettre `.change()` après une mise à jour serveur.
    # Son gr.State d'entrée est alors potentiellement en retard sur le worker
    # (notamment juste après la fin d'une transcription). L'état durable reste
    # donc la seule source d'autorité ; le paramètre est conservé pour ne pas
    # modifier le contrat public du callback Gradio.
    _ = workspace_state
    state = load_workspace()
    items = _workspace_items(state)
    library_order = _workspace_order(state)
    queue_order: list[str] = []
    timestamp = datetime.now().isoformat(timespec="seconds")

    for source in paths:
        item_id, stored_path = import_audio(source)
        previous = items.get(item_id, {})
        current_name = _display_audio_name(source, audio_names)
        proposed_name = str((audio_names or {}).get(_audio_key(source)) or "")
        previous_name = str(previous.get("name") or "")
        if previous_name and (not proposed_name or proposed_name == source.name):
            display_name = previous_name
        else:
            display_name = proposed_name or previous_name or current_name
        item = {
            "id": item_id,
            "path": str(stored_path),
            "source_path": str(source),
            "original_name": str(previous.get("original_name") or source.name),
            "name": display_name,
            "status": str(previous.get("status") or "En attente"),
            "progress": float(previous.get("progress") or 0.0),
            "processing_time": str(previous.get("processing_time") or "—"),
            "processing_seconds": float(previous.get("processing_seconds") or 0.0),
            "processing_base_seconds": float(
                previous.get("processing_base_seconds")
                if previous.get("processing_base_seconds") is not None
                else previous.get("processing_seconds") or 0.0
            ),
            "processing_started_at": str(
                previous.get("processing_started_at") or ""
            ),
            "partial": str(previous.get("partial") or ""),
            "stage": str(previous.get("stage") or ""),
            "settings": previous.get("settings") if isinstance(previous.get("settings"), dict) else {},
            "run_id": str(previous.get("run_id") or ""),
            "checkpoint_position": float(previous.get("checkpoint_position") or 0.0),
            "resume_count": int(previous.get("resume_count") or 0),
            "heartbeat": str(previous.get("heartbeat") or ""),
            "result": previous.get("result") if isinstance(previous.get("result"), dict) else None,
            "files": previous.get("files") if isinstance(previous.get("files"), list) else [],
            "comments": previous.get("comments") if isinstance(previous.get("comments"), list) else [],
            "speaker_roles": (
                previous.get("speaker_roles")
                if isinstance(previous.get("speaker_roles"), dict)
                else {}
            ),
            "assistant_queries": (
                previous.get("assistant_queries")
                if isinstance(previous.get("assistant_queries"), list)
                else []
            ),
            "error": str(previous.get("error") or ""),
            "created_at": str(previous.get("created_at") or timestamp),
            "updated_at": str(previous.get("updated_at") or timestamp),
        }
        items[item_id] = item
        if item_id not in library_order:
            library_order.append(item_id)
        completed = (
            str(item.get("status") or "") == "Terminé"
            and isinstance(item.get("result"), dict)
        )
        if not completed and item_id not in queue_order:
            queue_order.append(item_id)

    for active_id in JOB_MANAGER.active_item_ids():
        if active_id in items and active_id not in queue_order:
            queue_order.append(active_id)
    state["order"] = library_order
    state["queue_order"] = queue_order
    state["items"] = items
    state = _save_workspace_from_ui(state)
    queue_order = _workspace_queue_ids(state)
    selected = queue_order[0] if queue_order else (library_order[0] if library_order else None)
    payload = _fiche_payload(selected, state)
    return (
        state,
        _workspace_queue_rows(state),
        gr.Dropdown(
            choices=_workspace_choices(state),
            value=selected,
            interactive=bool(library_order),
        ),
        *payload,
        _dashboard_html(state),
    )


def sync_workspace_names_ui(
    audio_files: object,
    audio_names: dict[str, str] | None,
    workspace_state: dict[str, object] | None,
    selected_item: str | None,
):
    state = normalise_workspace(workspace_state or load_workspace())
    items = _workspace_items(state)
    order = _workspace_order(state)
    proposed_names = audio_names or {}
    for item_id in order:
        item = items.get(item_id)
        if not isinstance(item, dict):
            continue
        managed_path = Path(str(item.get("path") or ""))
        source_path = str(item.get("source_path") or "")
        display_name = proposed_names.get(str(managed_path)) or proposed_names.get(source_path)
        if not display_name:
            continue
        item["name"] = str(display_name)
        item["updated_at"] = datetime.now().isoformat(timespec="seconds")
        if managed_path.is_file():
            try:
                managed_path = rename_library_audio(
                    item_id,
                    managed_path,
                    str(display_name),
                )
            except (FileExistsError, OSError, ValueError):
                pass
        item["path"] = str(managed_path)
    state = _save_workspace_from_ui(state)
    queue_assets = _workspace_queue_assets(state)
    visible_paths = [path for _, path in queue_assets]
    visible_names = {
        str(path): str(_workspace_items(state)[item_id].get("name") or path.name)
        for item_id, path in queue_assets
    }
    selected = selected_item if selected_item in order else (order[0] if order else None)
    return (
        [str(path) for path in visible_paths],
        visible_names,
        state,
        _workspace_queue_rows(state),
        gr.Dropdown(
            choices=_workspace_choices(state),
            value=selected,
            interactive=bool(order),
        ),
        *_fiche_payload(selected, state),
        _dashboard_html(state),
    )


def load_fiche_ui(item_id: str | None, workspace_state: dict[str, object] | None):
    state = normalise_workspace(workspace_state or load_workspace())
    item = _workspace_item(item_id, state)
    if item_id and item:
        _persist_ui_navigation(fiche_view="detail", fiche_item_id=str(item_id))
    header, transcript, comments, comment_selector, add_button, resolve_button = _fiche_payload(
        item_id,
        state,
    )
    result = item.get("result") if item and isinstance(item.get("result"), dict) else None
    if result:
        advanced_rows = turns_to_rows(_turns_from_result_data(result))
        metadata = _metadata_from_result_data(result)
        files = [str(path) for path in item.get("files", [])]
    else:
        advanced_rows, metadata, files = [], {}, []
    return (
        header,
        str(item.get("path")) if item and Path(str(item.get("path") or "")).is_file() else None,
        transcript,
        comments,
        comment_selector,
        add_button,
        resolve_button,
        advanced_rows,
        metadata,
        files,
    )


def select_queue_fiche_ui(
    workspace_state: dict[str, object] | None,
    evt: gr.SelectData,
):
    index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    order = _workspace_queue_ids(workspace_state)
    try:
        item_id = order[int(index)]
    except (IndexError, TypeError, ValueError):
        item_id = order[0] if order else None
    return (
        gr.Dropdown(
            choices=_workspace_choices(workspace_state),
            value=item_id,
            interactive=bool(order),
        ),
        *load_fiche_ui(item_id, workspace_state),
    )


def select_library_fiche_ui(
    workspace_state: dict[str, object] | None,
    sort_by: str | None,
    evt: gr.SelectData,
):
    index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    ids = _workspace_library_ids(workspace_state, sort_by)
    try:
        item_id = ids[int(index)]
    except (IndexError, TypeError, ValueError):
        item_id = ids[0] if ids else None
    return (
        gr.Dropdown(
            choices=_workspace_choices(workspace_state),
            value=item_id,
            interactive=bool(ids),
        ),
        *load_fiche_ui(item_id, workspace_state),
    )


def select_library_fiche_by_index_ui(
    index_value: str | int | float | None,
    workspace_state: dict[str, object] | None,
    sort_by: str | None,
):
    """Open a library item after the UI has disambiguated click from double-click."""
    ids = _workspace_library_ids(workspace_state, sort_by)
    try:
        item_id = ids[int(float(str(index_value).strip()))]
    except (IndexError, TypeError, ValueError):
        item_id = ids[0] if ids else None
    return (
        gr.Dropdown(
            choices=_workspace_choices(workspace_state),
            value=item_id,
            interactive=bool(ids),
        ),
        *load_fiche_ui(item_id, workspace_state),
    )


def open_library_fiche_ui(
    workspace_state: dict[str, object] | None,
    sort_by: str | None,
    evt: gr.SelectData,
):
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=True),
        *select_library_fiche_ui(workspace_state, sort_by, evt),
    )


def open_library_fiche_by_index_ui(
    index_value: str | int | float | None,
    workspace_state: dict[str, object] | None,
    sort_by: str | None,
):
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=True),
        *select_library_fiche_by_index_ui(index_value, workspace_state, sort_by),
    )


def close_fiche_ui():
    return (
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def persist_fiche_library_ui() -> None:
    _persist_ui_navigation(fiche_view="library", fiche_item_id="")


def capture_selected_passage(text: str | None, evt: gr.SelectData) -> str:
    selected = evt.value if isinstance(evt.value, str) else ""
    if not selected and isinstance(evt.index, (list, tuple)) and len(evt.index) == 2:
        try:
            selected = (text or "")[int(evt.index[0]) : int(evt.index[1])]
        except (TypeError, ValueError):
            selected = ""
    return selected.strip()


def _normalise_anchor_payload(raw: object, passage: str) -> dict[str, object]:
    if isinstance(raw, str) and raw.strip():
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            raw = {}
    anchor = dict(raw) if isinstance(raw, dict) else {}
    anchor["quote"] = passage
    anchor["source"] = str(anchor.get("source") or "manual-text")
    for key in ("start_seconds", "end_seconds"):
        try:
            value = float(anchor.get(key))
        except (TypeError, ValueError):
            value = None
        anchor[key] = value if value is not None and value >= 0 else None
    if anchor["start_seconds"] is None:
        anchor["end_seconds"] = None
    elif anchor["end_seconds"] is not None:
        anchor["end_seconds"] = max(anchor["start_seconds"], anchor["end_seconds"])
    return anchor


def _comment_by_id(
    item: dict[str, object] | None,
    comment_id: str | None,
) -> dict[str, object] | None:
    comments = item.get("comments") if isinstance(item, dict) else None
    if not isinstance(comments, list):
        return None
    for comment in comments:
        if isinstance(comment, dict) and str(comment.get("id")) == str(comment_id):
            return comment
    return None


def _thread_context_html(
    item: dict[str, object] | None,
    comment_id: str | None,
) -> str:
    comment = _comment_by_id(item, comment_id)
    if not comment:
        return (
            "<section class='reply-context is-empty'>"
            "<span class='selection-kicker'>Réponse</span>"
            "<p>Choisissez « Répondre » sur une discussion pour poursuivre l’échange.</p>"
            "</section>"
        )
    passage = html.escape(str(comment.get("passage") or ""))
    author = html.escape(str(comment.get("author") or "Moi"))
    anchor = comment.get("anchor") if isinstance(comment.get("anchor"), dict) else {}
    return (
        "<section class='reply-context is-ready'>"
        f"<div><span class='selection-kicker'>Répondre à {author}</span>"
        f"<span class='selection-time'>{html.escape(_anchor_label(anchor))}</span></div>"
        f"<q>{passage}</q>"
        "</section>"
    )


def _save_root_comment(
    item_id: str,
    author: str | None,
    passage: str,
    body: str,
    anchor: dict[str, object],
    editing_id: str | None,
) -> tuple[dict[str, object], str]:
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    comment_id = str(editing_id or uuid.uuid4().hex)

    def save(current: dict[str, object]) -> None:
        current_item = _workspace_item(item_id, current)
        if not current_item:
            raise gr.Error("Cette fiche audio n'est plus disponible.")
        comments = current_item.get("comments")
        if not isinstance(comments, list):
            comments = []
            current_item["comments"] = comments
        if editing_id:
            comment = _comment_by_id(current_item, editing_id)
            if not comment:
                raise gr.Error("Cette discussion n'est plus disponible.")
            comment.update(
                {
                    "author": (author or "Moi").strip() or "Moi",
                    "passage": passage,
                    "body": body,
                    "anchor": anchor,
                    "updated_at": now,
                }
            )
        else:
            comments.append(
                {
                    "id": comment_id,
                    "author": (author or "Moi").strip() or "Moi",
                    "passage": passage,
                    "body": body,
                    "status": "À discuter",
                    "anchor": anchor,
                    "replies": [],
                    "created_at": now,
                    "updated_at": now,
                    "resolved_at": None,
                }
            )
        current_item["updated_at"] = now

    return mutate_workspace(save), comment_id


def add_comment_ui(
    item_id: str | None,
    author: str | None,
    passage: str | None,
    body: str | None,
    workspace_state: dict[str, object] | None,
):
    state = load_workspace()
    if item_id not in _workspace_items(state):
        state = normalise_workspace(workspace_state or state)
    item = _workspace_item(item_id, state)
    if not item:
        raise gr.Error("Ouvrez d'abord une fiche audio.")
    if not _commenting_enabled(item):
        raise gr.Error("La transcription doit avoir commencé avant de créer une discussion.")

    clean_passage = (passage or "").strip()
    clean_body = (body or "").strip()
    if not clean_passage:
        raise gr.Error("Sélectionnez ou copiez le passage concerné.")
    if not clean_body:
        raise gr.Error("Écrivez le commentaire à partager.")

    state, comment_id = _save_root_comment(
        str(item_id),
        author,
        clean_passage,
        clean_body,
        _normalise_anchor_payload({}, clean_passage),
        None,
    )
    item = _workspace_item(item_id, state)
    return (
        state,
        _comments_feed_html(item),
        gr.Dropdown(
            choices=_comment_choices(item),
            value=comment_id,
            interactive=True,
        ),
        _fiche_header_html(item),
        "",
        "",
        gr.Button(value="Marquer comme résolu", interactive=True),
        _dashboard_html(state),
    )


def save_comment_ui(
    item_id: str | None,
    author: str | None,
    passage: str | None,
    body: str | None,
    anchor_payload: object,
    draft_item_id: str | None,
    editing_id: str | None,
    workspace_state: dict[str, object] | None,
):
    state = load_workspace()
    if item_id not in _workspace_items(state):
        state = normalise_workspace(workspace_state or state)
    item = _workspace_item(item_id, state)
    if not item:
        raise gr.Error("Ouvrez d'abord une fiche audio.")
    if not _commenting_enabled(item):
        raise gr.Error("La transcription doit avoir commencé avant de créer une discussion.")
    if draft_item_id and str(draft_item_id) != str(item_id):
        raise gr.Error("Ce brouillon appartient à une autre fiche. Sélectionnez à nouveau le passage.")

    clean_passage = (passage or "").strip()
    clean_body = (body or "").strip()
    if not clean_passage:
        raise gr.Error("Sélectionnez le passage concerné dans la transcription.")
    if not clean_body:
        raise gr.Error("Écrivez votre message avant de l'ajouter à la discussion.")
    if len(clean_passage) > 4000 or len(clean_body) > 8000:
        raise gr.Error("Ce commentaire est trop long. Réduisez l'extrait ou le message.")

    anchor = _normalise_anchor_payload(anchor_payload, clean_passage)
    state, saved_id = _save_root_comment(
        str(item_id),
        author,
        clean_passage,
        clean_body,
        anchor,
        (editing_id or "").strip() or None,
    )
    item = _workspace_item(item_id, state)
    return (
        state,
        _comments_feed_html(item),
        gr.Dropdown(
            choices=_comment_choices(item),
            value=saved_id,
            interactive=True,
        ),
        _fiche_header_html(item),
        "",
        "",
        gr.Button(value=_comment_action_label(item, saved_id), interactive=True),
        _dashboard_html(state),
        "",
        "",
        "",
        _selection_context_html(),
        gr.Button(value="Créer la discussion", interactive=_commenting_enabled(item)),
        "",
        _thread_context_html(item, None),
        gr.Button(value="Répondre", interactive=False),
    )


def reset_comment_composer_ui(item_id: str | None):
    state = load_workspace()
    item = _workspace_item(item_id, state)
    choices = _comment_choices(item)
    selected = choices[0][1] if choices else None
    return (
        "",
        "",
        "",
        "",
        "",
        _selection_context_html(),
        gr.Button(value="Créer la discussion", interactive=_commenting_enabled(item)),
        gr.Dropdown(choices=choices, value=selected, interactive=bool(choices)),
        "",
        _thread_context_html(item, None),
        "",
        gr.Button(value="Répondre", interactive=False),
    )


def load_comment_action_ui(
    item_id: str | None,
    comment_id: str | None,
    workspace_state: dict[str, object] | None,
):
    item = _workspace_item(item_id, workspace_state)
    return gr.Button(
        value=_comment_action_label(item, comment_id),
        interactive=bool(item and comment_id),
    )


def toggle_comment_ui(
    item_id: str | None,
    comment_id: str | None,
    workspace_state: dict[str, object] | None,
):
    state = load_workspace()
    if item_id not in _workspace_items(state):
        state = normalise_workspace(workspace_state or state)
    item = _workspace_item(item_id, state)
    if not item or not comment_id:
        raise gr.Error("Choisissez un commentaire.")
    def toggle(current: dict[str, object]) -> None:
        current_item = _workspace_item(item_id, current)
        comments = current_item.get("comments") if current_item else None
        if isinstance(comments, list):
            for comment in comments:
                if isinstance(comment, dict) and str(comment.get("id")) == str(comment_id):
                    now = datetime.now().astimezone().isoformat(timespec="seconds")
                    resolved = comment.get("status") != "Résolu"
                    comment["status"] = "Résolu" if resolved else "À discuter"
                    comment["resolved_at"] = now if resolved else None
                    comment["updated_at"] = now
                    current_item["updated_at"] = now
                    return
        raise gr.Error("Ce commentaire n'est plus disponible.")

    state = mutate_workspace(toggle)
    item = _workspace_item(item_id, state)
    return (
        state,
        _comments_feed_html(item),
        _fiche_header_html(item),
        gr.Button(
            value=_comment_action_label(item, comment_id),
            interactive=True,
        ),
        _dashboard_html(state),
    )


def add_reply_ui(
    item_id: str | None,
    comment_id: str | None,
    author: str | None,
    body: str | None,
    workspace_state: dict[str, object] | None,
):
    state = load_workspace()
    if item_id not in _workspace_items(state):
        state = normalise_workspace(workspace_state or state)
    item = _workspace_item(item_id, state)
    comment = _comment_by_id(item, comment_id)
    clean_body = (body or "").strip()
    if not item or not comment:
        raise gr.Error("Choisissez la discussion à laquelle répondre.")
    if not clean_body:
        raise gr.Error("Écrivez votre réponse avant de l'envoyer.")
    if len(clean_body) > 8000:
        raise gr.Error("Cette réponse est trop longue.")

    reply_id = uuid.uuid4().hex
    now = datetime.now().astimezone().isoformat(timespec="seconds")

    def append_reply(current: dict[str, object]) -> None:
        current_item = _workspace_item(item_id, current)
        current_comment = _comment_by_id(current_item, comment_id)
        if not current_comment:
            raise gr.Error("Cette discussion n'est plus disponible.")
        replies = current_comment.get("replies")
        if not isinstance(replies, list):
            replies = []
            current_comment["replies"] = replies
        replies.append(
            {
                "id": reply_id,
                "author": (author or "Moi").strip() or "Moi",
                "body": clean_body,
                "created_at": now,
                "updated_at": now,
            }
        )
        current_comment["updated_at"] = now
        current_item["updated_at"] = now

    state = mutate_workspace(append_reply)
    item = _workspace_item(item_id, state)
    return (
        state,
        _comments_feed_html(item),
        _fiche_header_html(item),
        _dashboard_html(state),
        _thread_context_html(item, comment_id),
        "",
        gr.Button(value="Répondre", interactive=True),
    )


def _delete_comment(
    item_id: str,
    comment_id: str,
) -> dict[str, object]:
    def remove(current: dict[str, object]) -> None:
        current_item = _workspace_item(item_id, current)
        comments = current_item.get("comments") if current_item else None
        if not isinstance(comments, list):
            raise gr.Error("Cette discussion n'est plus disponible.")
        before = len(comments)
        current_item["comments"] = [
            comment
            for comment in comments
            if not isinstance(comment, dict) or str(comment.get("id")) != str(comment_id)
        ]
        if len(current_item["comments"]) == before:
            raise gr.Error("Cette discussion n'est plus disponible.")
        current_item["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")

    return mutate_workspace(remove)


def handle_comment_feed_action_ui(
    item_id: str | None,
    workspace_state: dict[str, object] | None,
    passage: str | None,
    body: str | None,
    anchor_payload: object,
    draft_item_id: str | None,
    editing_id: str | None,
    reply_target_id: str | None,
    reply_body: str | None,
    evt: gr.EventData,
):
    action = str(getattr(evt, "action", "") or "")
    comment_id = str(getattr(evt, "comment_id", "") or "")
    state = load_workspace()
    if item_id not in _workspace_items(state):
        state = normalise_workspace(workspace_state or state)
    item = _workspace_item(item_id, state)
    comment = _comment_by_id(item, comment_id)
    if not item or not comment:
        raise gr.Error("Cette discussion n'est plus disponible.")

    next_passage = passage or ""
    next_body = body or ""
    next_anchor = str(anchor_payload or "")
    next_draft_item = str(draft_item_id or "")
    next_editing = str(editing_id or "")
    next_reply_target = str(reply_target_id or "")
    next_reply_body = str(reply_body or "")
    if next_reply_target != comment_id:
        next_reply_body = ""
    selection_html = _selection_context_html(
        next_passage,
        _normalise_anchor_payload(next_anchor, next_passage),
        editing=bool(next_editing),
    )
    add_button = gr.Button(value="Créer la discussion", interactive=_commenting_enabled(item))

    if action != "reply":
        next_reply_target = ""
        next_reply_body = ""

    if action == "resolve":
        toggled = toggle_comment_ui(item_id, comment_id, state)
        state = toggled[0]
        item = _workspace_item(item_id, state)
    elif action == "delete":
        deleted_id = comment_id
        state = _delete_comment(str(item_id), comment_id)
        item = _workspace_item(item_id, state)
        choices = _comment_choices(item)
        comment_id = str(choices[0][1]) if choices else ""
        next_reply_body = ""
        if next_editing == deleted_id:
            next_passage = ""
            next_body = ""
            next_anchor = ""
            next_draft_item = ""
            next_editing = ""
            selection_html = _selection_context_html()
            add_button = gr.Button(
                value="Créer la discussion",
                interactive=_commenting_enabled(item),
            )
    elif action == "edit":
        next_passage = str(comment.get("passage") or "")
        next_body = str(comment.get("body") or "")
        next_anchor = json.dumps(comment.get("anchor") or {}, ensure_ascii=False)
        next_draft_item = str(item_id or "")
        next_editing = comment_id
        selection_html = _selection_context_html(
            next_passage,
            comment.get("anchor"),
            editing=True,
        )
        add_button = gr.Button(value="Enregistrer les modifications", interactive=True)
    elif action == "reply":
        next_reply_target = comment_id

    selected_comment = _comment_by_id(item, comment_id)
    choices = _comment_choices(item)
    selected_id = comment_id if selected_comment else (str(choices[0][1]) if choices else None)
    return (
        state,
        _comments_feed_html(item),
        gr.Dropdown(choices=choices, value=selected_id, interactive=bool(choices)),
        _fiche_header_html(item),
        gr.Button(
            value=_comment_action_label(item, selected_id),
            interactive=bool(selected_id),
        ),
        _dashboard_html(state),
        next_passage,
        next_body,
        next_anchor,
        next_draft_item,
        next_editing,
        selection_html,
        add_button,
        next_reply_target,
        _thread_context_html(item, next_reply_target),
        next_reply_body,
        gr.Button(value="Répondre", interactive=bool(next_reply_target)),
    )


def switch_fiche_context_ui(mode: str | None):
    selected = str(mode or "Assistant IA")
    normalised = {
        "Transcription": "transcript",
        "Assistant IA": "assistant",
        "Discussions": "discussions",
    }.get(selected, "assistant")
    _persist_ui_navigation(context_mode=normalised)
    return (
        gr.update(visible=selected == "Assistant IA"),
        gr.update(visible=selected == "Discussions"),
    )


def switch_fiche_context_when_ready_ui(mode: str | None, ready: bool):
    selected = str(mode or "Assistant IA")
    if ready:
        return switch_fiche_context_ui(selected)
    return (
        gr.update(visible=selected == "Assistant IA"),
        gr.update(visible=selected == "Discussions"),
    )


def load_assistant_ui(
    item_id: str | None,
    workspace_state: dict[str, object] | None,
):
    state = normalise_workspace(workspace_state or load_workspace())
    item = _workspace_item(item_id, state)
    return _assistant_panel_payload(item)


def _role_editor_controls_payload(
    item: dict[str, object] | None,
    *,
    visible: bool,
):
    choices = _speaker_choices(item)
    ready = len(choices) >= 2
    proposal = _assistant_role_proposal(item)
    client_speaker_id = proposal.get("client_speaker_id")
    master_speaker_id = proposal.get("master_speaker_id")
    return (
        gr.update(visible=bool(visible and ready)),
        gr.Dropdown(
            choices=choices,
            value=client_speaker_id,
            interactive=ready,
        ),
        gr.Dropdown(
            choices=choices,
            value=master_speaker_id,
            interactive=ready,
        ),
        _role_voice_excerpt_html(item, client_speaker_id, "Client"),
        _role_voice_excerpt_html(item, master_speaker_id, "Master"),
    )


def open_role_editor_ui(
    item_id: str | None,
    workspace_state: dict[str, object] | None,
):
    state = normalise_workspace(workspace_state or load_workspace())
    return _role_editor_controls_payload(_workspace_item(item_id, state), visible=True)


def cancel_role_editor_ui(
    item_id: str | None,
    workspace_state: dict[str, object] | None,
):
    state = normalise_workspace(workspace_state or load_workspace())
    return _role_editor_controls_payload(_workspace_item(item_id, state), visible=False)


def close_role_editor_ui():
    return gr.update(visible=False)


def clear_assistant_ui(
    item_id: str | None,
    workspace_state: dict[str, object] | None,
):
    state = normalise_workspace(workspace_state or load_workspace())
    item = _workspace_item(item_id, state)
    return (
        "",
        _assistant_empty_html(item) + _assistant_history_html(item, include_latest=True),
    )


def auto_detect_roles_ui(
    item_id: str | None,
    workspace_state: dict[str, object] | None,
):
    state = normalise_workspace(workspace_state or load_workspace())
    item = _workspace_item(item_id, state)
    if not item or not _item_turns(item):
        raise gr.Error("La détection des rôles sera disponible après la transcription.")
    proposal = suggest_speaker_roles(_item_turns(item))
    choices = _speaker_choices(item)
    client_speaker_id = proposal.get("client_speaker_id")
    master_speaker_id = proposal.get("master_speaker_id")
    return (
        gr.Dropdown(choices=choices, value=client_speaker_id, interactive=True),
        gr.Dropdown(choices=choices, value=master_speaker_id, interactive=True),
        _role_voice_excerpt_html(item, client_speaker_id, "Client"),
        _role_voice_excerpt_html(item, master_speaker_id, "Master"),
    )


def _persist_roles(
    item_id: str,
    client_speaker_id: str,
    master_speaker_id: str,
    *,
    source: str,
    confirmed: bool,
) -> dict[str, object]:
    if not client_speaker_id or not master_speaker_id:
        raise gr.Error("Attribuez une voix au Client et une voix au Master.")
    if client_speaker_id == master_speaker_id:
        raise gr.Error("Le Client et le Master doivent correspondre à deux voix différentes.")
    now = datetime.now().astimezone().isoformat(timespec="seconds")

    def save(current: dict[str, object]) -> None:
        item = _workspace_item(item_id, current)
        if not item:
            raise gr.Error("Cette fiche audio n'est plus disponible.")
        known_ids = {turn.speaker_id for turn in _item_turns(item)}
        if client_speaker_id not in known_ids or master_speaker_id not in known_ids:
            raise gr.Error("Une des voix choisies n'appartient plus à cette fiche.")
        existing = item.get("speaker_roles") if isinstance(item.get("speaker_roles"), dict) else {}
        roles = dict(existing)
        roles[client_speaker_id] = {
            **(roles.get(client_speaker_id) if isinstance(roles.get(client_speaker_id), dict) else {}),
            "speaker_id": client_speaker_id,
            "role": "Client",
            "confidence": 1.0 if confirmed else None,
            "source": source,
            "confirmed": confirmed,
            "updated_at": now,
        }
        roles[master_speaker_id] = {
            **(roles.get(master_speaker_id) if isinstance(roles.get(master_speaker_id), dict) else {}),
            "speaker_id": master_speaker_id,
            "role": "Master",
            "confidence": 1.0 if confirmed else None,
            "source": source,
            "confirmed": confirmed,
            "updated_at": now,
        }
        for speaker_id, payload in roles.items():
            if speaker_id not in {client_speaker_id, master_speaker_id} and isinstance(payload, dict):
                payload["role"] = "À confirmer"
                payload["confirmed"] = False
        item["speaker_roles"] = roles
        item["updated_at"] = now

    return mutate_workspace(save)


def confirm_roles_ui(
    item_id: str | None,
    client_speaker_id: str | None,
    master_speaker_id: str | None,
    workspace_state: dict[str, object] | None,
):
    state = load_workspace()
    if item_id not in _workspace_items(state):
        state = normalise_workspace(workspace_state or state)
    if not item_id:
        raise gr.Error("Ouvrez d'abord une fiche audio.")
    state = _persist_roles(
        str(item_id),
        str(client_speaker_id or ""),
        str(master_speaker_id or ""),
        source="manual",
        confirmed=True,
    )
    item = _workspace_item(item_id, state)
    return state, _role_summary_html(item), _item_transcript(item)


def swap_roles_ui(
    item_id: str | None,
    client_speaker_id: str | None,
    master_speaker_id: str | None,
    workspace_state: dict[str, object] | None,
):
    next_client = str(master_speaker_id or "")
    next_master = str(client_speaker_id or "")
    if not item_id:
        raise gr.Error("Ouvrez d'abord une fiche audio.")
    state = normalise_workspace(workspace_state or load_workspace())
    item = _workspace_item(item_id, state)
    if not item:
        raise gr.Error("Cette fiche audio n'est plus disponible.")
    choices = _speaker_choices(item)
    if len(choices) < 2 or not next_client or not next_master:
        raise gr.Error("L'attribution sera disponible après la séparation des deux voix.")
    return (
        gr.Dropdown(choices=choices, value=next_client, interactive=True),
        gr.Dropdown(choices=choices, value=next_master, interactive=True),
        _role_voice_excerpt_html(item, next_client, "Client"),
        _role_voice_excerpt_html(item, next_master, "Master"),
    )


def run_assistant_verification_ui(
    item_id: str | None,
    question: str | None,
    client_speaker_id: str | None,
    master_speaker_id: str | None,
    workspace_state: dict[str, object] | None,
):
    clean_question = str(question or "").strip()
    if not clean_question:
        raise gr.Error("Écrivez une question sur la consultation.")
    if len(clean_question) > 4000:
        raise gr.Error("Cette question est trop longue. Reformulez-la plus brièvement.")

    state = load_workspace()
    if item_id not in _workspace_items(state):
        state = normalise_workspace(workspace_state or state)
    item = _workspace_item(item_id, state)
    turns = _item_turns(item)
    if not item or not turns:
        raise gr.Error("La transcription doit être terminée avant de lancer cette vérification.")
    if not item_id:
        raise gr.Error("Ouvrez d'abord une fiche audio.")

    client_id = str(client_speaker_id or "")
    master_id = str(master_speaker_id or "")
    state = _persist_roles(
        str(item_id),
        client_id,
        master_id,
        source="manual",
        confirmed=True,
    )
    item = _workspace_item(item_id, state)
    turns = _item_turns(item)
    role_payload = {
        "client_speaker_id": client_id,
        "master_speaker_id": master_id,
    }
    audio_observations = None
    if question_needs_audio_analysis(clean_question):
        audio_path = str(item.get("path") or item.get("source_path") or "")
        audio_observations = analyze_audio_dynamics(audio_path, turns)
    result = verify_claim(
        clean_question,
        turns,
        role_payload,
        reasoner=get_default_reasoner(),
        audio_observations=audio_observations,
    )
    if audio_observations is not None:
        result["audio_observations"] = audio_observations
    result["answer"] = str(result.get("answer") or result.get("explanation") or "")
    coverage = result.get("coverage") if isinstance(result.get("coverage"), dict) else {}
    coverage["role_mapping_confirmed"] = True
    result["coverage"] = coverage
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    query = {
        "id": f"query-{uuid.uuid4().hex}",
        "question": clean_question,
        "status": "completed",
        "result": result,
        "created_at": now,
        "updated_at": now,
    }

    def save_query(current: dict[str, object]) -> None:
        current_item = _workspace_item(str(item_id), current)
        if not current_item:
            raise gr.Error("Cette fiche audio n'est plus disponible.")
        history = current_item.get("assistant_queries")
        if not isinstance(history, list):
            history = []
            current_item["assistant_queries"] = history
        history.append(query)
        current_item["updated_at"] = now

    state = mutate_workspace(save_query)
    item = _workspace_item(item_id, state)
    return (
        state,
        _role_summary_html(item),
        _assistant_result_html(result, item) + _assistant_history_html(item),
        _item_transcript(item),
    )


def prepare_discussion_from_assistant_ui(
    item_id: str | None,
    evt: gr.EventData,
):
    if str(getattr(evt, "action", "") or "") != "discuss":
        return (
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )
    quote = str(getattr(evt, "quote", "") or "").strip()
    if not quote or not item_id:
        raise gr.Error("Ce passage n'est plus disponible.")
    try:
        start = max(0.0, float(getattr(evt, "start_seconds", 0.0) or 0.0))
        end = max(start, float(getattr(evt, "end_seconds", start) or start))
    except (TypeError, ValueError):
        start = end = 0.0
    anchor = {
        "start_seconds": start,
        "end_seconds": end,
        "quote": quote,
        "source": "assistant-citation",
        "speaker_id": str(getattr(evt, "speaker_id", "") or ""),
        "speaker": str(getattr(evt, "role", "") or ""),
        "turn_ids": [],
    }
    return (
        quote,
        json.dumps(anchor, ensure_ascii=False),
        _selection_context_html(quote, anchor),
        str(item_id),
        "",
        "",
        gr.Button(value="Créer la discussion", interactive=True),
    )


def _prepare_transcription_workspace(
    sources: list[Path],
    audio_names: dict[str, str] | None,
    workspace_state: dict[str, object] | None,
) -> tuple[dict[str, object], list[str], bool]:
    persistent = workspace_state is not None
    state = normalise_workspace(workspace_state or blank_workspace())
    items = _workspace_items(state)
    library_order = _workspace_order(state)
    queue_order: list[str] = []
    timestamp = datetime.now().isoformat(timespec="seconds")
    for index, source in enumerate(sources):
        if persistent:
            item_id, stored_path = import_audio(source)
        else:
            item_id, stored_path = f"session-{index}-{uuid.uuid4().hex[:8]}", source
        previous = items.get(item_id, {})
        item = {
            "id": item_id,
            "path": str(stored_path),
            "source_path": str(source),
            "original_name": str(previous.get("original_name") or source.name),
            "name": _display_audio_name(source, audio_names),
            "status": str(previous.get("status") or "En attente"),
            "progress": float(previous.get("progress") or 0.0),
            "processing_time": str(previous.get("processing_time") or "—"),
            "processing_seconds": float(previous.get("processing_seconds") or 0.0),
            "processing_base_seconds": float(
                previous.get("processing_base_seconds")
                if previous.get("processing_base_seconds") is not None
                else previous.get("processing_seconds") or 0.0
            ),
            "processing_started_at": str(
                previous.get("processing_started_at") or ""
            ),
            "partial": str(previous.get("partial") or ""),
            "stage": str(previous.get("stage") or ""),
            "settings": previous.get("settings") if isinstance(previous.get("settings"), dict) else {},
            "run_id": str(previous.get("run_id") or ""),
            "checkpoint_position": float(previous.get("checkpoint_position") or 0.0),
            "resume_count": int(previous.get("resume_count") or 0),
            "heartbeat": str(previous.get("heartbeat") or ""),
            "result": previous.get("result") if isinstance(previous.get("result"), dict) else None,
            "files": previous.get("files") if isinstance(previous.get("files"), list) else [],
            "comments": previous.get("comments") if isinstance(previous.get("comments"), list) else [],
            "speaker_roles": (
                previous.get("speaker_roles")
                if isinstance(previous.get("speaker_roles"), dict)
                else {}
            ),
            "assistant_queries": (
                previous.get("assistant_queries")
                if isinstance(previous.get("assistant_queries"), list)
                else []
            ),
            "error": "",
            "created_at": str(previous.get("created_at") or timestamp),
            "updated_at": timestamp,
        }
        items[item_id] = item
        if item_id not in library_order:
            library_order.append(item_id)
        completed = (
            str(item.get("status") or "") == "Terminé"
            and isinstance(item.get("result"), dict)
        )
        if not completed and item_id not in queue_order:
            queue_order.append(item_id)

    state["items"] = items
    state["order"] = library_order
    state["queue_order"] = queue_order
    if persistent:
        state = _save_workspace_from_ui(state)
        queue_order = _workspace_queue_ids(state)
    return state, queue_order, persistent


def _update_transcription_workspace(
    state: dict[str, object],
    item_id: str,
    persistent: bool,
    **updates: object,
) -> dict[str, object]:
    if persistent:
        try:
            return update_workspace_item(item_id, **updates)
        except KeyError:
            pass
    current = load_workspace() if persistent else normalise_workspace(state)
    item = _workspace_item(item_id, current)
    if item is None:
        current = normalise_workspace(state)
        item = _workspace_item(item_id, current)
    if item is not None:
        item.update(updates)
        item["updated_at"] = datetime.now().isoformat(timespec="seconds")
    return _save_workspace_from_ui(current) if persistent else current


def _workspace_live_outputs(
    state: dict[str, object],
    item_id: str | None,
):
    return (
        state,
        gr.Dropdown(
            choices=_workspace_choices(state),
            value=item_id,
            interactive=bool(_workspace_order(state)),
        ),
        *_fiche_payload(item_id, state),
        _dashboard_html(state),
    )


def _batch_state_from_workspace(state: dict[str, object]) -> dict[str, object]:
    batch: dict[str, object] = {
        "order": [],
        "items": {},
        "batch_zip": state.get("batch_zip"),
    }
    order = batch["order"]
    batch_items = batch["items"]
    assert isinstance(order, list) and isinstance(batch_items, dict)
    items = _workspace_items(state)
    for index, item_id in enumerate(_workspace_order(state), start=1):
        item = items.get(item_id, {})
        result = item.get("result")
        if not isinstance(result, dict):
            continue
        label = f"{index:02d} · {item.get('name') or 'Audio'}"
        order.append(label)
        batch_items[label] = {
            "result": result,
            "files": [str(path) for path in item.get("files", [])],
        }
    return batch


def _workspace_status_markdown(state: dict[str, object]) -> str:
    items = _workspace_items(state)
    order = _workspace_order(state)
    active_ids = JOB_MANAGER.active_item_ids()
    active_id = next((item_id for item_id in order if item_id in active_ids), None)
    if active_id:
        item = items.get(active_id, {})
        name = str(item.get("name") or "Audio")
        stage = str(item.get("stage") or "Transcription en cours")
        position = float(item.get("checkpoint_position") or 0.0)
        checkpoint_note = (
            f" Dernier point sûr : **{format_duration(position)}**."
            if position > 0
            else ""
        )
        return (
            f"### Traitement en arrière-plan — {name}\n\n"
            f"{stage}.{checkpoint_note} Vous pouvez rafraîchir ou fermer cet onglet : "
            "le traitement local continue."
        )
    queue_ids = _workspace_queue_ids(state)
    resumable = [
        item_id
        for item_id in queue_ids
        if str(items.get(item_id, {}).get("status") or "") == "À reprendre"
    ]
    if resumable:
        count = len(resumable)
        noun = "audio" if count == 1 else "audios"
        verb = "possède" if count == 1 else "possèdent"
        return (
            "### Reprise disponible\n\n"
            f"**{count} {noun}** {verb} un checkpoint local. "
            "Relancez la file pour reprendre au dernier passage enregistré."
        )
    if queue_ids:
        count = len(queue_ids)
        noun = "audio" if count == 1 else "audios"
        failed = sum(
            1
            for item_id in queue_ids
            if str(items.get(item_id, {}).get("status") or "") == "Échec"
        )
        if failed:
            verb = "est" if count == 1 else "sont"
            return (
                "### Relance disponible\n\n"
                f"**{count} {noun}** {verb} dans la file, dont **{failed}** à réessayer."
            )
        verb = "attend" if count == 1 else "attendent"
        possessive = "son" if count == 1 else "leur"
        return (
            "### File prête\n\n"
            f"**{count} {noun}** {verb} {possessive} traitement dans l'ordre affiché."
        )
    completed = sum(
        1 for item_id in order if items.get(item_id, {}).get("status") == "Terminé"
    )
    if completed:
        noun = "transcription" if completed == 1 else "transcriptions"
        adjective = "disponible" if completed == 1 else "disponibles"
        return (
            "### Aucun audio en attente\n\n"
            f"**{completed} {noun}** {adjective} dans l'onglet Fiches."
        )
    if order:
        return (
            "### File vide\n\n"
            "Les enregistrements conservés restent disponibles dans l'onglet Fiches."
        )
    return "### Prêt\n\nLes modèles sont installés. Ajoutez vos audios pour commencer."


def restore_workspace_ui():
    JOB_MANAGER.recover_orphaned()
    state = load_workspace()
    items = _workspace_items(state)
    library_changed = False
    for item_id in _workspace_order(state):
        item = items.get(item_id)
        if not isinstance(item, dict):
            continue
        managed_path = Path(str(item.get("path") or ""))
        display_name = str(item.get("name") or managed_path.name)
        if not managed_path.is_file() or managed_path.name == display_name:
            continue
        try:
            renamed_path = rename_library_audio(item_id, managed_path, display_name)
        except (FileExistsError, OSError, ValueError):
            continue
        item["path"] = str(renamed_path)
        library_changed = library_changed or renamed_path != managed_path
    if library_changed:
        state = _save_workspace_from_ui(state)
        items = _workspace_items(state)
    available = _workspace_queue_assets(state)
    paths = [path for _, path in available]
    names = {
        str(path): str(items[item_id].get("name") or path.name)
        for item_id, path in available
    }
    order = _workspace_order(state)
    active_ids = JOB_MANAGER.active_item_ids()
    stored_item_id = _workspace_ui_navigation(state)["fiche_item_id"]
    selected = stored_item_id if stored_item_id in order else next(
        (
            item_id
            for item_id in order
            if item_id in active_ids
            or str(items.get(item_id, {}).get("status") or "") == "À reprendre"
        ),
        order[0] if order else None,
    )
    selected_item = _workspace_item(selected, state)
    payload = _fiche_payload(selected, state)

    rename_choices = _rename_choices(paths, names)
    rename_value = str(paths[0]) if paths else None
    rename_stem = _rename_stem(names[rename_value]) if rename_value else ""

    batch = _batch_state_from_workspace(state)
    batch_order = batch.get("order") if isinstance(batch.get("order"), list) else []
    result_label = batch_order[0] if batch_order else None
    if selected_item and isinstance(selected_item.get("result"), dict):
        result = selected_item["result"]
        advanced_rows = turns_to_rows(_turns_from_result_data(result))
        metadata = _metadata_from_result_data(result)
        files = [str(path) for path in selected_item.get("files", [])]
        legacy_transcript = _item_transcript(selected_item)
    else:
        advanced_rows, metadata, files, legacy_transcript = [], {}, [], ""

    status = _workspace_status_markdown(state)
    return (
        [str(path) for path in paths],
        str(paths[0]) if paths else None,
        _workspace_queue_rows(state),
        _queue_hint(paths, "File restaurée." if paths else ""),
        names,
        gr.Dropdown(
            choices=rename_choices,
            value=rename_value,
            interactive=bool(paths),
        ),
        gr.Textbox(value=rename_stem, interactive=bool(paths)),
        gr.Button(interactive=bool(paths)),
        state,
        gr.Dropdown(
            choices=_workspace_choices(state),
            value=selected,
            interactive=bool(_workspace_order(state)),
        ),
        *payload,
        gr.Dropdown(
            choices=batch_order,
            value=result_label,
            interactive=bool(batch_order),
        ),
        advanced_rows,
        legacy_transcript,
        files,
        batch,
        metadata,
        status,
        _dashboard_html(state),
        gr.Button(
            value="Lancer la file d'attente",
            interactive=bool(paths),
        ),
    )


_WORKSPACE_POLL_VOLATILE_ITEM_KEYS = frozenset(
    {
        "progress",
        "processing_time",
        "processing_seconds",
        "partial",
        "stage",
        "checkpoint_position",
        "heartbeat",
        "updated_at",
    }
)


def _workspace_poll_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _workspace_poll_stable_state(state: dict[str, object]) -> dict[str, object]:
    stable_state = copy.deepcopy(state)
    items = stable_state.get("items")
    if isinstance(items, dict):
        for item in items.values():
            if not isinstance(item, dict):
                continue
            for key in _WORKSPACE_POLL_VOLATILE_ITEM_KEYS:
                item.pop(key, None)
    return stable_state


def _workspace_poll_revision_parts(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"full": str(value)}
    if not isinstance(decoded, dict) or decoded.get("version") != 2:
        return {"full": str(value)}
    return {
        str(key): str(part)
        for key, part in decoded.items()
        if isinstance(key, str) and isinstance(part, (str, int))
    }


def poll_workspace_ui(
    selected_item: str | None,
    sort_by: str | None,
    last_revision: str | None = None,
    selected_comment: str | None = None,
    reply_target: str | None = None,
):
    """Reconnecte l'interface à l'état durable sans dépendre de la requête de lancement."""

    state = load_workspace()
    order = _workspace_order(state)
    if selected_item not in order:
        active_ids = JOB_MANAGER.active_item_ids()
        selected_item = next(
            (item_id for item_id in order if item_id in active_ids),
            order[0] if order else None,
        )
    item = _workspace_item(selected_item, state)
    batch = _batch_state_from_workspace(state)
    batch_order = batch.get("order") if isinstance(batch.get("order"), list) else []
    result_label = None
    if selected_item in order and item and isinstance(item.get("result"), dict):
        result_label = f"{order.index(selected_item) + 1:02d} · {item.get('name') or 'Audio'}"
    if result_label not in batch_order:
        result_label = batch_order[-1] if batch_order else None
    if item and isinstance(item.get("result"), dict):
        result = item["result"]
        advanced_rows = turns_to_rows(_turns_from_result_data(result))
        metadata = _metadata_from_result_data(result)
        files = [str(path) for path in item.get("files", [])]
        legacy_transcript = _item_transcript(item)
    else:
        advanced_rows, metadata, files = [], {}, []
        legacy_transcript = _item_transcript(item)
    comment_choices = _comment_choices(item)
    comment_ids = {str(value) for _, value in comment_choices}
    selected_comment_is_valid = str(selected_comment or "") in comment_ids
    selected_comment = (
        str(selected_comment)
        if selected_comment_is_valid
        else (str(comment_choices[0][1]) if comment_choices else None)
    )
    reply_target_is_valid = str(reply_target or "") in comment_ids
    reply_target = str(reply_target) if reply_target_is_valid else ""
    queue_rows = _workspace_queue_rows(state)
    library_rows = _workspace_library_rows(state, sort_by)
    fiche_header = _fiche_header_html(item)
    fiche_transcript = _item_transcript(item)
    workspace_status = _workspace_status_markdown(state)
    dashboard = _dashboard_html(state)
    assistant_payload = _assistant_panel_payload(item)
    poll_context = {
        "selected_item": selected_item,
        "sort_by": sort_by,
        "selected_comment": selected_comment,
        "reply_target": reply_target,
    }
    revision_parts = {
        "version": 2,
        "full": _workspace_poll_digest({"state": state, "context": poll_context}),
        "stable": _workspace_poll_digest(
            {
                "state": _workspace_poll_stable_state(state),
                "context": poll_context,
            }
        ),
        "queue": _workspace_poll_digest(queue_rows),
        "library": _workspace_poll_digest(library_rows),
        "header": _workspace_poll_digest(fiche_header),
        "transcript": _workspace_poll_digest(fiche_transcript),
        "status": _workspace_poll_digest(workspace_status),
        "dashboard": _workspace_poll_digest(dashboard),
        "assistant": _workspace_poll_digest(
            {
                "role_summary": assistant_payload[0],
                "speaker_choices": _speaker_choices(item),
                "proposal": _assistant_role_proposal(item),
                "client_excerpt": assistant_payload[3],
                "master_excerpt": assistant_payload[4],
                "result": assistant_payload[5],
                "submit_enabled": bool(_item_turns(item)),
            }
        ),
    }
    revision = json.dumps(revision_parts, sort_keys=True, separators=(",", ":"))
    previous_revision = _workspace_poll_revision_parts(last_revision)
    if previous_revision.get("full") == revision_parts["full"]:
        return (revision, *([gr.skip()] * 27))

    stable_changed = previous_revision.get("stable") != revision_parts["stable"]
    if not stable_changed:
        changed = lambda key: previous_revision.get(key) != revision_parts[key]
        transcript_update = fiche_transcript if changed("transcript") else gr.skip()
        return (
            revision,
            gr.skip(),
            queue_rows if changed("queue") else gr.skip(),
            library_rows if changed("library") else gr.skip(),
            fiche_header if changed("header") else gr.skip(),
            transcript_update,
            gr.skip(),
            gr.skip(),
            gr.skip(),
            transcript_update,
            gr.skip(),
            gr.skip(),
            gr.skip(),
            workspace_status if changed("status") else gr.skip(),
            dashboard if changed("dashboard") else gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            *(assistant_payload if changed("assistant") else ([gr.skip()] * 7)),
        )

    return (
        revision,
        state,
        queue_rows,
        library_rows,
        fiche_header,
        fiche_transcript,
        _comments_feed_html(item),
        gr.Dropdown(
            choices=batch_order,
            value=result_label,
            interactive=bool(batch_order),
        ),
        advanced_rows,
        legacy_transcript,
        files,
        batch,
        metadata,
        workspace_status,
        dashboard,
        gr.Dropdown(
            choices=comment_choices,
            value=selected_comment,
            interactive=bool(comment_choices),
        ),
        reply_target,
        _thread_context_html(item, reply_target),
        gr.Button(value="Répondre", interactive=reply_target_is_valid),
        gr.Button(
            value=_comment_action_label(item, selected_comment),
            interactive=bool(selected_comment),
        ),
        gr.skip() if reply_target_is_valid else "",
        *assistant_payload,
    )


def poll_workspace_queue_ui(audio_files: object = None):
    """Retire uniquement les audios terminés d'un uploader déjà cohérent.

    Une écriture serveur dans ``gr.File`` redéclenche son listener ``.change``.
    Le poll ne doit donc jamais restaurer la file durable pendant qu'un upload,
    une suppression, un vidage ou un glisser-déposer est encore en cours. Il
    n'écrit que lorsque tous les chemins visibles sont déjà connus, que les
    éléments encore actifs sont présents dans le même ordre et que les seuls
    éléments à retirer sont des transcriptions terminées.
    """

    if JOB_MANAGER.is_active():
        return (gr.skip(), gr.skip(), gr.skip())

    state = load_workspace()
    visible_paths = _normalise_audio_paths(audio_files)
    visible_ids: list[str] = []
    for path in visible_paths:
        item_id = _workspace_audio_id_for_path(state, path)
        if item_id is None or item_id in visible_ids:
            return (gr.skip(), gr.skip(), gr.skip())
        visible_ids.append(item_id)

    queue_assets = _workspace_queue_assets(state)
    queue_ids = [item_id for item_id, _ in queue_assets]
    queue_id_set = set(queue_ids)
    visible_queue_ids = [
        item_id for item_id in visible_ids if item_id in queue_id_set
    ]
    if visible_queue_ids != queue_ids:
        return (gr.skip(), gr.skip(), gr.skip())

    items = _workspace_items(state)
    completed_extras = [
        item_id for item_id in visible_ids if item_id not in queue_id_set
    ]
    if not completed_extras or any(
        str(items.get(item_id, {}).get("status") or "") != "Terminé"
        or not isinstance(items.get(item_id, {}).get("result"), dict)
        for item_id in completed_extras
    ):
        return (gr.skip(), gr.skip(), gr.skip())

    queue_paths = [path for _, path in queue_assets]
    return (
        [str(path) for path in queue_paths],
        _queue_hint(queue_paths),
        gr.Button(
            value="Lancer la file d'attente",
            interactive=bool(queue_paths),
        ),
    )


def load_selected_result_ui(label: str | None, batch_state: dict[str, object] | None):
    items = (batch_state or {}).get("items", {})
    item = items.get(label) if isinstance(items, dict) else None
    if not isinstance(item, dict) or not isinstance(item.get("result"), dict):
        raise gr.Error("Ce résultat n'est plus disponible.")

    data = item["result"]
    turns = _turns_from_result_data(data)
    downloads = [str(path) for path in item.get("files", [])]
    batch_zip = (batch_state or {}).get("batch_zip")
    if batch_zip:
        downloads.insert(0, str(batch_zip))
    return (
        turns_to_rows(turns),
        plain_transcript(turns),
        _metadata_from_result_data(data),
        downloads,
    )


def transcribe_ui(
    audio_files: object,
    profile: str,
    language_label: str,
    requested_speakers: int,
    speaker_1: str,
    speaker_2: str,
    vocabulary: str | None,
    audio_names: dict[str, str] | None = None,
    workspace_state: dict[str, object] | None = None,
    engine: str = DEFAULT_TRANSCRIPTION_ENGINE,
):
    sources = _normalise_audio_paths(audio_files)
    if not sources:
        raise gr.Error("Ajoutez au moins un fichier audio à la file d'attente.")

    missing = [source.name for source in sources if not source.is_file()]
    if missing:
        raise gr.Error("Fichier(s) introuvable(s) : " + ", ".join(missing))

    workspace, workspace_order, persistent_workspace = _prepare_transcription_workspace(
        sources,
        audio_names,
        workspace_state,
    )
    queue_rows = _initial_queue_rows(sources, audio_names)
    batch_state: dict[str, object] = {"order": [], "items": {}, "batch_zip": None}
    completed_files: list[list[Path]] = []
    completed_zips: list[Path] = []
    failures: list[str] = []
    total = len(sources)
    effective_engine = _effective_engine_choice(engine, profile)

    if persistent_workspace:
        settings = JobSettings(
            profile=profile,
            language=LANGUAGES.get(language_label),
            requested_speakers=int(requested_speakers),
            speaker_names=(speaker_1 or "", speaker_2 or ""),
            initial_prompt=vocabulary or "",
            engine=effective_engine,
        )
        added = JOB_MANAGER.enqueue(workspace_order, settings)
        workspace = load_workspace()
        batch_state = _batch_state_from_workspace(workspace)
        batch_order = batch_state.get("order")
        choices = list(batch_order) if isinstance(batch_order, list) else []
        selected = added[0] if added else (
            next(
                (
                    item_id
                    for item_id in workspace_order
                    if JOB_MANAGER.is_active(item_id)
                ),
                workspace_order[0] if workspace_order else None,
            )
        )
        if added:
            status = (
                "### Traitement lancé en arrière-plan\n\n"
                f"**{len(added)} audio(s)** sont dans la file. Vous pouvez changer "
                "d'onglet ou rafraîchir la page sans interrompre la transcription."
            )
        elif JOB_MANAGER.is_active():
            status = (
                "### Traitement déjà en cours\n\n"
                "La page est reconnectée au worker local ; aucun doublon n'a été lancé."
            )
        else:
            status = (
                "### Aucun audio en attente\n\n"
                "Les transcriptions terminées sont disponibles dans l'onglet Fiches."
            )
        yield (
            status,
            _workspace_queue_rows(workspace),
            gr.Dropdown(
                choices=choices,
                value=choices[-1] if choices else None,
                interactive=bool(choices),
            ),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            batch_state,
            gr.skip(),
            *_workspace_live_outputs(workspace, selected),
        )
        return

    yield (
        f"### File prête\n\n**{total} audio(s)** seront traités dans l'ordre affiché.",
        copy.deepcopy(queue_rows),
        gr.Dropdown(choices=[], value=None, interactive=False),
        [],
        "",
        [],
        copy.deepcopy(batch_state),
        {},
        *_workspace_live_outputs(workspace, workspace_order[0] if workspace_order else None),
    )

    for index, source in enumerate(sources):
        item_id = workspace_order[index]
        display_name = _display_audio_name(source, audio_names)
        started_at = time.monotonic()
        queue_rows[index][2] = "En cours"
        queue_rows[index][3] = "0 %"
        queue_rows[index][4] = "0 s"
        workspace = _update_transcription_workspace(
            workspace,
            item_id,
            persistent_workspace,
            name=display_name,
            status="En cours",
            progress=0.0,
            processing_time="0 s",
            partial="",
            error="",
        )
        events: queue.Queue[tuple[str, object, object | None]] = queue.Queue()

        def report(value: float, message: str) -> None:
            events.put(("progress", float(value), str(message)))

        def report_partial(text: str, position: float) -> None:
            events.put(("partial", text, float(position)))

        def worker(path: Path = source) -> None:
            try:
                result = run_pipeline(
                    path,
                    profile=profile,
                    asr_backend=effective_engine,
                    language=LANGUAGES.get(language_label),
                    requested_speakers=int(requested_speakers),
                    speaker_names=[speaker_1 or "", speaker_2 or ""],
                    initial_prompt=vocabulary or "",
                    progress=report,
                    partial=report_partial,
                )
                events.put(("done", result, None))
            except Exception as exc:  # transmis proprement au générateur Gradio
                events.put(("error", exc, None))

        threading.Thread(
            target=worker,
            daemon=True,
            name=f"transcription-worker-{index + 1}",
        ).start()

        file_finished = False
        while not file_finished:
            event_type, payload, extra = events.get()
            elapsed = time.monotonic() - started_at
            queue_rows[index][4] = format_duration(elapsed)

            if event_type == "progress":
                value = max(0.0, min(1.0, float(payload)))
                message = str(extra)
                queue_rows[index][3] = f"{value:.0%}"
                overall = (index + value) / total
                workspace = _update_transcription_workspace(
                    workspace,
                    item_id,
                    persistent_workspace,
                    status="En cours",
                    progress=value,
                    processing_time=format_duration(elapsed),
                )
                yield (
                    f"### Audio {index + 1}/{total} — {display_name}\n\n"
                    f"{message} · progression globale **{overall:.0%}**",
                    copy.deepcopy(queue_rows),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    *_workspace_live_outputs(workspace, item_id),
                )
            elif event_type == "partial":
                position = float(extra or 0.0)
                workspace = _update_transcription_workspace(
                    workspace,
                    item_id,
                    persistent_workspace,
                    status="En cours",
                    partial=str(payload),
                    processing_time=format_duration(elapsed),
                )
                yield (
                    f"### Audio {index + 1}/{total} — transcription en cours\n\n"
                    f"Texte reconnu jusqu'à **{format_duration(position)}**. "
                    "Les interlocuteurs seront attribués à la fin de cet audio.",
                    copy.deepcopy(queue_rows),
                    gr.skip(),
                    gr.skip(),
                    str(payload),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    *_workspace_live_outputs(workspace, item_id),
                )
            elif event_type == "done":
                result = payload
                result.audio_name = display_name
                try:
                    files = write_exports(result)
                except Exception as exc:
                    event_type = "error"
                    payload = exc
                else:
                    completed_files.append(files)
                    completed_zips.append(files[0])
                    queue_rows[index][2] = "Terminé"
                    queue_rows[index][3] = "100 %"
                    queue_rows[index][4] = format_duration(result.processing_seconds)

                    label = f"{index + 1:02d} · {display_name}"
                    order = batch_state["order"]
                    items = batch_state["items"]
                    assert isinstance(order, list) and isinstance(items, dict)
                    order.append(label)
                    items[label] = {
                        "result": result.to_dict(),
                        "files": [str(path) for path in files],
                    }
                    workspace = _update_transcription_workspace(
                        workspace,
                        item_id,
                        persistent_workspace,
                        name=display_name,
                        status="Terminé",
                        progress=1.0,
                        processing_time=format_duration(result.processing_seconds),
                        partial=plain_transcript(result.turns),
                        result=result.to_dict(),
                        files=[str(path) for path in files],
                        error="",
                    )

                    yield (
                        f"### Audio {index + 1}/{total} terminé — {display_name}\n\n"
                        + _status_markdown(result),
                        copy.deepcopy(queue_rows),
                        gr.Dropdown(choices=list(order), value=label, interactive=True),
                        turns_to_rows(result.turns),
                        plain_transcript(result.turns),
                        [str(path) for path in files],
                        copy.deepcopy(batch_state),
                        result.to_metadata(),
                        *_workspace_live_outputs(workspace, item_id),
                    )
                    file_finished = True

            if event_type == "error":
                exc = payload
                detail = f"{type(exc).__name__}: {exc}"
                failures.append(f"{display_name} — {detail}")
                queue_rows[index][2] = "Échec"
                queue_rows[index][3] = "—"
                queue_rows[index][4] = format_duration(elapsed)
                workspace = _update_transcription_workspace(
                    workspace,
                    item_id,
                    persistent_workspace,
                    status="Échec",
                    progress=0.0,
                    processing_time=format_duration(elapsed),
                    error=detail,
                )
                yield (
                    f"### Échec sur l'audio {index + 1}/{total} — {display_name}\n\n"
                    f"La file poursuit avec le fichier suivant. Détail technique : `{detail}`",
                    copy.deepcopy(queue_rows),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    *_workspace_live_outputs(workspace, item_id),
                )
                file_finished = True

    batch_zip: Path | None = None
    if completed_zips and total > 1:
        try:
            batch_zip = write_batch_archive(completed_zips)
            batch_state["batch_zip"] = str(batch_zip)
            workspace["batch_zip"] = str(batch_zip)
            if persistent_workspace:
                workspace = _save_workspace_from_ui(workspace)
        except Exception as exc:
            failures.append(f"Création du ZIP global — {type(exc).__name__}: {exc}")

    success_count = len(completed_files)
    if success_count:
        final_downloads: list[str]
        if total == 1:
            final_downloads = [str(path) for path in completed_files[0]]
        else:
            final_downloads = ([str(batch_zip)] if batch_zip else []) + [
                str(path) for path in completed_zips
            ]
        final_label = batch_state["order"][-1]
        status = (
            "### File d'attente terminée\n\n"
            f"**{success_count}/{total} audio(s)** transcrits. "
            "Vous pouvez choisir un résultat à relire ou télécharger le ZIP global du lot."
        )
        if failures:
            status += "\n\n> Attention : " + "  \n> ".join(failures)
        yield (
            status,
            copy.deepcopy(queue_rows),
            gr.Dropdown(
                choices=list(batch_state["order"]),
                value=final_label,
                interactive=True,
            ),
            gr.skip(),
            gr.skip(),
            final_downloads,
            copy.deepcopy(batch_state),
            gr.skip(),
            *_workspace_live_outputs(
                workspace,
                workspace_order[-1] if workspace_order else None,
            ),
        )
    else:
        yield (
            "### Aucun audio n'a pu être transcrit\n\n" + "  \n".join(failures),
            copy.deepcopy(queue_rows),
            gr.Dropdown(choices=[], value=None, interactive=False),
            [],
            "",
            [],
            copy.deepcopy(batch_state),
            {},
            *_workspace_live_outputs(
                workspace,
                workspace_order[-1] if workspace_order else None,
            ),
        )


def _table_rows(table: Any) -> list[list[object]]:
    if table is None:
        return []
    if hasattr(table, "values") and hasattr(table.values, "tolist"):
        return table.values.tolist()
    if isinstance(table, dict):
        data = table.get("data", [])
        return [list(row) for row in data]
    return [list(row) for row in table]


def export_corrections_ui(table: Any, metadata: dict[str, object] | None):
    if not metadata:
        raise gr.Error("Sélectionnez d'abord une transcription.")
    try:
        result = result_from_rows(_table_rows(table), metadata)
        files = write_exports(result, corrected=True)
        return (
            plain_transcript(result.turns),
            [str(path) for path in files],
            "### Corrections exportées\n\nLes nouveaux fichiers sont prêts au téléchargement.",
        )
    except Exception as exc:
        raise gr.Error(f"Impossible d'exporter les corrections : {exc}") from exc


def _build_app_legacy() -> gr.Blocks:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with gr.Blocks(title="Studio Audio", analytics_enabled=False) as app:
        batch_state = gr.State({"order": [], "items": {}, "batch_zip": None})
        current_metadata = gr.State({})

        gr.HTML(
            """
            <section class="hero">
              <div class="hero-top">
                <div>
                  <h1>Studio Audio</h1>
                  <p>Organisez plusieurs enregistrements, suivez leur transcription et récupérez des documents prêts à partager.</p>
                </div>
              </div>
              <div class="hero-badges">
                <span>🔒 Audios conservés sur ce PC</span>
                <span>👥 Séparation des interlocuteurs</span>
                <span>📚 Traitement par lot</span>
              </div>
            </section>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=420):
                audio_queue = gr.File(
                    file_count="multiple",
                    file_types=["audio"],
                    type="filepath",
                    label="1 · File d'attente audio",
                    height=240,
                    interactive=True,
                    allow_reordering=True,
                    elem_id="audio-queue",
                )
                queue_hint = gr.HTML(
                    "<div class='queue-count'><strong>File vide.</strong> "
                    "Déposez un ou plusieurs fichiers audio ci-dessus.</div>"
                )
                gr.Markdown(
                    "<span class='hint'>Formats courants : WAV, MP3, M4A, FLAC, OGG… "
                    "L'ordre visible au moment du lancement sera utilisé.</span>"
                )
                with gr.Accordion("Écouter le premier audio de la file", open=False):
                    preview = gr.Audio(
                        type="filepath",
                        label="Aperçu",
                        interactive=False,
                    )

            with gr.Column(scale=2, min_width=330):
                profile = gr.Radio(
                    choices=list(PROFILE_CONFIG),
                    value="Équilibré",
                    label="2 · Niveau de qualité",
                    info="Équilibré est recommandé pour les conversations téléphoniques.",
                )
                language = gr.Dropdown(
                    choices=list(LANGUAGES),
                    value="Français",
                    label="Langue parlée",
                )
                requested_speakers = gr.Slider(
                    minimum=1,
                    maximum=8,
                    step=1,
                    value=2,
                    label="Nombre d'interlocuteurs",
                    info="Indiquer le nombre connu améliore fortement la séparation.",
                )

        queue_table = gr.Dataframe(
            headers=QUEUE_HEADERS,
            datatype=["number", "str", "str", "str", "str", "str"],
            value=[],
            interactive=False,
            wrap=True,
            label="Suivi de la file d'attente",
            elem_id="queue-table",
        )

        with gr.Accordion("Noms et vocabulaire (facultatif)", open=False):
            with gr.Row():
                speaker_1 = gr.Textbox(
                    value="Interlocuteur 1",
                    label="Première voix entendue",
                )
                speaker_2 = gr.Textbox(
                    value="Interlocuteur 2",
                    label="Deuxième voix entendue",
                )
            vocabulary = gr.Textbox(
                label="Noms propres ou mots métier",
                placeholder="Ex. Dupont, Qualiopi, référence ZX-418…",
                info="Quelques termes difficiles aident Whisper à mieux les orthographier.",
                lines=2,
            )

        transcribe_button = gr.Button(
            "3 · Lancer la file d'attente",
            variant="primary",
            size="lg",
            elem_classes=["primary-btn"],
        )

        status = gr.Markdown(
            "### Prêt\n\nAjoutez vos audios et organisez-les. La première transcription "
            "télécharge les modèles ; les suivantes réutilisent le modèle déjà chargé."
        )

        with gr.Tabs():
            with gr.Tab("Suivi en direct"):
                transcript = gr.Textbox(
                    label="Transcription de l'audio en cours",
                    info="Le texte apparaît progressivement, avant l'attribution finale des voix.",
                    lines=18,
                    buttons=["copy"],
                    interactive=True,
                )
            with gr.Tab("Relire et corriger"):
                result_selector = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Audio à relire",
                    info="Chaque audio terminé reste disponible ici.",
                    interactive=False,
                )
                table = gr.Dataframe(
                    headers=["Début", "Fin", "Interlocuteur", "Texte"],
                    datatype=["str", "str", "str", "str"],
                    row_count=1,
                    column_count=4,
                    interactive=True,
                    wrap=True,
                    label="Corrigez directement les cellules si nécessaire",
                )
                export_button = gr.Button("Exporter les corrections", variant="secondary")
            with gr.Tab("Télécharger"):
                gr.Markdown(
                    "Le ZIP global contient un ZIP complet par audio. "
                    "Vous pouvez aussi télécharger séparément le résultat sélectionné."
                )
                downloads = gr.File(
                    label="Fichiers prêts à partager",
                    file_count="multiple",
                    interactive=False,
                )

        audio_queue.change(
            fn=preview_queue_ui,
            inputs=[audio_queue],
            outputs=[preview, queue_table, queue_hint],
            queue=False,
        )
        transcribe_button.click(
            fn=transcribe_ui,
            inputs=[
                audio_queue,
                profile,
                language,
                requested_speakers,
                speaker_1,
                speaker_2,
                vocabulary,
            ],
            outputs=[
                status,
                queue_table,
                result_selector,
                table,
                transcript,
                downloads,
                batch_state,
                current_metadata,
            ],
            show_progress="minimal",
        )
        result_selector.change(
            fn=load_selected_result_ui,
            inputs=[result_selector, batch_state],
            outputs=[table, transcript, current_metadata, downloads],
            show_progress="hidden",
        )
        export_button.click(
            fn=export_corrections_ui,
            inputs=[table, current_metadata],
            outputs=[transcript, downloads, status],
            show_progress="minimal",
        )

    return app


def _build_app_single_page() -> gr.Blocks:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_ready = models_ready()

    initial_workspace = load_workspace()
    initial_items = _workspace_items(initial_workspace)
    initial_order = _workspace_order(initial_workspace)
    initial_available = _workspace_queue_assets(initial_workspace)
    initial_paths = [path for _, path in initial_available]
    initial_names = {
        str(path): str(initial_items[item_id].get("name") or path.name)
        for item_id, path in initial_available
    }
    initial_selected = initial_order[0] if initial_order else None
    initial_item = _workspace_item(initial_selected, initial_workspace)
    initial_role_proposal = _assistant_role_proposal(initial_item)
    initial_speaker_choices = _speaker_choices(initial_item)
    initial_assistant_result = _assistant_latest_result(initial_item)
    initial_comment_choices = _comment_choices(initial_item)
    initial_comment = initial_comment_choices[0][1] if initial_comment_choices else None
    initial_comment_enabled = _commenting_enabled(initial_item)
    initial_batch = _batch_state_from_workspace(initial_workspace)
    initial_batch_order = (
        initial_batch.get("order") if isinstance(initial_batch.get("order"), list) else []
    )
    initial_result_label = initial_batch_order[0] if initial_batch_order else None
    initial_result = (
        initial_item.get("result")
        if initial_item and isinstance(initial_item.get("result"), dict)
        else None
    )
    if initial_result:
        initial_advanced_rows = turns_to_rows(_turns_from_result_data(initial_result))
        initial_metadata = _metadata_from_result_data(initial_result)
        initial_downloads = [str(path) for path in initial_item.get("files", [])]
    else:
        initial_advanced_rows, initial_metadata, initial_downloads = [], {}, []

    with gr.Blocks(title="Studio Audio", analytics_enabled=False) as app:
        with gr.Column(visible=not setup_ready, elem_id="setup-screen") as setup_screen:
            with gr.Group(elem_classes=["setup-card"]):
                gr.HTML(
                    """
                    <section class="setup-copy">
                      <span class="eyebrow">Premier lancement</span>
                      <h1>Préparation de l'application</h1>
                      <p>Les modèles de transcription et de séparation des voix sont installés maintenant. Vous n'aurez ainsi aucune attente imprévue au moment de traiter votre premier audio.</p>
                    </section>
                    """
                )
                setup_progress = gr.HTML(
                    _setup_progress_html(100 if setup_ready else 0)
                )
                setup_status = gr.Markdown(
                    "### Vérification de l'installation\n\nPréparation des moteurs locaux…"
                )
                setup_details = gr.HTML(_setup_details_html())
                setup_retry = gr.Button(
                    "Relancer la préparation",
                    variant="secondary",
                    visible=False,
                )
            gr.HTML(
                "<p class='setup-footnote'>Téléchargement unique · environ 8 Go · les modèles restent sur cet ordinateur</p>"
            )

        with gr.Column(visible=setup_ready, elem_id="app-screen") as main_screen:
            batch_state = gr.State(initial_batch)
            current_metadata = gr.State(initial_metadata)
            audio_names = gr.State(initial_names)
            workspace_state = gr.State(initial_workspace)
            legacy_dashboard = gr.State("")

            gr.HTML(
                f"""
                <header class="app-header">
                  <div class="app-header-top">
                    <div>
                      <span class="eyebrow">Studio de transcription local</span>
                      <h1>Workspace audio</h1>
                      <p>Transcrivez, relisez et documentez chaque conversation dans un espace de travail privé.</p>
                    </div>
                    <div class="header-pills">
                      <span class="privacy-pill">Données sur ce PC</span>
                      <span class="privacy-pill">Workspace privé</span>
                    </div>
                  </div>
                </header>
                """
            )

            with gr.Row(equal_height=False):
                with gr.Column(scale=3, min_width=420):
                    with gr.Group(elem_classes=["glass-card"]):
                        gr.HTML(
                            "<p class='step-title'>1. Organiser les audios</p>"
                            "<p class='step-copy'>Déposez plusieurs fichiers puis déplacez-les pour choisir leur ordre.</p>"
                        )
                        audio_queue = gr.File(
                            value=[str(path) for path in initial_paths],
                            file_count="multiple",
                            file_types=["audio"],
                            type="filepath",
                            label="File d'attente",
                            height=238,
                            interactive=True,
                            allow_reordering=True,
                            elem_id="audio-queue",
                        )
                        queue_hint = gr.HTML(
                            _queue_hint(
                                initial_paths,
                                "Espace de travail restauré." if initial_paths else "",
                            )
                        )
                        with gr.Accordion(
                            "Renommer un audio",
                            open=False,
                            elem_classes=["rename-panel"],
                        ):
                            rename_selector = gr.Dropdown(
                                choices=_rename_choices(initial_paths, initial_names),
                                value=str(initial_paths[0]) if initial_paths else None,
                                label="Audio à renommer",
                                interactive=bool(initial_paths),
                                filterable=False,
                                elem_id="rename-audio-select",
                            )
                            with gr.Row(elem_classes=["rename-row"]):
                                rename_input = gr.Textbox(
                                    value=(
                                        _rename_stem(initial_names[str(initial_paths[0])])
                                        if initial_paths
                                        else ""
                                    ),
                                    label="Nouveau nom (sans extension)",
                                    placeholder="Ex. Entretien client Dupont",
                                    interactive=bool(initial_paths),
                                    scale=3,
                                )
                                rename_button = gr.Button(
                                    "Enregistrer le nom",
                                    variant="secondary",
                                    interactive=bool(initial_paths),
                                    scale=1,
                                )
                        gr.Markdown(
                            "<span class='hint'>WAV, MP3, M4A, FLAC et OGG. "
                            "L'ordre visible au lancement sera conservé.</span>"
                        )
                        with gr.Accordion("Écouter le premier audio", open=False):
                            preview = gr.Audio(
                                value=str(initial_paths[0]) if initial_paths else None,
                                type="filepath",
                                label="Aperçu",
                                interactive=False,
                            )

                with gr.Column(scale=2, min_width=330):
                    with gr.Group(elem_classes=["glass-card"]):
                        gr.HTML(
                            "<p class='step-title'>2. Régler la transcription</p>"
                            "<p class='step-copy'>Le mode Équilibré convient à la plupart des conversations.</p>"
                        )
                        profile = gr.Radio(
                            choices=list(PROFILE_CONFIG),
                            value="Équilibré",
                            label="Niveau de qualité",
                            elem_classes=["quality-selector"],
                        )
                        language = gr.Dropdown(
                            choices=list(LANGUAGES),
                            value="Français",
                            label="Langue parlée",
                            interactive=True,
                            filterable=False,
                            elem_id="language-choice",
                        )
                        requested_speakers = gr.Slider(
                            minimum=1,
                            maximum=8,
                            step=1,
                            value=2,
                            label="Nombre d'interlocuteurs",
                            info="Un nombre connu améliore la séparation des voix.",
                        )

            with gr.Group(elem_classes=["glass-card"]):
                gr.HTML(
                    "<div class='section-heading'><div><span class='eyebrow'>File active</span>"
                    "<h2>Audios à traiter</h2></div>"
                    "<p>Les éléments terminés restent disponibles dans les fiches audio.</p></div>"
                )
                queue_table = gr.Dataframe(
                    headers=QUEUE_HEADERS,
                    datatype=["number", "str", "str", "str", "str", "str"],
                    value=_workspace_queue_rows(initial_workspace),
                    interactive=False,
                    wrap=True,
                    label="Audios du workspace",
                    elem_id="queue-table",
                )

            with gr.Accordion("Personnaliser les voix et le vocabulaire", open=False):
                with gr.Row():
                    speaker_1 = gr.Textbox(
                        value="Interlocuteur 1",
                        label="Première voix entendue",
                    )
                    speaker_2 = gr.Textbox(
                        value="Interlocuteur 2",
                        label="Deuxième voix entendue",
                    )
                vocabulary = gr.Textbox(
                    label="Noms propres ou mots métier",
                    placeholder="Ex. Dupont, Qualiopi, référence ZX-418…",
                    info="Quelques termes difficiles aident le moteur à mieux les orthographier.",
                    lines=2,
                )

            transcribe_button = gr.Button(
                "3. Lancer la transcription",
                variant="primary",
                size="lg",
                elem_classes=["primary-btn"],
                interactive=bool(initial_paths),
            )

            status = gr.Markdown(
                _workspace_status_markdown(initial_workspace),
                elem_classes=["status-copy", "glass-card", "status-banner"],
            )

            with gr.Group(elem_classes=["glass-card", "workspace-shell"]):
                gr.HTML(
                    """
                    <div class="section-heading workspace-heading">
                      <div>
                        <span class="eyebrow">Dossier de travail</span>
                        <h2>Fiches audio</h2>
                      </div>
                      <p>Une vue complète par enregistrement, de l'attente jusqu'à la validation.</p>
                    </div>
                    """
                )
                fiche_selector = gr.Dropdown(
                    choices=_workspace_choices(initial_workspace),
                    value=initial_selected,
                    label="Ouvrir une fiche audio",
                    interactive=bool(initial_order),
                    filterable=True,
                    elem_id="fiche-selector",
                )
                fiche_header = gr.HTML(_fiche_header_html(initial_item))

                with gr.Row(equal_height=False, elem_classes=["audio-detail-layout"]):
                    with gr.Column(scale=3, min_width=430, elem_classes=["transcript-pane"]):
                        gr.HTML(
                            "<div class='pane-heading'><span class='eyebrow'>Contenu</span>"
                            "<h3>Transcription</h3><p>Sélectionnez un passage pour le joindre automatiquement à une discussion.</p></div>"
                        )
                        fiche_transcript = gr.Textbox(
                            value=_item_transcript(initial_item),
                            label="Transcription de la fiche",
                            lines=22,
                            buttons=["copy"],
                            interactive=False,
                            elem_id="fiche-transcript",
                        )

                    with gr.Column(scale=2, min_width=360, elem_classes=["discussion-pane"]):
                        gr.HTML(
                            "<div class='pane-heading'><span class='eyebrow'>Collaboration</span>"
                            "<h3>Discussions</h3><p>Commentez un extrait puis clôturez le point lorsqu'il est traité.</p></div>"
                        )
                        comments_table = gr.HTML(
                            value=_comments_feed_html(initial_item),
                            elem_id="comments-feed",
                            js_on_load=COMMENTS_FEED_JS,
                        )
                        with gr.Group(elem_classes=["comment-composer"]):
                            comment_author = gr.Textbox(
                                value="Moi",
                                label="Votre nom",
                                placeholder="Ex. Camille",
                            )
                            comment_passage = gr.Textbox(
                                label="Passage concerné",
                                placeholder="Sélectionnez du texte à gauche ou collez un extrait ici.",
                                lines=3,
                            )
                            comment_body = gr.Textbox(
                                label="Commentaire",
                                placeholder="Quelle question, décision ou correction faut-il traiter ?",
                                lines=4,
                            )
                            comment_add_button = gr.Button(
                                "Ajouter à la discussion",
                                variant="primary",
                                interactive=initial_comment_enabled,
                                elem_classes=["comment-primary"],
                            )
                        with gr.Row(elem_classes=["comment-resolution"]):
                            comment_selector = gr.Dropdown(
                                choices=initial_comment_choices,
                                value=initial_comment,
                                label="Point à mettre à jour",
                                interactive=bool(initial_comment_choices),
                                elem_id="comment-selector",
                                scale=3,
                            )
                            comment_resolve_button = gr.Button(
                                _comment_action_label(initial_item, initial_comment),
                                variant="secondary",
                                interactive=bool(initial_comment_choices),
                                scale=2,
                            )

                transcript = gr.Textbox(
                    value=_item_transcript(initial_item),
                    visible=False,
                )

                with gr.Accordion("Édition avancée et fichiers", open=False):
                    with gr.Tabs():
                        with gr.Tab("Relire et corriger"):
                            result_selector = gr.Dropdown(
                                choices=initial_batch_order,
                                value=initial_result_label,
                                label="Audio à relire",
                                info="Chaque audio terminé reste disponible ici.",
                                interactive=bool(initial_batch_order),
                            )
                            table = gr.Dataframe(
                                headers=["Début", "Fin", "Interlocuteur", "Texte"],
                                datatype=["str", "str", "str", "str"],
                                value=initial_advanced_rows,
                                row_count=(1, "dynamic"),
                                column_count=4,
                                interactive=True,
                                wrap=True,
                                label="Corrigez directement les cellules si nécessaire",
                            )
                            export_button = gr.Button(
                                "Exporter les corrections",
                                variant="secondary",
                            )
                        with gr.Tab("Télécharger"):
                            gr.Markdown(
                                "Le ZIP global rassemble un dossier complet par audio. "
                                "Le résultat sélectionné reste aussi disponible séparément."
                            )
                            downloads = gr.File(
                                value=initial_downloads,
                                label="Fichiers prêts à partager",
                                file_count="multiple",
                                interactive=False,
                            )

            queue_change = audio_queue.change(
                fn=sync_queue_ui,
                inputs=[audio_queue, audio_names],
                outputs=[
                    preview,
                    queue_table,
                    queue_hint,
                    audio_names,
                    rename_selector,
                    rename_input,
                    rename_button,
                    transcribe_button,
                ],
                queue=False,
                show_progress="hidden",
                show_progress_on=[],
            )
            queue_change.then(
                fn=sync_workspace_ui,
                inputs=[audio_queue, audio_names, workspace_state],
                outputs=[
                    workspace_state,
                    queue_table,
                    fiche_selector,
                    fiche_header,
                    fiche_transcript,
                    comments_table,
                    comment_selector,
                    comment_add_button,
                    comment_resolve_button,
                ],
                queue=False,
                show_progress="hidden",
                show_progress_on=[],
            )
            inline_rename = inline_rename_submit.click(
                fn=rename_audio_by_index_ui,
                inputs=[
                    audio_queue,
                    inline_rename_index,
                    inline_rename_name,
                    audio_names,
                ],
                outputs=[
                    audio_names,
                    queue_table,
                    queue_hint,
                    rename_selector,
                    rename_input,
                    rename_button,
                ],
                queue=False,
            )
            inline_rename_sync = inline_rename.then(
                fn=sync_workspace_names_ui,
                inputs=[audio_queue, audio_names, workspace_state, fiche_selector],
                outputs=[
                    audio_queue,
                    audio_names,
                    workspace_state,
                    queue_table,
                    fiche_selector,
                    fiche_header,
                    fiche_transcript,
                    comments_table,
                    comment_selector,
                    comment_add_button,
                    comment_resolve_button,
                ],
                queue=False,
            )
            rename_submit = rename_input.submit(
                fn=rename_audio_ui,
                inputs=[audio_queue, rename_selector, rename_input, audio_names],
                outputs=[
                    audio_names,
                    queue_table,
                    queue_hint,
                    rename_selector,
                    rename_input,
                    rename_button,
                ],
                queue=False,
            )
            rename_submit.then(
                fn=sync_workspace_names_ui,
                inputs=[audio_queue, audio_names, workspace_state, fiche_selector],
                outputs=[
                    workspace_state,
                    queue_table,
                    fiche_selector,
                    fiche_header,
                    fiche_transcript,
                    comments_table,
                    comment_selector,
                    comment_add_button,
                    comment_resolve_button,
                ],
                queue=False,
            )
            inline_rename_sync.then(
                fn=sync_library_ui,
                inputs=[workspace_state, library_sort],
                outputs=[fiche_library_table],
                queue=False,
            )
            library_sort.change(
                fn=sync_library_ui,
                inputs=[workspace_state, library_sort],
                outputs=[fiche_library_table],
                queue=False,
            )
            workspace_state.change(
                fn=sync_library_ui,
                inputs=[workspace_state, library_sort],
                outputs=[fiche_library_table],
                queue=False,
            )
            fiche_library_table.select(
                fn=select_library_fiche_ui,
                inputs=[workspace_state, library_sort],
                outputs=[
                    fiche_selector,
                    fiche_header,
                    preview,
                    fiche_transcript,
                    comments_table,
                    comment_selector,
                    comment_add_button,
                    comment_resolve_button,
                    table,
                    current_metadata,
                    downloads,
                ],
                queue=False,
            )
            fiche_selector.input(
                fn=load_fiche_ui,
                inputs=[fiche_selector, workspace_state],
                outputs=[
                    fiche_header,
                    preview,
                    fiche_transcript,
                    comments_table,
                    comment_selector,
                    comment_add_button,
                    comment_resolve_button,
                    table,
                    current_metadata,
                    downloads,
                ],
                queue=False,
            )
            queue_table.select(
                fn=select_queue_fiche_ui,
                inputs=[workspace_state],
                outputs=[
                    fiche_selector,
                    fiche_header,
                    preview,
                    fiche_transcript,
                    comments_table,
                    comment_selector,
                    comment_add_button,
                    comment_resolve_button,
                    table,
                    current_metadata,
                    downloads,
                ],
                queue=False,
            )
            fiche_transcript.select(
                fn=capture_selected_passage,
                inputs=[fiche_transcript],
                outputs=[comment_passage],
                queue=False,
            )
            comment_add_button.click(
                fn=add_comment_ui,
                inputs=[
                    fiche_selector,
                    comment_author,
                    comment_passage,
                    comment_body,
                    workspace_state,
                ],
                outputs=[
                    workspace_state,
                    comments_table,
                    comment_selector,
                    fiche_header,
                    comment_passage,
                    comment_body,
                    comment_resolve_button,
                ],
                queue=False,
            )
            comment_selector.input(
                fn=load_comment_action_ui,
                inputs=[fiche_selector, comment_selector, workspace_state],
                outputs=[comment_resolve_button],
                queue=False,
            )
            comment_resolve_button.click(
                fn=toggle_comment_ui,
                inputs=[fiche_selector, comment_selector, workspace_state],
                outputs=[
                    workspace_state,
                    comments_table,
                    fiche_header,
                    comment_resolve_button,
                ],
                queue=False,
            )
            transcribe_button.click(
                fn=transcribe_ui,
                inputs=[
                    audio_queue,
                    profile,
                    language,
                    requested_speakers,
                    speaker_1,
                    speaker_2,
                    vocabulary,
                    audio_names,
                    workspace_state,
                ],
                outputs=[
                    status,
                    queue_table,
                    result_selector,
                    table,
                    transcript,
                    downloads,
                    batch_state,
                    current_metadata,
                    workspace_state,
                    fiche_selector,
                    fiche_header,
                    fiche_transcript,
                    comments_table,
                    comment_selector,
                    comment_add_button,
                    comment_resolve_button,
                ],
                show_progress="minimal",
                concurrency_id="transcription",
                concurrency_limit=1,
            )
            result_selector.input(
                fn=load_selected_result_ui,
                inputs=[result_selector, batch_state],
                outputs=[table, transcript, current_metadata, downloads],
                show_progress="hidden",
            )
            export_button.click(
                fn=export_corrections_ui,
                inputs=[table, current_metadata],
                outputs=[transcript, downloads, status],
                show_progress="minimal",
            )

        setup_outputs = [
            setup_progress,
            setup_status,
            setup_details,
            setup_screen,
            main_screen,
            setup_retry,
        ]
        app.load(
            fn=None,
            js=INLINE_RENAME_JS,
            queue=False,
        )
        app.load(
            fn=None,
            js=OVERFLOW_TOOLTIP_JS,
            queue=False,
        )
        app.load(
            fn=prepare_models_ui,
            outputs=setup_outputs,
            show_progress="hidden",
        )
        app.load(
            fn=restore_workspace_ui,
            outputs=[
                audio_queue,
                preview,
                queue_table,
                queue_hint,
                audio_names,
                rename_selector,
                rename_input,
                rename_button,
                workspace_state,
                fiche_selector,
                fiche_header,
                fiche_transcript,
                comments_table,
                comment_selector,
                comment_add_button,
                comment_resolve_button,
                result_selector,
                table,
                transcript,
                downloads,
                batch_state,
                current_metadata,
                status,
                legacy_dashboard,
                transcribe_button,
            ],
            show_progress="hidden",
        )
        app.load(
            fn=restore_engine_preference_ui,
            inputs=[profile, language, requested_speakers],
            outputs=[engine, engine_status, launch_summary],
            show_progress="hidden",
        )
        setup_retry.click(
            fn=prepare_models_ui,
            outputs=setup_outputs,
            show_progress="minimal",
        )

    return app


def build_app() -> gr.Blocks:
    """Construit le workspace SaaS local à navigation multi-onglets."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_ready = models_ready()

    initial_workspace = load_workspace()
    initial_navigation = _workspace_ui_navigation(initial_workspace)
    initial_engine = _workspace_engine_preference(initial_workspace)
    initial_items = _workspace_items(initial_workspace)
    initial_order = _workspace_order(initial_workspace)
    initial_available = _workspace_queue_assets(initial_workspace)
    initial_paths = [path for _, path in initial_available]
    initial_names = {
        str(path): str(initial_items[item_id].get("name") or path.name)
        for item_id, path in initial_available
    }
    stored_fiche_item = initial_navigation.get("fiche_item_id")
    initial_selected = (
        stored_fiche_item
        if stored_fiche_item in initial_items
        else (initial_order[0] if initial_order else None)
    )
    initial_item = _workspace_item(initial_selected, initial_workspace)
    initial_active_tab = initial_navigation["active_tab"]
    initial_detail_visible = (
        initial_navigation["fiche_view"] == "detail" and initial_item is not None
    )
    initial_context_mode = {
        "transcript": "Transcription",
        "assistant": "Assistant IA",
        "discussions": "Discussions",
    }[initial_navigation["context_mode"]]
    initial_role_proposal = _assistant_role_proposal(initial_item)
    initial_speaker_choices = _speaker_choices(initial_item)
    initial_assistant_result = _assistant_latest_result(initial_item)
    initial_comment_choices = _comment_choices(initial_item)
    initial_comment = initial_comment_choices[0][1] if initial_comment_choices else None
    initial_comment_enabled = _commenting_enabled(initial_item)
    initial_batch = _batch_state_from_workspace(initial_workspace)
    initial_batch_order = (
        initial_batch.get("order") if isinstance(initial_batch.get("order"), list) else []
    )
    initial_result_label = initial_batch_order[0] if initial_batch_order else None
    initial_result = (
        initial_item.get("result")
        if initial_item and isinstance(initial_item.get("result"), dict)
        else None
    )
    if initial_result:
        initial_advanced_rows = turns_to_rows(_turns_from_result_data(initial_result))
        initial_metadata = _metadata_from_result_data(initial_result)
        initial_downloads = [str(path) for path in initial_item.get("files", [])]
    else:
        initial_advanced_rows, initial_metadata, initial_downloads = [], {}, []

    with gr.Blocks(title="Studio Audio", analytics_enabled=False) as app:
        with gr.Column(visible=not setup_ready, elem_id="setup-screen") as setup_screen:
            with gr.Group(elem_classes=["setup-card"]):
                gr.HTML(
                    """
                    <section class="setup-copy">
                      <span class="eyebrow">Premier lancement</span>
                      <h1>Préparation de votre workspace</h1>
                      <p>Les moteurs de transcription et de séparation des voix sont installés une seule fois. L'application sera ensuite immédiatement prête pour tous les audios suivants.</p>
                    </section>
                    """
                )
                setup_progress = gr.HTML(_setup_progress_html(100 if setup_ready else 0))
                setup_status = gr.Markdown(
                    "### Vérification de l'installation\n\nPréparation des moteurs locaux…"
                )
                setup_details = gr.HTML(_setup_details_html())
                setup_retry = gr.Button(
                    "Relancer la préparation",
                    variant="secondary",
                    visible=False,
                )
            gr.HTML(
                "<p class='setup-footnote'>Téléchargement unique · environ 8 Go · modèles et données conservés sur ce PC</p>"
            )

        with gr.Column(visible=setup_ready, elem_id="app-screen") as main_screen:
            batch_state = gr.State(initial_batch)
            current_metadata = gr.State(initial_metadata)
            audio_names = gr.State(initial_names)
            workspace_state = gr.State(initial_workspace)
            delete_feedback = gr.State("")
            workspace_poll_revision = gr.State("")
            workspace_timer = gr.Timer(1.0)

            navigation_state = gr.HTML(
                "<span data-navigation-state='true' data-ready='false'></span>",
                elem_id="studio-audio-navigation-state",
            )
            navigation_ready = gr.State(False)

            gr.HTML(
                """
                <header class="app-header">
                  <div class="app-header-top">
                    <div class="app-header-copy">
                      <span class="eyebrow">Workspace conversationnel privé</span>
                      <h1>Studio Audio</h1>
                      <p>Organisez vos enregistrements, suivez leur transcription et transformez chaque échange en document exploitable.</p>
                    </div>
                    <div class="header-pills" role="list" aria-label="Confidentialité et disponibilité">
                      <span class="privacy-pill" role="listitem">Données sur ce PC</span>
                      <span class="privacy-pill" role="listitem">Fonctionne hors ligne</span>
                    </div>
                  </div>
                </header>
                """
            )
            inline_rename_index = gr.Textbox(
                value="",
                container=False,
                elem_id="inline-rename-index",
                elem_classes=["rename-bridge"],
            )
            inline_rename_name = gr.Textbox(
                value="",
                container=False,
                elem_id="inline-rename-name",
                elem_classes=["rename-bridge"],
            )
            inline_rename_submit = gr.Button(
                "Appliquer le nom",
                elem_id="inline-rename-submit",
                elem_classes=["rename-bridge"],
            )
            library_open_index = gr.Textbox(
                value="",
                container=False,
                elem_id="library-open-index",
                elem_classes=["rename-bridge"],
            )
            library_open_submit = gr.Button(
                "Ouvrir la fiche",
                elem_id="library-open-submit",
                elem_classes=["rename-bridge"],
            )
            delete_audio_target = gr.Textbox(
                value="",
                container=False,
                elem_id="delete-audio-target",
                elem_classes=["rename-bridge"],
            )
            delete_audio_submit = gr.Button(
                "Confirmer la suppression",
                elem_id="delete-audio-submit",
                elem_classes=["rename-bridge"],
            )
            navigation_tab_bridge = gr.Textbox(
                value="",
                container=False,
                elem_id="navigation-tab-bridge",
                elem_classes=["rename-bridge"],
            )
            navigation_tab_submit = gr.Button(
                "Mémoriser l'onglet",
                elem_id="navigation-tab-submit",
                elem_classes=["rename-bridge"],
            )
            navigation_fiche_close_submit = gr.Button(
                "Mémoriser le retour aux fiches",
                elem_id="navigation-fiche-close-submit",
                elem_classes=["rename-bridge"],
            )

            with gr.Tabs(selected=initial_active_tab, elem_id="saas-nav") as main_tabs:
                with gr.Tab("Accueil", id="dashboard") as dashboard_tab:
                    with gr.Column(elem_classes=["page-shell"]):
                        dashboard = gr.HTML(
                            _dashboard_html(initial_workspace),
                            elem_id="dashboard-view",
                        )
                        gr.HTML(
                            """
                            <div class="settings-note">
                              <strong>Workspace persistant.</strong> La file d'attente, les résultats et les discussions sont restaurés automatiquement à la prochaine ouverture de l'application.
                            </div>
                            """
                        )

                with gr.Tab("Audios", id="audios") as audios_tab:
                    with gr.Column(elem_classes=["page-shell"]):
                        gr.HTML(
                            """
                            <section class="page-hero">
                              <div><span class="eyebrow">Traitement audio</span><h2>File de traitement</h2><p>Ajoutez les prochains enregistrements et organisez leur ordre. Une fois terminés, ils restent disponibles dans Fiches.</p></div>
                              <span class="page-hero-note">Traitement séquentiel et local</span>
                            </section>
                            """
                        )
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=3, min_width=430):
                                with gr.Group(elem_classes=["glass-card"]):
                                    gr.HTML(
                                        "<p class='step-title'>Ajouter à la file</p>"
                                        "<p class='step-copy'>Seuls les audios encore à traiter apparaissent ici.</p>"
                                    )
                                    audio_queue = gr.File(
                                        value=[str(path) for path in initial_paths],
                                        file_count="multiple",
                                        file_types=["audio"],
                                        type="filepath",
                                        label="Audios à traiter",
                                        height=258,
                                        interactive=True,
                                        allow_reordering=True,
                                        elem_id="audio-queue",
                                    )
                                    queue_hint = gr.HTML(
                                        _queue_hint(
                                            initial_paths,
                                            "File restaurée." if initial_paths else "",
                                        )
                                    )
                                    gr.HTML(
                                        "<p class='inline-rename-hint'>Double-cliquez sur le nom d'un fichier pour le renommer directement.</p>"
                                    )
                                    rename_selector = gr.Dropdown(
                                        choices=_rename_choices(initial_paths, initial_names),
                                        value=str(initial_paths[0]) if initial_paths else None,
                                        visible=False,
                                    )
                                    rename_input = gr.Textbox(visible=False)
                                    rename_button = gr.Button(visible=False)
                                    gr.Markdown(
                                        "<span class='hint'>WAV, MP3, M4A, FLAC et OGG. L'ordre affiché est celui du traitement.</span>"
                                    )

                            with gr.Column(scale=2, min_width=330):
                                with gr.Group(elem_classes=["glass-card", "launch-card"]):
                                    gr.HTML(
                                        "<p class='step-title'>Lancer le traitement</p>"
                                        "<p class='step-copy'>Les audios seront traités un par un et la transcription apparaîtra dans leur fiche.</p>"
                                    )
                                    launch_summary = gr.HTML(
                                        _run_summary_html(
                                            "Équilibré",
                                            "Français",
                                            2,
                                            initial_engine,
                                        )
                                    )
                                    transcribe_button = gr.Button(
                                        "Lancer la file d'attente",
                                        variant="primary",
                                        size="lg",
                                        elem_classes=["primary-btn"],
                                        interactive=bool(initial_paths),
                                    )
                                    job_status = gr.Markdown(
                                        _workspace_status_markdown(initial_workspace),
                                        elem_id="queue-job-status",
                                    )

                        with gr.Group(
                            elem_id="queue-active-card",
                            elem_classes=["glass-card", "settings-card"],
                        ):
                            with gr.Row(elem_classes=["library-toolbar", "queue-toolbar"]):
                                gr.HTML(
                                    "<div class='library-toolbar-copy'><span class='eyebrow'>File active</span>"
                                    "<h3>Audios à traiter</h3><p>Suivez uniquement les enregistrements en attente, en cours ou à reprendre.</p></div>"
                                )
                                gr.HTML(
                                    "<div class='queue-flow-note'><span>Ordre de traitement</span>"
                                    "<strong>Du haut vers le bas</strong></div>"
                                )
                            gr.HTML(
                                "<div id='queue-empty-state' role='status' aria-live='polite'>"
                                "<div class='queue-empty-copy'><strong>La file est à jour</strong>"
                                "<span>Aucun audio ne reste à traiter. Retrouvez les transcriptions terminées dans Fiches.</span></div></div>"
                            )
                            queue_table = gr.Dataframe(
                                headers=QUEUE_HEADERS,
                                datatype=["number", "str", "str", "str", "str", "str"],
                                value=_workspace_queue_rows(initial_workspace),
                                interactive=False,
                                wrap=False,
                                label="Audios du workspace",
                                show_label=False,
                                column_widths=[56, 280, 120, 110, 160, 48],
                                buttons=[],
                                elem_id="queue-table",
                            )

                with gr.Tab("Fiches", id="fiche") as fiche_tab:
                    with gr.Column(elem_classes=["page-shell"]):
                        fiche_library_hero = gr.HTML(
                            """
                            <section class="page-hero">
                              <div><span class="eyebrow">Dossier de travail</span><h2>Fiches audio</h2><p>Retrouvez tous vos enregistrements, triez la bibliothèque puis ouvrez la fiche dont vous avez besoin.</p></div>
                              <span class="page-hero-note">Bibliothèque locale</span>
                            </section>
                            """,
                            elem_id="fiche-library-hero",
                            visible=not initial_detail_visible,
                        )
                        with gr.Group(
                            visible=not initial_detail_visible,
                            elem_id="fiche-library-view",
                            elem_classes=["glass-card", "settings-card"],
                        ) as fiche_library_view:
                            with gr.Row(elem_classes=["library-toolbar"]):
                                gr.HTML(
                                    "<div class='library-toolbar-copy'><span class='eyebrow'>Bibliothèque des fiches</span>"
                                    "<h3>Choisir un audio</h3><p>Triez la liste puis cliquez sur une ligne pour ouvrir son dossier.</p></div>"
                                )
                                library_sort = gr.Dropdown(
                                    choices=LIBRARY_SORTS,
                                    value=LIBRARY_SORTS[0],
                                    label="Trier par",
                                    interactive=True,
                                    filterable=False,
                                    elem_id="library-sort",
                                    scale=1,
                                    min_width=260,
                                )
                            fiche_library_table = gr.Dataframe(
                                headers=LIBRARY_HEADERS,
                                datatype=["str", "str", "str", "str", "number", "str"],
                                value=_workspace_library_rows(
                                    initial_workspace,
                                    LIBRARY_SORTS[0],
                                ),
                                interactive=False,
                                wrap=False,
                                label="Fiches audio",
                                show_label=False,
                                buttons=[],
                                elem_id="fiche-library-table",
                            )
                        fiche_detail_hero = gr.HTML(
                            """
                            <section class="page-hero">
                              <div><span class="eyebrow">Dossier audio</span><h2>Fiche de consultation</h2><p>Écoutez, vérifiez les affirmations et échangez sur les passages importants depuis un seul espace.</p></div>
                              <span class="page-hero-note">Tout reste sur ce PC</span>
                            </section>
                            """,
                            elem_id="fiche-detail-hero",
                            visible=initial_detail_visible,
                        )
                        with gr.Group(
                            visible=initial_detail_visible,
                            elem_id="fiche-detail-view",
                            elem_classes=["glass-card", "workspace-shell"],
                        ) as fiche_detail_view:
                            with gr.Row(elem_classes=["fiche-detail-toolbar"]):
                                fiche_back_button = gr.Button(
                                    "Retour aux fiches",
                                    variant="secondary",
                                    elem_classes=["fiche-back-button"],
                                    scale=0,
                                )
                                fiche_selector = gr.Dropdown(
                                    choices=_workspace_choices(initial_workspace),
                                    value=initial_selected,
                                    label="Fiche ouverte",
                                    show_label=False,
                                    container=False,
                                    interactive=bool(initial_order),
                                    filterable=True,
                                    elem_id="fiche-selector",
                                    scale=1,
                                )
                            fiche_header = gr.HTML(_fiche_header_html(initial_item))
                            assistant_role_summary = gr.HTML(
                                _role_summary_html(initial_item, initial_role_proposal),
                                elem_id="assistant-role-summary",
                            )
                            assistant_role_edit = gr.Button(
                                "Ouvrir l'éditeur des rôles",
                                variant="secondary",
                                elem_classes=["role-edit-bridge"],
                                elem_id="assistant-role-edit-bridge",
                                scale=0,
                            )
                            with gr.Group(
                                visible=False,
                                elem_id="assistant-role-editor",
                                elem_classes=["role-editor"],
                            ) as assistant_role_editor:
                                gr.HTML(
                                    "<div class='role-editor-heading'>"
                                    "<div><span class='role-editor-kicker'>Attribution des voix</span>"
                                    "<h3>Attribuer chaque voix</h3></div>"
                                    "<p>Comparez les extraits, choisissez le Client et le Master, puis enregistrez. "
                                    "L'attribution s'appliquera à toute la transcription.</p>"
                                    "</div>"
                                )
                                with gr.Row(elem_classes=["role-selector-grid"]):
                                    with gr.Column(elem_classes=["role-selector-column"], min_width=0):
                                        assistant_client_role = gr.Dropdown(
                                            choices=initial_speaker_choices,
                                            value=initial_role_proposal.get("client_speaker_id"),
                                            label="Client",
                                            interactive=bool(initial_speaker_choices),
                                            filterable=False,
                                            elem_id="assistant-client-role",
                                        )
                                        assistant_client_excerpt = gr.HTML(
                                            _role_voice_excerpt_html(
                                                initial_item,
                                                initial_role_proposal.get("client_speaker_id"),
                                                "Client",
                                            ),
                                            elem_id="assistant-client-excerpt",
                                        )
                                    with gr.Column(
                                        elem_classes=["role-swap-column"],
                                        scale=0,
                                        min_width=0,
                                    ):
                                        assistant_swap_roles = gr.Button(
                                            "Inverser les rôles",
                                            variant="secondary",
                                            elem_classes=["role-action-button"],
                                            elem_id="assistant-swap-roles",
                                        )
                                    with gr.Column(elem_classes=["role-selector-column"], min_width=0):
                                        assistant_master_role = gr.Dropdown(
                                            choices=initial_speaker_choices,
                                            value=initial_role_proposal.get("master_speaker_id"),
                                            label="Master",
                                            interactive=bool(initial_speaker_choices),
                                            filterable=False,
                                            elem_id="assistant-master-role",
                                        )
                                        assistant_master_excerpt = gr.HTML(
                                            _role_voice_excerpt_html(
                                                initial_item,
                                                initial_role_proposal.get("master_speaker_id"),
                                                "Master",
                                            ),
                                            elem_id="assistant-master-excerpt",
                                        )
                                with gr.Row(elem_classes=["role-editor-actions"]):
                                    assistant_auto_roles = gr.Button(
                                        "Relancer la détection",
                                        variant="secondary",
                                        elem_classes=["role-action-button"],
                                        elem_id="assistant-auto-roles",
                                    )
                                    assistant_cancel_roles = gr.Button(
                                        "Annuler",
                                        variant="secondary",
                                        elem_classes=["role-action-button"],
                                        elem_id="assistant-cancel-roles",
                                    )
                                    assistant_confirm_roles = gr.Button(
                                        "Enregistrer les rôles",
                                        variant="primary",
                                        elem_classes=["role-action-button"],
                                        elem_id="assistant-confirm-roles",
                                    )
                            with gr.Group(elem_classes=["audio-player-card"]):
                                gr.HTML(AUDIO_TIMELINE_HTML)
                                preview = gr.Audio(
                                    value=(
                                        str(initial_item.get("path"))
                                        if initial_item
                                        and Path(str(initial_item.get("path") or "")).is_file()
                                        else None
                                    ),
                                    type="filepath",
                                    label="Écouter l'enregistrement",
                                    interactive=False,
                                    buttons=["download"],
                                    waveform_options=gr.WaveformOptions(
                                        waveform_color="#b9a58f",
                                        waveform_progress_color="#65584c",
                                        trim_region_color="#e7dccd",
                                        skip_length=1,
                                    ),
                                    elem_id="fiche-audio-player",
                                )

                            with gr.Row(elem_classes=["fiche-context-nav"]):
                                gr.HTML(
                                    "<div class='fiche-context-nav-copy'><strong>Espace de travail</strong>"
                                    "<span>La sélection reste disponible lorsque vous naviguez entre les onglets.</span></div>"
                                )
                                fiche_context_mode = gr.Radio(
                                    choices=["Transcription", "Assistant IA", "Discussions"],
                                    value=initial_context_mode,
                                    show_label=False,
                                    container=False,
                                    elem_id="fiche-context-mode",
                                )

                            with gr.Row(equal_height=False, elem_classes=["fiche-workbench"]):
                                with gr.Column(
                                    scale=3,
                                    min_width=430,
                                    elem_classes=["transcript-pane"],
                                ):
                                    gr.HTML(
                                        "<div class='pane-heading'><span class='eyebrow'>Contenu</span>"
                                        "<h3>Transcription</h3><p id='transcript-selection-help'>Sélectionnez une phrase pour conserver son timecode et ouvrir une discussion.</p></div>"
                                    )
                                    fiche_transcript = gr.Textbox(
                                        value=_item_transcript(initial_item),
                                        label="Transcription de la fiche",
                                        show_label=False,
                                        lines=24,
                                        buttons=["copy"],
                                        interactive=True,
                                        elem_id="fiche-transcript",
                                    )

                                with gr.Column(
                                    scale=2,
                                    min_width=360,
                                    elem_classes=["context-rail"],
                                ):
                                    with gr.Group(elem_classes=["shared-context-shell"]):
                                        comment_selection_context = gr.HTML(
                                            _selection_context_html(),
                                            elem_id="comment-selection-context",
                                        )

                                    with gr.Group(
                                        visible=initial_context_mode == "Assistant IA",
                                        elem_id="assistant-panel",
                                        elem_classes=["assistant-panel"],
                                    ) as assistant_panel:
                                        gr.HTML(
                                            "<section class='assistant-intro' aria-labelledby='assistant-verification-title'><div>"
                                            "<div class='assistant-intro-title'><span class='assistant-step'>Assistant audio</span>"
                                            "<h3 id='assistant-verification-title'>Interroger la consultation</h3></div>"
                                            "<span class='assistant-local-badge'>Analyse locale</span></div>"
                                            "<p>Demandez une synthèse, recherchez un thème ou vérifiez précisément ce qui a été dit. "
                                            "Chaque réponse reste reliée aux passages et timecodes de l'audio.</p></section>"
                                        )
                                        assistant_question = gr.Textbox(
                                            label="Votre question",
                                            placeholder="Ex. De quoi parle principalement cette consultation ?",
                                            lines=3,
                                            elem_id="assistant-question",
                                        )
                                        gr.HTML(
                                            "<p class='assistant-form-hint'>Exemples : « Est-ce que ça parle d'amour ? » "
                                            "« Le ton monte-t-il durant l'échange ? » ou "
                                            "« Le Master a-t-il annoncé une séparation ? »</p>"
                                        )
                                        with gr.Row(elem_classes=["assistant-actions"]):
                                            assistant_submit = gr.Button(
                                                "Analyser l'audio",
                                                variant="primary",
                                                interactive=bool(_item_turns(initial_item)),
                                                elem_id="assistant-submit",
                                                elem_classes=["comment-primary"],
                                            )
                                            assistant_clear = gr.Button(
                                                "Réinitialiser",
                                                variant="secondary",
                                                elem_id="assistant-reset",
                                            )
                                        assistant_result = gr.HTML(
                                            _assistant_result_html(initial_assistant_result, initial_item)
                                            + _assistant_history_html(initial_item),
                                            elem_id="assistant-result",
                                            js_on_load=ASSISTANT_RESULT_JS,
                                        )

                                    with gr.Group(
                                        visible=initial_context_mode == "Discussions",
                                        elem_id="discussion-panel",
                                        elem_classes=["discussion-panel"],
                                    ) as discussion_panel:
                                        gr.HTML(
                                            "<div class='discussion-toolbar'><div><span class='eyebrow'>Collaboration</span>"
                                            "<strong>Discussions liées aux passages</strong></div>"
                                            "<span>Ancrées aux timecodes</span></div>"
                                        )
                                        comments_table = gr.HTML(
                                            value=_comments_feed_html(initial_item),
                                            elem_id="comments-feed",
                                            js_on_load=COMMENTS_FEED_JS,
                                        )
                                        comment_reply_target = gr.State("")
                                        with gr.Column(elem_classes=["comment-reply-composer"]):
                                            comment_thread_context = gr.HTML(
                                                _thread_context_html(initial_item, None)
                                            )
                                            comment_reply_body = gr.Textbox(
                                                label="Votre réponse",
                                                placeholder="Répondez à cette discussion…",
                                                lines=3,
                                                elem_id="comment-reply",
                                            )
                                            comment_reply_button = gr.Button(
                                                "Répondre",
                                                variant="secondary",
                                                interactive=False,
                                                elem_id="comment-reply-submit",
                                            )
                                        with gr.Group(
                                            elem_id="comment-composer",
                                            elem_classes=["comment-composer", "is-compact"],
                                        ):
                                            comment_body = gr.Textbox(
                                                label="Votre message",
                                                placeholder="Ajoutez une question, une décision ou une correction…",
                                                lines=4,
                                                elem_id="comment-body",
                                            )
                                            with gr.Accordion("Ajuster l'extrait sélectionné", open=False):
                                                comment_passage = gr.Textbox(
                                                    label="Passage concerné",
                                                    placeholder="Sélectionnez du texte à gauche ou collez un extrait ici.",
                                                    lines=3,
                                                )
                                            with gr.Accordion(
                                                "Identité de contribution",
                                                open=False,
                                                elem_classes=["comment-identity"],
                                            ):
                                                comment_author = gr.Textbox(
                                                    value="Moi",
                                                    label="Votre nom",
                                                    placeholder="Ex. Camille",
                                                    elem_id="comment-author",
                                                )
                                            comment_anchor = gr.State("")
                                            comment_draft_item = gr.State("")
                                            comment_editing_id = gr.State("")
                                            with gr.Row(elem_classes=["comment-composer-actions"]):
                                                comment_add_button = gr.Button(
                                                    "Créer la discussion",
                                                    variant="primary",
                                                    interactive=initial_comment_enabled,
                                                    elem_id="comment-submit",
                                                    elem_classes=["comment-primary"],
                                                    scale=3,
                                                )
                                                comment_cancel_button = gr.Button(
                                                    "Effacer",
                                                    variant="secondary",
                                                    scale=1,
                                                )
                                        comment_selector = gr.Dropdown(
                                            choices=initial_comment_choices,
                                            value=initial_comment,
                                            label="Discussion sélectionnée",
                                            interactive=bool(initial_comment_choices),
                                            visible="hidden",
                                            elem_id="comment-selector",
                                        )
                                        comment_resolve_button = gr.Button(
                                            _comment_action_label(initial_item, initial_comment),
                                            interactive=bool(initial_comment_choices),
                                            visible="hidden",
                                        )

                with gr.Tab("Rapports", id="reports") as reports_tab:
                    with gr.Column(elem_classes=["page-shell"]):
                        gr.HTML(
                            """
                            <section class="page-hero">
                              <div><span class="eyebrow">Livrables</span><h2>Rapports et exports</h2><p>Corrigez les tours de parole puis récupérez les formats prêts à partager avec vos collègues.</p></div>
                              <span class="page-hero-note">TXT · DOCX · JSON · SRT · ZIP</span>
                            </section>
                            """
                        )
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=3, min_width=520):
                                with gr.Group(elem_classes=["glass-card", "settings-card"]):
                                    result_selector = gr.Dropdown(
                                        choices=initial_batch_order,
                                        value=initial_result_label,
                                        label="Audio à relire",
                                        info="Chaque audio terminé reste disponible ici.",
                                        interactive=bool(initial_batch_order),
                                    )
                                    gr.HTML(
                                        "<div class='report-editor-intro'><div><strong>Éditeur de transcription</strong>"
                                        "<span>Chargez l'éditeur uniquement lorsque vous souhaitez corriger les prises de parole.</span>"
                                        "</div><span class='report-editor-status'>Chargement à la demande</span></div>"
                                    )
                                    report_editor_reveal = gr.Button(
                                        "Ouvrir l'éditeur de transcription",
                                        variant="secondary",
                                        elem_classes=["report-editor-launch"],
                                    )
                                    table = gr.Dataframe(
                                        headers=["Début", "Fin", "Interlocuteur", "Texte"],
                                        datatype=["str", "str", "str", "str"],
                                        value=initial_advanced_rows,
                                        row_count=(1, "dynamic"),
                                        column_count=4,
                                        interactive=True,
                                        visible=False,
                                        max_height=460,
                                        wrap=False,
                                        column_widths=[92, 92, 145, 620],
                                        label="Corriger la transcription",
                                    )
                                    export_button = gr.Button(
                                        "Exporter les corrections",
                                        variant="primary",
                                        elem_classes=["primary-btn"],
                                    )
                                    report_status = gr.Markdown(
                                        "Les corrections exportées sont ajoutées aux fichiers disponibles.",
                                        elem_classes=["hint"],
                                    )
                            with gr.Column(scale=2, min_width=330):
                                with gr.Group(elem_classes=["glass-card", "settings-card"]):
                                    gr.HTML(
                                        "<p class='step-title'>Fichiers prêts à partager</p>"
                                        "<p class='step-copy'>Le ZIP global rassemble un dossier complet par audio.</p>"
                                    )
                                    downloads = gr.File(
                                        value=initial_downloads,
                                        label="Téléchargements",
                                        file_count="multiple",
                                        interactive=False,
                                        elem_classes=["report-downloads"],
                                    )
                        transcript = gr.Textbox(
                            value=_item_transcript(initial_item),
                            visible=False,
                        )

                with gr.Tab("Paramètres", id="settings") as settings_tab:
                    with gr.Column(elem_classes=["page-shell"]):
                        gr.HTML(
                            """
                            <section class="page-hero">
                              <div><span class="eyebrow">Configuration</span><h2>Paramètres de transcription</h2><p>Choisissez le compromis qualité-vitesse, la langue et les informations qui aident le moteur à mieux structurer la conversation.</p></div>
                              <span class="page-hero-note">Réglages appliqués au prochain lancement</span>
                            </section>
                            """
                        )
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=3, min_width=430):
                                with gr.Group(elem_classes=["glass-card", "settings-card"]):
                                    gr.HTML(
                                        "<p class='step-title'>Moteur de transcription</p>"
                                        "<p class='step-copy'>Le mode Équilibré est recommandé pour les conversations téléphoniques.</p>"
                                    )
                                    profile = gr.Radio(
                                        choices=list(PROFILE_CONFIG),
                                        value="Équilibré",
                                        label="Niveau de qualité",
                                        elem_classes=["quality-selector"],
                                    )
                                    gr.HTML(_engine_settings_copy())
                                    engine = gr.Radio(
                                        choices=ENGINE_CHOICES,
                                        value=initial_engine,
                                        label="Moteur d'exécution",
                                        elem_classes=["engine-selector"],
                                    )
                                    engine_status = gr.HTML(
                                        _engine_status_html(initial_engine, "Équilibré")
                                    )
                                    engine_refresh = gr.Button(
                                        "Actualiser la détection",
                                        variant="secondary",
                                        size="sm",
                                        elem_classes=["engine-refresh"],
                                        visible=openvino_arc_supported(),
                                    )
                                    gr.Markdown(
                                        "<span class='hint'>Ce réglage ne télécharge aucun modèle. Il est mémorisé uniquement sur ce PC.</span>"
                                    )
                                    language = gr.Dropdown(
                                        choices=list(LANGUAGES),
                                        value="Français",
                                        label="Langue parlée",
                                        interactive=True,
                                        filterable=False,
                                        elem_id="language-choice",
                                    )
                                    requested_speakers = gr.Slider(
                                        minimum=1,
                                        maximum=8,
                                        step=1,
                                        value=2,
                                        label="Nombre d'interlocuteurs",
                                        info="Un nombre connu améliore la séparation des voix.",
                                    )
                            with gr.Column(scale=2, min_width=350):
                                with gr.Group(elem_classes=["glass-card", "settings-card"]):
                                    gr.HTML(
                                        "<p class='step-title'>Voix et vocabulaire</p>"
                                        "<p class='step-copy'>Ces indices améliorent les noms des locuteurs et les termes spécifiques.</p>"
                                    )
                                    speaker_1 = gr.Textbox(
                                        value="Interlocuteur 1",
                                        label="Première voix entendue",
                                    )
                                    speaker_2 = gr.Textbox(
                                        value="Interlocuteur 2",
                                        label="Deuxième voix entendue",
                                    )
                                    vocabulary = gr.Textbox(
                                        label="Noms propres ou mots métier",
                                        placeholder="Ex. Dupont, Qualiopi, référence ZX-418…",
                                        info="Quelques termes difficiles aident le moteur à mieux les orthographier.",
                                        lines=3,
                                    )
                        gr.HTML(
                            """
                            <div class="settings-note">
                              <strong>Confidentialité locale.</strong> Les modèles, les copies audio, les transcriptions et les commentaires sont stockés uniquement dans le dossier de l'application. Aucun fichier n'est envoyé vers un service cloud.
                            </div>
                            """
                        )

            navigation_tab_submit.click(
                fn=persist_active_tab_ui,
                inputs=[navigation_tab_bridge],
                queue=False,
            )
            navigation_fiche_close_submit.click(
                fn=persist_fiche_library_ui,
                queue=False,
            )
            dashboard_tab.select(
                fn=lambda ready: persist_active_tab_when_ready_ui("dashboard", ready),
                inputs=[navigation_ready],
                queue=False,
            )
            audios_tab.select(
                fn=lambda ready: persist_active_tab_when_ready_ui("audios", ready),
                inputs=[navigation_ready],
                queue=False,
            )
            fiche_tab.select(
                fn=lambda ready: persist_active_tab_when_ready_ui("fiche", ready),
                inputs=[navigation_ready],
                queue=False,
            )
            reports_tab.select(
                fn=lambda ready: persist_active_tab_when_ready_ui("reports", ready),
                inputs=[navigation_ready],
                queue=False,
            )
            settings_tab.select(
                fn=lambda ready: persist_active_tab_when_ready_ui("settings", ready),
                inputs=[navigation_ready],
                queue=False,
            )
            report_editor_reveal.click(
                fn=lambda: (
                    gr.update(visible=False),
                    gr.update(visible=True),
                ),
                outputs=[report_editor_reveal, table],
                queue=False,
            )

            def collapse_report_editor_ui():
                """Keep every return to Reports lightweight after editing."""
                return gr.update(visible=True), gr.update(visible=False)

            for destination_tab in (
                dashboard_tab,
                audios_tab,
                fiche_tab,
                settings_tab,
            ):
                destination_tab.select(
                    fn=collapse_report_editor_ui,
                    outputs=[report_editor_reveal, table],
                    queue=False,
                )

            queue_change = audio_queue.change(
                fn=sync_queue_ui,
                inputs=[audio_queue, audio_names],
                outputs=[
                    preview,
                    queue_table,
                    queue_hint,
                    audio_names,
                    rename_selector,
                    rename_input,
                    rename_button,
                    transcribe_button,
                ],
                queue=False,
                show_progress="hidden",
                show_progress_on=[],
            )
            queue_change.then(
                fn=sync_workspace_ui,
                inputs=[audio_queue, audio_names, workspace_state],
                outputs=[
                    workspace_state,
                    queue_table,
                    fiche_selector,
                    fiche_header,
                    fiche_transcript,
                    comments_table,
                    comment_selector,
                    comment_add_button,
                    comment_resolve_button,
                    dashboard,
                ],
                queue=False,
                show_progress="hidden",
                show_progress_on=[],
            )
            inline_rename = inline_rename_submit.click(
                fn=rename_audio_by_target_ui,
                inputs=[
                    audio_queue,
                    inline_rename_index,
                    inline_rename_name,
                    audio_names,
                    workspace_state,
                    library_sort,
                ],
                outputs=[
                    audio_names,
                    queue_table,
                    queue_hint,
                    rename_selector,
                    rename_input,
                    rename_button,
                ],
                queue=False,
            )
            inline_rename_sync = inline_rename.then(
                fn=sync_workspace_names_ui,
                inputs=[audio_queue, audio_names, workspace_state, fiche_selector],
                outputs=[
                    audio_queue,
                    audio_names,
                    workspace_state,
                    queue_table,
                    fiche_selector,
                    fiche_header,
                    fiche_transcript,
                    comments_table,
                    comment_selector,
                    comment_add_button,
                    comment_resolve_button,
                    dashboard,
                ],
                queue=False,
            )
            inline_rename_sync.then(
                fn=sync_library_ui,
                inputs=[workspace_state, library_sort],
                outputs=[fiche_library_table],
                queue=False,
            )
            library_sort.change(
                fn=sync_library_ui,
                inputs=[workspace_state, library_sort],
                outputs=[fiche_library_table],
                queue=False,
            )
            workspace_state.change(
                fn=sync_library_ui,
                inputs=[workspace_state, library_sort],
                outputs=[fiche_library_table],
                queue=False,
                show_progress="hidden",
            )
            workspace_state.change(
                fn=load_assistant_ui,
                inputs=[fiche_selector, workspace_state],
                outputs=[
                    assistant_role_summary,
                    assistant_client_role,
                    assistant_master_role,
                    assistant_client_excerpt,
                    assistant_master_excerpt,
                    assistant_result,
                    assistant_submit,
                ],
                queue=False,
                show_progress="hidden",
            )
            library_open_action = library_open_submit.click(
                fn=open_library_fiche_by_index_ui,
                inputs=[library_open_index, workspace_state, library_sort],
                outputs=[
                    fiche_library_hero,
                    fiche_library_view,
                    fiche_detail_hero,
                    fiche_detail_view,
                    fiche_selector,
                    fiche_header,
                    preview,
                    fiche_transcript,
                    comments_table,
                    comment_selector,
                    comment_add_button,
                    comment_resolve_button,
                    table,
                    current_metadata,
                    downloads,
                ],
                queue=False,
                scroll_to_output=True,
                js=OPEN_FICHE_JS,
            )
            library_open_action.then(
                fn=restore_ui_navigation_state_ui,
                outputs=[navigation_state],
                queue=False,
                show_progress="hidden",
            )
            delete_action = delete_audio_submit.click(
                fn=delete_audio_by_target_ui,
                inputs=[delete_audio_target, library_sort, workspace_state],
                outputs=[delete_feedback],
                queue=False,
            )
            delete_refresh = delete_action.then(
                fn=restore_workspace_ui,
                outputs=[
                    audio_queue,
                    preview,
                    queue_table,
                    queue_hint,
                    audio_names,
                    rename_selector,
                    rename_input,
                    rename_button,
                    workspace_state,
                    fiche_selector,
                    fiche_header,
                    fiche_transcript,
                    comments_table,
                    comment_selector,
                    comment_add_button,
                    comment_resolve_button,
                    result_selector,
                    table,
                    transcript,
                    downloads,
                    batch_state,
                    current_metadata,
                    job_status,
                    dashboard,
                    transcribe_button,
                ],
                queue=False,
            )
            delete_refresh.then(
                fn=lambda message: message,
                inputs=[delete_feedback],
                outputs=[job_status],
                queue=False,
            ).then(
                fn=sync_library_ui,
                inputs=[workspace_state, library_sort],
                outputs=[fiche_library_table],
                queue=False,
            )
            library_select_action = fiche_library_table.select(
                fn=open_library_fiche_ui,
                inputs=[workspace_state, library_sort],
                outputs=[
                    fiche_library_hero,
                    fiche_library_view,
                    fiche_detail_hero,
                    fiche_detail_view,
                    fiche_selector,
                    fiche_header,
                    preview,
                    fiche_transcript,
                    comments_table,
                    comment_selector,
                    comment_add_button,
                    comment_resolve_button,
                    table,
                    current_metadata,
                    downloads,
                ],
                queue=False,
                scroll_to_output=True,
                js=OPEN_FICHE_JS,
            )
            library_select_action.then(
                fn=restore_ui_navigation_state_ui,
                outputs=[navigation_state],
                queue=False,
                show_progress="hidden",
            )
            fiche_close_action = fiche_back_button.click(
                fn=close_fiche_ui,
                outputs=[
                    fiche_library_hero,
                    fiche_library_view,
                    fiche_detail_hero,
                    fiche_detail_view,
                ],
                js=CLOSE_FICHE_JS,
                queue=False,
            )
            fiche_close_action.then(
                fn=persist_fiche_library_ui,
                queue=False,
            ).then(
                fn=restore_ui_navigation_state_ui,
                outputs=[navigation_state],
                queue=False,
                show_progress="hidden",
            )
            fiche_selector.input(
                fn=load_fiche_ui,
                inputs=[fiche_selector, workspace_state],
                outputs=[
                    fiche_header,
                    preview,
                    fiche_transcript,
                    comments_table,
                    comment_selector,
                    comment_add_button,
                    comment_resolve_button,
                    table,
                    current_metadata,
                    downloads,
                ],
                queue=False,
            )
            queue_table.select(
                fn=select_queue_fiche_ui,
                inputs=[workspace_state],
                outputs=[
                    fiche_selector,
                    fiche_header,
                    preview,
                    fiche_transcript,
                    comments_table,
                    comment_selector,
                    comment_add_button,
                    comment_resolve_button,
                    table,
                    current_metadata,
                    downloads,
                ],
                queue=False,
            )
            fiche_transcript.select(
                fn=capture_selected_passage_ui,
                inputs=[fiche_selector, fiche_transcript, workspace_state],
                outputs=[
                    comment_passage,
                    comment_anchor,
                    comment_selection_context,
                    comment_draft_item,
                ],
                queue=False,
            )
            fiche_context_change = fiche_context_mode.change(
                fn=switch_fiche_context_when_ready_ui,
                inputs=[fiche_context_mode, navigation_ready],
                outputs=[assistant_panel, discussion_panel],
                queue=False,
            )
            fiche_context_change.then(
                fn=restore_ui_navigation_state_ui,
                outputs=[navigation_state],
                queue=False,
                show_progress="hidden",
            )
            assistant_auto_roles.click(
                fn=auto_detect_roles_ui,
                inputs=[fiche_selector, workspace_state],
                outputs=[
                    assistant_client_role,
                    assistant_master_role,
                    assistant_client_excerpt,
                    assistant_master_excerpt,
                ],
                queue=False,
            )
            assistant_role_edit.click(
                fn=open_role_editor_ui,
                inputs=[fiche_selector, workspace_state],
                outputs=[
                    assistant_role_editor,
                    assistant_client_role,
                    assistant_master_role,
                    assistant_client_excerpt,
                    assistant_master_excerpt,
                ],
                queue=False,
            )
            assistant_cancel_roles.click(
                fn=cancel_role_editor_ui,
                inputs=[fiche_selector, workspace_state],
                outputs=[
                    assistant_role_editor,
                    assistant_client_role,
                    assistant_master_role,
                    assistant_client_excerpt,
                    assistant_master_excerpt,
                ],
                queue=False,
            )
            assistant_client_role.input(
                fn=role_voice_excerpts_ui,
                inputs=[
                    fiche_selector,
                    assistant_client_role,
                    assistant_master_role,
                    workspace_state,
                ],
                outputs=[assistant_client_excerpt, assistant_master_excerpt],
                queue=False,
            )
            assistant_master_role.input(
                fn=role_voice_excerpts_ui,
                inputs=[
                    fiche_selector,
                    assistant_client_role,
                    assistant_master_role,
                    workspace_state,
                ],
                outputs=[assistant_client_excerpt, assistant_master_excerpt],
                queue=False,
            )
            save_roles = assistant_confirm_roles.click(
                fn=confirm_roles_ui,
                inputs=[
                    fiche_selector,
                    assistant_client_role,
                    assistant_master_role,
                    workspace_state,
                ],
                outputs=[workspace_state, assistant_role_summary, fiche_transcript],
                queue=False,
            )
            save_roles.then(
                fn=close_role_editor_ui,
                outputs=[assistant_role_editor],
                queue=False,
            )
            assistant_swap_roles.click(
                fn=swap_roles_ui,
                inputs=[
                    fiche_selector,
                    assistant_client_role,
                    assistant_master_role,
                    workspace_state,
                ],
                outputs=[
                    assistant_client_role,
                    assistant_master_role,
                    assistant_client_excerpt,
                    assistant_master_excerpt,
                ],
                queue=False,
            )
            assistant_run = assistant_submit.click(
                fn=run_assistant_verification_ui,
                inputs=[
                    fiche_selector,
                    assistant_question,
                    assistant_client_role,
                    assistant_master_role,
                    workspace_state,
                ],
                outputs=[
                    workspace_state,
                    assistant_role_summary,
                    assistant_result,
                    fiche_transcript,
                ],
                show_progress="minimal",
                concurrency_id="assistant-analysis",
                concurrency_limit=1,
            )
            assistant_question.submit(
                fn=run_assistant_verification_ui,
                inputs=[
                    fiche_selector,
                    assistant_question,
                    assistant_client_role,
                    assistant_master_role,
                    workspace_state,
                ],
                outputs=[
                    workspace_state,
                    assistant_role_summary,
                    assistant_result,
                    fiche_transcript,
                ],
                show_progress="minimal",
                concurrency_id="assistant-analysis",
                concurrency_limit=1,
            )
            assistant_clear.click(
                fn=clear_assistant_ui,
                inputs=[fiche_selector, workspace_state],
                outputs=[assistant_question, assistant_result],
                queue=False,
            )
            assistant_result.click(
                fn=prepare_discussion_from_assistant_ui,
                inputs=[fiche_selector],
                outputs=[
                    comment_passage,
                    comment_anchor,
                    comment_selection_context,
                    comment_draft_item,
                    comment_editing_id,
                    comment_body,
                    comment_add_button,
                ],
                queue=False,
            )
            comment_add_button.click(
                fn=save_comment_ui,
                inputs=[
                    fiche_selector,
                    comment_author,
                    comment_passage,
                    comment_body,
                    comment_anchor,
                    comment_draft_item,
                    comment_editing_id,
                    workspace_state,
                ],
                outputs=[
                    workspace_state,
                    comments_table,
                    comment_selector,
                    fiche_header,
                    comment_passage,
                    comment_body,
                    comment_resolve_button,
                    dashboard,
                    comment_anchor,
                    comment_draft_item,
                    comment_editing_id,
                    comment_selection_context,
                    comment_add_button,
                    comment_reply_target,
                    comment_thread_context,
                    comment_reply_button,
                ],
                queue=False,
            )
            comment_cancel_button.click(
                fn=reset_comment_composer_ui,
                inputs=[fiche_selector],
                outputs=[
                    comment_passage,
                    comment_body,
                    comment_anchor,
                    comment_draft_item,
                    comment_editing_id,
                    comment_selection_context,
                    comment_add_button,
                    comment_selector,
                    comment_reply_target,
                    comment_thread_context,
                    comment_reply_body,
                    comment_reply_button,
                ],
                queue=False,
            )
            comments_table.click(
                fn=handle_comment_feed_action_ui,
                inputs=[
                    fiche_selector,
                    workspace_state,
                    comment_passage,
                    comment_body,
                    comment_anchor,
                    comment_draft_item,
                    comment_editing_id,
                    comment_reply_target,
                    comment_reply_body,
                ],
                outputs=[
                    workspace_state,
                    comments_table,
                    comment_selector,
                    fiche_header,
                    comment_resolve_button,
                    dashboard,
                    comment_passage,
                    comment_body,
                    comment_anchor,
                    comment_draft_item,
                    comment_editing_id,
                    comment_selection_context,
                    comment_add_button,
                    comment_reply_target,
                    comment_thread_context,
                    comment_reply_body,
                    comment_reply_button,
                ],
                queue=False,
            )
            comment_reply_button.click(
                fn=add_reply_ui,
                inputs=[
                    fiche_selector,
                    comment_reply_target,
                    comment_author,
                    comment_reply_body,
                    workspace_state,
                ],
                outputs=[
                    workspace_state,
                    comments_table,
                    fiche_header,
                    dashboard,
                    comment_thread_context,
                    comment_reply_body,
                    comment_reply_button,
                ],
                queue=False,
            )
            fiche_selector.change(
                fn=reset_comment_composer_ui,
                inputs=[fiche_selector],
                outputs=[
                    comment_passage,
                    comment_body,
                    comment_anchor,
                    comment_draft_item,
                    comment_editing_id,
                    comment_selection_context,
                    comment_add_button,
                    comment_selector,
                    comment_reply_target,
                    comment_thread_context,
                    comment_reply_body,
                    comment_reply_button,
                ],
                queue=False,
            )
            fiche_selector.change(
                fn=load_assistant_ui,
                inputs=[fiche_selector, workspace_state],
                outputs=[
                    assistant_role_summary,
                    assistant_client_role,
                    assistant_master_role,
                    assistant_client_excerpt,
                    assistant_master_excerpt,
                    assistant_result,
                    assistant_submit,
                ],
                queue=False,
            )
            comment_selector.input(
                fn=load_comment_action_ui,
                inputs=[fiche_selector, comment_selector, workspace_state],
                outputs=[comment_resolve_button],
                queue=False,
            )
            comment_resolve_button.click(
                fn=toggle_comment_ui,
                inputs=[fiche_selector, comment_selector, workspace_state],
                outputs=[
                    workspace_state,
                    comments_table,
                    fiche_header,
                    comment_resolve_button,
                    dashboard,
                ],
                queue=False,
            )
            transcribe_button.click(
                fn=transcribe_ui,
                inputs=[
                    audio_queue,
                    profile,
                    language,
                    requested_speakers,
                    speaker_1,
                    speaker_2,
                    vocabulary,
                    audio_names,
                    workspace_state,
                    engine,
                ],
                outputs=[
                    job_status,
                    queue_table,
                    result_selector,
                    table,
                    transcript,
                    downloads,
                    batch_state,
                    current_metadata,
                    workspace_state,
                    fiche_selector,
                    fiche_header,
                    fiche_transcript,
                    comments_table,
                    comment_selector,
                    comment_add_button,
                    comment_resolve_button,
                    dashboard,
                ],
                show_progress="minimal",
                concurrency_id="transcription",
                concurrency_limit=1,
            )
            result_selector.input(
                fn=load_selected_result_ui,
                inputs=[result_selector, batch_state],
                outputs=[table, transcript, current_metadata, downloads],
                show_progress="hidden",
            )
            export_button.click(
                fn=export_corrections_ui,
                inputs=[table, current_metadata],
                outputs=[transcript, downloads, report_status],
                show_progress="minimal",
            )
            for settings_component in (profile, language, requested_speakers, engine):
                settings_component.change(
                    fn=_run_summary_html,
                    inputs=[profile, language, requested_speakers, engine],
                    outputs=[launch_summary],
                    queue=False,
                )

            engine.change(
                fn=save_engine_preference_ui,
                inputs=[engine, profile],
                outputs=[workspace_state, engine_status],
                queue=False,
            )
            profile.change(
                fn=_engine_status_html,
                inputs=[engine, profile],
                outputs=[engine_status],
                queue=False,
            )
            engine_refresh.click(
                fn=_engine_status_html,
                inputs=[engine, profile],
                outputs=[engine_status],
                queue=False,
            )

            workspace_timer.tick(
                fn=poll_workspace_ui,
                inputs=[
                    fiche_selector,
                    library_sort,
                    workspace_poll_revision,
                    comment_selector,
                    comment_reply_target,
                ],
                outputs=[
                    workspace_poll_revision,
                    workspace_state,
                    queue_table,
                    fiche_library_table,
                    fiche_header,
                    fiche_transcript,
                    comments_table,
                    result_selector,
                    table,
                    transcript,
                    downloads,
                    batch_state,
                    current_metadata,
                    job_status,
                    dashboard,
                    comment_selector,
                    comment_reply_target,
                    comment_thread_context,
                    comment_reply_button,
                    comment_resolve_button,
                    comment_reply_body,
                    assistant_role_summary,
                    assistant_client_role,
                    assistant_master_role,
                    assistant_client_excerpt,
                    assistant_master_excerpt,
                    assistant_result,
                    assistant_submit,
                ],
                queue=False,
                trigger_mode="always_last",
                show_progress="hidden",
                show_progress_on=[],
            )
            workspace_timer.tick(
                fn=poll_workspace_queue_ui,
                inputs=[audio_queue],
                outputs=[audio_queue, queue_hint, transcribe_button],
                queue=False,
                trigger_mode="always_last",
                show_progress="hidden",
                show_progress_on=[],
            )

        setup_outputs = [
            setup_progress,
            setup_status,
            setup_details,
            setup_screen,
            main_screen,
            setup_retry,
        ]
        app.load(
            fn=restore_ui_navigation_state_ui,
            outputs=[navigation_state],
            show_progress="hidden",
        )
        app.load(
            fn=None,
            js=INLINE_RENAME_JS,
            queue=False,
        )
        app.load(
            fn=None,
            js=OVERFLOW_TOOLTIP_JS,
            queue=False,
        )
        app.load(
            fn=None,
            js=DELETE_ACTION_JS,
            queue=False,
        )
        app.load(
            fn=None,
            js=TIMELINE_INIT_JS,
            queue=False,
        )
        app.load(
            fn=None,
            js=TRANSCRIPT_DISCUSSION_JS,
            queue=False,
        )
        app.load(
            fn=None,
            js=DESKTOP_VISIBILITY_SYNC_JS,
            queue=False,
        )
        app.load(
            fn=None,
            js=PROCESSING_CLOCK_JS,
            queue=False,
        )
        app.load(
            fn=None,
            js=FICHE_CONTEXT_JS,
            queue=False,
        )
        app.load(
            fn=None,
            js=ROLE_EDITOR_JS,
            queue=False,
        )
        app.load(
            fn=prepare_models_ui,
            outputs=setup_outputs,
            show_progress="hidden",
        )
        app.load(
            fn=restore_workspace_ui,
            outputs=[
                audio_queue,
                preview,
                queue_table,
                queue_hint,
                audio_names,
                rename_selector,
                rename_input,
                rename_button,
                workspace_state,
                fiche_selector,
                fiche_header,
                fiche_transcript,
                comments_table,
                comment_selector,
                comment_add_button,
                comment_resolve_button,
                result_selector,
                table,
                transcript,
                downloads,
                batch_state,
                current_metadata,
                job_status,
                dashboard,
                transcribe_button,
            ],
            show_progress="hidden",
        )
        navigation_restore = app.load(
            fn=restore_navigation_components_ui,
            outputs=[
                main_tabs,
                fiche_library_hero,
                fiche_library_view,
                fiche_detail_hero,
                fiche_detail_view,
                fiche_context_mode,
                assistant_panel,
                discussion_panel,
            ],
            show_progress="hidden",
        )
        navigation_restore.then(
            fn=lambda: True,
            outputs=[navigation_ready],
            queue=False,
            show_progress="hidden",
        )
        setup_retry.click(
            fn=prepare_models_ui,
            outputs=setup_outputs,
            show_progress="minimal",
        )

    return app


def launch(*, host: str = "127.0.0.1", port: int = 7860, inbrowser: bool = True) -> None:
    app = build_app()
    app.queue(default_concurrency_limit=1, max_size=8)
    app.launch(
        server_name=host,
        server_port=port,
        inbrowser=inbrowser,
        share=False,
        allowed_paths=[
            str(RESULTS_DIR.resolve()),
            str(DATA_DIR.resolve()),
            str(ASSETS_DIR.resolve()),
        ],
        show_error=True,
        css=CUSTOM_CSS,
        footer_links=[],
    )
