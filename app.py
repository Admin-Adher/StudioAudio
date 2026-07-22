"""Point d'entrée de l'application de transcription locale."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

from transcription_locale.ui import launch


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcription audio locale avec séparation des voix")
    parser.add_argument("--host", default="127.0.0.1", help="Adresse d'écoute")
    parser.add_argument("--port", default=7860, type=int, help="Port de l'interface")
    parser.add_argument("--sans-navigateur", action="store_true", help="Ne pas ouvrir le navigateur")
    args = parser.parse_args()
    launch(host=args.host, port=args.port, inbrowser=not args.sans_navigateur)


if __name__ == "__main__":
    main()
