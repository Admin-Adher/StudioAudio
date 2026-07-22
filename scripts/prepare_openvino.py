"""Télécharge et vérifie explicitement le modèle du pilote OpenVINO."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY = "OpenVINO/whisper-large-v3-turbo-int8-ov"
REVISION = "4929ae83ea2d1df59f4b5898a9aab8aa1c29e711"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "modeles" / "openvino" / "whisper-large-v3-turbo-int8-ov"
MARKER = MODEL_DIR / "installation-locale.json"
REQUIRED_FILES = (
    "openvino_encoder_model.xml",
    "openvino_encoder_model.bin",
    "openvino_decoder_model.xml",
    "openvino_decoder_model.bin",
    "openvino_tokenizer.xml",
    "openvino_detokenizer.xml",
)


def main() -> int:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")

    from huggingface_hub import snapshot_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPOSITORY,
        revision=REVISION,
        local_dir=MODEL_DIR,
    )

    missing = [name for name in REQUIRED_FILES if not (MODEL_DIR / name).is_file()]
    if missing:
        raise RuntimeError("Fichiers OpenVINO manquants : " + ", ".join(missing))

    marker = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "installed_at": datetime.now(timezone.utc).isoformat(),
        "required_files": list(REQUIRED_FILES),
    }
    MARKER.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Modèle vérifié : {MODEL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
