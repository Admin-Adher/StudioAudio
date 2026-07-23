"""Télécharge le modèle local de l'Assistant Premium avec reprise automatique."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


REQUIRED_MODEL_FILES = (
    "config.json",
    "chat_template.jinja",
    "openvino_config.json",
    "openvino_detokenizer.xml",
    "openvino_detokenizer.bin",
    "openvino_model.xml",
    "openvino_model.bin",
    "openvino_tokenizer.xml",
    "openvino_tokenizer.bin",
    "tokenizer.json",
    "tokenizer_config.json",
)
WEIGHTS_FILENAME = "openvino_model.bin"
MINIMUM_WEIGHTS_BYTES = 4_000_000_000
EXPECTED_WEIGHTS_SHA256 = (
    "f2e6960fde61d24a83c477c46f9ee7158748762089f2018fc2bb6a6d6c7f2a7e"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_is_complete(destination: Path) -> bool:
    return all(
        (destination / name).is_file()
        for name in REQUIRED_MODEL_FILES
    )


def _weights_are_valid(path: Path) -> bool:
    return (
        path.is_file()
        and path.stat().st_size >= MINIMUM_WEIGHTS_BYTES
        and _sha256_file(path) == EXPECTED_WEIGHTS_SHA256
    )


def _validate_model(destination: Path) -> None:
    missing = [
        name
        for name in REQUIRED_MODEL_FILES
        if not (destination / name).is_file()
    ]
    if missing:
        raise RuntimeError(
            "Le modèle assistant téléchargé est incomplet : "
            + ", ".join(missing)
        )
    weights_path = destination / WEIGHTS_FILENAME
    actual_hash = _sha256_file(weights_path)
    if (
        weights_path.stat().st_size < MINIMUM_WEIGHTS_BYTES
        or actual_hash != EXPECTED_WEIGHTS_SHA256
    ):
        raise RuntimeError(
            "Le SHA-256 des poids OpenVINO est invalide : "
            f"attendu {EXPECTED_WEIGHTS_SHA256}, obtenu {actual_hash}."
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--repository",
        default="OpenVINO/Qwen3-8B-int4-ov",
    )
    parser.add_argument(
        "--revision",
        default="5c47abf4b8e12ebe8e99745bb0c1ec17e0c0abcc",
    )
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    args.destination.mkdir(parents=True, exist_ok=True)
    weights_path = args.destination / WEIGHTS_FILENAME
    manifest_path = args.destination / "studio-audio-model.json"
    # Un manifeste ancien ne doit jamais faire croire qu'un modèle corrompu
    # ayant échoué à la validation est utilisable.
    manifest_path.unlink(missing_ok=True)
    weights_valid = _weights_are_valid(weights_path)
    model_is_reusable = _model_is_complete(args.destination) and weights_valid
    if not model_is_reusable:
        # Un poids de bonne taille mais au contenu corrompu pourrait être
        # considéré comme déjà présent par le cache local Hugging Face.
        if weights_path.is_file() and not weights_valid:
            weights_path.unlink()
        snapshot_download(
            repo_id=args.repository,
            revision=args.revision,
            local_dir=args.destination,
        )
        _validate_model(args.destination)
    manifest_path.write_text(
        json.dumps(
            {
                "format": 1,
                "name": "Qwen3 8B INT4",
                "repository": args.repository,
                "revision": args.revision,
                "runtime": "openvino",
                "weights_sha256": EXPECTED_WEIGHTS_SHA256,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(args.destination.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
