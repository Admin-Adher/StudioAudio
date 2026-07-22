"""Chemins stables de l'application, en source comme une fois empaquetée.

PyInstaller extrait le code dans un dossier temporaire. Les données utilisateur
ne doivent donc jamais être écrites à côté de ``__file__`` dans un exécutable.
En développement, on conserve volontairement les dossiers historiques du dépôt
afin de ne pas déplacer silencieusement le workspace existant.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path


APP_DIRECTORY_NAME = "TranscriptionLocale"
DATA_ROOT_ENV = "TRANSCRIPTION_APP_DATA_DIR"
RESOURCE_ROOT_ENV = "TRANSCRIPTION_APP_RESOURCES_DIR"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_app_data_root(
    *,
    frozen: bool | None = None,
    platform_name: str | None = None,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    project_root: Path | None = None,
) -> Path:
    """Retourne le dossier durable propre à l'utilisateur.

    ``TRANSCRIPTION_APP_DATA_DIR`` reste prioritaire pour les tests, les postes
    administrés et une éventuelle migration. Hors exécutable, le dossier du
    projet reste utilisé pour préserver le comportement actuel.
    """

    environment = os.environ if environ is None else environ
    configured = str(environment.get(DATA_ROOT_ENV, "")).strip()
    if configured:
        return Path(configured).expanduser().resolve()

    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    source_root = PROJECT_ROOT if project_root is None else Path(project_root)
    if not is_frozen:
        return source_root.expanduser().resolve()

    platform_value = sys.platform if platform_name is None else platform_name
    user_home = Path.home() if home is None else Path(home)
    if platform_value == "win32":
        base = str(environment.get("LOCALAPPDATA", "")).strip()
        return (
            Path(base) if base else user_home / "AppData" / "Local"
        ) / APP_DIRECTORY_NAME
    if platform_value == "darwin":
        return user_home / "Library" / "Application Support" / APP_DIRECTORY_NAME

    xdg_data_home = str(environment.get("XDG_DATA_HOME", "")).strip()
    base = Path(xdg_data_home) if xdg_data_home else user_home / ".local" / "share"
    return base / APP_DIRECTORY_NAME


def resolve_resource_root(
    *,
    environ: Mapping[str, str] | None = None,
    bundle_root: Path | None = None,
) -> Path:
    """Retourne le dossier en lecture seule contenant les ressources livrées."""

    environment = os.environ if environ is None else environ
    configured = str(environment.get(RESOURCE_ROOT_ENV, "")).strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if bundle_root is not None:
        return Path(bundle_root).expanduser().resolve()
    extracted = getattr(sys, "_MEIPASS", None)
    return Path(extracted).resolve() if extracted else PROJECT_ROOT


APP_DATA_ROOT = resolve_app_data_root()
RESOURCE_ROOT = resolve_resource_root()


__all__ = [
    "APP_DATA_ROOT",
    "APP_DIRECTORY_NAME",
    "DATA_ROOT_ENV",
    "PROJECT_ROOT",
    "RESOURCE_ROOT",
    "RESOURCE_ROOT_ENV",
    "resolve_app_data_root",
    "resolve_resource_root",
]
