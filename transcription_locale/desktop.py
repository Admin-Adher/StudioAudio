"""Hôte bureau natif pour l'interface Gradio locale.

La fenêtre pywebview ne contient aucun moteur distant : elle affiche le serveur
Gradio lié exclusivement à ``127.0.0.1``. Le serveur est démarré dans un thread
de fond et fermé avec la fenêtre. Les checkpoints existants rendent un arrêt en
cours de transcription reprenable au prochain lancement.
"""

from __future__ import annotations

import base64
import html
import importlib
import importlib.util
import inspect
import json
import logging
import logging.handlers
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .runtime_paths import APP_DATA_ROOT, RESOURCE_ROOT


APP_TITLE = "Studio Audio"
WINDOW_TITLE = "Studio Audio"
BACKGROUND_COLOR = "#f3eee5"
SESSION_FILENAME = "desktop-session.json"
SMOKE_TEST_FLAG = "--smoke-test"
SMOKE_IMPORTS = (
    "groovy",
    "gradio",
    "gradio_client",
    "safehttpx",
    "fastapi",
    "webview",
)


class AlreadyRunningError(RuntimeError):
    """Une autre fenêtre bureau détient déjà le verrou local."""


class DesktopSmokeError(RuntimeError):
    """Le paquet bureau n'est pas capable de démarrer hors interface native."""


class AppInstanceLock:
    """Verrou inter-processus léger, sans dépendance supplémentaire."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._handle: Any | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            handle.close()
            raise AlreadyRunningError(
                "Studio Audio est déjà ouvert sur cet ordinateur."
            ) from exc
        self._handle = handle

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            handle.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            handle.close()
            self._handle = None

    def __enter__(self) -> AppInstanceLock:
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def configure_desktop_environment(data_root: Path = APP_DATA_ROOT) -> dict[str, Path]:
    """Prépare les dossiers durables avant l'import de Gradio et des modèles."""

    root = Path(data_root).expanduser().resolve()
    directories = {
        "root": root,
        "data": root / "donnees",
        "models": root / "modeles",
        "results": root / "resultats",
        "cache": root / "cache",
        "logs": root / "logs",
        "webview": root / "cache" / "webview",
        "gradio_temp": root / "cache" / "gradio",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TRANSCRIPTION_APP_DATA_DIR", str(root))
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("GRADIO_TEMP_DIR", str(directories["gradio_temp"]))
    os.environ.setdefault("HF_HOME", str(directories["cache"] / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(directories["models"] / "torch"))
    os.environ.setdefault("WESPEAKER_HOME", str(directories["models"] / "wespeaker"))
    os.environ.setdefault(
        "TRANSCRIPTION_OPENVINO_MODEL_DIR",
        str(
            directories["models"]
            / "openvino"
            / "whisper-large-v3-turbo-int8-ov"
        ),
    )
    os.environ.setdefault(
        "OMP_NUM_THREADS",
        str(max(1, min(os.cpu_count() or 4, 8))),
    )
    return directories


def configure_logging(log_directory: Path) -> Path:
    log_path = Path(log_directory) / "application.log"
    handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s · %(message)s")
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    return log_path


def find_available_port(host: str = "127.0.0.1") -> int:
    """Demande au système un port de boucle locale disponible."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((host, 0))
        return int(listener.getsockname()[1])


def _package_directory(package_name: str) -> Path:
    """Localise un paquet sans exécuter son ``__init__``.

    Cette étape est volontairement effectuée avant les imports du smoke test :
    le paquet ``groovy`` lit ``version.txt`` dès son import et produisait donc
    auparavant une erreur peu exploitable lorsque PyInstaller oubliait ce
    fichier de données.
    """

    try:
        specification = importlib.util.find_spec(package_name)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        raise DesktopSmokeError(
            f"Paquet requis introuvable : {package_name} ({exc})"
        ) from exc
    if specification is None:
        raise DesktopSmokeError(f"Paquet requis introuvable : {package_name}")
    locations = list(specification.submodule_search_locations or ())
    if locations:
        return Path(locations[0]).resolve()
    if specification.origin:
        return Path(specification.origin).resolve().parent
    raise DesktopSmokeError(
        f"Le chemin du paquet requis {package_name} est indéterminable."
    )


def validate_desktop_runtime(
    resource_root: Path = RESOURCE_ROOT,
) -> dict[str, str]:
    """Vérifie les ressources et imports critiques réellement livrés.

    La fonction fonctionne aussi depuis ``sys._MEIPASS``. Elle permet donc au
    binaire construit — et pas seulement au code source — de signaler une
    ressource PyInstaller manquante avant de tenter d'ouvrir une fenêtre.
    """

    root = Path(resource_root).resolve()
    icon_path = root / "assets" / "app-icon.png"
    if not icon_path.is_file():
        raise DesktopSmokeError(
            f"Ressource packagée manquante : {icon_path}"
        )

    groovy_version_path = _package_directory("groovy") / "version.txt"
    if not groovy_version_path.is_file():
        raise DesktopSmokeError(
            f"Ressource packagée manquante : {groovy_version_path}"
        )
    groovy_version = groovy_version_path.read_text(encoding="utf-8").strip()
    if not groovy_version:
        raise DesktopSmokeError(
            f"Ressource packagée vide : {groovy_version_path}"
        )

    imported: list[str] = []
    for package_name in SMOKE_IMPORTS:
        try:
            importlib.import_module(package_name)
        except Exception as exc:
            raise DesktopSmokeError(
                f"Import bureau impossible : {package_name} ({exc})"
            ) from exc
        imported.append(package_name)
    return {
        "resource_root": str(root),
        "groovy_version": groovy_version,
        "imports": ", ".join(imported),
    }


def wait_for_loopback_server(
    url: str,
    *,
    timeout_seconds: float = 45.0,
) -> int:
    """Attend une réponse HTTP du serveur local sans utiliser de proxy."""

    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
        raise DesktopSmokeError(
            f"Le serveur bureau doit rester sur la boucle locale : {url}"
        )
    deadline = time.monotonic() + max(0.1, float(timeout_seconds))
    last_error: BaseException | None = None
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "StudioAudioDesktopSmoke/1.0"},
    )
    while time.monotonic() < deadline:
        try:
            with opener.open(request, timeout=1.0) as response:
                status = int(getattr(response, "status", response.getcode()))
            if 200 <= status < 400:
                return status
            last_error = DesktopSmokeError(f"Réponse HTTP inattendue : {status}")
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(0.1)
    detail = f" ({last_error})" if last_error else ""
    raise DesktopSmokeError(
        f"Le serveur local ne répond pas après {timeout_seconds:.1f} s{detail}"
    )


def _loading_html() -> str:
    try:
        icon_data = base64.b64encode(
            (RESOURCE_ROOT / "assets" / "app-icon.png").read_bytes()
        ).decode("ascii")
        mark = f'<img class="app-icon" src="data:image/png;base64,{icon_data}" alt="">'
    except OSError:
        mark = '<div class="mark"></div>'
    template = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<style>
:root{color-scheme:light;font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;color:#3d352e;
background:radial-gradient(circle at 50% 35%,#fffdf9 0,#f7f2ea 48%,#efe7dc 100%)}
.panel{width:min(520px,calc(100vw - 48px));padding:46px;border:1px solid rgba(112,94,76,.16);border-radius:28px;
background:rgba(255,255,255,.72);box-shadow:0 28px 80px rgba(79,63,47,.14);backdrop-filter:blur(18px);text-align:center}
.app-icon{display:block;width:72px;height:72px;object-fit:contain;margin:0 auto 22px;border-radius:20px}
.mark{width:54px;height:54px;margin:0 auto 24px;border-radius:18px;background:#6e6052;position:relative;box-shadow:0 12px 28px rgba(78,64,51,.24)}
.mark:after{content:"";position:absolute;inset:15px;border:3px solid #fff;border-top-color:transparent;border-radius:50%;animation:spin .85s linear infinite}
h1{font-size:26px;letter-spacing:-.03em;margin:0 0 10px}p{margin:0;color:#756b61;line-height:1.55}.hint{font-size:13px;margin-top:22px;color:#958a7e}
@keyframes spin{to{transform:rotate(360deg)}}
</style></head><body><main class="panel">__APP_MARK__<h1>Préparation du studio audio</h1>
<p>Votre espace local et vos fiches sont en cours de chargement.</p><p class="hint">Les données restent sur cet ordinateur.</p></main></body></html>"""
    return template.replace("__APP_MARK__", mark)


def _message_html(title: str, message: str, detail: str = "") -> str:
    safe_title = html.escape(title)
    safe_message = html.escape(message)
    safe_detail = html.escape(detail)
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><style>
body{{margin:0;min-height:100vh;display:grid;place-items:center;background:#f3eee5;color:#3d352e;
font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{width:min(620px,calc(100vw - 48px));
padding:40px;border-radius:24px;background:#fffdf9;border:1px solid #dfd5ca;box-shadow:0 24px 70px #75655422}}
h1{{font-size:24px;margin:0 0 12px}}p{{line-height:1.55;color:#6f655b}}code{{display:block;margin-top:20px;padding:13px 15px;
border-radius:12px;background:#f5efe7;color:#564a3f;overflow-wrap:anywhere}}</style></head><body><main>
<h1>{safe_title}</h1><p>{safe_message}</p>{f'<code>{safe_detail}</code>' if safe_detail else ''}
</main></body></html>"""


class DesktopController:
    def __init__(self, directories: dict[str, Path], log_path: Path) -> None:
        self.directories = directories
        self.log_path = log_path
        self.port = find_available_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.session_path = directories["root"] / SESSION_FILENAME
        self._app: Any | None = None
        self._guard = threading.RLock()
        self._closing = threading.Event()

    def _write_session(self) -> None:
        temporary = self.session_path.with_suffix(".json.tmp")
        payload = {
            "pid": os.getpid(),
            "port": self.port,
            "url": self.url,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.session_path)

    def start_server(self) -> bool:
        """Construit puis lance le véritable serveur Gradio sur loopback.

        Cette méthode ne dépend pas de pywebview. Le smoke test du binaire peut
        ainsi exercer exactement le même runtime local, puis le fermer, sans
        créer de fenêtre native.
        """

        from .exports import RESULTS_DIR
        from .ui import ASSETS_DIR, CUSTOM_CSS, build_app
        from .workspace import DATA_DIR

        app = build_app()
        app.queue(default_concurrency_limit=1, max_size=8)
        with self._guard:
            self._app = app
        if self._closing.is_set():
            self.stop()
            return False
        try:
            app.launch(
                server_name="127.0.0.1",
                server_port=self.port,
                inbrowser=False,
                share=False,
                prevent_thread_lock=True,
                allowed_paths=[
                    str(RESULTS_DIR.resolve()),
                    str(DATA_DIR.resolve()),
                    str(ASSETS_DIR.resolve()),
                ],
                show_error=True,
                css=CUSTOM_CSS,
                footer_links=[],
                quiet=True,
            )
        except Exception:
            with self._guard:
                if self._app is app:
                    self._app = None
            try:
                app.close()
            except Exception:
                logging.exception("Nettoyage impossible après l'échec du serveur local")
            raise
        if self._closing.is_set():
            self.stop()
            return False
        self._write_session()
        logging.info("Serveur bureau prêt sur %s", self.url)
        return True

    def bootstrap(self, window: Any) -> None:
        """Démarre Gradio après l'affichage immédiat de la fenêtre de chargement."""

        try:
            if not self.start_server():
                return
            window.load_url(self.url)
            logging.info("Fenêtre bureau prête sur %s", self.url)
        except Exception as exc:  # pragma: no cover - vérifié par le smoke test packagé
            logging.exception("Échec du démarrage de l'application bureau")
            try:
                window.load_html(
                    _message_html(
                        "Le studio audio n'a pas pu démarrer",
                        str(exc) or "Une erreur inattendue s'est produite.",
                        f"Journal technique : {self.log_path}",
                    )
                )
            except Exception:
                logging.exception("Impossible d'afficher la page d'erreur native")

    def stop(self, *_: object) -> bool:
        clean = True
        self._closing.set()
        with self._guard:
            app = self._app
            self._app = None
        if app is not None:
            try:
                app.close()
            except Exception:
                clean = False
                logging.exception("Le serveur local ne s'est pas fermé proprement")
        try:
            self.session_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            clean = False
            logging.warning("Le fichier de session n'a pas pu être supprimé")
        return clean


def run_desktop_smoke_test(
    directories: dict[str, Path],
    log_path: Path,
    *,
    timeout_seconds: float = 45.0,
) -> int:
    """Teste le runtime packagé complet sans ouvrir de fenêtre native."""

    controller = DesktopController(directories, log_path)
    report: dict[str, object] = {
        "mode": "desktop-smoke",
        "status": "error",
    }
    exit_code = 3
    try:
        report.update(validate_desktop_runtime())
        if not controller.start_server():
            raise DesktopSmokeError("Le serveur local s'est arrêté avant son démarrage.")
        report["http_status"] = wait_for_loopback_server(
            controller.url,
            timeout_seconds=timeout_seconds,
        )
        report["url"] = controller.url
        report["status"] = "ok"
        exit_code = 0
    except Exception as exc:
        report["error"] = str(exc) or exc.__class__.__name__
        logging.exception("Échec du smoke test du paquet bureau")
    finally:
        clean_shutdown = controller.stop()
        report["clean_shutdown"] = clean_shutdown
        if controller.session_path.exists():
            report["status"] = "error"
            report["error"] = "Le fichier de session persiste après l'arrêt."
            exit_code = 4
        elif not clean_shutdown:
            report["status"] = "error"
            report["error"] = "Le serveur local ne s'est pas fermé proprement."
            exit_code = 4
    logging.info("Résultat du smoke test bureau : %s", report["status"])
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return exit_code


def _start_webview(webview: Any, controller: DesktopController, window: Any) -> None:
    start_kwargs: dict[str, object] = {
        "debug": False,
        "private_mode": False,
    }
    parameters = inspect.signature(webview.start).parameters
    if "storage_path" in parameters:
        start_kwargs["storage_path"] = str(controller.directories["webview"])
    if sys.platform == "win32" and "gui" in parameters:
        start_kwargs["gui"] = "edgechromium"
    webview.start(controller.bootstrap, (window,), **start_kwargs)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    directories = configure_desktop_environment()
    log_path = configure_logging(directories["logs"])
    if SMOKE_TEST_FLAG in arguments:
        logging.info("Démarrage du smoke test de %s", WINDOW_TITLE)
        return run_desktop_smoke_test(directories, log_path)

    logging.info("Démarrage de %s", WINDOW_TITLE)

    try:
        import webview
    except ImportError:
        logging.exception("pywebview est absent du paquet bureau")
        return 2

    instance_lock = AppInstanceLock(directories["root"] / "desktop.lock")
    try:
        instance_lock.acquire()
    except AlreadyRunningError as exc:
        window = webview.create_window(
            APP_TITLE,
            html=_message_html("Application déjà ouverte", str(exc)),
            width=560,
            height=360,
            resizable=False,
            background_color=BACKGROUND_COLOR,
        )
        webview.start(debug=False)
        return 0

    controller = DesktopController(directories, log_path)
    try:
        window = webview.create_window(
            WINDOW_TITLE,
            html=_loading_html(),
            width=1280,
            height=820,
            min_size=(1024, 720),
            resizable=True,
            fullscreen=False,
            background_color=BACKGROUND_COLOR,
            text_select=True,
            zoomable=False,
        )
        window.events.closed += controller.stop
        _start_webview(webview, controller, window)
        return 0
    finally:
        controller.stop()
        instance_lock.release()
        logging.info("Application bureau fermée")


__all__ = [
    "APP_TITLE",
    "AlreadyRunningError",
    "AppInstanceLock",
    "DesktopController",
    "DesktopSmokeError",
    "SMOKE_IMPORTS",
    "SMOKE_TEST_FLAG",
    "configure_desktop_environment",
    "find_available_port",
    "main",
    "run_desktop_smoke_test",
    "validate_desktop_runtime",
    "wait_for_loopback_server",
]
