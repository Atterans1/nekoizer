"""Base models of the RVC engine: pinned sources, first-use download with SHA256 checks, and a status for the UI.

The files are the RVC project's HuBERT content model (lj1995/VoiceConversionWebUI, transformers layout) and its RMVPE pitch model.
Nothing is bundled. The first RVC voice that is used starts the download in a background thread; Ensure() never blocks.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import appdata
import download_weights
import integrity

LOG = logging.getLogger("seedvc.rvc_weights")

# ---- PUBLIC
HF_REVISION = "e6d0c1a17da07c33557852f9dfa2bd44cc75737d"  # lj1995/VoiceConversionWebUI, pinned commit
_BASE = f"https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/{HF_REVISION}/"
MODELS = (
    {"name": "RVC HuBERT config", "relpath": "hubert_base/config.json", "url": _BASE + "hubert_base/config.json",
     "size": 1492, "sha256": "0346950779dfb7f9316fa74ed846e2b8a22a08eedfdc5387b73f327cb1a4a7cf"},
    {"name": "RVC HuBERT content model", "relpath": "hubert_base/pytorch_model.bin", "url": _BASE + "hubert_base/pytorch_model.bin",
     "size": 189206711, "sha256": "cc8c20f4b90a520757260197a3ff2505705a7adbd20ad9eeaa4e1a9b38442ef5"},
    {"name": "RVC RMVPE pitch model", "relpath": "rmvpe.pt", "url": _BASE + "rmvpe.pt",
     "size": 181184272, "sha256": "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193"},
)


def WeightsDir() -> Path:
    return appdata.DataDir() / "rvc_weights"


def Status() -> dict:
    """state: missing | downloading | ready | error, with the file and byte counts of the current transfer."""
    with _LOCK:
        return dict(_STATE)


def Ensure(models: tuple = MODELS, weights_dir: Path | None = None) -> dict:
    """Start (once) the download of missing base models and their verification. Returns Status() at once."""
    global _THREAD
    folder = Path(weights_dir) if weights_dir is not None else WeightsDir()
    with _LOCK:
        if _THREAD is not None and _THREAD.is_alive():
            return dict(_STATE)
        if _STATE["state"] == "ready" and _STATE["dir"] == str(folder):
            return dict(_STATE)
        _STATE.update(state="downloading", file=None, done=0, total=0, message=None, dir=str(folder))
        _THREAD = threading.Thread(target=_Work, args=(models, folder), name="rvc-weights", daemon=True)
        _THREAD.start()
        return dict(_STATE)


def Reset() -> None:
    """Forget the cached state (tests and a changed weights folder)."""
    with _LOCK:
        _STATE.update(state="missing", file=None, done=0, total=0, message=None, dir=None)


ModelPaths = dict  # relpath -> absolute path under the weights folder


def PathOf(relpath: str, weights_dir: Path | None = None) -> Path:
    return (Path(weights_dir) if weights_dir is not None else WeightsDir()) / relpath


# ---- PRIVATE
_LOCK = threading.Lock()
_THREAD: threading.Thread | None = None
_STATE: dict = {"state": "missing", "file": None, "done": 0, "total": 0, "message": None, "dir": None}


def _Work(models: tuple, folder: Path) -> None:
    try:
        cache = folder.parent / integrity.CACHE_NAME
        for model in models:
            if (folder / model["relpath"]).exists():
                continue
            with _LOCK:
                _STATE.update(file=model["name"], done=0, total=model["size"])

            def progress(name: str, done: int, total: int) -> None:
                with _LOCK:
                    _STATE.update(file=name, done=done, total=total)

            LOG.info("downloading %s", model["name"])
            download_weights.Fetch(model, folder, cache, progress)
        manifest = integrity.Manifest("rvc", "", {}, tuple(models))
        verifier = integrity.ModelVerifier(manifest, folder, cache)
        verifier.Start()
        verifier.Wait()
        with _LOCK:
            _STATE.update(state="ready", file=None, done=0, total=0, message=None)
    except (download_weights.DownloadError, integrity.IntegrityError) as exc:
        LOG.error("RVC base models: %s", exc)
        with _LOCK:
            _STATE.update(state="error", message=str(exc))
