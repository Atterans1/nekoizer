"""Tamper-evidence for the shipped config files and the downloaded model files. This is not DRM.

installer/make_manifest.py records the SHA256 and size of every config file shipped inside the app and of every model file the
first-run download fetches in manifest.json, and bakes the SHA256 of manifest.json into the executable (build_meta.py). At start-up
the app checks manifest.json against that baked hash, the shipped files against the manifest, and each model file against the
manifest before the engine may load it. A file that does not match is refused with a message that names it.

No secret is stored in the binary. Whoever can rebuild or patch the executable, or edit the verification cache, gets past these
checks, and so does an edit that restores a model's size and modification time (later starts only stat a model that was hashed
once; download_weights.py --full-check hashes everything again). The checks catch damaged downloads and casual edits.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOG = logging.getLogger("seedvc.integrity")

# ---------------------------------------------------------------- PUBLIC
CACHE_NAME = "verify_cache.json"


class IntegrityError(RuntimeError):
    """A shipped or model file does not match the build manifest. The message is written for the user."""


@dataclass(frozen=True)
class Manifest:
    version: str
    sha256: str  # SHA256 of the manifest file itself
    files: dict[str, dict[str, Any]]  # shipped config files: path relative to the app folder -> {size, sha256}
    models: tuple[dict[str, Any], ...]  # name, relpath (inside the models folder), url, size, sha256


def Sha256File(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    buffer = bytearray(8 << 20)
    view = memoryview(buffer)
    with open(path, "rb", buffering=0) as handle:
        while True:
            count = handle.readinto(buffer)
            if not count:
                return digest.hexdigest()
            digest.update(view[:count])


def LoadManifest(path: str | os.PathLike, baked_sha256: str | None) -> Manifest:
    """Read manifest.json. When baked_sha256 is given (a packaged build) the file must hash to exactly that."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise IntegrityError(f"The build manifest {path} cannot be read ({exc}). Reinstall Nekoizer.") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if baked_sha256 is not None and digest != baked_sha256:
        raise IntegrityError("manifest.json does not match the hash recorded in this build, so the installation was modified or "
                             "is damaged. Reinstall Nekoizer.")
    try:
        doc = json.loads(raw)
        files = {str(rel): {"size": int(meta["size"]), "sha256": str(meta["sha256"])} for rel, meta in doc["files"].items()}
        models = tuple({"name": str(m["name"]), "relpath": str(m["relpath"]), "url": str(m["url"]), "size": int(m["size"]),
                        "sha256": str(m["sha256"])} for m in doc["models"])
        return Manifest(str(doc["version"]), digest, files, models)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise IntegrityError(f"manifest.json is not a valid build manifest ({type(exc).__name__}: {exc}). "
                             "Reinstall Nekoizer.") from exc


def VerifyShippedFiles(manifest: Manifest, app_dir: str | os.PathLike) -> None:
    """Hash every shipped config file against the manifest (a few KB in total)."""
    bad = []
    for rel, meta in sorted(manifest.files.items()):
        path = Path(app_dir) / rel
        try:
            intact = path.stat().st_size == meta["size"] and Sha256File(path) == meta["sha256"]
        except OSError:
            bad.append(f"{rel} is missing")
            continue
        if not intact:
            bad.append(f"{rel} was modified")
    if bad:
        raise IntegrityError("These shipped files do not match the build manifest: " + "; ".join(bad) +
                             ". Reinstall Nekoizer to restore them.")


def RecordVerified(cache_path: str | os.PathLike, path: str | os.PathLike, sha256: str) -> None:
    """Remember that this file, at its current size and mtime, hashed to sha256 (the downloader calls this after verifying)."""
    size, mtime_ns = _Stamp(path)
    _Remember(Path(cache_path), {_Key(path): {"size": size, "mtime_ns": mtime_ns, "sha256": sha256}})


class ModelVerifier:
    """Checks every present model file against the manifest on a background thread.

    Start() returns at once. Wait() blocks until the check ends and raises IntegrityError when a file has the wrong size or SHA256.
    A missing file is not an error here, the downloader fetches it. A file hashed before (same size and mtime in the cache) is only
    stat-ed, unless use_cache is False.
    """

    def __init__(self, manifest: Manifest, models_dir: str | os.PathLike, cache_path: str | os.PathLike, use_cache: bool = True):
        self._manifest = manifest
        self._models_dir = Path(models_dir)
        self._cache_path = Path(cache_path)
        self._use_cache = use_cache
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._status: dict[str, str] = {}
        self._bad: list[str] = []
        self._crash: BaseException | None = None
        self._hashed_bytes = 0
        self._seconds = 0.0

    def Start(self) -> None:
        self._thread = threading.Thread(target=self._Run, name="seedvc-model-verify", daemon=True)
        self._thread.start()

    def Wait(self, timeout: float | None = None) -> None:
        if not self._done.wait(timeout):
            raise IntegrityError("The model check did not finish in time.")
        if self._crash is not None:
            raise IntegrityError(f"The model check failed unexpectedly: {type(self._crash).__name__}: {self._crash}")
        if self._bad:
            many = len(self._bad) > 1
            raise IntegrityError(f"Model check failed, nothing was loaded: {'; '.join(self._bad)}. Delete the file{'s' if many else ''} "
                                 f"from {self._models_dir} and start the app again to download {'them' if many else 'it'} again, "
                                 "or reinstall the app.")

    def Stats(self) -> dict[str, Any]:
        states = list(self._status.values())
        return {"status": dict(self._status), "cached": states.count("cached"), "hashed": states.count("hashed"),
                "hashed_mb": round(self._hashed_bytes / 2**20, 1), "seconds": round(self._seconds, 4)}

    # ---- PRIVATE
    def _Run(self) -> None:
        started = time.perf_counter()
        try:
            entries = _LoadCache(self._cache_path) if self._use_cache else {}
            fresh: dict[str, dict[str, Any]] = {}
            for model in self._manifest.models:
                self._status[model["relpath"]] = self._CheckOne(model, entries, fresh)
            _Remember(self._cache_path, fresh)
        except BaseException as exc:  # reported by Wait(); a dead thread must not leave Wait() hanging
            self._crash = exc
        finally:
            self._seconds = time.perf_counter() - started
            self._done.set()

    def _CheckOne(self, model: dict[str, Any], entries: dict[str, dict[str, Any]], fresh: dict[str, dict[str, Any]]) -> str:
        rel, path = model["relpath"], self._models_dir / model["relpath"]
        try:
            size, mtime_ns = _Stamp(path)
        except FileNotFoundError:
            return "missing"
        if size != model["size"]:
            self._bad.append(f"{rel} has the wrong size ({size:,} bytes, expected {model['size']:,})")
            return "bad"
        known = entries.get(_Key(path))
        if known and known.get("size") == size and known.get("mtime_ns") == mtime_ns and known.get("sha256") == model["sha256"]:
            return "cached"
        digest = Sha256File(path)
        if digest != model["sha256"]:
            self._bad.append(f"{rel} has the wrong SHA256 ({digest[:12]}..., expected {model['sha256'][:12]}...)")
            return "bad"
        fresh[_Key(path)] = {"size": size, "mtime_ns": mtime_ns, "sha256": digest}
        self._hashed_bytes += size
        return "hashed"


# ---------------------------------------------------------------- PRIVATE
_CACHE_LOCK = threading.Lock()


def _Key(path: str | os.PathLike) -> str:
    return os.path.normcase(os.path.abspath(path))


def _Stamp(path: str | os.PathLike) -> tuple[int, int]:
    info = os.stat(path)
    return info.st_size, info.st_mtime_ns


def _LoadCache(cache_path: Path) -> dict[str, dict[str, Any]]:
    try:
        entries = json.loads(cache_path.read_text(encoding="utf-8"))["entries"]
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    return entries if isinstance(entries, dict) else {}


def _Remember(cache_path: Path, fresh: dict[str, dict[str, Any]]) -> None:
    if not fresh:
        return
    with _CACHE_LOCK:
        entries = _LoadCache(cache_path)
        entries.update(fresh)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_name(cache_path.name + ".tmp")
            temporary.write_text(json.dumps({"schema": 1, "entries": entries}, indent=1), encoding="utf-8")
            os.replace(temporary, cache_path)
        except OSError as exc:
            LOG.warning("could not write the verification cache %s: %s", cache_path, exc)
