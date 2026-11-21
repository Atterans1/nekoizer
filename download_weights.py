"""Download the Seed-VC model files listed in the build manifest, with resume and SHA256 verification.

  python download_weights.py [--full-check]

Files go to appdata.ModelsDir(). Each download resumes from the chunks its .part.json lists, is hashed, and only then renamed into place. --full-check
hashes every present file again instead of trusting the verification cache. Exit code 0 means every file is present and verified.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import appdata
import integrity

# ---------------------------------------------------------------- PUBLIC
CHUNK_BYTES = 8 << 20  # one Range request; the host throttles each connection, so several run at once
WORKERS = 4
READ_BYTES = 1 << 20
RETRIES = 6


class DownloadError(RuntimeError):
    """A model file could not be fetched or did not verify. The message is written for the user."""


def CopyShipped(manifest: integrity.Manifest, models_dir: Path) -> None:
    """The engine reads its configs and weights from one folder: copy the shipped configs next to the weights."""
    for rel in manifest.files:
        if rel.startswith("assets/seedvc/"):
            dest = models_dir / rel[len("assets/seedvc/"):]
            dest.parent.mkdir(parents=True, exist_ok=True)
            source = appdata.AppDir() / rel
            if not dest.exists() or integrity.Sha256File(dest) != integrity.Sha256File(source):
                shutil.copyfile(source, dest)


def Fetch(model: dict, models_dir: Path, cache_path: Path, progress=None) -> Path:
    """Download one model file in parallel 8 MB ranges (resuming from the chunks a .part.json lists), verify size and SHA256, move it into place."""
    dest = models_dir / model["relpath"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, RETRIES + 1):
        try:
            _Transfer(model, part, progress)
            break
        except OSError as exc:
            if attempt == RETRIES:
                raise DownloadError(f"Could not download {model['name']} ({exc}). Check your internet connection and start the app again; "
                                    "the download resumes where it stopped.") from exc
            time.sleep(min(2 ** attempt, 30))
    digest = integrity.Sha256File(part)
    if part.stat().st_size != model["size"] or digest != model["sha256"]:
        part.unlink()
        _Sidecar(part).unlink(missing_ok=True)
        raise DownloadError(f"{model['name']} downloaded damaged (SHA256 {digest[:12]}..., expected {model['sha256'][:12]}...). Start the app again to retry.")
    os.replace(part, dest)
    _Sidecar(part).unlink(missing_ok=True)
    integrity.RecordVerified(cache_path, dest, digest)
    return dest


def Missing(manifest: integrity.Manifest, models_dir: Path) -> list[dict]:
    return [m for m in manifest.models if not (models_dir / m["relpath"]).exists()]


def Main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full-check", action="store_true", help="hash every present file again")
    args = ap.parse_args(argv)
    manifest = integrity.LoadManifest(appdata.ManifestPath(), appdata.BakedManifestSha256())
    models_dir, cache = appdata.ModelsDir(), appdata.DataDir() / integrity.CACHE_NAME
    models_dir.mkdir(parents=True, exist_ok=True)
    CopyShipped(manifest, models_dir)
    try:
        for model in Missing(manifest, models_dir):
            print(f"downloading {model['name']} ({model['size'] / 2**20:.0f} MB)", flush=True)
            Fetch(model, models_dir, cache, _Print)
        verifier = integrity.ModelVerifier(manifest, models_dir, cache, use_cache=not args.full_check)
        verifier.Start()
        verifier.Wait()
    except (DownloadError, integrity.IntegrityError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    print("models ready in", models_dir)
    return 0


# ---------------------------------------------------------------- PRIVATE
def _Sidecar(part: Path) -> Path:
    return part.with_name(part.name + ".json")


def _Transfer(model: dict, part: Path, progress) -> None:
    """Fetch every chunk the sidecar does not list as done, WORKERS at a time, writing each at its own offset of the preallocated .part."""
    size = model["size"]
    count = -(-size // CHUNK_BYTES)
    sidecar = _Sidecar(part)
    done: set[int] = set()
    if part.exists() and part.stat().st_size == size and sidecar.exists():
        try:
            done = {int(i) for i in json.loads(sidecar.read_text())["done"] if 0 <= int(i) < count}
        except (ValueError, KeyError, TypeError):
            done = set()
    if not done or not part.exists() or part.stat().st_size != size:
        done = set()
        with open(part, "wb") as handle:
            handle.truncate(size)
    lock = threading.Lock()
    got = [sum(min(CHUNK_BYTES, size - i * CHUNK_BYTES) for i in done)]

    def chunk(index: int) -> None:
        start = index * CHUNK_BYTES
        end = min(start + CHUNK_BYTES, size) - 1
        request = urllib.request.Request(model["url"], headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 206:
                raise OSError(f"the server ignored the byte range (HTTP {response.status})")
            data = response.read()
        if len(data) != end - start + 1:
            raise OSError("a chunk arrived short")
        with lock:
            with open(part, "r+b") as handle:
                handle.seek(start)
                handle.write(data)
            done.add(index)
            sidecar.write_text(json.dumps({"done": sorted(done)}))
            got[0] += len(data)
            if progress:
                progress(model["name"], got[0], size)

    todo = [i for i in range(count) if i not in done]
    if progress and not todo:
        progress(model["name"], size, size)
    with ThreadPoolExecutor(WORKERS) as pool:
        for future in [pool.submit(chunk, i) for i in todo]:
            future.result()


def _Print(name: str, done: int, total: int) -> None:
    if done % (64 << 20) < CHUNK_BYTES:
        print(f"  {name}: {done * 100 // total}%", flush=True)


if __name__ == "__main__":
    sys.exit(Main())
