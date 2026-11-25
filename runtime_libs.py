"""PyTorch CUDA runtime libraries, fetched on first run instead of shipped in the installer.

The packaged app leaves torch/lib out of its bundle (build_exe.py). installer/torch_lib_manifest.json lists every file of that folder with its
member in the official PyTorch wheel: byte offset of the deflated data, compressed size, CRC-32, size and SHA256. A missing or damaged member
is downloaded in parallel ranged chunks of the wheel, inflated, verified and placed in DataDir()/runtime/torch_lib. Link() then makes
<bundle>/torch/lib a directory junction to that folder, so torch finds its DLLs where it expects them. Nothing here imports torch.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import appdata

# ---------------------------------------------------------------- PUBLIC
CHUNK_BYTES = 8 << 20  # one Range request of the compressed member
CONNECTIONS = 4
RETRIES = 6
TIMEOUT_SECONDS = 30
RETRY_BACKOFF_SECONDS = 2  # the wait doubles per attempt, capped at 30 s
READ_BYTES = 1 << 20
OUT_BYTES = 16 << 20  # largest inflated piece held in memory at once
CHUNK_DEADLINE_SECONDS = 300  # R: one 8 MB chunk must finish within five minutes; a trickling connection raises and is retried
MANIFEST_NAME = "torch_lib_manifest.json"
MANIFEST_FORMAT = 1
VERIFIED_NAME = "verified.json"


class RuntimeLibsError(RuntimeError):
    """A torch runtime library could not be fetched, verified or linked. The message is written for the user."""


class HttpRangeFile(io.RawIOBase):
    """A read-only file object over one URL of known size. Every read is one HTTP Range request, so zipfile can read a wheel's
    central directory without downloading the wheel. A server that ignores Range (HTTP 200) or sends another file is refused."""

    def __init__(self, url: str, size: int):
        self._url = url
        self._size = int(size)
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        if base + offset < 0:
            raise OSError("seek before the start of the file")
        self._pos = base + offset
        return self._pos

    def readinto(self, buffer) -> int:
        want = min(len(buffer), self._size - self._pos)
        if want <= 0:
            return 0
        start = self._pos
        request = urllib.request.Request(self._url, headers={"Range": f"bytes={start}-{start + want - 1}"})
        started = time.monotonic()  # R: a trickling server feeds every socket op inside the timeout but never finishes the read
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            if response.status != 206:
                raise OSError(f"the server ignored the byte range (HTTP {response.status})")
            _CheckContentRange(response.headers.get("Content-Range"), start, self._size)
            data = response.read()
        elapsed = time.monotonic() - started
        if elapsed > CHUNK_DEADLINE_SECONDS:
            raise OSError(f"the range read took {elapsed:.0f} s, beyond the {CHUNK_DEADLINE_SECONDS} s deadline")
        if len(data) != want:
            raise OSError("a byte range arrived short")
        buffer[:want] = data
        self._pos += want
        return want


def LoadManifest(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Read torch_lib_manifest.json (default: the copy shipped beside manifest.json)."""
    source = Path(path) if path else _DefaultManifest()
    try:
        doc = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeLibsError(f"The PyTorch library list {source} is unreadable ({exc}). Reinstall Nekoizer.") from exc
    if doc.get("format") != MANIFEST_FORMAT:
        raise RuntimeLibsError(f"The PyTorch library list {source} has an unknown format. Reinstall Nekoizer.")
    return doc


def Missing(manifest_path: str | os.PathLike | None = None, runtime_dir: str | os.PathLike | None = None) -> list[dict[str, Any]]:
    """The members the runtime folder lacks or holds damaged. Each entry carries its wheel URL and size. A file verified before is
    recognised by its size and mtime, so a normal start hashes nothing."""
    doc = LoadManifest(manifest_path)
    folder = _Folder(runtime_dir)
    folder.mkdir(parents=True, exist_ok=True)
    stamps = _LoadStamps(folder)
    out = []
    for entry in doc["files"]:
        if not _Verified(folder / entry["relpath"], entry, stamps):
            out.append(_WithWheel(entry, doc["wheel"]))
    _SaveStamps(folder, stamps)
    return out


def FetchMember(entry: dict[str, Any], runtime_dir: str | os.PathLike | None = None, progress: Callable[[str, int, int], None] | None = None) -> None:
    """Download one member as parallel ranged chunks of its compressed bytes, inflate it, check size, CRC-32 and SHA256, and place it.
    A partial download resumes from the chunks its sidecar lists. A damaged member is deleted and refused."""
    folder = _Folder(runtime_dir)
    dest = folder / entry["relpath"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    zpart = folder / (entry["relpath"] + ".z.part")
    sidecar = zpart.with_name(zpart.name + ".json")
    part = dest.with_name(dest.name + ".part")
    size = entry["compress_size"]
    count = -(-size // CHUNK_BYTES)
    done = _ResumePoint(zpart, sidecar, size, count)
    if not done:
        with open(zpart, "wb") as handle:
            handle.truncate(size)
    _Download(entry, zpart, sidecar, done, progress)
    _Inflate(entry, zpart, part, sidecar)
    os.replace(part, dest)
    zpart.unlink(missing_ok=True)
    sidecar.unlink(missing_ok=True)
    stamps = _LoadStamps(folder)
    stamps[entry["relpath"]] = _Stamp(dest, entry["sha256"])
    _SaveStamps(folder, stamps)


def Link(runtime_dir: str | os.PathLike | None = None, bundle_lib: str | os.PathLike | None = None) -> None:
    """Make bundle_lib (default <bundle>/torch/lib) a directory junction to the runtime folder. A correct link is left alone; a wrong
    link is replaced; an empty folder is removed; a non-empty real folder is renamed aside, never deleted."""
    target = _Folder(runtime_dir).resolve()
    link = Path(bundle_lib) if bundle_lib else _BundleLib()
    if link.is_symlink() or _IsJunction(link):
        if _SameFolder(link, target):
            return
        os.rmdir(link)  # removes the link only; the folder it pointed at stays
    elif link.is_dir():
        if any(link.iterdir()):
            aside = link.with_name(f"{link.name}.replaced-{int(time.time())}")
            try:
                link.rename(aside)
            except OSError as exc:
                raise RuntimeLibsError(f"Could not move {link} aside ({exc}). Reinstall Nekoizer into a writable folder.") from exc
        else:
            os.rmdir(link)
    elif link.exists():
        raise RuntimeLibsError(f"{link} is a file, not the PyTorch library folder. Reinstall Nekoizer.")
    try:
        link.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeLibsError(f"Could not create {link.parent} ({exc}). Reinstall Nekoizer into a writable folder.") from exc
    _MakeLink(link, target)
    if not _SameFolder(link, target):
        raise RuntimeLibsError(f"{link} does not lead to {target} after linking. Reinstall Nekoizer.")


def Ensure(progress: Callable[[str, int, int], None] | None = None, manifest_path: str | os.PathLike | None = None,
           runtime_dir: str | os.PathLike | None = None, bundle_lib: str | os.PathLike | None = None) -> None:
    """Fetch every missing member, then link the bundle folder. The windowed app runs the same steps itself, inside one progress window."""
    for entry in Missing(manifest_path, runtime_dir):
        FetchMember(entry, runtime_dir, progress)
    Link(runtime_dir, bundle_lib)


# ---------------------------------------------------------------- PRIVATE
def _DefaultManifest() -> Path:
    return appdata.AppDir() / "installer" / MANIFEST_NAME


def _BundleLib() -> Path:
    return appdata.AppDir() / "torch" / "lib"


def _Folder(runtime_dir: str | os.PathLike | None) -> Path:
    return Path(runtime_dir) if runtime_dir else appdata.DataDir() / "runtime" / "torch_lib"


def _WithWheel(entry: dict[str, Any], wheel: dict[str, Any]) -> dict[str, Any]:
    return {**entry, "url": wheel["url"], "wheel_size": wheel["size"]}


def _CheckContentRange(value: str | None, start: int, total: int) -> None:
    if not value or not value.startswith("bytes "):
        raise OSError("the server sent no Content-Range")
    span, _, size = value[len("bytes "):].partition("/")
    if int(span.split("-")[0]) != start or int(size) != total:
        raise OSError("the server sent another byte range or another file")


def _Sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb", buffering=0) as handle:
        while True:
            block = handle.read(8 << 20)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _Stamp(path: Path, sha256: str) -> dict[str, Any]:
    info = path.stat()
    return {"size": info.st_size, "mtime_ns": info.st_mtime_ns, "sha256": sha256}


def _LoadStamps(folder: Path) -> dict[str, dict[str, Any]]:
    try:
        doc = json.loads((folder / VERIFIED_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _SaveStamps(folder: Path, stamps: dict[str, dict[str, Any]]) -> None:
    temp = folder / (VERIFIED_NAME + ".tmp")
    temp.write_text(json.dumps(stamps, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(temp, folder / VERIFIED_NAME)


def _Verified(path: Path, entry: dict[str, Any], stamps: dict[str, dict[str, Any]]) -> bool:
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    info = path.stat()
    known = stamps.get(entry["relpath"])
    if known and known.get("size") == info.st_size and known.get("mtime_ns") == info.st_mtime_ns and known.get("sha256") == entry["sha256"]:
        return True
    if _Sha256(path) != entry["sha256"]:
        return False
    stamps[entry["relpath"]] = _Stamp(path, entry["sha256"])
    return True


def _ChunkLength(index: int, size: int) -> int:
    return min(CHUNK_BYTES, size - index * CHUNK_BYTES)


def _ResumePoint(zpart: Path, sidecar: Path, size: int, count: int) -> set[int]:
    """Chunks a previous run finished, or an empty set when the partial file cannot be trusted."""
    if not (zpart.exists() and sidecar.exists() and zpart.stat().st_size == size):
        return set()
    try:
        done = {int(i) for i in json.loads(sidecar.read_text(encoding="utf-8"))["done"] if 0 <= int(i) < count}
    except (ValueError, KeyError, TypeError):
        return set()
    return done


def _Download(entry: dict[str, Any], zpart: Path, sidecar: Path, done: set[int], progress) -> None:
    size = entry["compress_size"]
    lock = threading.Lock()
    got = [sum(_ChunkLength(i, size) for i in done)]

    def chunk(index: int) -> None:
        length = _ChunkLength(index, size)
        start = entry["data_offset"] + index * CHUNK_BYTES
        data = _FetchRange(entry, start, length)
        with lock:
            with open(zpart, "r+b") as handle:
                handle.seek(index * CHUNK_BYTES)
                handle.write(data)
            done.add(index)
            sidecar.write_text(json.dumps({"done": sorted(done)}), encoding="utf-8")
            got[0] += length
            if progress:
                progress(entry["member"], got[0], size)

    todo = [i for i in range(-(-size // CHUNK_BYTES)) if i not in done]
    if progress and not todo:
        progress(entry["member"], size, size)
    with ThreadPoolExecutor(CONNECTIONS) as pool:
        for future in [pool.submit(chunk, i) for i in todo]:
            future.result()


def _FetchRange(entry: dict[str, Any], start: int, length: int) -> bytes:
    for attempt in range(1, RETRIES + 1):
        try:
            reader = HttpRangeFile(entry["url"], entry["wheel_size"])  # each connection has its own reader
            reader.seek(start)
            data = reader.read(length)
            if len(data) != length:
                raise OSError("a chunk arrived short")
            return data
        except OSError as exc:
            if attempt == RETRIES:
                raise RuntimeLibsError(f"Could not download {entry['member']} ({exc}). Check your internet connection and start the app again; "
                                       "the download resumes where it stopped.") from exc
            time.sleep(min(RETRY_BACKOFF_SECONDS * 2 ** attempt, 30))
    raise AssertionError("unreachable")


def _Inflate(entry: dict[str, Any], zpart: Path, part: Path, sidecar: Path) -> None:
    """Inflate the compressed file into part, checking size, CRC-32 and SHA256 as it goes. Damage removes every partial file."""
    inflater = zlib.decompressobj(-15)
    digest = hashlib.sha256()
    crc = 0
    written = 0
    try:
        with open(zpart, "rb") as source, open(part, "wb") as out:
            def emit(piece: bytes) -> None:
                nonlocal crc, written
                out.write(piece)
                digest.update(piece)
                crc = zlib.crc32(piece, crc)
                written += len(piece)

            for block in iter(lambda: source.read(READ_BYTES), b""):
                pending = block
                while pending:
                    emit(inflater.decompress(pending, OUT_BYTES))
                    pending = inflater.unconsumed_tail
            emit(inflater.flush())
        intact = inflater.eof and written == entry["size"] and crc == entry["crc32"] and digest.hexdigest() == entry["sha256"]
        reason = f"SHA256 {digest.hexdigest()[:12]}..., expected {entry['sha256'][:12]}..."
    except zlib.error as exc:
        intact, reason = False, str(exc)
    if not intact:
        part.unlink(missing_ok=True)
        zpart.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        raise RuntimeLibsError(f"{entry['member']} downloaded damaged ({reason}). Start the app again to retry.")


def _IsJunction(path: Path) -> bool:
    return os.name == "nt" and os.path.isjunction(path)


def _SameFolder(link: Path, target: Path) -> bool:
    return os.path.normcase(os.path.realpath(link)) == os.path.normcase(os.path.realpath(target))


def _MakeLink(link: Path, target: Path) -> None:
    if os.name != "nt":
        os.symlink(target, link, target_is_directory=True)
        return
    result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeLibsError(f"Could not link {link} to {target} ({detail}). The install folder must be writable; reinstall Nekoizer into a writable folder.")
