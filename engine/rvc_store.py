"""Imported RVC voices, one folder per voice under DataDir()/rvc_models.

A voice folder holds model.pth (weights that passed LoadCheckpoint), model.index (optional FAISS index) and meta.json.
A writer builds the folder under a .tmp- name inside the store and renames it into place, so a voice is either complete
or absent. Names from a zip or a link are never used as paths.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import uuid
import zipfile
import zlib
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

import appdata
from engine.rvc_checkpoint import FEATURE_WIDTH, LoadCheckpoint, UnsafeCheckpointError
from net_guard import CheckUrl, FetchToFile, GuardError

# ---------------------------------------------------------------- PUBLIC
MAX_MODEL_BYTES = 400 * 1024 * 1024
MAX_ZIP_BYTES = 600 * 1024 * 1024
MAX_ZIP_MEMBERS = 64
MAX_NAME_CHARS = 64
INDEX_MAGIC = b"IwF"  # the FAISS IVF family, the only index kind RVC writes

LOG = logging.getLogger("seedvc.rvc_store")


class RvcImportError(ValueError):
    """An import or lookup was refused. The message is written for the user and shown as it is."""


def StoreDir() -> Path:
    """The folder of imported voices. Not created here: the writers create it."""
    return appdata.DataDir() / "rvc_models"


def IncomingDir() -> Path:
    """The folder for partial uploads and downloads, on the store's volume. Created on demand."""
    path = appdata.DataDir() / "rvc_incoming"
    path.mkdir(parents=True, exist_ok=True)
    return path


def List() -> list[dict]:
    """Every complete voice, sorted by name. A broken folder is logged and skipped."""
    store = StoreDir()
    if not store.is_dir():
        return []
    voices = []
    for folder in store.iterdir():
        if folder.is_dir() and not folder.name.startswith("."):
            meta = _ReadMeta(folder)
            if meta is not None:
                voices.append(_Public(meta))
    return sorted(voices, key=lambda voice: voice["name"])


def Get(name: str) -> dict:
    """One voice with its file paths. The paths are for the engine only and never go to the browser."""
    folder = _FolderFor(name)
    meta = _ReadMeta(folder)
    if meta is None:
        raise RvcImportError(f"no imported voice named {name}")
    return {
        **_Public(meta),
        "model_path": str((folder / _MODEL).resolve()),
        "index_path": str((folder / _INDEX).resolve()) if meta["has_index"] else None,
    }


def Delete(name: str) -> None:
    """Remove a voice folder. Refuses any name that is not a stored voice."""
    folder = _FolderFor(name)
    try:
        shutil.rmtree(folder)
    except OSError as exc:
        raise RvcImportError(f"could not delete {name}: {exc.strerror or exc}") from exc


def ImportPth(pth_path: str | Path, index_path: str | Path | None = None, name: str | None = None) -> dict:
    """Import one exported RVC voice. The caller's files are read, never changed and never deleted."""
    pth_copy = index_copy = None
    try:
        pth_copy = _CopyFile(Path(pth_path), MAX_MODEL_BYTES)
        if index_path:
            index_copy = _CopyFile(Path(index_path), MAX_MODEL_BYTES)
        return _ImportCheckpoint(pth_copy, index_copy, name or Path(pth_path).stem, "upload")
    finally:
        _Remove(pth_copy)
        _Remove(index_copy)


def ImportZip(zip_path: str | Path, name: str | None = None) -> dict:
    """Import the one .pth voice, and its optional .index, from a zip. Every other member is ignored, never extracted."""
    return _ImportZipFile(Path(zip_path), name, "zip")


def ImportLink(url: str, index_url: str | None = None, name: str | None = None) -> dict:
    """Import a voice from a direct https link to a .pth or .zip file, with an optional https link to its .index."""
    _CheckLink(url)
    link_path = _UrlPath(url)
    is_zip = link_path.endswith(".zip")
    if not (is_zip or link_path.endswith(".pth")):
        raise RvcImportError(
            "the link must point straight at a .pth or .zip file: a web page or a share page cannot be imported, "
            "so download the file and drop it on the slot"
        )
    if index_url:
        if is_zip:
            raise RvcImportError("a zip carries its own index: remove index_url")
        _CheckLink(index_url)
        if not _UrlPath(index_url).endswith(".index"):
            raise RvcImportError("the index link must point straight at a .index file")
    default = _UrlStem(url)
    temp = _Scratch()
    temp_index = _Scratch() if index_url else None
    try:
        info = _Download(url, temp, MAX_ZIP_BYTES if is_zip else MAX_MODEL_BYTES)
        _RefuseWebPage(info)
        if is_zip:
            return _ImportZipFile(temp, name or default or None, "link")
        if index_url:
            _Download(index_url, temp_index, MAX_MODEL_BYTES)
        return _ImportCheckpoint(temp, temp_index, name or default, "link")
    finally:
        _Remove(temp)
        _Remove(temp_index)


# ---------------------------------------------------------------- PRIVATE
_CHUNK = 1024 * 1024
_MODEL = "model.pth"
_INDEX = "model.index"
_META = "meta.json"
_PUBLIC_KEYS = ("name", "version", "sr", "f0", "has_index", "size_mb", "source", "imported_at")
_META_TYPES = {
    "name": str,
    "version": str,
    "sr": int,
    "f0": bool,
    "has_index": bool,
    "size_mb": (int, float),
    "source": str,
    "imported_at": str,
}
_NAME_BAD = re.compile(r"[^A-Za-z0-9 _.\-]")
_NAME_DOTS = re.compile(r"\.{2,}")
_RESERVED = frozenset({"CON", "PRN", "AUX", "NUL", *(f"COM{n}" for n in range(1, 10)), *(f"LPT{n}" for n in range(1, 10))})
_DRIVE = re.compile(r"[A-Za-z]:")


def _CleanName(raw: str) -> str:
    """A name that is safe as one folder name: allowed characters only, no dot runs, no reserved Windows stems."""
    name = _NAME_DOTS.sub("_", _NAME_BAD.sub("_", raw)).lstrip(" .")
    name = name[:MAX_NAME_CHARS].rstrip(" .")
    if not name:
        raise RvcImportError("the voice needs a name")
    if name.split(".", 1)[0].upper() in _RESERVED:
        raise RvcImportError("that name is reserved by Windows: choose another")
    return name


def _FolderFor(name: str) -> Path:
    """The folder of a stored voice. A name the store could not have written is refused as unknown."""
    missing = RvcImportError(f"no imported voice named {name}")
    if not isinstance(name, str):
        raise missing
    try:
        clean = _CleanName(name)
    except RvcImportError:
        raise missing from None
    store = StoreDir()
    folder = store / clean
    if clean != name or not folder.resolve().is_relative_to(store.resolve()) or not folder.is_dir():
        raise missing
    return folder


def _ReadMeta(folder: Path) -> dict | None:
    """The meta.json of a complete voice folder, or None with a log line when the folder is broken."""
    try:
        meta = json.loads((folder / _META).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = None
    if _Usable(meta, folder):
        return meta
    LOG.warning("skipping voice folder %s: meta.json is missing or unreadable, or its model files are incomplete", folder.name)
    return None


def _Usable(meta, folder: Path) -> bool:
    if not isinstance(meta, dict):
        return False
    if not all(isinstance(meta.get(key), kind) for key, kind in _META_TYPES.items()):
        return False
    if meta["name"] != folder.name or meta["version"] not in FEATURE_WIDTH:
        return False
    return (folder / _MODEL).is_file() and (not meta["has_index"] or (folder / _INDEX).is_file())


def _Public(meta: dict) -> dict:
    return {key: meta[key] for key in _PUBLIC_KEYS}


def _Scratch() -> Path:
    return IncomingDir() / f"{uuid.uuid4().hex}.part"


def _Remove(path: Path | None) -> None:
    if path is not None:
        with suppress(OSError):
            path.unlink(missing_ok=True)


def _CopyFile(source: Path, cap: int) -> Path:
    """Copy a file the caller named into IncomingDir. A file larger than cap is refused before any byte is copied."""
    try:
        handle = source.open("rb")
    except OSError as exc:
        raise RvcImportError(f"{source.name} cannot be read ({exc.strerror or exc})") from exc
    with handle:
        size = os.fstat(handle.fileno()).st_size
        if size > cap:
            raise RvcImportError(f"{source.name} is {size / 2**20:.0f} MB, larger than the {cap // 2**20} MB limit")
        return _CopyCapped(handle, cap, source.name)


def _CopyCapped(stream, cap: int, what: str) -> Path:
    """Copy a binary stream into a new IncomingDir file. Past cap bytes the copy stops and the partial file is deleted."""
    target = _Scratch()
    total = 0
    try:
        with target.open("wb") as out:
            while chunk := stream.read(_CHUNK):
                total += len(chunk)
                if total > cap:
                    raise RvcImportError(f"{what} is larger than the {cap // 2**20} MB limit")
                out.write(chunk)
    except BaseException:
        _Remove(target)
        raise
    return target


def _Sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _ImportCheckpoint(pth_copy: Path, index_copy: Path | None, name: str, source: str) -> dict:
    """Validate the two copies, then move them into a new voice folder with one rename. The caller deletes its copies."""
    try:
        checkpoint = LoadCheckpoint(pth_copy, max_bytes=MAX_MODEL_BYTES, label=f"{name}.pth")
    except UnsafeCheckpointError as exc:
        raise RvcImportError(str(exc)) from exc
    if index_copy is not None:
        with index_copy.open("rb") as handle:
            if handle.read(len(INDEX_MAGIC)) != INDEX_MAGIC:
                raise RvcImportError("the index file is not a FAISS index written by RVC")
    clean = _CleanName(name)
    final = StoreDir() / clean
    if final.exists():
        raise RvcImportError(f"a voice named {clean} is already imported: delete it first")
    temp = StoreDir() / f".tmp-{uuid.uuid4().hex}"
    try:
        temp.mkdir(parents=True)
        os.replace(pth_copy, temp / _MODEL)
        model_bytes = (temp / _MODEL).stat().st_size
        index_bytes = 0
        if index_copy is not None:
            os.replace(index_copy, temp / _INDEX)
            index_bytes = (temp / _INDEX).stat().st_size
        meta = {
            "name": clean,
            "version": checkpoint.version,
            "sr": checkpoint.sr,
            "f0": checkpoint.f0,
            "has_index": index_copy is not None,
            "size_mb": round((model_bytes + index_bytes) / 2**20, 2),
            "source": source,
            "imported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_sha256": _Sha256(temp / _MODEL),
            "info": checkpoint.info,
        }
        (temp / _META).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        os.replace(temp, final)
    except OSError as exc:
        raise RvcImportError(f"could not save the voice on disk: {exc.strerror or exc}") from exc
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    return _Public(meta)


def _ImportZipFile(zip_path: Path, name: str | None, source: str) -> dict:
    """The zip rules shared by ImportZip and ImportLink. Only the chosen .pth and .index members are ever extracted."""
    try:
        size = zip_path.stat().st_size
    except OSError as exc:
        raise RvcImportError(f"{zip_path.name} cannot be read ({exc.strerror or exc})") from exc
    if size > MAX_ZIP_BYTES:
        raise RvcImportError(f"the zip is {size / 2**20:.0f} MB, larger than the {MAX_ZIP_BYTES // 2**20} MB limit")
    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise RvcImportError("the file is not a zip archive") from exc
    with archive:
        members = archive.infolist()
        if len(members) > MAX_ZIP_MEMBERS:
            raise RvcImportError(f"the zip holds {len(members)} entries: the limit is {MAX_ZIP_MEMBERS}")
        pth_info, index_info = _PickMembers(members)
        copies: list[Path] = []
        try:
            pth_copy = _ExtractMember(archive, pth_info)
            copies.append(pth_copy)
            index_copy = None
            if index_info is not None:
                index_copy = _ExtractMember(archive, index_info)
                copies.append(index_copy)
            return _ImportCheckpoint(pth_copy, index_copy, name or PurePosixPath(pth_info.filename).stem, source)
        finally:
            for copy in copies:
                _Remove(copy)


def _PickMembers(members: list[zipfile.ZipInfo]) -> tuple[zipfile.ZipInfo, zipfile.ZipInfo | None]:
    """Check every entry name and link before anything is extracted, then choose the one .pth and the optional .index."""
    pth, index = [], []
    for info in members:
        _CheckEntryName(info.orig_filename)  # as stored: ZipFile turns a backslash into "/" on Windows before it reaches .filename
        if ((info.external_attr >> 16) & 0o170000) == 0o120000:
            raise RvcImportError("the zip holds a symbolic link, refused")
        if info.filename.endswith("/"):
            continue
        base = PurePosixPath(info.filename).name.lower()
        if base.endswith(".pth"):
            pth.append(info)
        elif base.endswith(".index"):
            index.append(info)
    if not pth:
        raise RvcImportError("the zip holds no .pth voice model")
    if len(pth) > 1:
        raise RvcImportError(f"the zip holds {len(pth)} .pth files: import one voice per zip")
    if len(index) > 1:
        raise RvcImportError(f"the zip holds {len(index)} .index files: keep one")
    return pth[0], (index[0] if index else None)


def _CheckEntryName(name: str) -> None:
    """Refuse absolute, drive-letter, backslash, empty, dot and dot-dot names. A member name is never used as a path."""
    body = name[:-1] if name.endswith("/") else name
    segments = body.split("/")
    unsafe = "\\" in name or name.startswith("/") or _DRIVE.match(name) is not None or any(seg in ("", ".", "..") for seg in segments)
    if unsafe:
        raise RvcImportError("the zip holds an unsafe entry name, refused")


def _ExtractMember(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> Path:
    """Stream one chosen member into an IncomingDir file under a uuid name."""
    what = "the voice file inside the zip"
    if info.file_size > MAX_MODEL_BYTES:
        raise RvcImportError(f"{what} is larger than the {MAX_MODEL_BYTES // 2**20} MB limit")
    try:
        with archive.open(info) as stream:
            return _CopyCapped(stream, MAX_MODEL_BYTES, what)
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError, zlib.error) as exc:
        raise RvcImportError(f"the zip could not be read: {exc}") from exc


def _Download(url: str, target: Path, cap: int) -> dict:
    try:
        return FetchToFile(url, target, max_bytes=cap)
    except GuardError as exc:
        raise RvcImportError(str(exc)) from exc


def _CheckLink(link: str) -> None:
    try:
        CheckUrl(link)
    except GuardError as exc:
        raise RvcImportError(str(exc)) from exc


def _RefuseWebPage(info: dict) -> None:
    head = info["head"].lstrip().lower()
    if info["content_type"].startswith("text/html") or head.startswith((b"<!doctype", b"<html")):
        raise RvcImportError("that link is a web page, not a voice file: download the file and drop it on the slot")


def _UrlPath(link: str) -> str:
    return urlsplit(link).path.lower()


def _UrlStem(link: str) -> str:
    return PurePosixPath(unquote(urlsplit(link).path)).stem
