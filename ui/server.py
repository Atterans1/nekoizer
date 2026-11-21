"""REST and WebSocket control surface for the Seed-VC pipeline, plus the static dark page.

CreateApp(pipeline) drives any object with the Pipeline interface (pipeline.Pipeline or a test double). Routes follow
docs/seedvc/CONTRACT.md. app_seedvc.py binds 127.0.0.1 only. Requests whose Host or Origin is not a loopback name are
refused, so a page on another origin cannot drive the microphone through this server.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import time
import uuid
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from python_multipart import MultipartParser
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import parse_options_header
from starlette.concurrency import run_in_threadpool

import seedvc_params as P
from engine import rvc_store
from net_guard import FetchBytes, GuardError

LOG = logging.getLogger("seedvc.ui")

# ---- PUBLIC
def CreateApp(pipeline: Any) -> FastAPI:
    """Build the FastAPI app for one pipeline. Blocking pipeline calls run in the threadpool."""
    app = FastAPI(title="Seed-VC voice changer", docs_url=None, redoc_url=None, openapi_url=None)
    recording: dict[str, int | None] = {"slot": None}

    @app.middleware("http")
    async def LoopbackOnly(request: Request, call_next):
        if not _IsLoopback(request.headers):
            return JSONResponse({"detail": "only loopback Host and Origin are accepted"}, status_code=403)
        return await call_next(request)

    @app.get("/", include_in_schema=False)
    def Index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html", headers=_PAGE_HEADERS)

    @app.get("/api/state")
    async def GetState() -> dict:
        return await run_in_threadpool(_StateDocument, pipeline, recording["slot"])

    @app.post("/api/param")
    async def PostParam(request: Request) -> dict:
        body = await _JsonObject(request)
        name = body.get("name")
        if not isinstance(name, str) or name not in P.ParamSpec:
            raise HTTPException(400, f"unknown parameter {name!r}")
        try:
            value = P.Validate(name, body.get("value"))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        stored = await _Call(pipeline.SetParam, name, value)
        return {"ok": True, "name": name, "value": _Plain(stored)}

    @app.post("/api/preset")
    async def PostPreset(request: Request) -> dict:
        body = await _JsonObject(request)
        name = body.get("name")
        if name == P.TRUNCATED_PRESET_LABEL:
            raise HTTPException(409, "this preset is cut off in the source and has no values")
        if not isinstance(name, str) or name not in P.PRESETS:
            raise HTTPException(404, f"unknown preset {name!r}")
        if P.PRESETS[name] is None:
            raise HTTPException(409, f"preset {name!r} has no values")
        params = await _Call(pipeline.ApplyPreset, name)
        return {"ok": True, "name": name, "params": _Plain(params)}

    @app.post("/api/manual_slot")
    async def PostManualSlot(request: Request) -> dict:
        body = await _JsonObject(request)
        idx = body.get("idx")
        if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < len(P.SLOT_NAMES):
            raise HTTPException(400, "idx must be an integer from 0 to 3")
        await _Call(pipeline.SetManualSlot, idx)
        return {"ok": True, "idx": idx}

    @app.get("/api/devices")
    async def GetDevices() -> dict:
        return {"devices": _Plain(await _Call(pipeline.ListDevices))}

    @app.post("/api/slot/{idx}")
    async def PostSlot(idx: int, request: Request) -> dict:
        slot = _SlotIndex(idx)
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(400, 'send the .wav as multipart field "file"')
        data = await upload.read(_MAX_UPLOAD_BYTES + 1)
        await form.close()
        if len(data) > _MAX_UPLOAD_BYTES:
            raise HTTPException(413, "the wav is larger than 64 MiB")
        audio, sr = await run_in_threadpool(_DecodeWav, data)
        await _Call(pipeline.SetSlotFromWav, slot, audio, sr)
        return {"ok": True, "slot": _SlotRows(pipeline)[slot]}

    @app.post("/api/slot/{idx}/url")
    async def PostSlotUrl(idx: int, request: Request) -> dict:
        """Load a reference clip from a direct https link to an audio file (wav, flac, ogg or mp3)."""
        slot = _SlotIndex(idx)
        body = await request.json()
        url = body.get("url") if isinstance(body, dict) else None
        if not isinstance(url, str):
            raise HTTPException(400, 'send {"url": "https://..."}')
        data = await run_in_threadpool(_FetchAudio, url)
        audio, sr = await run_in_threadpool(_DecodeWav, data)
        await _Call(pipeline.SetSlotFromWav, slot, audio, sr)
        return {"ok": True, "slot": _SlotRows(pipeline)[slot]}

    @app.post("/api/open_voice_library")
    async def OpenVoiceLibrary() -> dict:
        """Open the voice-models.com library in the user's browser (a fixed address, never one the page supplies)."""
        await run_in_threadpool(webbrowser.open, VOICE_LIBRARY_URL)
        return {"ok": True, "url": VOICE_LIBRARY_URL}

    @app.delete("/api/slot/{idx}")
    async def DeleteSlot(idx: int) -> dict:
        slot = _SlotIndex(idx)
        await _Call(pipeline.ClearSlot, slot)
        return {"ok": True, "slot": _SlotRows(pipeline)[slot]}

    @app.post("/api/slot/{idx}/record/start")
    async def RecordStart(idx: int) -> dict:
        slot = _SlotIndex(idx)
        if recording["slot"] is not None:
            raise HTTPException(409, f"slot {recording['slot']} is already recording")
        await _Call(pipeline.StartRecording, slot)
        recording["slot"] = slot
        return {"ok": True, "slot": slot, "max_seconds": P.RECORDER_MAX_S}

    @app.post("/api/slot/{idx}/record/stop")
    async def RecordStop(idx: int) -> dict:
        slot = _SlotIndex(idx)
        if recording["slot"] != slot:
            raise HTTPException(409, f"slot {slot} is not recording")
        try:
            await _Call(pipeline.StopRecording, slot)
        finally:
            recording["slot"] = None
        return {"ok": True, "slot": _SlotRows(pipeline)[slot]}

    @app.get("/api/rvc/models")
    async def GetRvcModels() -> dict:
        return await run_in_threadpool(_RvcDocument, pipeline)

    @app.post("/api/slot/{idx}/rvc/import")
    async def PostRvcImport(idx: int, request: Request) -> Any:
        """Import an uploaded .zip, or a .pth with an optional .index, then point the slot at it."""
        slot = _SlotIndex(idx)
        upload = _RvcUpload(request.headers)
        try:
            await upload.Receive(request)
            kind, source, index_path, name = upload.Source()
            if kind == "zip":
                model = await _StoreCall(rvc_store.ImportZip, source, name=name)
            else:
                model = await _StoreCall(rvc_store.ImportPth, source, index_path=index_path, name=name)
            return await _AssignRvc(pipeline, slot, model)
        finally:
            upload.Close()

    @app.post("/api/slot/{idx}/rvc/link")
    async def PostRvcLink(idx: int, request: Request) -> Any:
        """Import a voice from a direct https link to a .zip or .pth file, then point the slot at it."""
        slot = _SlotIndex(idx)
        body = await _JsonObject(request)
        url, index_url, name = body.get("url"), body.get("index_url"), body.get("name")
        if not isinstance(url, str):
            raise HTTPException(400, 'send {"url": "https://..."} with optional "index_url" and "name"')
        for key, value in (("index_url", index_url), ("name", name)):
            if value is not None and not isinstance(value, str):
                raise HTTPException(400, f"{key} must be text")
        model = await _StoreCall(rvc_store.ImportLink, url, index_url=index_url, name=name)
        return await _AssignRvc(pipeline, slot, model)

    @app.post("/api/slot/{idx}/rvc/use")
    async def PostRvcUse(idx: int, request: Request) -> dict:
        slot = _SlotIndex(idx)
        body = await _JsonObject(request)
        name = body.get("name")
        if not isinstance(name, str) or not name.strip():
            raise HTTPException(400, 'send {"name": "..."} with the name of an imported voice')
        await _Call(pipeline.SetSlotFromRvc, slot, name)
        model = await _StoreCall(rvc_store.Get, name)
        return {"ok": True, "slot": _SlotRows(pipeline)[slot], "model": _WithSlots(model, pipeline.SlotInfo())}

    @app.delete("/api/rvc/models/{name}")
    async def DeleteRvcVoice(name: str) -> dict:
        cleared = await _Call(pipeline.DeleteRvcModel, name)
        return {"ok": True, "cleared_slots": cleared}

    @app.websocket("/ws")
    async def Live(ws: WebSocket) -> None:
        if not _IsLoopback(ws.headers):
            await ws.close(code=1008)
            return
        await ws.accept()
        try:
            while True:
                payload = await run_in_threadpool(_LivePayload, pipeline, recording["slot"])
                await ws.send_json(payload)
                await asyncio.sleep(_TELEMETRY_PERIOD_S)
        except WebSocketDisconnect:
            return
        except Exception:  # not a clean disconnect: keep the traceback visible
            LOG.exception("telemetry socket failed")

    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
    return app


# ---- PRIVATE
_STATIC_DIR = Path(__file__).resolve().parent / "static"
_PAGE_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                                "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'"),
    "X-Content-Type-Options": "nosniff",
}
_LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})
_MAX_UPLOAD_BYTES = 64 * 1024 * 1024
_MiB = 1024 * 1024
_MODEL_SENTENCE = "send one .zip, or one .pth with an optional .index"
_RVC_PARTS = frozenset({"zip", "pth", "index", "name"})
_NAME_MAX_CHARS = 200
_NAME_MAX_BYTES = 800  # at most 4 UTF-8 bytes per character
_MODEL_PATH_KEYS = ("model_path", "index_path")
VOICE_LIBRARY_URL = "https://voice-models.com/"
_TELEMETRY_PERIOD_S = 0.05  # 20 Hz (UI.md)


def _IsLoopback(headers: Any) -> bool:
    if _HostName(headers.get("host")) not in _LOOPBACK_NAMES:
        return False
    origin = headers.get("origin")
    return origin is None or (urlsplit(origin).hostname or "") in _LOOPBACK_NAMES


def _HostName(host: str | None) -> str:
    if not host:
        return ""
    host = host.strip().lower()
    if host.startswith("["):
        return host[1:host.find("]")] if "]" in host else ""
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def _SlotIndex(idx: int) -> int:
    if not 0 <= idx < len(P.SLOT_NAMES):
        raise HTTPException(404, f"slot {idx} does not exist; slots are 0..{len(P.SLOT_NAMES) - 1}")
    return idx


async def _JsonObject(request: Request) -> dict[str, Any]:
    try:
        body = json.loads(await request.body())
    except ValueError as exc:
        raise HTTPException(400, "body must be a JSON object") from exc
    if not isinstance(body, dict):
        raise HTTPException(400, "body must be a JSON object")
    return body


async def _Call(fn: Any, *args: Any) -> Any:
    try:
        return await run_in_threadpool(fn, *args)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        LOG.exception("pipeline call %s failed", getattr(fn, "__name__", fn))
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc


def _DecodeWav(data: bytes) -> tuple[np.ndarray, int]:
    if not _IsAudio(data):
        raise HTTPException(415, "only wav, flac, ogg or mp3 audio files are accepted")
    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:  # soundfile raises its own error types
        raise HTTPException(400, f"could not decode the wav: {exc}") from exc
    if audio.shape[0] == 0:
        raise HTTPException(400, "the wav holds no samples")
    mono = audio.mean(axis=1, dtype=np.float32) if audio.shape[1] > 1 else audio[:, 0]
    return np.ascontiguousarray(mono, dtype=np.float32), int(sr)


def _IsAudio(data: bytes) -> bool:
    """Container magic of the formats libsndfile reads that people actually have: wav, flac, ogg, mp3 (ID3 tag or MPEG frame sync)."""
    return (data[:4] == b"RIFF" and data[8:12] == b"WAVE") or data[:4] in (b"fLaC", b"OggS") or data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0)


class _RvcUpload:
    """Streams one multipart/form-data body into temp files in rvc_store.IncomingDir(), enforcing each part's cap as bytes arrive.

    Caps are read from rvc_store when the request starts. The caller runs Close() in a finally block; it deletes every temp file.
    """

    def __init__(self, headers: Any) -> None:
        _, options = parse_options_header(headers.get("content-type"))
        boundary = options.get(b"boundary")
        if not boundary:
            raise HTTPException(400, "send multipart/form-data with the parts zip, pth, index or name")
        self._body_cap = rvc_store.MAX_ZIP_BYTES + _MiB
        declared = headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > self._body_cap:
            raise HTTPException(413, f"the upload is larger than {self._body_cap / _MiB:g} MB")
        self._caps = {"zip": rvc_store.MAX_ZIP_BYTES, "pth": rvc_store.MAX_MODEL_BYTES, "index": rvc_store.MAX_MODEL_BYTES}
        self._body_bytes = 0
        self._sizes: dict[str, int] = {}
        self._paths: dict[str, Path] = {}
        self._temp_files: list[Path] = []
        self._name_bytes = bytearray()
        self._part: str | None = None
        self._header_field = bytearray()
        self._header_value = bytearray()
        self._file: Any = None
        self._parser = MultipartParser(boundary, {
            "on_part_begin": self._PartBegin,
            "on_part_data": self._PartData,
            "on_part_end": self._PartEnd,
            "on_header_field": self._HeaderField,
            "on_header_value": self._HeaderValue,
            "on_header_end": self._HeaderEnd,
            "on_headers_finished": self._HeadersFinished,
        })

    async def Receive(self, request: Request) -> None:
        try:
            async for chunk in request.stream():
                self._body_bytes += len(chunk)
                if self._body_bytes > self._body_cap:
                    raise HTTPException(413, f"the upload is larger than {self._body_cap / _MiB:g} MB")
                self._parser.write(chunk)
            self._parser.finalize()
        except MultipartParseError as exc:
            raise HTTPException(400, "the upload is not valid multipart/form-data") from exc

    def Source(self) -> tuple[str, Path, Path | None, str | None]:
        """The one archive or checkpoint the upload carries: (kind, file, index file or None, name or None)."""
        self._CloseFile()
        has_zip, has_pth = "zip" in self._paths, "pth" in self._paths
        if has_zip == has_pth or (has_zip and "index" in self._paths):
            raise HTTPException(400, _MODEL_SENTENCE)
        name = self._Name()
        if has_zip:
            return "zip", self._paths["zip"], None, name
        return "pth", self._paths["pth"], self._paths.get("index"), name

    def Close(self) -> None:
        self._CloseFile()
        for path in self._temp_files:
            path.unlink(missing_ok=True)

    def _Name(self) -> str | None:
        if not self._name_bytes:
            return None
        try:
            text = bytes(self._name_bytes).decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise HTTPException(400, "the name must be UTF-8 text") from exc
        if len(text) > _NAME_MAX_CHARS:
            raise HTTPException(400, f"the name is longer than {_NAME_MAX_CHARS} characters")
        return text or None

    def _CloseFile(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def _PartBegin(self) -> None:
        self._part = None
        self._header_field.clear()
        self._header_value.clear()

    def _HeaderField(self, data: Any, start: int, end: int) -> None:
        self._header_field += data[start:end]

    def _HeaderValue(self, data: Any, start: int, end: int) -> None:
        self._header_value += data[start:end]

    def _HeaderEnd(self) -> None:
        if bytes(self._header_field).lower() == b"content-disposition":
            _, options = parse_options_header(bytes(self._header_value))
            self._part = options.get(b"name", b"").decode("latin-1")
        self._header_field.clear()
        self._header_value.clear()

    def _HeadersFinished(self) -> None:
        part = self._part
        if part not in _RVC_PARTS:
            raise HTTPException(400, f"unknown form part {part!r}; send zip, pth, index or name")
        if part in self._sizes:
            raise HTTPException(400, f"send the {part} part once")
        self._sizes[part] = 0
        if part == "name":
            return
        path = rvc_store.IncomingDir() / f"{uuid.uuid4().hex}-{part}"
        self._temp_files.append(path)
        self._paths[part] = path
        self._file = path.open("wb")

    def _PartData(self, data: Any, start: int, end: int) -> None:
        part = self._part
        if part == "name":
            if len(self._name_bytes) + (end - start) > _NAME_MAX_BYTES:
                raise HTTPException(400, f"the name is longer than {_NAME_MAX_CHARS} characters")
            self._name_bytes += data[start:end]
            return
        self._sizes[part] += end - start
        if self._sizes[part] > self._caps[part]:
            raise HTTPException(413, f"the {part} file is larger than {self._caps[part] / _MiB:g} MB")
        self._file.write(data[start:end])

    def _PartEnd(self) -> None:
        self._CloseFile()


def _FetchAudio(url: str) -> bytes:
    """Download a direct audio link through net_guard: https only, public hosts only (the connection and every redirect hop), at most
    _MAX_UPLOAD_BYTES, 20 s per read. A web page is refused, not scraped."""
    try:
        data, _ = FetchBytes(url.strip(), _MAX_UPLOAD_BYTES, timeout_s=20.0)
    except GuardError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    if not _IsAudio(data):
        raise HTTPException(415, "that link is not a direct wav, flac, ogg or mp3 file (a web page cannot be imported; download the clip and drop it on a slot)")
    return data


def _SlotRows(pipeline: Any) -> list[dict[str, Any]]:
    """Per-slot state comes from the pipeline's SlotInfo(); the server keeps no copy."""
    by_idx: dict[int, dict[str, Any]] = {}
    for pos, info in enumerate(pipeline.SlotInfo()):
        by_idx[int(info.get("idx", pos))] = info
    rows = []
    for i, name in enumerate(P.SLOT_NAMES):
        info = by_idx.get(i, {})
        loaded = bool(info.get("loaded", False))
        seconds = info.get("seconds") if loaded else None
        rows.append({"idx": i, "name": name, "loaded": loaded, "kind": info.get("kind"), "model": info.get("model"),
                     "seconds": None if seconds is None else float(seconds),
                     "active": bool(info.get("active", False)), "selected": bool(info.get("selected", False)),
                     "engine": _Plain(info)})
    return rows


def _Public(model: dict[str, Any]) -> dict[str, Any]:
    """The store's model dict without absolute paths: the only form the server returns."""
    return {k: v for k, v in model.items() if k not in _MODEL_PATH_KEYS}


def _WithSlots(model: dict[str, Any], slot_rows: list[dict[str, Any]]) -> dict[str, Any]:
    slots = [int(row["idx"]) for row in slot_rows if row.get("model") == model["name"]]
    return {**_Public(model), "slots": slots}


def _Weights(pipeline: Any) -> dict[str, Any]:
    try:
        return pipeline.RvcStatus()["weights"]
    except Exception as exc:  # a failed status read must not hide the rest of the state
        LOG.warning("RVC base-model status failed: %s", exc)
        return {"state": "error", "file": None, "done": 0, "total": 0, "message": str(exc)}


def _RvcDocument(pipeline: Any) -> dict[str, Any]:
    """Imported voices, each with the slots holding it, and the base-model download state."""
    rows = pipeline.SlotInfo()
    return {"models": [_WithSlots(m, rows) for m in rvc_store.List()], "weights": _Weights(pipeline)}


async def _StoreCall(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a blocking rvc_store call in the threadpool. GuardError keeps its own status; other refusals are 400."""
    try:
        return await run_in_threadpool(fn, *args, **kwargs)
    except GuardError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


async def _AssignRvc(pipeline: Any, slot: int, model: dict[str, Any]) -> Any:
    """Point a slot at a model already on disk. A refusal keeps the import and returns the model with the message."""
    try:
        await run_in_threadpool(pipeline.SetSlotFromRvc, slot, model["name"])
    except (ValueError, RuntimeError) as exc:
        return JSONResponse({"detail": str(exc), "model": _WithSlots(model, pipeline.SlotInfo())}, status_code=400)
    return {"ok": True, "slot": _SlotRows(pipeline)[slot], "model": _WithSlots(model, pipeline.SlotInfo())}


def _PresetRows() -> list[dict[str, Any]]:
    rows = [{"name": name, "label": name, "disabled": values is None, "values": values}
            for name, values in P.PRESETS.items()]
    rows.append({"name": P.TRUNCATED_PRESET_LABEL,
                 "label": f"{P.TRUNCATED_PRESET_LABEL}... (name cut off in spec)",
                 "disabled": True, "values": None})
    return rows


def _StateDocument(pipeline: Any, recording_slot: int | None) -> dict[str, Any]:
    return _Plain({
        "params": pipeline.GetParams(),
        "ranges": P.ParamSpec,
        "defaults": P.Defaults(),
        "slot_names": list(P.SLOT_NAMES),
        "slots": _SlotRows(pipeline),
        "presets": _PresetRows(),
        "recording": {"slot": recording_slot, "max_seconds": P.RECORDER_MAX_S},
        "hotkeys": P.HOTKEY_MODIFIERS,
        "telemetry": _TelemetryOrError(pipeline),
        "rvc": _RvcDocument(pipeline),
    })


def _TelemetryOrError(pipeline: Any) -> dict[str, Any]:
    try:
        return pipeline.Telemetry()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _LivePayload(pipeline: Any, recording_slot: int | None) -> dict[str, Any]:
    try:
        levels = pipeline.LevelSnapshot()
    except Exception:  # no input chain yet, or it has stopped
        levels = None
    return _Plain({"t": time.time(), "telemetry": _TelemetryOrError(pipeline), "levels": levels,
                   "recording": recording_slot})


def _Plain(value: Any) -> Any:
    """Convert numpy values to JSON-safe Python values; NaN and infinity become None."""
    if isinstance(value, dict):
        return {str(k): _Plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_Plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _Plain(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
