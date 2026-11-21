"""FastAPI backend for the S2S voice changer: HTTP control plane + 10 Hz telemetry WebSocket."""
import asyncio
import contextlib
import json
import os
import re
import shutil
import sys
import threading
import uuid
from collections import OrderedDict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

import params
from audio_engine import AudioEngine
from vc_inference import VCEngine

# ---------------------------------------------------------------- PUBLIC

Host = "127.0.0.1"
Port = 8765
MaxUploadBytes = 5 * 1024 * 1024
TelemetryHz = 10

Root = Path(__file__).resolve().parent
FrontendDir = Root / "frontend"


def _ResolveDataRoot() -> Path:
    """Mutable data (presets, references, created voices) never lives in the bundled install folder."""
    override = os.environ.get("S2S_DATA_DIR")
    if override:
        return Path(override)
    if getattr(sys, "frozen", False):
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "S2SVoiceChanger"
    return Root


DataRoot = _ResolveDataRoot()
PresetsDir = DataRoot / "presets"
ReferenceDir = DataRoot / "reference"
CurrentReference = ReferenceDir / "current.wav"
VoicesDir = DataRoot / "voices"

PresetNamePattern = re.compile(r"[A-Za-z0-9_-]{1,40}")
JobIdPattern = re.compile(r"[0-9a-f]{12}")
VoiceGenMaxJobs = 20
VoiceGenMaxLogLines = 200
VoiceGenPromptRange = (3, 300)
VoiceGenFeedbackMax = 300
EngineRunningMessage = "Stop the live engine first so CPU inference cannot cause audio dropouts"


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


class Client:
    """WebSocket wrapper serialising sends so acks and telemetry never interleave."""

    def __init__(self, socket: WebSocket):
        self.Socket = socket
        self._lock = asyncio.Lock()

    async def Send(self, message: dict) -> None:
        text = json.dumps(message, default=_Coerce)
        async with self._lock:
            await self.Socket.send_text(text)


class VoiceGenJobs:
    """Bounded in-memory store of voice-creation jobs; at most one job runs at a time."""

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._jobs = OrderedDict()
        self._lock = threading.Lock()

    def Create(self, prompt: str, feedback: str, parent_id, rejected_ids: list) -> dict:
        with self._lock:
            if any(job["status"] in ("queued", "running") for job in self._jobs.values()):
                raise ApiError("a voice creation job is already running", 409)
            job = {
                "id": uuid.uuid4().hex[:12],
                "status": "queued",
                "log": [],
                "result": None,
                "error": None,
                "prompt": prompt,
                "feedback": feedback,
                "parent_id": parent_id,
                "rejected_ids": list(rejected_ids),
                "clip_path": None,
            }
            self._jobs[job["id"]] = job
            while len(self._jobs) > self._capacity:
                _, evicted = self._jobs.popitem(last=False)
                if evicted["clip_path"]:
                    Path(evicted["clip_path"]).unlink(missing_ok=True)
            return job

    def Get(self, job_id: str) -> dict:
        if not JobIdPattern.fullmatch(job_id):
            raise ApiError("invalid job id")
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise ApiError("voice creation job not found: " + job_id, 404)
        return job

    def Active(self) -> bool:
        with self._lock:
            return any(job["status"] in ("queued", "running") for job in self._jobs.values())

    def Log(self, job: dict, message: str) -> None:
        with self._lock:
            job["log"].append(str(message))
            del job["log"][:-VoiceGenMaxLogLines]

    def SetStatus(self, job: dict, status: str, result=None, error=None, clip_path=None) -> None:
        with self._lock:
            job["status"] = status
            job["result"] = result
            job["error"] = error
            job["clip_path"] = clip_path

    def Snapshot(self, job: dict) -> dict:
        with self._lock:
            return {
                "job_id": job["id"],
                "status": job["status"],
                "log": list(job["log"]),
                "result": job["result"],
                "error": job["error"],
                "parent_id": job["parent_id"],
                "prompt": job["prompt"],
            }


class AppState:
    def __init__(self):
        self.Vc = None
        self.Engine = None
        self.Reference = None
        self.Clients = set()
        self.TelemetryTask = None
        self.ReferenceLock = threading.Lock()
        self.VoiceGen = VoiceGenJobs(VoiceGenMaxJobs)


State = AppState()


def ListPresets() -> list:
    return sorted(p.stem for p in PresetsDir.glob("*.wav") if PresetNamePattern.fullmatch(p.stem))


def BuildRanges() -> dict:
    ranges = {key: dict(spec) for key, spec in params.Ranges().items()}
    chunk_ms = State.Engine.GetParams()["chunk_ms"]
    if "crossfade_ms" in ranges:
        ranges["crossfade_ms"]["max"] = min(ranges["crossfade_ms"]["max"], chunk_ms // 2)
    return ranges


def BuildState() -> dict:
    return {
        "params": State.Engine.GetParams(),
        "running": bool(State.Engine.IsRunning),
        "reference": State.Reference,
        "presets": ListPresets(),
        "ranges": BuildRanges(),
    }


@contextlib.asynccontextmanager
async def Lifespan(app: FastAPI):
    PresetsDir.mkdir(parents=True, exist_ok=True)
    ReferenceDir.mkdir(parents=True, exist_ok=True)
    VoicesDir.mkdir(exist_ok=True)
    State.Vc = await run_in_threadpool(VCEngine)
    await run_in_threadpool(State.Vc.Load)
    State.Engine = await run_in_threadpool(AudioEngine, State.Vc, params.Defaults())
    State.TelemetryTask = asyncio.create_task(_TelemetryLoop())
    try:
        yield
    finally:
        State.TelemetryTask.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await State.TelemetryTask
        if State.Engine.IsRunning:
            await run_in_threadpool(State.Engine.Stop)


app = FastAPI(lifespan=Lifespan)
app.mount("/static", StaticFiles(directory=FrontendDir), name="static")


@app.exception_handler(ApiError)
async def _OnApiError(request, exc: ApiError):
    return JSONResponse({"error": exc.message}, status_code=exc.status)


@app.exception_handler(StarletteHTTPException)
async def _OnHttpError(request, exc: StarletteHTTPException):
    return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def _OnValidationError(request, exc: RequestValidationError):
    return JSONResponse({"error": "invalid request: " + str(exc.errors()[0].get("msg", "bad input"))}, status_code=400)


@app.get("/")
async def Index():
    return FileResponse(FrontendDir / "index.html")


@app.get("/api/devices")
async def GetDevices():
    return await run_in_threadpool(State.Engine.ListDevices)


@app.get("/api/state")
async def GetState():
    return BuildState()


@app.get("/api/presets")
async def GetPresets():
    return ListPresets()


@app.post("/api/reference")
async def PostReference(file: UploadFile = File(...)):
    filename = file.filename or ""
    if not filename.lower().endswith(".wav"):
        raise ApiError("reference must be a .wav file")
    data = await file.read(MaxUploadBytes + 1)
    if len(data) > MaxUploadBytes:
        raise ApiError("reference exceeds the 5 MB upload limit", 413)
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ApiError("file is not a valid RIFF/WAVE")
    staged = ReferenceDir / "upload.tmp.wav"
    await run_in_threadpool(_StageAndLoad, data, staged, Path(filename).stem[:40])
    return State.Reference


@app.post("/api/presets/save")
async def SavePreset(body: dict):
    name = _PresetName(body)
    if State.Reference is None or not CurrentReference.is_file():
        raise ApiError("load a reference clip before saving a preset")
    await run_in_threadpool(shutil.copyfile, CurrentReference, PresetsDir / (name + ".wav"))
    State.Reference = {**State.Reference, "name": name}
    return {"presets": ListPresets(), "reference": State.Reference}


@app.post("/api/presets/load")
async def LoadPreset(body: dict):
    name = _PresetName(body)
    path = PresetsDir / (name + ".wav")
    if not path.is_file():
        raise ApiError("preset not found: " + name, 404)
    await run_in_threadpool(_LoadReferenceFile, path, name)
    return State.Reference


@app.delete("/api/presets/{name}")
async def DeletePreset(name: str):
    if not PresetNamePattern.fullmatch(name):
        raise ApiError("invalid preset name")
    path = PresetsDir / (name + ".wav")
    if not path.is_file():
        raise ApiError("preset not found: " + name, 404)
    path.unlink()
    return {"presets": ListPresets()}


@app.get("/api/presets/{name}/download")
async def DownloadPreset(name: str):
    if not PresetNamePattern.fullmatch(name):
        raise ApiError("invalid preset name")
    path = PresetsDir / (name + ".wav")
    if not path.is_file():
        raise ApiError("preset not found: " + name, 404)
    return FileResponse(path, media_type="audio/wav", filename=name + ".wav")


@app.post("/api/voicegen/start")
async def PostVoiceGenStart(body: dict):
    job = _StartVoiceGen(body.get("prompt"), body.get("feedback"), body.get("parent_id"))
    return {"job_id": job["id"]}


@app.get("/api/voicegen/{job_id}")
async def GetVoiceGen(job_id: str):
    return State.VoiceGen.Snapshot(State.VoiceGen.Get(job_id))


@app.get("/api/voicegen/{job_id}/clip.wav")
async def GetVoiceGenClip(job_id: str):
    job = State.VoiceGen.Get(job_id)
    if job["status"] != "ready" or not job["clip_path"] or not Path(job["clip_path"]).is_file():
        raise ApiError("voice clip is not ready", 409)
    return FileResponse(job["clip_path"], media_type="audio/wav")


@app.post("/api/voicegen/{job_id}/accept")
async def PostVoiceGenAccept(job_id: str, body: dict):
    name = _PresetName(body)
    job = State.VoiceGen.Get(job_id)
    if job["status"] != "ready" or not Path(job["clip_path"]).is_file():
        raise ApiError("voice clip is not ready", 409)
    await run_in_threadpool(_AcceptClip, Path(job["clip_path"]), name)
    return {"presets": ListPresets(), "reference": State.Reference}


@app.post("/api/voicegen/{job_id}/reject")
async def PostVoiceGenReject(job_id: str, body: dict):
    parent = State.VoiceGen.Get(job_id)
    if parent["status"] != "ready":
        raise ApiError("only a finished voice can be rejected", 409)
    job = _StartVoiceGen(parent["prompt"], body.get("feedback"), parent["id"])
    return {"job_id": job["id"]}


@app.post("/api/start")
async def PostStart(body: dict):
    input_id = body.get("input_id")
    output_id = body.get("output_id")
    for label, value in (("input_id", input_id), ("output_id", output_id)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ApiError(label + " must be an integer device id")
    if State.Reference is None:
        raise ApiError("load a reference clip before starting")
    if State.Engine.IsRunning:
        raise ApiError("engine is already running", 409)
    if State.VoiceGen.Active():
        raise ApiError("wait for voice creation to finish before starting the engine", 409)
    try:
        await run_in_threadpool(State.Engine.Start, input_id, output_id)
    except (ValueError, RuntimeError, OSError) as exc:
        raise ApiError("start failed: " + str(exc), 400)
    return {"running": bool(State.Engine.IsRunning)}


@app.post("/api/stop")
async def PostStop():
    await run_in_threadpool(State.Engine.Stop)
    return {"running": bool(State.Engine.IsRunning)}


@app.websocket("/ws")
async def Socket(socket: WebSocket):
    await socket.accept()
    client = Client(socket)
    State.Clients.add(client)
    try:
        while True:
            raw = await socket.receive_text()
            await _HandleMessage(client, raw)
    except WebSocketDisconnect:
        pass
    finally:
        State.Clients.discard(client)


def main() -> None:
    uvicorn.run(app, host=Host, port=Port, log_level="warning")


# ---------------------------------------------------------------- PRIVATE

def _Coerce(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError("not JSON serialisable: " + type(value).__name__)


def _PresetName(body: dict) -> str:
    name = body.get("name") if isinstance(body, dict) else None
    if not isinstance(name, str) or not PresetNamePattern.fullmatch(name):
        raise ApiError("preset name must match [A-Za-z0-9_-]{1,40}")
    return name


def _StartVoiceGen(prompt, feedback, parent_id):
    if not isinstance(prompt, str) or not VoiceGenPromptRange[0] <= len(prompt.strip()) <= VoiceGenPromptRange[1]:
        raise ApiError("prompt must be %d to %d characters" % VoiceGenPromptRange)
    if feedback is not None and (not isinstance(feedback, str) or len(feedback) > VoiceGenFeedbackMax):
        raise ApiError("feedback must be at most %d characters" % VoiceGenFeedbackMax)
    if parent_id is not None and not isinstance(parent_id, str):
        raise ApiError("parent_id must be a string")
    rejected_ids = []
    if parent_id is not None:
        parent = State.VoiceGen.Get(parent_id)
        rejected_ids = list(parent["rejected_ids"])
        if parent["result"] and parent["result"]["source_id"] not in rejected_ids:
            rejected_ids.append(parent["result"]["source_id"])
    if State.Engine.IsRunning:
        raise ApiError(EngineRunningMessage, 409)
    job = State.VoiceGen.Create(prompt.strip(), (feedback or "").strip(), parent_id, rejected_ids)
    threading.Thread(target=_RunVoiceGen, args=(job,), name="voicegen-" + job["id"], daemon=True).start()
    return job


def _RunVoiceGen(job: dict) -> None:
    jobs = State.VoiceGen
    jobs.SetStatus(job, "running")
    try:
        from voice_creator import VoiceCreator
        creator = VoiceCreator(State.Vc, str(VoicesDir))
        result = creator.Run(
            job["id"],
            job["prompt"],
            feedback=job["feedback"] or None,
            rejected_ids=list(job["rejected_ids"]),
            on_progress=lambda message: jobs.Log(job, message),
        )
        clip = Path(result["clip_path"]).resolve()
        if clip.parent != VoicesDir.resolve() or not clip.is_file():
            raise RuntimeError("voice creator returned a clip outside the voices folder")
        _AdoptReference(clip, "voice-" + job["id"], result["reference_info"])
        public = {key: value for key, value in result.items() if key != "clip_path"}
        jobs.SetStatus(job, "ready", result=public, clip_path=str(clip))
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        jobs.Log(job, "failed: " + str(exc))
        jobs.SetStatus(job, "failed", error=str(exc))
    except Exception as exc:
        jobs.Log(job, "crashed: %s: %s" % (type(exc).__name__, exc))
        jobs.SetStatus(job, "failed", error="voice creation crashed: %s: %s" % (type(exc).__name__, exc))


def _AdoptReference(clip: Path, name: str, info: dict) -> None:
    with State.ReferenceLock:
        shutil.copyfile(clip, CurrentReference)
        State.Reference = {"name": name, **info}


def _AcceptClip(clip: Path, name: str) -> None:
    target = PresetsDir / (name + ".wav")
    shutil.copyfile(clip, target)
    _LoadReferenceFile(target, name)


def _StageAndLoad(data: bytes, staged: Path, display_name: str) -> None:
    with State.ReferenceLock:
        staged.write_bytes(data)
        try:
            info = State.Vc.LoadReference(str(staged))
        except ValueError as exc:
            staged.unlink(missing_ok=True)
            raise ApiError(str(exc), 400)
        os.replace(staged, CurrentReference)
        State.Reference = {"name": display_name, **info}


def _LoadReferenceFile(path: Path, name: str) -> None:
    with State.ReferenceLock:
        try:
            info = State.Vc.LoadReference(str(path))
        except ValueError as exc:
            raise ApiError(str(exc), 400)
        shutil.copyfile(path, CurrentReference)
        State.Reference = {"name": name, **info}


async def _HandleMessage(client: Client, raw: str) -> None:
    try:
        message = json.loads(raw)
    except ValueError:
        await client.Send({"type": "error", "key": None, "message": "malformed JSON"})
        return
    if not isinstance(message, dict) or message.get("type") != "set":
        await client.Send({"type": "error", "key": None, "message": "unsupported message"})
        return
    key = message.get("key")
    try:
        if not isinstance(key, str):
            raise ValueError("key must be a string")
        value = params.Validate(key, message.get("value"))
        await run_in_threadpool(State.Engine.SetParam, key, value)
    except (ValueError, KeyError) as exc:
        await client.Send({"type": "error", "key": key if isinstance(key, str) else None, "message": str(exc)})
        return
    applied = State.Engine.GetParams()[key]
    await asyncio.gather(
        *(c.Send({"type": "ack", "key": key, "value": applied}) for c in list(State.Clients)),
        return_exceptions=True,
    )


async def _TelemetryLoop() -> None:
    interval = 1.0 / TelemetryHz
    while True:
        await asyncio.sleep(interval)
        if not State.Clients:
            continue
        message = {"type": "telemetry", **State.Engine.Telemetry(), **State.Engine.Levels(), "running": bool(State.Engine.IsRunning)}
        clients = list(State.Clients)
        results = await asyncio.gather(*(c.Send(message) for c in clients), return_exceptions=True)
        for client, result in zip(clients, results):
            if isinstance(result, Exception):
                State.Clients.discard(client)


if __name__ == "__main__":
    main()
