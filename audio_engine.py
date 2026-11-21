"""Real-time audio engine: mic -> VAD -> VC -> SOLA -> virtual cable.

Threading model
  input callback  -> InputRing (producer)         [PortAudio thread]
  worker thread   -> reads InputRing history, VAD, VC, SOLA, writes OutputRing
  output callback <- OutputRing (consumer)        [PortAudio thread]

The rings are single-producer / single-consumer with preallocated numpy storage and integer
counters. They contain no lock, but they are NOT a hardware lock-free structure: they are only
safe because CPython's GIL makes each counter store atomic, and each counter has exactly one
writer. The worker polls with a 1 ms sleep instead of waiting on an Event, so no lock is taken
inside an audio callback. queue.Queue is not used anywhere on the audio path.
"""
from __future__ import annotations

import collections
import logging
import threading
import time

import numpy as np
import psutil
import sounddevice as sd

import params as P
from vad_gate import VadGate

LOG = logging.getLogger("audio")

IN_SR = 16000
OUT_SR = 24000
SEARCH_MS = 5
CONTEXT_MS = 800
RING_SECONDS = 12
PREBUFFER_MS = 40
MAX_BACKLOG_HOPS = 3
LATENCY_EMA = 0.2
SILENCE_DB = -120.0
HOST_APIS = ("WASAPI", "ASIO")


# ---------------------------------------------------------------- PUBLIC
def SolaJoin(prev_tail, y, hop, fade, search):
    """Align y to prev_tail by normalised cross-correlation and crossfade.

    prev_tail: float32 (fade,) or None. y: float32 of length >= hop + fade + search.
    Returns (emit, new_tail, offset) with len(emit) == hop and len(new_tail) == fade.
    """
    need = hop + fade + search
    if len(y) < need:
        raise ValueError(f"SolaJoin needs {need} samples, got {len(y)}")
    t = np.arange(fade, dtype=np.float32) / max(fade, 1)
    fade_in = np.sin(0.5 * np.pi * t) ** 2
    if prev_tail is None:
        aligned = y[:hop + fade].copy()
        aligned[:fade] *= fade_in
        return aligned[:hop], aligned[hop:hop + fade].copy(), 0
    seg = np.lib.stride_tricks.sliding_window_view(y[:fade + search], fade)  # (search+1, fade)
    num = seg @ prev_tail
    den = np.sqrt((seg * seg).sum(axis=1)) + 1e-8
    off = int(np.argmax(num / den))
    aligned = y[off:off + hop + fade]
    emit = aligned[:hop].copy()
    emit[:fade] = prev_tail * (1.0 - fade_in) + aligned[:fade] * fade_in
    return emit, aligned[hop:hop + fade].copy(), off


class SpscRing:
    """Single-producer single-consumer float32 ring with absolute sample counters (see module docstring)."""

    def __init__(self, capacity: int):
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._cap = capacity
        self.written = 0
        self.read = 0

    def Write(self, data: np.ndarray) -> None:
        n = len(data)
        if n > self._cap:
            data = data[-self._cap:]
            n = self._cap
        start = self.written % self._cap
        first = min(n, self._cap - start)
        self._buf[start:start + first] = data[:first]
        if first < n:
            self._buf[:n - first] = data[first:]
        self.written += n

    def Available(self) -> int:
        return self.written - self.read

    def Pop(self, n: int) -> np.ndarray:
        """Consume up to n samples; returns fewer than n if the ring runs dry."""
        n = min(n, self.Available())
        out = self.Peek(self.read, n)
        self.read += n
        return out

    def Peek(self, abs_start: int, n: int) -> np.ndarray:
        if abs_start < self.written - self._cap or abs_start + n > self.written:
            raise IndexError("requested samples are not (or no longer) in the ring")
        idx = (abs_start + np.arange(n)) % self._cap
        return self._buf[idx]


class AudioEngine:
    def __init__(self, vc, params: dict | None = None):
        self._vc = vc
        self._params = dict(params or P.Defaults())
        self._params_lock = threading.Lock()
        self._vad = VadGate(self._params["vad_threshold"])
        self._in_ring = SpscRing(IN_SR * RING_SECONDS)
        self._out_ring = SpscRing(OUT_SR * RING_SECONDS)
        self._adc_marks = collections.deque(maxlen=512)
        self._out_tags = collections.deque(maxlen=512)
        self._in_clock_offset = 0.0
        self._out_clock_offset = 0.0
        self._in_stream = None
        self._out_stream = None
        self._worker = None
        self._stop = threading.Event()
        self._out_started = False
        self._out_channels = 1
        self._tap = None
        self._warmed = None
        self._stats = self._FreshStats()
        self._in_db = SILENCE_DB
        self._out_db = SILENCE_DB
        psutil.cpu_percent(None)

    @property
    def IsRunning(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def ListDevices(self) -> dict:
        apis = sd.query_hostapis()
        out = {"inputs": [], "outputs": []}
        for i, dev in enumerate(sd.query_devices()):
            api = apis[dev["hostapi"]]["name"]
            if not any(h in api for h in HOST_APIS):
                continue
            row = {"id": i, "name": dev["name"], "hostapi": api, "default_samplerate": dev["default_samplerate"]}
            if dev["max_input_channels"] > 0:
                out["inputs"].append(row)
            if dev["max_output_channels"] > 0:
                out["outputs"].append(row)
        return out

    def GetParams(self) -> dict:
        with self._params_lock:
            return dict(self._params)

    def SetParam(self, key: str, value):
        with self._params_lock:
            value = P.Validate(key, value, self._params)
            self._params[key] = value
        if key == "vad_threshold":
            self._vad.SetThreshold(value)
        return value

    def Start(self, input_id: int, output_id: int) -> None:
        if self.IsRunning:
            raise RuntimeError("Engine is already running")
        devices = {d["id"]: d for d in self.ListDevices()["inputs"] + self.ListDevices()["outputs"]}
        if input_id not in devices or output_id not in devices:
            raise ValueError("Input and output must be WASAPI or ASIO devices from ListDevices()")
        if input_id == output_id:
            raise ValueError("Input and output must be different devices")
        self._ResetState()
        wasapi = sd.WasapiSettings(auto_convert=True) if "WASAPI" in devices[input_id]["hostapi"] else None
        wasapi_out = sd.WasapiSettings(auto_convert=True) if "WASAPI" in devices[output_id]["hostapi"] else None
        self._in_stream = sd.InputStream(
            device=input_id, samplerate=IN_SR, channels=1, dtype="float32",
            latency="low", callback=self._OnInput, extra_settings=wasapi,
        )
        self._out_stream = self._OpenOutput(output_id, wasapi_out)
        self._WarmUp(self.GetParams())
        self._stop.clear()
        self._worker = threading.Thread(target=self._WorkerLoop, name="vc-worker", daemon=True)
        self._worker.start()
        self._out_stream.start()
        self._in_stream.start()

    def Stop(self) -> None:
        self._stop.set()
        for stream in (self._in_stream, self._out_stream):
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception as exc:
                    LOG.warning("stream close: %s", exc)
        self._in_stream = self._out_stream = None
        if self._worker is not None:
            self._worker.join(timeout=3)
        self._worker = None

    def StartTap(self) -> None:
        """Diagnostic: keep a copy of every sample handed to the output device (not used in normal operation)."""
        self._tap = []

    def StopTap(self) -> np.ndarray:
        chunks, self._tap = self._tap or [], None
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

    def Levels(self) -> dict:
        return {"in_db": round(self._in_db, 1), "out_db": round(self._out_db, 1)}

    def Telemetry(self) -> dict:
        s = self._stats
        chunk_ms = self.GetParams()["chunk_ms"]
        t = {
            "e2e_latency_ms": round(s["e2e_ema"], 1),
            "e2e_latency_last_ms": round(s["e2e_last"], 1),
            "inference_ms": round(s["inference_ms"], 1),
            "chunk_ms": chunk_ms,
            "realtime_ok": bool(s["inference_ms"] < chunk_ms) if s["model_calls"] else None,
            "vad_skipped_chunks": s["vad_skipped"],
            "model_calls": s["model_calls"],
            "underruns": s["underruns"],
            "dropped_chunks": s["dropped"],
            "cpu_pct": psutil.cpu_percent(None),
            "running": self.IsRunning,
            "last_error": s["error"],
        }
        t.update(self._vc.Telemetry())
        t["inference_ms"] = round(s["inference_ms"], 1)
        return t

    def ProcessOffline(self, audio16k: np.ndarray, params: dict | None = None):
        """Run the identical VAD -> VC -> SOLA path over an array (no devices). Returns (audio24k, stats)."""
        if params is not None:
            with self._params_lock:
                self._params.update({k: P.Validate(k, v, self._params) for k, v in params.items()})
            self._vad.SetThreshold(self._params["vad_threshold"])
        self._ResetState()
        p = self.GetParams()
        look = (p["crossfade_ms"] + SEARCH_MS) * IN_SR // 1000
        padded = np.concatenate([audio16k.astype(np.float32), np.zeros(look + CONTEXT_MS * IN_SR // 1000, np.float32)])
        ring = SpscRing(len(padded) + IN_SR)
        ring.Write(np.zeros(CONTEXT_MS * IN_SR // 1000, np.float32))
        ring.Write(padded)
        start = CONTEXT_MS * IN_SR // 1000
        outs = []
        ctx = _StepState()
        while start + p["chunk_ms"] * IN_SR // 1000 + look <= ring.written - 0:
            res = self._Step(ring, start, ctx, p)
            outs.append(res)
            start += p["chunk_ms"] * IN_SR // 1000
            if start + p["chunk_ms"] * IN_SR // 1000 > len(audio16k) + CONTEXT_MS * IN_SR // 1000:
                break
        audio = np.concatenate(outs) if outs else np.zeros(0, np.float32)
        calls = max(self._stats["model_calls"], 1)
        stats = {k: self._stats[k] for k in ("model_calls", "vad_skipped", "inference_ms")}
        stats["inference_ms_mean"] = self._stats["inference_sum"] / calls
        return audio, stats

    # ---------------------------------------------------------------- PRIVATE
    @staticmethod
    def _FreshStats() -> dict:
        return {"e2e_ema": 0.0, "e2e_last": 0.0, "inference_ms": 0.0, "model_calls": 0,
                "vad_skipped": 0, "underruns": 0, "dropped": 0, "error": None, "inference_sum": 0.0}

    def _ResetState(self) -> None:
        self._in_ring = SpscRing(IN_SR * RING_SECONDS)
        self._out_ring = SpscRing(OUT_SR * RING_SECONDS)
        self._adc_marks.clear()
        self._out_tags.clear()
        self._out_started = False
        self._stats = self._FreshStats()
        self._vad.Reset()

    def _OpenOutput(self, device: int, extra):
        for channels in (1, 2):
            try:
                self._out_channels = channels
                return sd.OutputStream(
                    device=device, samplerate=OUT_SR, channels=channels, dtype="float32",
                    latency="low", callback=self._OnOutput, extra_settings=extra,
                )
            except sd.PortAudioError as exc:
                last = exc
        raise RuntimeError(f"Could not open output device at {OUT_SR} Hz: {last}")

    def _OnInput(self, indata, frames, time_info, status) -> None:
        now = time.monotonic()
        self._in_clock_offset = now - time_info.currentTime
        adc = time_info.inputBufferAdcTime + self._in_clock_offset
        mono = indata[:, 0]
        self._adc_marks.append((self._in_ring.written, adc))
        self._in_ring.Write(mono)
        rms = float(np.sqrt(np.mean(mono * mono))) + 1e-9
        self._in_db = 20.0 * np.log10(rms)

    def _OnOutput(self, outdata, frames, time_info, status) -> None:
        now = time.monotonic()
        self._out_clock_offset = now - time_info.currentTime
        if not self._out_started:
            if self._out_ring.Available() >= PREBUFFER_MS * OUT_SR // 1000:
                self._out_started = True
            else:
                outdata.fill(0)
                return
        start = self._out_ring.read
        data = self._out_ring.Pop(frames)
        if len(data) < frames:
            self._stats["underruns"] += 1
            data = np.concatenate([data, np.zeros(frames - len(data), np.float32)])
        dac = time_info.outputBufferDacTime + self._out_clock_offset
        while self._out_tags and self._out_tags[0][0] < start + frames:
            idx, adc = self._out_tags.popleft()
            if idx >= start:
                lat = (dac + (idx - start) / OUT_SR - adc) * 1000.0
                s = self._stats
                s["e2e_last"] = lat
                s["e2e_ema"] = lat if s["e2e_ema"] == 0.0 else (1 - LATENCY_EMA) * s["e2e_ema"] + LATENCY_EMA * lat
        if self._tap is not None:
            self._tap.append(data.copy())
        outdata[:] = data[:, None] if self._out_channels == 1 else np.repeat(data[:, None], self._out_channels, axis=1)
        rms = float(np.sqrt(np.mean(data * data))) + 1e-9
        self._out_db = 20.0 * np.log10(rms)

    def _WarmUp(self, p: dict) -> None:
        """Run the model once on silence for the current window shape so CUDA-graph capture and kernel
        autotuning happen before audio is flowing (a capture inside the stream stalls it for ~1-2 s)."""
        hop = p["chunk_ms"] * IN_SR // 1000
        look = (p["crossfade_ms"] + SEARCH_MS) * IN_SR // 1000
        n_out = (hop + look) * OUT_SR // IN_SR + (SEARCH_MS * OUT_SR // 1000)
        window = np.zeros(CONTEXT_MS * IN_SR // 1000 + hop + look, np.float32)
        for _ in range(2):
            self._vc.ConvertWindow(window, n_out, p)
        self._warmed = (p["chunk_ms"], p["crossfade_ms"])

    def _AdcOf(self, abs_idx: int) -> float:
        for start, adc in reversed(self._adc_marks):
            if start <= abs_idx:
                return adc + (abs_idx - start) / IN_SR
        return self._adc_marks[0][1] if self._adc_marks else time.monotonic()

    def _WorkerLoop(self) -> None:
        ctx = _StepState()
        start = None
        try:
            while not self._stop.is_set():
                p = self.GetParams()
                if self._warmed != (p["chunk_ms"], p["crossfade_ms"]):
                    self._WarmUp(p)
                hop = p["chunk_ms"] * IN_SR // 1000
                look = (p["crossfade_ms"] + SEARCH_MS) * IN_SR // 1000
                if start is None:
                    if self._in_ring.written < hop + look:
                        time.sleep(0.001)
                        continue
                    start = self._in_ring.written - hop - look
                if self._in_ring.written < start + hop + look:
                    time.sleep(0.001)
                    continue
                backlog = self._in_ring.written - (start + hop + look)
                if backlog > MAX_BACKLOG_HOPS * hop:
                    skip = backlog // hop
                    start += skip * hop
                    self._stats["dropped"] += int(skip)
                    ctx.prev_tail = None
                adc = self._AdcOf(start)
                out = self._Step(self._in_ring, start, ctx, p)
                self._out_tags.append((self._out_ring.written, adc))
                self._out_ring.Write(out * p["output_gain"])
                start += hop
        except Exception as exc:
            LOG.exception("worker died")
            self._stats["error"] = f"{type(exc).__name__}: {exc}"

    def _Step(self, ring: SpscRing, start: int, ctx: "_StepState", p: dict) -> np.ndarray:
        """One hop: VAD on the hop, VC (or zeros) over hop+lookahead with context, SOLA join. Returns hop*1.5 samples."""
        hop = p["chunk_ms"] * IN_SR // 1000
        fade = p["crossfade_ms"] * OUT_SR // 1000
        search = SEARCH_MS * OUT_SR // 1000
        look = (p["crossfade_ms"] + SEARCH_MS) * IN_SR // 1000
        hop_out = hop * OUT_SR // IN_SR
        if ctx.prev_tail is not None and len(ctx.prev_tail) != fade:
            ctx.prev_tail = None
        n_out = hop_out + fade + search
        speech = self._vad.IsSpeech(ring.Peek(start, hop))
        if speech:
            ctx_len = min(CONTEXT_MS * IN_SR // 1000, start)
            window = ring.Peek(start - ctx_len, ctx_len + hop + look)
            y = self._vc.ConvertWindow(window, n_out, p)
            self._stats["model_calls"] += 1
            self._stats["inference_ms"] = self._vc.Telemetry()["last_inference_ms"]
            self._stats["inference_sum"] += self._stats["inference_ms"]
        else:
            y = np.zeros(n_out, dtype=np.float32)
            self._stats["vad_skipped"] += 1
        emit, ctx.prev_tail, _ = SolaJoin(ctx.prev_tail, y, hop_out, fade, search)
        return emit


class _StepState:
    def __init__(self):
        self.prev_tail = None
