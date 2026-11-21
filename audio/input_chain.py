"""Microphone capture for the Seed-VC voice changer.

WASAPI only: Exclusive first, then Low-Latency Shared with the raw stream option. The capture callback only copies
into a preallocated ring; a Janitor-registered worker runs HPF 80 Hz -> compressor 2.5:1 -> sidetone MonitorRing (device
rate) -> resample_poly to 16 kHz -> Silero VAD gate -> a 16 kHz block queue read through ReadBlock().
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

import seedvc_params as P
from audio.dsp import HighPass, SoftCompressor
from audio.streaming import OpenStarted, Ring, StreamResampler
from audio.vad import SileroVad

# ---- PUBLIC
RECORD_SR = 24000            # StopRecording() returns float32 mono at this rate
CAPTURE_RING_S = 2.0         # capture ring at the device rate
OUTPUT_RING_S = 6.0          # 16 kHz output ring (oldest audio is dropped and counted when the reader falls behind)
MONITOR_RING_S = 1.0
MONITOR_MAX_LAG_S = 0.08     # a monitor reader further behind than this skips to the newest MONITOR_RESYNC_FILL_S
MONITOR_RESYNC_FILL_S = 0.02
LEVEL_WINDOW_S = 0.5         # LevelSnapshot covers about the last 0.5 s
WAVEFORM_POINTS = 100
CLIP_LEVEL = 0.999           # spec: clipping = any |sample| >= 0.999
TIMESTAMP_SLOTS = 2048       # capture-callback history used to map device frames to time.perf_counter()
WORKER_POLL_S = 0.002
WORKER_MAX_CHUNK_S = 0.05
EXCLUSIVE_RATES = (48000, 44100, 96000, 32000, 24000, 22050, 16000)   # tried after the device's native rate


@dataclass
class InputBlock:
    samples16k: np.ndarray   # float32, exactly n samples, exact zeros where the VAD is inactive
    active: bool             # any speech in the block or its hang tail
    t_capture: float         # time.perf_counter() of the block's first sample at the ADC
    rms_db: float            # of the block before gating
    peak: float              # 0..1, before gating


@dataclass(frozen=True)
class _OpenPlan:
    key: str            # name under which a failure is reported in open_errors
    mode: str           # "exclusive" or "shared_lowlatency"
    rate: int
    channels: int
    settings: object    # sd.WasapiSettings
    raw: bool           # whether the raw stream option was requested


class MonitorRing:
    """Post-compressor, pre-VAD mono signal at the device rate for OutputChain's sidetone stream.

    Write() is called by the InputChain worker, Read(n) by the monitor stream callback. Nobody ever blocks: the
    ring overwrites when unread, a reader more than MONITOR_MAX_LAG_S behind skips to the newest samples (skips),
    and a reader that finds fewer than n samples gets the rest as zeros (underruns).
    """

    def __init__(self, sample_rate: int = 48000):
        self.Configure(sample_rate)

    def Configure(self, sample_rate: int) -> None:
        """Set the device rate (InputChain.Start() calls this once the stream format is known) and clear the ring."""
        self.sample_rate = sample_rate
        self._ring = Ring(int(sample_rate * MONITOR_RING_S))
        self._w = 0
        self._r = 0
        self.underruns = 0
        self.skips = 0

    def Write(self, x: np.ndarray) -> None:
        cap = self._ring.capacity
        if x.size > cap:
            self._w += x.size - cap
            x = x[-cap:]
        self._ring.Put(self._w, x)
        self._w += x.size

    def Read(self, n: int) -> np.ndarray:
        out = np.zeros(n, np.float32)
        w = self._w
        avail = w - self._r
        if avail > max(int(self.sample_rate * MONITOR_MAX_LAG_S), 3 * n):
            self._r = w - max(int(self.sample_rate * MONITOR_RESYNC_FILL_S), n)
            avail = w - self._r
            self.skips += 1
        k = min(n, avail)
        if k < n:
            self.underruns += 1
        if k > 0:
            self._ring.Get(self._r, k, out)
            self._r += k
        return out


class InputChain:
    """Opens the microphone (WASAPI only) and serves processed 16 kHz blocks; see the module docstring."""

    # ---- PUBLIC
    def __init__(self, params, janitor, device_name: str | None = None):
        self._params = params
        self._device_filter = device_name
        self._life = threading.Lock()
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._vad = SileroVad()
        self._monitor = MonitorRing()
        self._running = False
        self._stream = None
        self._worker = None
        self._rec = None
        self._worker_error = None
        self._mode = None
        self._device = None
        self._device_id = None
        self._rate = None
        self._channels = None
        self._bypassed = None
        self._raw_requested = False
        self._stream_latency_ms = None
        self._attempts = {}
        self._comp = None
        self._cb_frames = 0
        self._pa_overruns = self._ring_overruns = self._out_overruns = 0
        self._out_dropped = 0
        self._ts_fallbacks = 0
        self._work_s = self._work_max_s = 0.0
        self._work_n = 0
        janitor.Add(self.Stop)

    @property
    def MonitorRing(self) -> MonitorRing:
        """Exists from construction; its sample_rate is final once Start() has returned."""
        return self._monitor

    def ListDevices(self) -> list[dict]:
        wasapi = self._WasapiIndex()
        host = sd.query_hostapis(wasapi)["name"]
        return [{"id": i, "name": d["name"], "hostapi": host, "default_samplerate": d["default_samplerate"],
                 "max_input_channels": d["max_input_channels"]}
                for i, d in enumerate(sd.query_devices()) if d["hostapi"] == wasapi and d["max_input_channels"] > 0]

    def Start(self) -> None:
        with self._life:
            if self._running:
                return
            dev = self._ResolveDevice()
            info = sd.query_devices(dev)
            native_sr, native_ch = int(round(info["default_samplerate"])), int(info["max_input_channels"])
            stream, mode, rate, ch, raw = self._OpenStream(dev, native_sr, native_ch)
            self._stream, self._mode, self._rate, self._channels = stream, mode, rate, ch
            self._device, self._device_id, self._raw_requested = info["name"], dev, raw
            self._bypassed = True if mode == "exclusive" else None   # shared: PortAudio does not report whether Windows honoured RAW
            self._stream_latency_ms = round(float(stream.latency) * 1000.0, 2)
            self._BuildChain(rate)
            self._stop.clear()
            self._worker = threading.Thread(target=self._Run, name="InputChainWorker", daemon=True)
            self._worker.start()
            self._running = True

    def Stop(self) -> None:
        with self._life:
            if not self._running and self._stream is None:
                return
            self._running = False
            self._stop.set()
            stream, self._stream = self._stream, None
            try:
                if stream is not None:
                    stream.stop()
            finally:
                if stream is not None:
                    stream.close()
                with self._cond:
                    self._cond.notify_all()
                worker, self._worker = self._worker, None
                if worker is not None and worker is not threading.current_thread():
                    worker.join(timeout=2.0)
                self._rec = None

    def ReadBlock(self, n_samples16k: int, timeout_s: float) -> InputBlock | None:
        """Blocks until n post-chain 16 kHz samples exist; None on timeout, when stopped, or when not started."""
        if not self._running:
            return None
        if not 0 < n_samples16k <= self._out_cap:
            raise ValueError(f"n_samples16k must be in 1..{self._out_cap}")
        deadline = time.perf_counter() + timeout_s
        with self._cond:
            while self._ow - self._or < n_samples16k:
                if self._worker_error:
                    raise RuntimeError(f"InputChain worker died: {self._worker_error}")
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0 or self._stop.is_set():
                    return None
                self._cond.wait(remaining)
            start = self._or
            pre = self._pre.Get(start, n_samples16k)
            gated = self._gated.Get(start, n_samples16k)
            flags = self._flags.Get(start, n_samples16k)
            self._or = start + n_samples16k
        rms = float(np.sqrt(np.mean(np.square(pre, dtype=np.float64))))
        return InputBlock(samples16k=gated, active=bool(flags.any()),
                          t_capture=self._CaptureTime(start * self._resampler.down / self._resampler.up),
                          rms_db=20.0 * math.log10(max(rms, 1e-9)), peak=min(1.0, float(np.abs(pre).max())))

    def Telemetry(self) -> dict:
        comp = self._comp.GetState() if self._comp is not None else {"gain_reduction_db": 0.0, "makeup_db": 0.0}
        return {
            "mode": self._mode,
            "device": self._device,
            "samplerate": self._rate,
            "buffer_ms": round(self._cb_frames / self._rate * 1000.0, 2) if self._rate else 0.0,
            "vad_prob": self._vad.last_prob,
            "vad_active": self._vad.last_active,
            "gain_reduction_db": comp["gain_reduction_db"],
            "makeup_db": comp["makeup_db"],
            "overruns": self._pa_overruns + self._ring_overruns + self._out_overruns,
            "enhancements_bypassed": self._bypassed,
        }

    def Diagnostics(self) -> dict:
        """Extra detail beyond the contract's Telemetry(): negotiation errors, overrun breakdown, worker timing."""
        return {
            "device_id": self._device_id, "channels": self._channels, "raw_requested": self._raw_requested,
            "open_errors": dict(self._attempts), "stream_latency_ms": self._stream_latency_ms,
            "callback_frames": self._cb_frames, "overruns_portaudio": self._pa_overruns,
            "overruns_capture_ring": self._ring_overruns, "overruns_output_ring": self._out_overruns,
            "dropped_16k_samples": self._out_dropped, "timestamp_fallbacks": self._ts_fallbacks,
            "worker_ms_mean": round(1000.0 * self._work_s / self._work_n, 3) if self._work_n else 0.0,
            "worker_ms_max": round(1000.0 * self._work_max_s, 3), "worker_chunks": self._work_n,
            "worker_error": self._worker_error, "monitor_underruns": self._monitor.underruns,
            "monitor_skips": self._monitor.skips, "adc_minus_current_ms": self._AdcMinusCurrentMs(),
        }

    def StartRecording(self, max_seconds: float = P.RECORDER_MAX_S) -> None:
        if not self._running:
            raise RuntimeError("InputChain.Start() must be called before StartRecording()")
        self._rec = _Recorder(self._rate, min(max_seconds, P.RECORDER_MAX_S))

    def StopRecording(self) -> np.ndarray:
        """Raw mic (HPF only: no compressor, no gate) as float32 mono at 24000 Hz."""
        rec, self._rec = self._rec, None
        return rec.Result() if rec is not None else np.zeros(0, np.float32)

    def LevelSnapshot(self) -> dict:
        """peak and clipping come from the raw device samples (max over channels), waveform from the HPF signal."""
        if not self._running:
            return {"peak": 0.0, "clipping": False, "waveform": [0.0] * WAVEFORM_POINTS}
        peak = float(self._lvl_raw.Snapshot().max())
        wave = self._lvl_wave.Snapshot().reshape(WAVEFORM_POINTS, -1)
        picked = wave[np.arange(WAVEFORM_POINTS), np.abs(wave).argmax(axis=1)]
        return {"peak": peak, "clipping": peak >= CLIP_LEVEL, "waveform": [round(float(v), 4) for v in picked]}

    # ---- PRIVATE
    @staticmethod
    def _WasapiIndex() -> int:
        for i, h in enumerate(sd.query_hostapis()):
            if "WASAPI" in h["name"]:
                return i
        raise RuntimeError("PortAudio has no WASAPI host API")

    def _ResolveDevice(self) -> int:
        devices = self.ListDevices()
        if self._device_filter:
            want = self._device_filter.lower()
            hits = [d for d in devices if d["name"].lower() == want] or [d for d in devices if want in d["name"].lower()]
            if not hits:
                raise RuntimeError(f"no WASAPI input device matches {self._device_filter!r}; available: {[d['name'] for d in devices]}")
            return hits[0]["id"]
        default = sd.query_hostapis(self._WasapiIndex())["default_input_device"]
        names = {d["id"]: d["name"].lower() for d in devices}
        if default in names and "cable" not in names[default]:
            return default
        others = [i for i, n in names.items() if "cable" not in n]   # a VB-Audio CABLE input would capture our own output
        if not others:
            raise RuntimeError(f"no usable WASAPI microphone; available: {list(names.values())}")
        return others[0]

    def _ExclusiveFormat(self, dev: int, native_sr: int, native_ch: int):
        rates = [native_sr] + [r for r in EXCLUSIVE_RATES if r != native_sr]
        chans = [native_ch] + [c for c in (2, 1) if c < native_ch]
        last = "no format tried"
        for rate in rates:
            for ch in chans:
                try:
                    sd.check_input_settings(device=dev, channels=ch, dtype="float32", samplerate=rate,
                                            extra_settings=sd.WasapiSettings(exclusive=True))
                    return rate, ch
                except sd.PortAudioError as e:
                    last = str(e)
        self._attempts["exclusive"] = last
        return None

    @staticmethod
    def _RequestRaw(settings) -> bool:
        """AUDCLNT_STREAMOPTIONS_RAW through PortAudio's PaWasapiStreamInfo.streamOption (sounddevice has no keyword for it)."""
        try:
            settings._streaminfo.streamOption = sd._lib.eStreamOptionRaw
        except (AttributeError, TypeError):
            return False
        return True

    def _OpenStream(self, dev: int, native_sr: int, native_ch: int):
        """Try each open plan in order; the first stream that starts wins. Failures are kept per plan in open_errors."""
        self._attempts = {}
        for plan in self._OpenPlans(dev, native_sr, native_ch):
            try:
                stream = self._TryOpen(dev, plan.rate, plan.channels, plan.settings)
                return stream, plan.mode, plan.rate, plan.channels, plan.raw
            except sd.PortAudioError as e:
                self._attempts[plan.key] = str(e)
        raise RuntimeError(f"could not open WASAPI input device {dev}: {self._attempts}")

    def _OpenPlans(self, dev: int, native_sr: int, native_ch: int):
        """Exclusive first; then Low-Latency Shared at the device's native format with the raw option; then plain shared.
        Lazy, so a plan's probing (exclusive format negotiation, raw option) only runs when the plans before it have failed."""
        fmt = self._ExclusiveFormat(dev, native_sr, native_ch)
        if fmt is not None:
            yield _OpenPlan("exclusive", "exclusive", fmt[0], fmt[1], sd.WasapiSettings(exclusive=True), False)
        shared = sd.WasapiSettings(exclusive=False)
        raw = self._RequestRaw(shared)
        yield _OpenPlan("shared_lowlatency_raw" if raw else "shared_lowlatency", "shared_lowlatency", native_sr, native_ch, shared, raw)
        if raw:
            yield _OpenPlan("shared_lowlatency", "shared_lowlatency", native_sr, native_ch, sd.WasapiSettings(exclusive=False), False)

    def _TryOpen(self, dev: int, rate: int, ch: int, settings):
        self._Allocate(rate, ch)
        return OpenStarted(lambda: sd.InputStream(device=dev, samplerate=rate, channels=ch, dtype="float32", blocksize=0,
                                                  latency="low", callback=self._Callback, extra_settings=settings))

    def _Allocate(self, rate: int, ch: int) -> None:
        self._cap = int(CAPTURE_RING_S * rate)
        self._ring = Ring(self._cap, (ch,))
        self._w = 0
        self._r = 0
        self._ts_frame = np.zeros(TIMESTAMP_SLOTS, np.int64)
        self._ts_len = np.zeros(TIMESTAMP_SLOTS, np.int64)
        self._ts_adc = np.zeros(TIMESTAMP_SLOTS)
        self._ts_cur = np.zeros(TIMESTAMP_SLOTS)
        self._ts_now = np.zeros(TIMESTAMP_SLOTS)
        self._ts_n = 0
        self._pa_overruns = self._ring_overruns = self._cb_frames = 0

    def _BuildChain(self, rate: int) -> None:
        self._hpf = HighPass(rate)
        self._comp = SoftCompressor(rate)
        self._vad.Reset()
        self._resampler = StreamResampler(rate, P.INPUT_SR)
        self._monitor.Configure(rate)
        self._max_chunk = max(1, int(rate * WORKER_MAX_CHUNK_S))
        self._out_cap = int(OUTPUT_RING_S * P.INPUT_SR)
        self._pre = Ring(self._out_cap)
        self._gated = Ring(self._out_cap)
        self._flags = Ring(self._out_cap, dtype=np.uint8)
        self._ow = 0
        self._or = 0
        self._out_overruns = self._out_dropped = 0
        self._acc = np.zeros(8192, np.float32)
        self._acc_fill = 0
        window = WAVEFORM_POINTS * max(1, round(rate * LEVEL_WINDOW_S / WAVEFORM_POINTS))
        self._lvl_raw = _LevelRing(window)
        self._lvl_wave = _LevelRing(window)
        self._ts_fallbacks = 0
        self._work_s = self._work_max_s = 0.0
        self._work_n = 0
        self._worker_error = None
        self._rec = None

    def _Callback(self, indata, frames, time_info, status) -> None:
        # PortAudio thread: copy into the ring and note timestamps. No locks, no allocation beyond scalars.
        if status and status.input_overflow:
            self._pa_overruns += 1
        w = self._w
        cap = self._cap
        if w - self._r + frames > cap:
            self._ring_overruns += 1
            return
        self._ring.Put(w, indata)
        k = self._ts_n % TIMESTAMP_SLOTS
        self._ts_frame[k] = w
        self._ts_len[k] = frames
        self._ts_adc[k] = time_info.inputBufferAdcTime
        self._ts_cur[k] = time_info.currentTime
        self._ts_now[k] = time.perf_counter()
        self._ts_n += 1
        self._cb_frames = frames
        self._w = w + frames

    def _Run(self) -> None:
        try:
            while not self._stop.is_set():
                avail = self._w - self._r
                if avail <= 0:
                    time.sleep(WORKER_POLL_S)
                    continue
                t0 = time.perf_counter()
                self._Process(self._ReadCapture(min(avail, self._max_chunk)))
                dt = time.perf_counter() - t0
                self._work_s += dt
                self._work_n += 1
                if dt > self._work_max_s:
                    self._work_max_s = dt
        except Exception as e:  # noqa: BLE001 - surfaced to the reader through ReadBlock
            self._worker_error = f"{type(e).__name__}: {e}"
            with self._cond:
                self._cond.notify_all()

    def _ReadCapture(self, take: int) -> np.ndarray:
        out = self._ring.Get(self._r, take)
        self._r += take
        return out

    def _Process(self, raw: np.ndarray) -> None:
        self._lvl_raw.Write(np.abs(raw).max(axis=1))
        mono = np.ascontiguousarray(raw[:, 0]) if raw.shape[1] == 1 else raw.mean(axis=1, dtype=np.float32)
        hp = self._hpf.Process(mono)
        self._lvl_wave.Write(hp)
        rec = self._rec
        if rec is not None:
            rec.Feed(hp)
        comp = self._comp.Process(hp)
        self._monitor.Write(comp)
        self._GateFrames(self._resampler.Process(comp))

    def _GateFrames(self, y16: np.ndarray) -> None:
        frame = P.VAD_FRAME_SAMPLES
        fill = self._acc_fill
        self._acc[fill:fill + y16.size] = y16
        fill += y16.size
        k = fill // frame
        if k:
            pre = self._acc[:k * frame].copy()
            gated = np.empty_like(pre)
            flags = np.empty(k * frame, np.uint8)
            for i in range(k):
                sl = slice(i * frame, (i + 1) * frame)
                gated[sl], _, active = self._vad.Gate(pre[sl])
                flags[sl] = active
            self._acc[:fill - k * frame] = self._acc[k * frame:fill]
            self._OutputAppend(pre, gated, flags)
        self._acc_fill = fill - k * frame

    def _OutputAppend(self, pre: np.ndarray, gated: np.ndarray, flags: np.ndarray) -> None:
        n = pre.size
        with self._cond:
            over = self._ow - self._or + n - self._out_cap
            if over > 0:
                self._or += over
                self._out_dropped += over
                self._out_overruns += 1
            for ring, src in ((self._pre, pre), (self._gated, gated), (self._flags, flags)):
                ring.Put(self._ow, src)
            self._ow += n
            self._cond.notify_all()

    def _AdcMinusCurrentMs(self) -> dict | None:
        count = min(self._ts_n, TIMESTAMP_SLOTS) if self._rate else 0
        if count == 0:
            return None
        d = (self._ts_adc[:count] - self._ts_cur[:count]) * 1000.0
        return {"min": round(float(d.min()), 2), "mean": round(float(d.mean()), 2), "max": round(float(d.max()), 2)}

    def _CaptureTime(self, device_frame: float) -> float:
        """time.perf_counter() of a device frame: the PortAudio ADC time of its callback, moved onto the perf_counter
        timeline by the smallest (perf_counter - stream clock) offset seen (the callback with the least scheduling delay)."""
        n = self._ts_n
        count = min(n, TIMESTAMP_SLOTS)
        if count == 0:
            return time.perf_counter()
        idx = (np.arange(count) + (n - count)) % TIMESTAMP_SLOTS
        frames, adc, cur, now, lens = (a[idx] for a in (self._ts_frame, self._ts_adc, self._ts_cur, self._ts_now, self._ts_len))
        j = max(int(np.searchsorted(frames, device_frame, side="right")) - 1, 0)
        offset = float(np.min(now - cur))
        if 0.0 <= cur[j] - adc[j] < 1.0:
            first_sample = float(adc[j]) + offset
        else:
            # PortAudio WASAPI reports inputBufferAdcTime after currentTime (measured +8..12 ms), so it is not a capture
            # time. Use the callback's stream-clock time minus the stream's input latency instead (the age of a buffer's
            # first sample when its callback fires).
            first_sample = float(cur[j]) + offset - self._stream_latency_ms / 1000.0
            self._ts_fallbacks += 1
        return first_sample + (device_frame - float(frames[j])) / self._rate


# ---- PRIVATE
class _LevelRing:
    """Sliding window of the most recent samples (about LEVEL_WINDOW_S) for LevelSnapshot()."""

    def __init__(self, size: int):
        self._buf = np.zeros(size, np.float32)

    def Write(self, x: np.ndarray) -> None:
        k = min(x.size, self._buf.size)
        if k:
            self._buf[:-k] = self._buf[k:]
            self._buf[-k:] = x[-k:]

    def Snapshot(self) -> np.ndarray:
        return self._buf.copy()


class _Recorder:
    """Collects the HPF-only mic signal at 24 kHz, up to max_seconds. Fed by the worker, read by StopRecording()."""

    def __init__(self, rate_in: int, max_seconds: float):
        self._resampler = StreamResampler(rate_in, RECORD_SR)
        self._max = int(round(max_seconds * RECORD_SR))
        self._chunks: list[np.ndarray] = []
        self._n = 0
        self._lock = threading.Lock()

    def Feed(self, hp: np.ndarray) -> None:
        with self._lock:
            if self._n >= self._max:
                return
            y = self._resampler.Process(hp)[:self._max - self._n]
            if y.size:
                self._chunks.append(y.copy())
                self._n += y.size

    def Result(self) -> np.ndarray:
        with self._lock:
            return np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)
