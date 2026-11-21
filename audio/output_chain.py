"""OutputChain: Sola -> resample to the device rate -> WASAPI 'CABLE Input', plus the sidetone monitor stream.

The output callback never takes a lock: a single-producer ring with preallocated storage and two monotonically growing
frame counters (producer owns the write counter, the callback owns the read counter). The callback counts underruns
(it needed more than the ring held), fills the rest with exact zeros and measures when each block's first sample
reaches the DAC.
"""
from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

import seedvc_params as P
from audio.balancer import AutoBalancer
from audio.sola import Sola
from audio.streaming import OpenStarted, Ring, StreamResampler

# ---- PUBLIC

PLAYOUT_CUSHION_MS = 40.0  # choice: silence queued before a block that restarts playback from an empty ring, so later blocks may arrive this late
RING_SECONDS = 2.0
MAIN_DEVICE_FORBIDDEN = ("16ch",)  # the main stream goes to 'CABLE Input' only


@dataclass
class BlockMeta:
    t_capture: float  # time.perf_counter() at capture of the block's first sample
    block_ms: float
    active: bool = True
    inference_s: float = 0.0


def ResolveWasapiOutput(name_substring: str | None, forbid: tuple = ()) -> dict:
    """Find a WASAPI output device by case-insensitive name substring (None = the WASAPI default output device).
    Raises RuntimeError listing the available names; there is no fallback to another device."""
    apis = [i for i, h in enumerate(sd.query_hostapis()) if "WASAPI" in h["name"]]
    if not apis:
        raise RuntimeError("PortAudio has no Windows WASAPI host API")
    api, devices = apis[0], sd.query_devices()
    outputs = [i for i, d in enumerate(devices) if d["hostapi"] == api and d["max_output_channels"] > 0]
    if name_substring is None:
        index = sd.query_hostapis(api)["default_output_device"]
        if index < 0:
            raise RuntimeError("WASAPI has no default output device")
    else:
        needle = name_substring.lower()
        found = [i for i in outputs if needle in devices[i]["name"].lower()
                 and not any(f.lower() in devices[i]["name"].lower() for f in forbid)]
        if not found:
            raise RuntimeError(f"no WASAPI output device matches {name_substring!r} (forbidden: {forbid}); "
                               f"available: {[devices[i]['name'] for i in outputs]}")
        index = found[0]
    d = devices[index]
    return {"index": index, "name": d["name"], "samplerate": int(round(d["default_samplerate"])), "channels": int(d["max_output_channels"])}


class OutputChain:
    def __init__(self, params: P.ParamStore, janitor: P.Janitor, device_name: str = P.CABLE_OUTPUT_NAME,
                 monitor_device_name: str | None = None, monitor_ring=None):
        self._params, self._janitor = params, janitor
        self._device_name, self._monitor_device_name, self._monitor_ring = device_name, monitor_device_name, monitor_ring
        self.Balancer = AutoBalancer(params)
        self._sola = Sola(P.MODEL_SR, params.Get("block_ms"), params.Get("overlap_ms"), P.SOLA_SEARCH_MS)
        self._sola_dirty = False
        self._mon_enabled = bool(params.Get("monitor_enabled"))
        self._mon_volume = float(params.Get("monitor_volume"))
        self._param_handlers = {  # live parameter -> how this chain reacts; parameters not listed here need no reaction
            "block_ms": self._MarkSolaDirty,
            "overlap_ms": self._MarkSolaDirty,
            "monitor_enabled": self._SetMonitorEnabled,
            "monitor_volume": self._SetMonitorVolume,
        }
        self._lifecycle = threading.Lock()  # Start and Stop only; no callback takes it
        self._main = self._mon = self._ring = self._resampler = None
        self._running = self._registered = False
        self._device, self._sr, self._ch = None, 0, 0
        self._mon_device, self._mon_sr, self._mon_ch, self._mon_error = None, 0, 0, None
        self._ResetCounters()
        params.Subscribe(self._OnParam)

    def Start(self) -> None:
        with self._lifecycle:
            if self._running:
                return
            dev = ResolveWasapiOutput(self._device_name, forbid=MAIN_DEVICE_FORBIDDEN)
            self._device, self._sr, self._ch = dev["name"], dev["samplerate"], dev["channels"]
            self._ring = _Fifo(int(RING_SECONDS * self._sr), self._ch)
            self._resampler = StreamResampler(P.MODEL_SR, self._sr)
            self._sola.Reset()
            self._ResetCounters()
            self._main = OpenStarted(lambda: sd.OutputStream(device=dev["index"], samplerate=self._sr, channels=self._ch, dtype="float32",
                                                             latency="low", callback=self._OnMain,
                                                             extra_settings=sd.WasapiSettings(auto_convert=True)))
            self._running = True
            if not self._registered:
                self._janitor.Add(self.Stop)
                self._registered = True
            if self._monitor_ring is not None:
                self._OpenMonitor()

    def Stop(self) -> None:
        with self._lifecycle:
            streams = [s for s in (self._mon, self._main) if s is not None]
            self._mon = self._main = None
            self._running = False
        first_error = None
        for s in streams:
            for close in (s.abort, s.close):
                try:
                    close()
                except Exception as e:
                    first_error = first_error or e
        if first_error is not None:
            raise first_error

    def Submit(self, wav22k: np.ndarray, meta: BlockMeta) -> None:
        self._RequireRunning()
        self._ApplyParamChanges()
        self._Enqueue(self._resampler.Process(self._sola.Process(wav22k)), meta, meta.active)

    def SubmitSilence(self, meta: BlockMeta) -> None:
        self._RequireRunning()
        self._ApplyParamChanges()
        self._sola.Reset()
        self._Enqueue(self._resampler.Process(np.zeros(self._sola.BlockSamples, dtype=np.float32)), meta, False)

    def EffectiveSteps(self, requested: int) -> int:
        return self.Balancer.EffectiveSteps(requested)

    def ReportInference(self, inference_s: float, block_s: float) -> None:
        self.Balancer.Update(inference_s, block_s)

    def Telemetry(self) -> dict:
        ring, main, mon = self._ring, self._main, self._mon
        n = self._e2e_n
        mon_n = self._mon_delay_n
        return {
            "device": self._device,
            "samplerate": self._sr,
            "channels": self._ch,
            "underruns": self._underruns,
            "underrun_events": self._underrun_events,
            "zero_filled_ms": self._zero_frames * 1000.0 / self._sr if self._sr else 0.0,
            "buffer_ms": (ring.w - ring.r) * 1000.0 / self._sr if ring is not None else 0.0,
            "e2e_ms_last": self._e2e_last,
            "e2e_ms_mean": self._e2e_sum / n if n else 0.0,
            "e2e_ms_max": self._e2e_max,
            "e2e_blocks": n,
            "e2e_invalid": self._e2e_invalid,
            "blocks_out": self._blocks_out,
            "overflow_frames": self._overflow_frames,
            "callback_errors": self._cb_errors,
            "device_underflows": self._device_underflows,
            "cushion_ms": PLAYOUT_CUSHION_MS,
            "stream_latency_ms": main.latency * 1000.0 if main is not None else 0.0,
            "monitor": {
                "enabled": self._mon_enabled,
                "volume": self._mon_volume,
                "device": self._mon_device if mon is not None else None,
                "samplerate": self._mon_sr,
                "channels": self._mon_ch,
                "buffer_ms": self._mon_frames * 1000.0 / self._mon_sr if self._mon_sr else 0.0,
                "delay_ms": self._mon_delay_sum * 1000.0 / mon_n if mon_n else 0.0,
                "stream_latency_ms": mon.latency * 1000.0 if mon is not None else 0.0,
                "short_reads": self._mon_short,
                "callback_errors": self._mon_cb_errors,
                "error": self._mon_error,
            },
            "balancer": self.Balancer.State(),
        }

    # ---- PRIVATE

    def _ResetCounters(self) -> None:
        self._markers = collections.deque()  # (first frame index, t_capture, active), appended before the frames are published
        self._started = False  # a block has been queued: from now on a short ring is an underrun
        self._needs_prime = True
        self._starving = False
        self._underruns = self._underrun_events = self._zero_frames = self._blocks_out = 0
        self._overflow_frames = self._cb_errors = self._device_underflows = 0
        self._e2e_last = self._e2e_sum = self._e2e_max = 0.0
        self._e2e_n = self._e2e_invalid = 0
        self._mon_frames = self._mon_short = self._mon_cb_errors = self._mon_delay_n = 0
        self._mon_delay_sum = 0.0

    def _RequireRunning(self) -> None:
        if not self._running:
            raise RuntimeError("OutputChain is not started")

    def _OnParam(self, name: str, value) -> None:
        handler = self._param_handlers.get(name)
        if handler is not None:
            handler(value)

    def _MarkSolaDirty(self, _value) -> None:
        self._sola_dirty = True

    def _SetMonitorEnabled(self, value) -> None:
        self._mon_enabled = bool(value)

    def _SetMonitorVolume(self, value) -> None:
        self._mon_volume = float(value)

    def _ApplyParamChanges(self) -> None:
        if self._sola_dirty:
            self._sola_dirty = False
            self._sola.Rebuild(self._params.Get("block_ms"), self._params.Get("overlap_ms"))

    def _Enqueue(self, out: np.ndarray, meta: BlockMeta, active: bool) -> None:
        if out.size == 0:
            return
        ring = self._ring
        pad = 0
        if self._needs_prime:
            self._needs_prime = False
            pad = int(round(PLAYOUT_CUSHION_MS * self._sr / 1000.0))
        free = ring.cap - (ring.w - ring.r)
        if pad + out.size > free:
            self._overflow_frames += pad + out.size - free
            pad = min(pad, free)
            out = out[: free - pad]
            if out.size == 0:
                return
        self._markers.append((ring.w + pad, meta.t_capture, active))
        ring.Write(out, pad)
        self._started = True

    def _OnMain(self, outdata, frames, time_info, status) -> None:
        try:
            now = time.perf_counter()
            if status is not None and status.output_underflow:
                self._device_underflows += 1  # PortAudio says the device ran dry: this callback missed its deadline
            ring = self._ring
            r0 = ring.r
            avail = ring.w - r0
            n = frames if avail >= frames else avail
            if n:
                ring.Read(outdata, n)
            if n < frames:
                outdata[n:frames] = 0.0
                if self._started:
                    self._underruns += 1
                    self._zero_frames += frames - n
                    if not self._starving:
                        self._underrun_events += 1
                        self._starving = True
                    self._needs_prime = True
            else:
                self._starving = False
            markers = self._markers
            if markers:
                end = r0 + n
                delta = time_info.outputBufferDacTime - time_info.currentTime
                while markers and markers[0][0] < end:
                    start, t_capture, active = markers.popleft()
                    self._blocks_out += 1
                    if not active:
                        continue
                    if 0.0 <= delta < 1.0:
                        e2e = (now + delta + max(0, start - r0) / self._sr - t_capture) * 1000.0
                        self._e2e_last = e2e
                        self._e2e_sum += e2e
                        self._e2e_max = max(self._e2e_max, e2e)
                        self._e2e_n += 1
                    else:
                        self._e2e_invalid += 1
        except Exception:
            outdata[:] = 0.0
            self._cb_errors += 1

    def _OpenMonitor(self) -> None:
        try:
            dev = ResolveWasapiOutput(self._monitor_device_name)
            ring_sr = getattr(self._monitor_ring, "samplerate", None)
            sr = int(ring_sr) if ring_sr else dev["samplerate"]
            stream = OpenStarted(lambda: sd.OutputStream(device=dev["index"], samplerate=sr, channels=dev["channels"], dtype="float32",
                                                         latency="low", callback=self._OnMonitor,
                                                         extra_settings=sd.WasapiSettings(auto_convert=True)))
        except Exception as e:
            self._mon_error = f"{type(e).__name__}: {e}"
            return
        self._mon_error = None
        self._mon_device, self._mon_sr, self._mon_ch, self._mon = dev["name"], sr, dev["channels"], stream

    def _OnMonitor(self, outdata, frames, time_info, status) -> None:
        try:
            mono = self._monitor_ring.Read(frames)  # always drained so the sidetone is fresh the moment it is enabled
            n = min(len(mono), frames)
            if n < frames:
                self._mon_short += 1
            gain = self._mon_volume if self._mon_enabled else 0.0
            outdata[:n] = (mono[:n] * gain)[:, None]
            outdata[n:frames] = 0.0
            self._mon_frames = frames
            delta = time_info.outputBufferDacTime - time_info.currentTime
            if 0.0 <= delta < 1.0:
                self._mon_delay_sum += delta
                self._mon_delay_n += 1
        except Exception:
            outdata[:] = 0.0
            self._mon_cb_errors += 1


class _Fifo:
    """Single-producer single-consumer frame FIFO over a Ring. w (producer) and r (consumer) only ever grow."""

    def __init__(self, capacity: int, channels: int):
        self.ring = Ring(capacity, (channels,))
        self.cap = capacity
        self.w = 0
        self.r = 0

    def Write(self, mono: np.ndarray, pad: int = 0) -> None:
        if pad:
            self.ring.Fill(self.w, pad)
        self.ring.Put(self.w + pad, mono[:, None])
        self.w += pad + mono.size

    def Read(self, out: np.ndarray, n: int) -> None:
        self.ring.Get(self.r, n, out)
        self.r += n
