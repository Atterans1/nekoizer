"""Pipeline: wires InputChain -> EngineHost (Seed-VC clip engine or RVC voice engine) -> OutputChain and owns the worker thread.

The slot bank holds reference clips and RVC voices side by side; the host routes each block to the engine of the active slot.
Interfaces are fixed by docs/seedvc/CONTRACT.md and docs/rvc/ (the addendum). This module only coordinates; it holds no DSP or model code.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

import seedvc_params as P

LOG = logging.getLogger("seedvc.pipeline")


# ---------------------------------------------------------------- PUBLIC
class Pipeline:
    """Live pipeline. Construct, Start(), then drive it from the UI; Stop() releases every device and thread."""

    def __init__(self, assets_dir: str | Path, params: P.ParamStore | None = None, input_device: str | None = None,
                 output_device: str = P.CABLE_OUTPUT_NAME, monitor_device: str | None = None, device: str = "cuda",
                 engines: tuple | None = None):
        """engines: optional (clip engine, RVC engine) built by the caller; None builds the real ones. Nothing is loaded here."""
        from engine.engine_host import EngineHost
        from engine.rvc_engine import RvcEngine
        from engine.seedvc_expressive import SeedVCExpressive

        self.assets_dir = Path(assets_dir)
        self.params = params or P.ParamStore()
        self.janitor = P.Janitor()
        self._input_device = input_device
        self._output_device = output_device
        self._monitor_device = monitor_device
        self._device = device
        clip, rvc = engines if engines is not None else (SeedVCExpressive(self.params, self.janitor, self.assets_dir, device=device),
                                                          RvcEngine(self.params, self.janitor, device=device))
        self._host = EngineHost(self.params, self.janitor, clip, rvc, device)
        self._input = None
        self._output = None
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._running = False
        self._lock = threading.Lock()
        self._last_error: str | None = None
        self._stats = {"blocks": 0, "active_blocks": 0, "silent_blocks": 0, "convert_errors": 0, "last_inference_s": 0.0,
                       "inference_s_sum": 0.0, "load": None}
        self._record_slot: int | None = None

    def Start(self) -> dict:
        with self._lock:
            if self._running:
                return {"already_running": True}
            from audio.input_chain import InputChain
            from audio.output_chain import OutputChain
            self._stats["load"] = self._host.LoadClip()
            self._input = InputChain(self.params, self.janitor, self._input_device)
            self._output = OutputChain(self.params, self.janitor, self._output_device, self._monitor_device,
                                       monitor_ring=self._input.MonitorRing)
            self._output.Start()
            self._input.Start()
            self._stop.clear()
            self._worker = threading.Thread(target=self._Run, name="seedvc-worker", daemon=True)
            self._worker.start()
            self.janitor.Add(self._StopWorker)
            self._running = True
            return {"started": True, "load": self._stats["load"]}

    def Stop(self) -> None:
        with self._lock:
            self._running = False
        self._stop.set()
        self.janitor.Cleanup()

    # ---- parameters / presets / reference bank
    def GetParams(self) -> dict[str, Any]:
        return self.params.Snapshot()

    def SetParam(self, name: str, value: Any) -> Any:
        return self.params.Set(name, value)

    def ApplyPreset(self, name: str) -> dict[str, Any]:
        return self.params.ApplyPreset(name)

    def SetSlotFromWav(self, idx: int, wav: np.ndarray, sr: int) -> dict:
        return self._host.SetClip(idx, wav, sr)

    def SetSlotFromRvc(self, idx: int, name: str) -> dict:
        from engine import rvc_store
        rvc_store.Get(name)  # ValueError for an unknown voice
        self._host.SetRvc(idx, name)
        return self._host.SlotRows()[idx]

    def ClearSlot(self, idx: int) -> None:
        self._host.ClearSlot(idx)

    def DeleteRvcModel(self, name: str) -> list[int]:
        from engine import rvc_store
        rvc_store.Get(name)
        cleared = self._host.ClearRvcModel(name)
        rvc_store.Delete(name)
        return cleared

    def RvcStatus(self) -> dict:
        from engine import rvc_weights
        return {"weights": rvc_weights.Status()}

    def SetManualSlot(self, idx: int) -> None:
        self.params.Set("manual_slot", idx)
        self.params.Set("emotion_mode", "manual")

    def SlotInfo(self) -> list[dict]:
        return self._host.SlotRows()

    # ---- recording into a slot
    def StartRecording(self, idx: int) -> None:
        if not 0 <= idx < len(P.SLOT_NAMES):
            raise ValueError("slot index must be 0..3")
        self._record_slot = idx
        self._RequireInput().StartRecording(P.RECORDER_MAX_S)

    def StopRecording(self, idx: int | None = None) -> dict:
        slot = self._record_slot if idx is None else idx
        wav = self._RequireInput().StopRecording()
        self._record_slot = None
        if slot is None or wav is None or len(wav) == 0:
            raise RuntimeError("no recording to store")
        return self.SetSlotFromWav(slot, np.asarray(wav, dtype=np.float32), 24000)

    def LevelSnapshot(self) -> dict:
        return self._RequireInput().LevelSnapshot()

    def ListDevices(self) -> list[dict]:
        if self._input is not None:
            return self._input.ListDevices()
        from audio.input_chain import InputChain
        probe = InputChain(self.params, P.Janitor(), None)
        return probe.ListDevices()

    # ---- telemetry
    def Telemetry(self) -> dict:
        s = dict(self._stats)
        blocks = max(1, s["active_blocks"])
        s["inference_s_mean"] = s["inference_s_sum"] / blocks
        return {
            "running": self._running,
            "error": self._last_error,
            "pipeline": s,
            "input": self._input.Telemetry() if self._input else None,
            "output": self._output.Telemetry() if self._output else None,
            "engine": self._host.Telemetry(),
            "params": self.params.Snapshot(),
        }

    # ---- offline path (same routing, engines and SOLA, no devices)
    def ProcessOffline(self, audio16k: np.ndarray) -> tuple[np.ndarray, dict]:
        from audio.sola import Sola
        self._host.ResetStreams()
        block_ms = self.params.Get("block_ms")
        overlap_ms = self.params.Get("overlap_ms")
        n = int(round(block_ms * P.INPUT_SR / 1000))
        sola = Sola(P.MODEL_SR, block_ms, overlap_ms, P.SOLA_SEARCH_MS)
        outs, times, kinds = [], [], []
        for start in range(0, len(audio16k) - n + 1, n):
            res = self._host.ConvertBlock(audio16k[start:start + n].astype(np.float32), True)
            kinds.append(self._host.Telemetry()["kind"])
            times.append(res.inference_s)
            outs.append(sola.Process(res.wav22k))
        audio = np.concatenate(outs) if outs else np.zeros(0, np.float32)
        arr = np.asarray(times) if times else np.zeros(1)
        return audio, {"blocks": len(times), "inference_s_median": float(np.median(arr)), "inference_s_p95": float(np.percentile(arr, 95)),
                       "inference_s_max": float(arr.max()), "kinds": kinds}

    # ---- PRIVATE
    def _RequireInput(self):
        if self._input is None:
            raise RuntimeError("pipeline not started: the input chain is not open")
        return self._input

    def _StopWorker(self) -> None:
        self._stop.set()
        w = self._worker
        if w is not None and w.is_alive() and w is not threading.current_thread():
            w.join(timeout=5.0)

    def _DrainInput(self) -> None:
        """Drop the input backlog that piled up while an engine was switching, so the next block is the live one."""
        n = int(round(self.params.Get("block_ms") * P.INPUT_SR / 1000))
        for _ in range(10_000):
            if self._input.ReadBlock(n, timeout_s=0.0) is None:
                return

    def _Run(self) -> None:
        from audio.output_chain import BlockMeta
        while not self._stop.is_set():
            block_ms = self.params.Get("block_ms")
            n = int(round(block_ms * P.INPUT_SR / 1000))
            try:
                blk = self._input.ReadBlock(n, timeout_s=0.5)
            except Exception as exc:  # noqa: BLE001
                self._last_error = f"input: {exc}"
                time.sleep(0.05)
                continue
            if blk is None:
                continue
            meta = BlockMeta(t_capture=blk.t_capture, block_ms=block_ms, active=blk.active, inference_s=0.0)
            self._stats["blocks"] += 1
            if not blk.active:
                self._stats["silent_blocks"] += 1
                self._output.SubmitSilence(meta)
                continue
            try:
                steps = self._output.EffectiveSteps(self.params.Get("diffusion_steps"))
                switches = self._host.SwitchCount()
                res = self._host.ConvertBlock(blk.samples16k, True, steps=steps)
            except Exception as exc:  # noqa: BLE001
                self._stats["convert_errors"] += 1
                self._last_error = f"convert: {type(exc).__name__}: {exc}"
                LOG.exception("convert failed")
                self._output.SubmitSilence(meta)
                continue
            if self._host.SwitchCount() != switches:
                # an engine swap happened in this block: the audio so far is stale and the SOLA tail must restart
                self._DrainInput()
                self._output.SubmitSilence(meta)
                continue
            self._stats["active_blocks"] += 1
            self._stats["last_inference_s"] = res.inference_s
            self._stats["inference_s_sum"] += res.inference_s
            self._output.ReportInference(res.inference_s, block_ms / 1000.0)
            meta.inference_s = res.inference_s
            self._output.Submit(res.wav22k, meta)
