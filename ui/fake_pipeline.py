"""TEST-ONLY stand-in for pipeline.Pipeline. Product code never imports this module.

It keeps the real ParamStore and preset table, so parameter and preset behaviour is the real one. Audio, slots and
devices are synthetic and labelled FAKE; nothing here opens a microphone, a speaker or a model.
Run "python -m ui.fake_pipeline --port-file <path>" to serve the real UI on 127.0.0.1 for browser checks.
"""
from __future__ import annotations

import argparse
import math
import socket
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

import seedvc_params as P
from engine import rvc_store

FAKE_DEVICES = [
    {"id": 0, "name": "FAKE microphone", "hostapi": "FAKE", "default_samplerate": 48000.0, "max_input_channels": 1},
]


# ---- PUBLIC
class FakePipeline:
    """Pipeline-interface double. self.calls records each state-changing call as a plain tuple."""

    def __init__(self, params: P.ParamStore | None = None, clip: bool = False, weights_state: str = "missing"):
        self.params = params if params is not None else P.ParamStore()
        self.clip = clip
        self.weights_state = weights_state
        self.calls: list[tuple] = []
        self._slots: dict[int, dict[str, Any]] = {}
        self._recording: tuple[int, float] | None = None
        self._lock = threading.Lock()

    def GetParams(self) -> dict[str, Any]:
        return self.params.Snapshot()

    def SetParam(self, name: str, value: Any) -> Any:
        self.calls.append(("SetParam", name, value))
        return self.params.Set(name, value)

    def ApplyPreset(self, name: str) -> dict[str, Any]:
        self.calls.append(("ApplyPreset", name))
        return self.params.ApplyPreset(name)

    def SetSlotFromWav(self, idx: int, wav: np.ndarray, sr: int) -> dict[str, Any]:
        if not 0 <= idx < len(P.SLOT_NAMES):
            raise ValueError(f"slot index must be 0..{len(P.SLOT_NAMES) - 1}")
        self.calls.append(("SetSlotFromWav", idx, int(wav.size), str(wav.dtype), int(wav.ndim), int(sr)))
        seconds = wav.size / sr
        info = {**_EMPTY_INFO, "kind": "clip", "seconds": seconds, "seconds_used": min(seconds, P.PROMPT_MAX_S), "prepared": True}
        with self._lock:
            self._slots[idx] = info
        return {"idx": idx, "name": P.SLOT_NAMES[idx], "seconds_used": info["seconds_used"],
                "embedding_dim": 0, "prompt_tokens": 0, "cached": False}

    def ClearSlot(self, idx: int) -> None:
        self.calls.append(("ClearSlot", idx))
        with self._lock:
            self._slots.pop(idx, None)

    def SetSlotFromRvc(self, idx: int, name: str) -> dict[str, Any]:
        if not 0 <= idx < len(P.SLOT_NAMES):
            raise ValueError(f"slot index must be 0..{len(P.SLOT_NAMES) - 1}")
        model = rvc_store.Get(name)
        self.calls.append(("SetSlotFromRvc", idx, name))
        info = {**_EMPTY_INFO, "kind": "rvc", "model": name, "version": model["version"], "sr": model["sr"],
                "f0": model["f0"], "has_index": model["has_index"], "seconds": 0.0, "seconds_used": 0.0}
        with self._lock:
            self._slots[idx] = info
        return self.SlotInfo()[idx]

    def DeleteRvcModel(self, name: str) -> list[int]:
        rvc_store.Get(name)  # raises ValueError for an unknown voice
        self.calls.append(("DeleteRvcModel", name))
        with self._lock:
            cleared = sorted(i for i, info in self._slots.items() if info["model"] == name)
            for i in cleared:
                del self._slots[i]
        rvc_store.Delete(name)
        return cleared

    def RvcStatus(self) -> dict[str, Any]:
        return {"weights": {"state": self.weights_state, "file": None, "done": 0, "total": 0, "message": None}}

    def SetManualSlot(self, idx: int) -> None:
        self.calls.append(("SetManualSlot", idx))
        self.params.Set("manual_slot", idx)
        self.params.Set("emotion_mode", "manual")

    def SlotInfo(self) -> list[dict[str, Any]]:
        with self._lock:
            active = self._ActiveLocked()
            selected = self.params.Get("manual_slot")
            rows = []
            for i in range(len(P.SLOT_NAMES)):
                info = self._slots.get(i)
                rows.append({"idx": i, "name": P.SLOT_NAMES[i], "loaded": info is not None,
                             **_EMPTY_INFO, **(info or {}),
                             "active": i == active, "selected": selected == i})
            return rows

    def StartRecording(self, idx: int) -> None:
        self.calls.append(("StartRecording", idx))
        with self._lock:
            if self._recording is not None:
                raise RuntimeError("a recording is already running")
            self._recording = (idx, time.monotonic())

    def StopRecording(self, idx: int | None = None) -> dict[str, Any]:
        with self._lock:
            if self._recording is None:
                raise RuntimeError("no recording to store")
            slot, started = self._recording
            self._recording = None
        self.calls.append(("StopRecording", slot))
        elapsed = min(P.RECORDER_MAX_S, time.monotonic() - started)
        n = int(elapsed * 24000)
        tone = (0.25 * np.sin(2 * np.pi * 220.0 * np.arange(n) / 24000)).astype(np.float32)
        return self.SetSlotFromWav(slot, tone, 24000)

    def LevelSnapshot(self) -> dict[str, Any]:
        with self._lock:
            rec = self._recording
        if rec is None:
            return {"peak": 0.0, "clipping": False, "waveform": []}
        t = time.monotonic() - rec[1]
        waveform = [round(0.5 * math.sin(2 * math.pi * 3.0 * (t - k * 0.01)), 4) for k in range(50)]
        peak = 1.0 if self.clip else max(abs(v) for v in waveform)
        return {"peak": peak, "clipping": bool(self.clip), "waveform": waveform}

    def ListDevices(self) -> list[dict[str, Any]]:
        return [dict(d) for d in FAKE_DEVICES]

    def Telemetry(self) -> dict[str, Any]:
        with self._lock:
            loaded = len(self._slots)
            active = self._ActiveLocked()
            kind = self._slots[active]["kind"] if active is not None else None
        return {
            "running": True,
            "error": None,
            "fake": True,
            "pipeline": {"fake": True, "blocks": 0, "slots_loaded": loaded},
            "input": {"mode": "FAKE", "device": FAKE_DEVICES[0]["name"], "samplerate": 48000, "buffer_ms": 0.0,
                      "vad_prob": 0.0, "vad_active": False, "gain_reduction_db": 0.0, "makeup_db": 0.0,
                      "overruns": 0, "enhancements_bypassed": None},
            "output": {"device": "FAKE output (no audio device)", "samplerate": 48000, "underruns": 0,
                       "buffer_ms": 0.0, "e2e_ms_last": None, "e2e_ms_mean": None, "e2e_ms_max": None,
                       "blocks_out": 0,
                       "monitor": {"enabled": self.params.Get("monitor_enabled"), "device": None, "buffer_ms": 0.0},
                       "balancer": {"enabled": self.params.Get("auto_balancer"),
                                    "steps": self.params.Get("diffusion_steps"), "throttled": False}},
            "engine": {"kind": kind, "switching": False, "waiting": None,
                       "error": None, "unused_controls": ["cfg_rate", "diffusion_steps", "timbre_mix", "prosody_follow"] if kind == "rvc" else [], "switches": 0, "last_switch": None,
                       "resident": {"clip": False, "rvc": False}, "vram_mb": {"clip": 0.0, "rvc": 0.0}},
            "params": self.params.Snapshot(),
        }

    def _ActiveLocked(self) -> int | None:
        """The slot the engine converts with, by the real pipeline's rule. Caller holds self._lock."""
        loaded = sorted(self._slots)
        if self.params.Get("emotion_mode") == "manual":
            manual = self.params.Get("manual_slot")
            if manual in self._slots:
                return manual
            return loaded[0] if loaded else None
        clips = [i for i in loaded if self._slots[i]["kind"] == "clip"]
        if clips:
            return clips[0]
        return loaded[0] if loaded else None


def main(argv: list[str] | None = None) -> None:
    from ui.server import CreateApp
    import uvicorn

    parser = argparse.ArgumentParser(description="FAKE pipeline UI for browser checks (test only)")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--port-file", type=Path, default=None)
    parser.add_argument("--clip", action="store_true")
    args = parser.parse_args(argv)
    port = args.port or _FreePort()
    if args.port_file is not None:
        args.port_file.write_text(str(port), encoding="utf-8")
    uvicorn.run(CreateApp(FakePipeline(clip=args.clip)), host="127.0.0.1", port=port, log_level="warning")


# ---- PRIVATE
_EMPTY_INFO = {"kind": None, "model": None, "seconds": None, "seconds_used": 0.0, "embedding_dim": 0,
               "prompt_tokens": 0, "content_hash": "", "version": None, "sr": None, "f0": None,
               "has_index": None, "prepared": None}


def _FreePort() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


if __name__ == "__main__":
    main()
