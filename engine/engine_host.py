"""EngineHost: the four voice slots (reference clip or RVC voice), the routing of every block to the engine that owns the active
slot, and the GPU memory policy for switching between the two engines.

Routing, per block, from the parameters only: manual mode converts with the manual slot (or the fallback slot the emotion selector
would pick: Neutral, else the first loaded slot); auto mode converts with the reference clips (emotion picks among them) and falls
back to an RVC voice only when no clip is loaded. The clip engine's bank holds clip slots only, so an RVC slot can never reach the
Seed-VC prompt path, and the emotion selector never sees one.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch

import seedvc_params as P
from engine.voice_engine import KIND_CLIP, KIND_RVC, AllocatedMb, MakeGrid, SilentResult

LOG = logging.getLogger("seedvc.engine_host")

# ---- PUBLIC
MARGIN_MB = 300.0  # free device memory that must remain after a load
MB = 2**20


@dataclass
class Slot:
    kind: str | None = None
    name: str | None = None  # RVC model name
    wav: np.ndarray | None = None  # clip audio, kept so a released Seed-VC can re-encode it
    sr: int = 0
    prepared: bool = False


class EngineHost:
    def __init__(self, params: P.ParamStore, janitor: P.Janitor, clip, rvc, device: str = "cuda"):
        self._params = params
        self._clip = clip
        self._rvc = rvc
        self._device = torch.device(device)
        self._slots = [Slot() for _ in P.SLOT_NAMES]
        self._lock = threading.RLock()
        self._active: str | None = None
        self._estimate = {KIND_CLIP: clip.vram_estimate_mb, KIND_RVC: rvc.vram_estimate_mb}  # MB a load of each engine needs
        self._footprint = {KIND_CLIP: 0.0, KIND_RVC: 0.0}  # MB each engine holds on the device while resident, as measured
        self._opened: dict[str, float | None] = {KIND_CLIP: None, KIND_RVC: None}  # allocated MB when a load began, until the first converted block
        self._switches = 0
        self._last_switch: dict | None = None
        self._error: str | None = None
        self._failed_route: tuple | None = None
        self._waiting: str | None = None
        self._switching = False
        janitor.Add(self._ReleaseAll)

    # ---- slots
    def SetClip(self, idx: int, wav: np.ndarray, sr: int) -> dict:
        with self._lock:
            self._CheckIdx(idx)
            data = np.ascontiguousarray(np.asarray(wav, dtype=np.float32))
            self._slots[idx] = Slot(KIND_CLIP, None, data, int(sr), False)
            if self._clip.Loaded():
                info = self._clip.SetSlot(idx, data, int(sr))
                self._slots[idx].prepared = True
                return info | {"prepared": True}
            seconds = data.shape[0] / int(sr)
            return {"idx": idx, "name": P.SLOT_NAMES[idx], "seconds": seconds, "seconds_used": min(seconds, P.PROMPT_MAX_S),
                    "embedding_dim": 0, "prompt_tokens": 0, "content_hash": "", "cached": False, "prepared": False}

    def SetRvc(self, idx: int, name: str) -> None:
        with self._lock:
            self._CheckIdx(idx)
            if self._slots[idx].kind == KIND_CLIP and self._clip.Loaded():
                self._clip.ClearSlot(idx)
            self._slots[idx] = Slot(KIND_RVC, name)

    def ClearSlot(self, idx: int) -> None:
        with self._lock:
            self._CheckIdx(idx)
            if self._slots[idx].kind == KIND_CLIP and self._clip.Loaded():
                self._clip.ClearSlot(idx)
            self._slots[idx] = Slot()

    def ClearRvcModel(self, name: str) -> list[int]:
        with self._lock:
            cleared = [i for i, s in enumerate(self._slots) if s.kind == KIND_RVC and s.name == name]
            for i in cleared:
                self._slots[i] = Slot()
            voice = self._rvc.Voice()
            if voice is not None and voice["name"] == name:
                self._rvc.UnloadVoice()
            return cleared

    def SlotRows(self) -> list[dict]:
        with self._lock:
            route_kind, route_idx = self.Route()
            manual = self._params.Get("manual_slot")
            bank = {int(r["idx"]): r for r in self._clip.SlotInfo()} if self._clip.Loaded() else {}
            rows = []
            for i, slot in enumerate(self._slots):
                row = {"idx": i, "name": P.SLOT_NAMES[i], "loaded": slot.kind is not None, "kind": slot.kind, "model": slot.name,
                       "seconds": None, "seconds_used": 0.0, "embedding_dim": 0, "prompt_tokens": 0, "content_hash": "",
                       "version": None, "sr": None, "f0": None, "has_index": None, "prepared": None,
                       "active": False, "selected": manual == i}
                if slot.kind == KIND_CLIP:
                    b = bank.get(i, {})
                    row.update(seconds=slot.wav.shape[0] / slot.sr if slot.wav is not None else None,
                               seconds_used=float(b.get("seconds_used", 0.0)), embedding_dim=int(b.get("embedding_dim", 0)),
                               prompt_tokens=int(b.get("prompt_tokens", 0)), content_hash=str(b.get("content_hash", "")),
                               prepared=slot.prepared)
                    row["active"] = route_kind == KIND_CLIP and (route_idx is None or route_idx == i)
                elif slot.kind == KIND_RVC:
                    facts = self._RvcFacts(slot.name)
                    row.update(facts)
                    row["active"] = route_kind == KIND_RVC and route_idx == i
                rows.append(row)
            return rows

    # ---- routing
    def Route(self) -> tuple[str | None, int | None]:
        snap = self._params.Snapshot()
        slots = self._slots
        if snap["emotion_mode"] == "manual":
            m = snap["manual_slot"]
            if slots[m].kind is not None:
                return slots[m].kind, m
            loaded = [i for i, s in enumerate(slots) if s.kind is not None]
            if not loaded:
                return None, None
            first = 0 if slots[0].kind is not None else loaded[0]
            return slots[first].kind, first
        if any(s.kind == KIND_CLIP for s in slots):
            return KIND_CLIP, None
        rvc = [i for i, s in enumerate(slots) if s.kind == KIND_RVC]
        return (KIND_RVC, rvc[0]) if rvc else (None, None)

    def ConvertBlock(self, block16k: np.ndarray, active: bool, steps: int | None = None):
        with self._lock:
            p = self._params.Snapshot()
            grid = MakeGrid(p["block_ms"], p["overlap_ms"])
            kind, idx = self.Route()
            if kind is None:
                self._active = None
                return SilentResult(grid.out22)
            if (kind, idx) == self._failed_route or not self._Ready(kind, idx):
                return SilentResult(grid.out22)
            result = (self._clip if kind == KIND_CLIP else self._rvc).ConvertBlock(block16k, active, steps)
            if active:
                self._CloseFootprint(kind)
            return result

    def SwitchCount(self) -> int:
        return self._switches

    def Telemetry(self) -> dict:
        with self._lock:
            unused = list(self._rvc.unused_controls) if self._active == KIND_RVC else []
            return {"kind": self._active, "switching": self._switching, "waiting": self._waiting, "error": self._error,
                    "unused_controls": unused, "switches": self._switches, "last_switch": self._last_switch,
                    "resident": {KIND_CLIP: self._clip.Loaded(), KIND_RVC: self._rvc.Loaded()},
                    "vram_mb": {KIND_CLIP: self._FootprintMb(KIND_CLIP, self._clip), KIND_RVC: self._FootprintMb(KIND_RVC, self._rvc)}}

    def LoadClip(self) -> dict:
        with self._lock:
            return self._LoadKind(KIND_CLIP)

    def ResetStreams(self) -> None:
        with self._lock:
            if self._clip.Loaded():
                self._clip.ResetStream()
            if self._rvc.Loaded():
                self._rvc.ResetStream()

    # ---- PRIVATE
    @staticmethod
    def _CheckIdx(idx: int) -> None:
        if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < len(P.SLOT_NAMES):
            raise ValueError(f"slot index must be an int 0..{len(P.SLOT_NAMES) - 1}, got {idx!r}")

    def _RvcFacts(self, name: str) -> dict:
        from engine import rvc_store
        try:
            m = rvc_store.Get(name)
        except ValueError:
            return {"version": None, "sr": None, "f0": None, "has_index": None}
        return {"version": m["version"], "sr": m["sr"], "f0": m["f0"], "has_index": m["has_index"]}

    def _Ready(self, kind: str, idx: int | None) -> bool:
        """True when the engine of kind is resident and holds the voice of idx. Otherwise switches (may load) and returns the result."""
        loaded_ok = self._clip.Loaded() if kind == KIND_CLIP else self._rvc.Loaded()
        voice_ok = True
        if kind == KIND_RVC:
            voice = self._rvc.Voice()
            voice_ok = voice is not None and voice["name"] == self._slots[idx].name
        if loaded_ok and voice_ok and self._active == kind:
            return True
        try:
            ok = self._Switch(kind, idx)
        except Exception as exc:  # the switch failed: silence, and no retry until the route changes
            LOG.exception("switch to %s failed", kind)
            self._error = f"{type(exc).__name__}: {exc}"
            self._failed_route = (kind, idx)
            self._switching = False
            return False
        self._failed_route = None if ok else self._failed_route
        return ok

    def _ResidentKind(self) -> str | None:
        """The one engine whose weights are on the device now: None when neither or both are."""
        resident = [k for k, engine in ((KIND_CLIP, self._clip), (KIND_RVC, self._rvc)) if engine.Loaded()]
        return resident[0] if len(resident) == 1 else None

    def _Switch(self, kind: str, idx: int | None) -> bool:
        started = time.perf_counter()
        previous = self._active or self._ResidentKind()
        state = (self._rvc if kind == KIND_RVC else self._clip).Prepare()
        if state["state"] == "error":
            raise RuntimeError(state["message"])
        if state["state"] != "ready":
            done, total = state.get("done") or 0, state.get("total") or 0
            self._waiting = f"{state.get('label', kind)} {state['state']}" + (f" {100 * done // total}%" if total else "")
            return False
        self._waiting = None
        self._switching = True
        released: list[str] = []
        try:
            if not (self._clip.Loaded() if kind == KIND_CLIP else self._rvc.Loaded()):
                released += self._MakeRoom(kind)
                self._LoadKind(kind)
            if kind == KIND_RVC:
                from engine import rvc_store
                name = self._slots[idx].name
                voice = self._rvc.Voice()
                if voice is None or voice["name"] != name:
                    self._rvc.LoadVoice(rvc_store.Get(name) | {"name": name})
            (self._rvc if kind == KIND_RVC else self._clip).ResetStream()
            self._active = kind
            self._error = None
            self._switches += 1
        finally:
            self._switching = False
        ms = (time.perf_counter() - started) * 1000.0
        self._last_switch = {"from": previous, "to": kind, "ms": ms, "released": released}
        LOG.info("engine switch %s -> %s in %.0f ms (released %s)", previous, kind, ms, released)
        return True

    def _LoadKind(self, kind: str) -> dict:
        if self._device.type == "cuda" and self._footprint[kind] == 0.0:
            self._opened[kind] = AllocatedMb(self._device)
        if kind == KIND_CLIP:
            info = self._clip.Load()
            self._PrepareClips()
        else:
            info = self._rvc.Load()
        return info

    def _CloseFootprint(self, kind: str) -> None:
        """The first converted block after a load has made every lazy allocation (CUDA-graph pools, the voice's synthesizer). What the
        process gained since the load began is this engine's footprint: the engines load and convert one at a time under the host lock,
        so the other engine allocated nothing meanwhile. The next load of this engine is planned with that figure."""
        opened = self._opened[kind]
        if opened is not None:
            self._opened[kind] = None
            self._footprint[kind] = self._estimate[kind] = max(0.0, AllocatedMb(self._device) - opened)

    def _FootprintMb(self, kind: str, engine) -> float:
        if not engine.Loaded():
            return 0.0
        opened = self._opened[kind]
        return self._footprint[kind] if opened is None else max(0.0, AllocatedMb(self._device) - opened)

    def _PrepareClips(self) -> None:
        for i, slot in enumerate(self._slots):
            if slot.kind == KIND_CLIP and not slot.prepared:
                self._clip.SetSlot(i, slot.wav, slot.sr)
                slot.prepared = True

    def _MakeRoom(self, kind: str) -> list[str]:
        """Free the other engine's models when the device cannot hold both (or when the device has no room at all)."""
        if self._device.type != "cuda":
            return []
        need = self._estimate[kind]
        if self._FreeMb() - need >= MARGIN_MB:
            return []
        other = KIND_RVC if kind == KIND_CLIP else KIND_CLIP
        engine = self._rvc if other == KIND_RVC else self._clip
        released = []
        if engine.Loaded():
            engine.Release()
            self._footprint[other], self._opened[other] = 0.0, None
            if other == KIND_CLIP:
                for slot in self._slots:
                    if slot.kind == KIND_CLIP:
                        slot.prepared = False
            released.append(other)
        if self._FreeMb() - need < MARGIN_MB:
            raise RuntimeError(f"not enough GPU memory for the {kind} engine: {self._FreeMb():.0f} MB free, about {need:.0f} MB needed")
        return released

    def _FreeMb(self) -> float:
        if self._device.type != "cuda":
            return float("inf")
        free, _ = torch.cuda.mem_get_info()
        return free / MB

    def _ReleaseAll(self) -> None:
        with self._lock:
            self._rvc.Release()
            self._clip.Release()
            self._footprint = {KIND_CLIP: 0.0, KIND_RVC: 0.0}
            self._opened = {KIND_CLIP: None, KIND_RVC: None}
            self._active = None
