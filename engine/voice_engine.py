"""The engine interface the pipeline routes blocks through, and the sample grid both engines share."""
from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np
import torch

from seedvc_params import EXTRA_CONTEXT_RIGHT_S, INPUT_SR, MODEL_SR, SOLA_SEARCH_MS

# ---- PUBLIC
KIND_CLIP = "clip"  # a reference clip converted by Seed-VC
KIND_RVC = "rvc"  # an imported RVC voice model
PROSODY_CONTEXT_S = 0.5  # converted audio before the returned region that the WORLD stage may use


def AllocatedMb(device: torch.device) -> float:
    """Device memory the caching allocator has handed out to this process's tensors, in MB (0.0 off the GPU). Both engines share the
    process, so this is a process total: an engine's own share is the difference between two readings the EngineHost takes around it."""
    return torch.cuda.memory_allocated() / 2**20 if device.type == "cuda" else 0.0


@dataclass
class ConvertResult:
    wav22k: object  # float32, block + overlap + search samples at MODEL_SR, before SOLA
    inference_s: float
    steps_used: int
    slot_weights: list
    f0_median_hz: float
    pitch_std_st: float
    rms_db: float
    stage_applied: bool


@dataclass(frozen=True)
class Grid:
    """Sample counts of one block at 16 kHz (input side) and at MODEL_SR (output side)."""

    block16: int
    overlap16: int
    search16: int
    right16: int
    block22: int
    overlap22: int
    search22: int
    right22: int
    out22: int  # block + overlap + search at MODEL_SR
    prosody_ctx22: int


def MakeGrid(block_ms: float, overlap_ms: float) -> Grid:
    block22 = int(round(block_ms / 1000.0 * MODEL_SR))
    overlap22 = int(round(overlap_ms / 1000.0 * MODEL_SR))
    search22 = int(round(SOLA_SEARCH_MS / 1000.0 * MODEL_SR))
    return Grid(
        block16=int(round(block_ms / 1000.0 * INPUT_SR)),
        overlap16=int(round(overlap_ms / 1000.0 * INPUT_SR)),
        search16=int(round(SOLA_SEARCH_MS / 1000.0 * INPUT_SR)),
        right16=int(round(EXTRA_CONTEXT_RIGHT_S * INPUT_SR)),
        block22=block22,
        overlap22=overlap22,
        search22=search22,
        right22=int(round(EXTRA_CONTEXT_RIGHT_S * MODEL_SR)),
        out22=block22 + overlap22 + search22,
        prosody_ctx22=int(round(PROSODY_CONTEXT_S * MODEL_SR)),
    )


def SilentResult(out22: int) -> ConvertResult:
    return ConvertResult(np.zeros(out22, np.float32), 0.0, 0, [0.0] * 4, 0.0, 0.0, -120.0, False)


class VoiceEngine(abc.ABC):
    """One conversion engine. The EngineHost routes each block to the engine of the slot that is active."""

    kind: str = ""
    unused_controls: tuple = ()  # parameter names this engine ignores
    vram_estimate_mb: float = 0.0  # what a first load is expected to take on the device, until the EngineHost has measured it

    @abc.abstractmethod
    def Load(self) -> dict:
        """Make the engine resident (weights on the device). Returns a dict with vram_mb and load_seconds."""

    @abc.abstractmethod
    def Release(self) -> None:
        """Drop every model and device buffer this engine holds."""

    def Prepare(self) -> dict:
        """Start fetching whatever Load() needs and return at once. state: ready | missing | downloading | error, with message, done, total and
        label for the UI while it is not ready. An engine with nothing to fetch is always ready."""
        return {"state": "ready"}

    @abc.abstractmethod
    def Loaded(self) -> bool:
        """True while the engine's models are resident."""

    @abc.abstractmethod
    def ConvertBlock(self, block16k: np.ndarray, active: bool, steps: int | None = None) -> ConvertResult:
        """Convert one block of 16 kHz audio into out22 samples at MODEL_SR."""

    @abc.abstractmethod
    def ResetStream(self) -> None:
        """Forget the streaming context (ring buffer, pitch and emotion state)."""

    @abc.abstractmethod
    def Telemetry(self) -> dict:
        """Engine-specific numbers for the UI and the report."""
