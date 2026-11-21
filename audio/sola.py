"""SOLA joiner for the converted 22.05 kHz blocks, ported from Seed-VC real-time-gui.py (SOLA after DDSP-SVC).

The next chunk is aligned to the carried tail by normalised cross-correlation over the search window, the overlap is
crossfaded with a sin^2 (fade in) and cos^2 (fade out) pair, exactly one block is returned and the following overlap is
kept as the new tail. The two windows sum to 1 (amplitude-complementary), so a coherent signal keeps its level.
"""
from __future__ import annotations

import numpy as np

# ---- PUBLIC


class Sola:
    """Stateful joiner. Input chunk = block + overlap + search samples, output = exactly block samples.

    Not thread-safe: Process, Rebuild and Reset must be called from one thread."""

    def __init__(self, sr: int, block_ms: float, overlap_ms: float, search_ms: float):
        self.sr = int(sr)
        self._search = self._Samples(search_ms)
        self._tail = np.zeros(0, dtype=np.float32)
        self._Configure(block_ms, overlap_ms)

    @property
    def BlockSamples(self) -> int:
        return self._block

    @property
    def OverlapSamples(self) -> int:
        return self._overlap

    @property
    def SearchSamples(self) -> int:
        return self._search

    @property
    def InputSamples(self) -> int:
        return self._block + self._overlap + self._search

    def Process(self, wav: np.ndarray) -> np.ndarray:
        x = np.asarray(wav, dtype=np.float32)
        need = self.InputSamples
        if x.ndim != 1 or x.size < need:
            raise ValueError(
                f"Sola.Process needs a 1-D chunk of at least {need} samples "
                f"(block {self._block} + overlap {self._overlap} + search {self._search}), got shape {x.shape}"
            )
        ov, search, block = self._overlap, self._search, self._block
        head = x[: ov + search].astype(np.float64)
        num = np.correlate(head, self._tail.astype(np.float64), "valid")
        csum = np.concatenate(([0.0], np.cumsum(head * head)))
        den = np.sqrt(csum[ov : ov + search + 1] - csum[: search + 1] + 1e-8)
        offset = int(np.argmax(num / den))
        y = x[offset : offset + block + ov].copy()
        y[:ov] = y[:ov] * self._fade_in + self._tail * self._fade_out
        self._tail = y[block : block + ov].copy()
        return y[:block].copy()

    def Rebuild(self, block_ms: float, overlap_ms: float) -> None:
        """Apply a new block or overlap length and keep the carried tail: the part next to the emitted block is kept,
        extra overlap is zero-padded at its far end."""
        self._Configure(block_ms, overlap_ms)

    def Reset(self) -> None:
        self._tail = np.zeros(self._overlap, dtype=np.float32)

    # ---- PRIVATE

    def _Samples(self, ms: float) -> int:
        n = int(np.floor(float(ms) * self.sr / 1000.0 + 0.5))
        if n < 0:
            raise ValueError(f"{ms} ms is negative")
        return n

    def _Configure(self, block_ms: float, overlap_ms: float) -> None:
        block, overlap = self._Samples(block_ms), self._Samples(overlap_ms)
        if block < 1 or overlap < 1:
            raise ValueError(f"block_ms={block_ms} and overlap_ms={overlap_ms} must each be at least one sample at {self.sr} Hz")
        tail = np.zeros(overlap, dtype=np.float32)
        keep = min(overlap, self._tail.size)
        tail[:keep] = self._tail[:keep]
        self._block, self._overlap, self._tail = block, overlap, tail
        ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
        self._fade_in = (np.sin(0.5 * np.pi * ramp) ** 2).astype(np.float32)
        self._fade_out = (1.0 - self._fade_in).astype(np.float32)
