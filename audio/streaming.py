"""Primitives the capture and playback chains share: a wrap-around store, a streaming resampler, and opening a started stream.

Before this module each chain carried its own copy of the wrap-around copy loop (five of them) and of the streaming resampler,
and the same open-start-close-on-failure block three times.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import resample_poly


# ---- PUBLIC
class Ring:
    """Fixed-capacity circular store addressed by absolute frame positions that only ever grow.

    It owns the wrap-around arithmetic and nothing else: the owner keeps the write and read counters and decides what a full
    ring means (overwrite the oldest, drop the newest, refuse). Put, Fill and Get never block and allocate only what they return.
    """

    def __init__(self, capacity: int, tail_shape: tuple = (), dtype=np.float32):
        self.capacity = capacity
        self.buf = np.zeros((capacity, *tail_shape), dtype)

    def Put(self, pos: int, x: np.ndarray) -> None:
        """Store x (at most capacity frames) at absolute position pos. x may broadcast across the frame shape."""
        for dst, src in self._Spans(pos, len(x)):
            self.buf[dst] = x[src]

    def Fill(self, pos: int, n: int, value: float = 0.0) -> None:
        for dst, _ in self._Spans(pos, n):
            self.buf[dst] = value

    def Get(self, pos: int, n: int, out: np.ndarray | None = None) -> np.ndarray:
        """Copy n frames starting at pos into out[:n] (a new array when out is None) and return out."""
        if out is None:
            out = np.empty((n, *self.buf.shape[1:]), self.buf.dtype)
        for dst, src in self._Spans(pos, n):
            out[src] = self.buf[dst]
        return out

    def _Spans(self, pos: int, n: int):
        """(ring slice, caller slice) pairs covering n frames from pos: one span, or two when the range wraps."""
        i = pos % self.capacity
        first = min(n, self.capacity - i)
        if first == n:
            return ((slice(i, i + n), slice(0, n)),)
        return ((slice(i, i + first), slice(0, first)), (slice(0, n - first), slice(first, n)))


class StreamResampler:
    """resample_poly one block at a time with the exact result of resampling the whole signal at once.

    Input history is kept (aligned to a multiple of 'down' so the polyphase phase is right) and only the newest samples the
    FIR still needs to see ahead are held back: half a filter length, about 0.6 ms at 22.05 kHz or 48 kHz. Equal rates pass the
    array through untouched."""

    def __init__(self, rate_in: int, rate_out: int):
        g = math.gcd(rate_in, rate_out)
        self.up, self.down = rate_out // g, rate_in // g
        self._half = -(-10 * max(self.up, self.down) // self.up)  # input samples the FIR reaches to either side
        self._buf = np.zeros(0, dtype=np.float32)
        self._base = 0  # absolute input index of _buf[0], a multiple of 'down'
        self._emitted = 0  # absolute output samples returned so far

    def Process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.up == self.down:
            return x
        self._buf = np.concatenate([self._buf, x])
        avail = self._base + self._buf.size
        target = max(0, -(-(avail - self._half) * self.up // self.down))  # outputs whose filter support is complete
        if target <= self._emitted:
            return np.zeros(0, dtype=np.float32)
        y = resample_poly(self._buf, self.up, self.down)
        first = self._base // self.down * self.up  # absolute output index of y[0]
        out = np.asarray(y[self._emitted - first : target - first], dtype=np.float32)
        self._emitted = target
        keep_from = (self._emitted * self.down // self.up - self._half) // self.down * self.down
        if keep_from > self._base:
            self._buf = self._buf[keep_from - self._base :]
            self._base = keep_from
        return out


def OpenStarted(make):
    """Create a stream with make() and start it; a stream that fails to start is closed before the error propagates."""
    stream = make()
    try:
        stream.start()
    except Exception:
        stream.close()
        raise
    return stream
