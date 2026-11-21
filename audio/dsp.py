"""Input-side DSP: 80 Hz high-pass and a soft-knee 2.5:1 compressor with automatic makeup gain.

Both classes keep their state across blocks, so processing a signal block by block gives the same samples as
processing it in one piece. Every tunable value comes from seedvc_params (R8, R9); the few choices that file
leaves open are the module constants below and are reported in the task notes.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

import seedvc_params as P

# ---- PUBLIC
ACTIVE_FLOOR_DB = -50.0     # choice: a hop whose peak is below this is silence, not "active speech" (R9 leaves it open)
MAKEUP_WINDOW_S = 3.0       # R9: average gain reduction over the last 3 s
MAKEUP_MIN_ACTIVE_S = 0.2   # choice: below this much active speech in the window the makeup target is held, not updated
MAKEUP_TAU_S = 0.3          # choice: the applied makeup follows its target with this time constant (no steps)
HOP_S = 0.001               # choice: control rate of the level detector and gain computer (1 ms)
GAIN_SMOOTH_S = 0.0003      # choice: per-sample one-pole smoothing of the 1 ms gain steps (removes zipper noise)


def CompressorStaticCurveDb(level_db: float, threshold_db: float = P.COMP_THRESHOLD_DB, ratio: float = P.COMP_RATIO,
                            knee_db: float = P.COMP_KNEE_DB) -> float:
    """Output level (dB) for an input level (dB): unity below the knee, slope 1/ratio above it, quadratic inside."""
    over = level_db - threshold_db
    half = knee_db / 2.0
    if over <= -half:
        return level_db
    if over >= half:
        return threshold_db + over / ratio
    return level_db + (1.0 / ratio - 1.0) * (over + half) ** 2 / (2.0 * knee_db)


class HighPass:
    """2nd-order Butterworth high-pass (R8) with filter state kept across blocks."""

    # ---- PUBLIC
    def __init__(self, sr: int, cutoff_hz: float = P.HPF_HZ):
        self._sos = butter(2, cutoff_hz, btype="highpass", fs=sr, output="sos")
        self._zi = np.zeros((self._sos.shape[0], 2))

    def Process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x.astype(np.float32)
        y, self._zi = sosfilt(self._sos, x, zi=self._zi)
        return y.astype(np.float32)

    def Reset(self) -> None:
        self._zi[:] = 0.0


class SoftCompressor:
    """Feed-forward peak compressor, soft knee, 2.5:1 (R9), automatic makeup = average gain reduction on active
    speech over the last 3 s, capped at COMP_MAKEUP_MAX_DB.

    The level is the sample peak of each 1 ms hop (dBFS). The gain reduction follows attack/release smoothing at that
    hop rate; the resulting gain is applied per sample through a short one-pole smoother. There is no look-ahead.
    """

    # ---- PUBLIC
    def __init__(self, sr: int):
        self._hop = max(1, round(sr * HOP_S))
        dt = self._hop / sr
        self._a_attack = math.exp(-dt / (P.COMP_ATTACK_MS / 1000.0))
        self._a_release = math.exp(-dt / (P.COMP_RELEASE_MS / 1000.0))
        self._a_makeup = math.exp(-dt / MAKEUP_TAU_S)
        self._a_smooth = math.exp(-1.0 / (sr * GAIN_SMOOTH_S))
        self._window_hops = max(1, round(MAKEUP_WINDOW_S / dt))
        self._min_active_hops = max(1, round(MAKEUP_MIN_ACTIVE_S / dt))
        self.Reset()

    def Process(self, x: np.ndarray) -> np.ndarray:
        n = x.size
        if n == 0:
            return x.astype(np.float32)
        ax = np.abs(x)
        gain_step = np.empty(n)
        pos = 0
        while pos < n:
            take = min(self._hop - self._fill, n - pos)
            peak = float(ax[pos:pos + take].max())
            if peak > self._peak:
                self._peak = peak
            gain_step[pos:pos + take] = self._gain
            self._fill += take
            pos += take
            if self._fill == self._hop:
                self._CompleteHop()
        gain, self._smooth_zi = lfilter([1.0 - self._a_smooth], [1.0, -self._a_smooth], gain_step, zi=self._smooth_zi)
        return (x * gain).astype(np.float32)

    def GetState(self) -> dict:
        return {"gain_reduction_db": self._gr_db, "makeup_db": self._makeup_db}

    def Reset(self) -> None:
        self._gr_db = 0.0
        self._makeup_db = 0.0
        self._target_makeup_db = 0.0
        self._gain = 1.0
        self._fill = 0
        self._peak = 0.0
        self._smooth_zi = np.array([self._a_smooth])
        self._win_gr = [0.0] * self._window_hops
        self._win_active = [0] * self._window_hops
        self._win_pos = 0
        self._win_sum = 0.0
        self._win_count = 0

    # ---- PRIVATE
    def _CompleteHop(self) -> None:
        level_db = 20.0 * math.log10(max(self._peak, 1e-9))
        target = max(0.0, level_db - CompressorStaticCurveDb(level_db))
        a = self._a_attack if target > self._gr_db else self._a_release
        self._gr_db = a * self._gr_db + (1.0 - a) * target
        active = 1 if level_db >= ACTIVE_FLOOR_DB else 0
        i = self._win_pos
        self._win_sum += active * self._gr_db - self._win_active[i] * self._win_gr[i]
        self._win_count += active - self._win_active[i]
        self._win_gr[i] = self._gr_db
        self._win_active[i] = active
        self._win_pos = (i + 1) % self._window_hops
        if self._win_count >= self._min_active_hops:
            self._target_makeup_db = min(P.COMP_MAKEUP_MAX_DB, max(0.0, self._win_sum / self._win_count))
        self._makeup_db = self._a_makeup * self._makeup_db + (1.0 - self._a_makeup) * self._target_makeup_db
        self._gain = 10.0 ** ((self._makeup_db - self._gr_db) / 20.0)
        self._fill = 0
        self._peak = 0.0
