"""Emotion reference selection (SPEC_RESOLUTIONS R11 and R12).

Per active block the engine measures rms_db and pitch_std_st; EmotionSelector turns them into one weight per reference
slot. Pure Python with an injectable clock, so every rule is testable with literal feature sequences.
"""
from __future__ import annotations

import math
import statistics
import time
from collections import deque
from typing import Callable, Sequence

from seedvc_params import (
    EMOTION_ANGRY_DB_ABOVE_MEDIAN,
    EMOTION_ANGRY_PITCH_STD_ST,
    EMOTION_DWELL_MS,
    EMOTION_EMBED_TAU_S,
    EMOTION_HYPED_PITCH_STD_ST,
    EMOTION_PITCH_WINDOW_S,
    EMOTION_RMS_WINDOW_S,
    EMOTION_WHISPER_DB_BELOW_MEDIAN,
    SLOT_NAMES,
)

# ---------------------------------------------------------------- PUBLIC
assert len(SLOT_NAMES) >= 4  # the auto rules are named for slots 0..3; slots 4+ are manual-only
NEUTRAL, HYPED, WHISPER, ANGRY = 0, 1, 2, 3  # indices into SLOT_NAMES, fixed by name
MIN_VOICED_FRAMES = 10  # choice: fewer voiced F0 frames than this in the window give no pitch spread (None)


class PitchSpreadTracker:
    """Standard deviation of voiced F0 in semitones over the last EMOTION_PITCH_WINDOW_S of F0 frames (R11)."""

    def __init__(self, hop_s: float = 0.01):
        self._frames: deque[float] = deque(maxlen=max(1, int(round(EMOTION_PITCH_WINDOW_S / hop_s))))

    def Add(self, f0_hz) -> None:
        """Append one frame per element. Values <= 0 or non-finite are unvoiced."""
        for hz in f0_hz:
            self._frames.append(12.0 * math.log2(hz) if hz > 0.0 and math.isfinite(hz) else math.nan)

    def StdSemitones(self) -> float | None:
        """Population std of the voiced frames in the window, or None with fewer than MIN_VOICED_FRAMES."""
        voiced = [v for v in self._frames if v == v]
        if len(voiced) < MIN_VOICED_FRAMES:
            return None
        return statistics.pstdev(voiced)

    def Reset(self) -> None:
        self._frames.clear()


# R11 auto-mode rules as data, in priority order: the first whose test holds picks the slot, Neutral otherwise.
# Each test takes (rms_db, median_rms_db, pitch_std_st).
_AUTO_RULES = (
    (WHISPER, lambda rms, median, spread: rms < median - EMOTION_WHISPER_DB_BELOW_MEDIAN),
    (ANGRY, lambda rms, median, spread: rms > median + EMOTION_ANGRY_DB_ABOVE_MEDIAN and spread > EMOTION_ANGRY_PITCH_STD_ST),
    (HYPED, lambda rms, median, spread: spread > EMOTION_HYPED_PITCH_STD_ST),
)


class EmotionSelector:
    """Picks the reference slot and the smoothed per-slot weights.

    Auto mode, rules in order (R11): Whisper/Soft if rms_db < median - 8 dB; Angry/Intense if rms_db > median + 6 dB and
    pitch_std_st > 2.5; Hyped/Laughing if pitch_std_st > 3.5; else Neutral. The median is taken over the last 30 s of
    active-block rms_db values, the current block included.
    Dwell (R11, hysteresis): a candidate different from the selection must hold on every update for EMOTION_DWELL_MS
    before the selection changes; a block whose candidate equals the selection, or differs from the pending one, restarts
    the timer. The selection starts at Neutral. Empty slots are skipped: the candidate falls back to Neutral, then to the
    first loaded slot. A selection whose slot was cleared moves to that fallback at once.
    Manual mode: weights are exactly one-hot on manual_slot (same empty-slot fallback), with no dwell and no smoothing.
    Auto-mode weights follow the one-hot target with time constant EMOTION_EMBED_TAU_S (R12).
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self.Reset()

    def Reset(self) -> None:
        self._rms: deque[tuple[float, float]] = deque()
        self._selected = -1
        self._candidate = -1
        self._pending: tuple[int, float] | None = None
        self._weights = [0.0] * len(SLOT_NAMES)
        self._last_t: float | None = None

    def Update(
        self,
        rms_db: float,
        pitch_std_st: float | None,
        loaded: Sequence[bool],
        mode: str,
        manual_slot: int,
        active: bool = True,
    ) -> list[float]:
        """Feed one block. mode is the emotion_mode parameter ('auto' or 'manual'), manual_slot the manual_slot parameter.
        Returns the weights, one per slot, summing to 1 (all zero when no slot is loaded)."""
        loaded = [bool(v) for v in loaded]
        if len(loaded) != len(SLOT_NAMES):
            raise ValueError(f"loaded needs {len(SLOT_NAMES)} entries")
        if mode not in ("auto", "manual"):
            raise ValueError("mode must be 'auto' or 'manual'")
        if not 0 <= manual_slot < len(SLOT_NAMES):
            raise ValueError(f"manual_slot must be 0..{len(SLOT_NAMES) - 1}")
        now = self._clock()
        if not any(loaded):
            self._selected, self._candidate, self._pending = -1, -1, None
            self._weights, self._last_t = [0.0] * len(SLOT_NAMES), now
            return list(self._weights)
        if mode == "manual":
            self._selected = self._candidate = self._Resolve(manual_slot, loaded)
            self._pending = None
            self._weights, self._last_t = self._OneHot(self._selected), now
            return list(self._weights)
        self._ChooseAuto(now, rms_db, pitch_std_st, loaded, active)
        self._Smooth(now, loaded)
        return list(self._weights)

    def Selected(self) -> int:
        """Slot index driving the prompt latents (-1 before the first update or with no slot loaded)."""
        return self._selected

    def Weights(self) -> list[float]:
        return list(self._weights)

    def State(self) -> dict:
        return {
            "selected": self._selected,
            "candidate": self._candidate,
            "pending_slot": None if self._pending is None else self._pending[0],
            "median_rms_db": statistics.median(v for _, v in self._rms) if self._rms else None,
            "weights": list(self._weights),
        }

    # ---------------------------------------------------------------- PRIVATE
    @staticmethod
    def _OneHot(slot: int) -> list[float]:
        return [1.0 if i == slot else 0.0 for i in range(len(SLOT_NAMES))]

    @staticmethod
    def _Resolve(slot: int, loaded: list[bool]) -> int:
        if loaded[slot]:
            return slot
        return NEUTRAL if loaded[NEUTRAL] else loaded.index(True)

    def _Rule(self, rms_db: float, pitch_std_st: float | None) -> int:
        median = statistics.median(v for _, v in self._rms)
        spread = 0.0 if pitch_std_st is None else pitch_std_st
        return next((slot for slot, applies in _AUTO_RULES if applies(rms_db, median, spread)), NEUTRAL)

    def _ChooseAuto(self, now: float, rms_db: float, pitch_std_st: float | None, loaded: list[bool], active: bool) -> None:
        if self._selected < 0 or not loaded[self._selected]:
            self._selected = self._Resolve(NEUTRAL, loaded)
            self._pending = None
        if not active or not math.isfinite(rms_db):
            return
        self._rms.append((now, rms_db))
        while now - self._rms[0][0] > EMOTION_RMS_WINDOW_S:
            self._rms.popleft()
        candidate = self._Resolve(self._Rule(rms_db, pitch_std_st), loaded)
        self._candidate = candidate
        if candidate == self._selected:
            self._pending = None
        elif self._pending is None or self._pending[0] != candidate:
            self._pending = (candidate, now)
        elif (now - self._pending[1]) * 1000.0 >= EMOTION_DWELL_MS:
            self._selected, self._pending = candidate, None

    def _Smooth(self, now: float, loaded: list[bool]) -> None:
        target = self._OneHot(self._selected)
        if self._last_t is None or sum(self._weights) <= 0.0:
            weights = target
        else:
            alpha = 1.0 - math.exp(-max(0.0, now - self._last_t) / EMOTION_EMBED_TAU_S)
            weights = [w + (t - w) * alpha for w, t in zip(self._weights, target)]
        weights = [w if ok else 0.0 for w, ok in zip(weights, loaded)]
        total = sum(weights)
        self._weights = [w / total for w in weights] if total > 0.0 else target
        self._last_t = now
