"""Auto-balancer (SPEC_RESOLUTIONS R17): keep inference inside the block time by dropping diffusion steps for a while.

The exponential moving average of inference_s / block_s is updated after every block. When it reaches
BALANCER_THRESHOLD the effective step count becomes requested - BALANCER_STEP_DROP (never below the ParamSpec floor and
never above the request) for BALANCER_HOLD_S, then the requested count is restored. It can trigger again straight away.
"""
from __future__ import annotations

import threading
import time

import seedvc_params as P

# ---- PUBLIC


class AutoBalancer:
    """Thread-safe. 'now' arguments are seconds on any monotonic clock; omit them to use time.monotonic()."""

    def __init__(self, params: P.ParamStore):
        self._params = params
        self._lock = threading.Lock()
        self._ema: float | None = None
        self._ratio = 0.0
        self._until: float | None = None  # end of the current step drop, None when not throttled
        self._triggers = 0

    def Update(self, inference_s: float, block_s: float, now: float | None = None) -> None:
        if block_s <= 0.0:
            raise ValueError(f"block_s must be positive, got {block_s}")
        t = time.monotonic() if now is None else float(now)
        enabled = bool(self._params.Get("auto_balancer"))
        with self._lock:
            self._ratio = float(inference_s) / float(block_s)
            a = P.BALANCER_EMA_ALPHA
            self._ema = self._ratio if self._ema is None else a * self._ratio + (1.0 - a) * self._ema
            self._Expire(t, enabled)
            if enabled and self._until is None and self._ema >= P.BALANCER_THRESHOLD:
                self._until = t + P.BALANCER_HOLD_S
                self._triggers += 1

    def EffectiveSteps(self, requested: int, now: float | None = None) -> int:
        t = time.monotonic() if now is None else float(now)
        enabled = bool(self._params.Get("auto_balancer"))
        requested = int(requested)
        with self._lock:
            self._Expire(t, enabled)
            if self._until is None:
                return requested
            return min(requested, max(P.ParamSpec["diffusion_steps"]["min"], requested - P.BALANCER_STEP_DROP))

    def State(self, now: float | None = None) -> dict:
        t = time.monotonic() if now is None else float(now)
        enabled = bool(self._params.Get("auto_balancer"))
        with self._lock:
            self._Expire(t, enabled)
            return {
                "enabled": enabled,
                "ema": self._ema,
                "ratio": self._ratio,
                "throttled": self._until is not None,
                "throttle_remaining_s": 0.0 if self._until is None else self._until - t,
                "triggers": self._triggers,
                "threshold": P.BALANCER_THRESHOLD,
                "step_drop": P.BALANCER_STEP_DROP,
                "hold_s": P.BALANCER_HOLD_S,
            }

    # ---- PRIVATE

    def _Expire(self, t: float, enabled: bool) -> None:
        if self._until is not None and (not enabled or t >= self._until):
            self._until = None
