"""Pure monitoring maths: percentiles, counter deltas, block-window summaries and the health rules.

No clocks, threads or I/O: every function maps plain values to plain values, so the tests use literal numbers.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

import seedvc_params as P

# ---- PUBLIC
OK, WARN, FAIL = "ok", "warn", "fail"

WINDOW_S = 60.0       # every "last minute" metric looks at this many seconds
HISTORY_MAX_S = 1800  # the 1 Hz series keeps this many seconds

# Health thresholds. Literal on purpose: the tests, the UI and the docs quote them.
INFERENCE_WARN_RATIO = P.BALANCER_THRESHOLD  # p95 inference above 85 percent of the block time: where the balancer reacts
INFERENCE_FAIL_RATIO = 1.0                   # p95 inference above the block time cannot keep up in real time
UNDERRUN_WARN = 1                            # output underruns in the window
UNDERRUN_FAIL = 10
E2E_WARN_MS = 450.0                          # p95 end-to-end latency in the window
E2E_FAIL_MS = 600.0
OVERRUN_WARN = 1                             # input overruns in the window
OVERRUN_FAIL = 10
ERROR_FAIL = 1                               # convert errors in the window: each one dropped a block
SILENT_INPUT_DB = -70.0                      # VAD active on a block quieter than this (dBFS rms) is a mismatch
SPEECH_INPUT_DB = -45.0                      # input at least this loud is speech ...
SILENT_OUTPUT_PEAK = 1e-3                    # ... so an output peak below this (-60 dBFS) means nothing came out
MISMATCH_WARN = 3                            # mismatching blocks in the window
MISMATCH_FAIL = 10

LIMITS = {"e2e_warn_ms": E2E_WARN_MS, "e2e_fail_ms": E2E_FAIL_MS}  # sent with the history so the page never hardcodes them
RULE_NAMES = ("inference_p95", "underruns", "e2e_p95", "input_overruns", "convert_errors", "vad_level_mismatch")
# Columns of the 1 Hz series that TelemetryRecorder.History returns. ui/fake_pipeline.py builds the same keys.
HISTORY_FIELDS = ("t", "block_ms", "blocks", "active_blocks", "inference_ms_mean", "inference_ms_max", "e2e_ms_mean",
                  "e2e_ms_max", "underruns", "input_overruns", "convert_errors", "balancer_triggers", "vram_used_mb",
                  "vram_total_mb", "vram_proc_mb", "gpu_util_pct", "proc_cpu_pct", "proc_rss_mb")
HISTORY_INT_FIELDS = frozenset({"blocks", "active_blocks", "underruns", "input_overruns", "convert_errors",
                                "balancer_triggers"})


def Percentile(values: Sequence[float] | np.ndarray, q: float) -> float | None:
    """q-th percentile (linear interpolation, like numpy) of the finite values; None when there are none."""
    a = _Finite(values)
    return float(np.percentile(a, q)) if a.size else None


def Summarize(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """{'n', 'p50', 'p95', 'max'} of the finite values; the statistics are None when there are none."""
    a = _Finite(values)
    if not a.size:
        return {"n": 0, "p50": None, "p95": None, "max": None}
    return {"n": int(a.size), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def CounterDelta(times: Sequence[float], totals: Sequence[float], now: float, window_s: float) -> int:
    """Growth of a cumulative counter over the last window_s seconds before now.

    The baseline is the newest sample at or before now - window_s, or the first sample when the series is younger than
    the window. A counter that went down was reset, so what it holds now counts as new."""
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(totals, dtype=np.float64)
    if t.size == 0:
        return 0
    first = max(int(np.searchsorted(t, now - window_s, side="right")) - 1, 0)
    delta = v[-1] - v[first]
    return int(round(v[-1] if delta < 0 else delta))


def BlockWindowStats(active: Sequence[float], inference_s: Sequence[float], e2e_ms: Sequence[float],
                     in_rms_db: Sequence[float], out_peak: Sequence[float]) -> dict[str, Any]:
    """Summaries of the blocks of one window (equal-length arrays, oldest first). NaN means unknown.

    Inference statistics cover active blocks only: a silent block never runs the model."""
    act = np.asarray(active, dtype=np.float64) > 0.5
    inference = np.asarray(inference_s, dtype=np.float64)
    rms = np.asarray(in_rms_db, dtype=np.float64)
    peak = np.asarray(out_peak, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        silent_input = int(np.count_nonzero(act & (rms < SILENT_INPUT_DB)))
        silent_output = int(np.count_nonzero(act & (rms > SPEECH_INPUT_DB) & (peak < SILENT_OUTPUT_PEAK)))
    n, k = int(act.size), int(act.sum())
    return {"blocks": n, "active_blocks": k, "active_ratio": (k / n) if n else None,
            "inference_ms": Summarize(inference[act] * 1000.0), "e2e_ms": Summarize(e2e_ms),
            "silent_input_active": silent_input, "silent_output_active": silent_output}


def EvaluateHealth(stats: dict[str, Any]) -> dict[str, Any]:
    """Apply the six health rules to one window of statistics.

    stats keys: block_ms, window_s, inference_ms and e2e_ms (Summarize dicts), underruns, input_overruns, convert_errors,
    silent_input_active, silent_output_active (counts in the window). Missing data never fails a rule: no evidence, no fault.
    Returns {'status': ok|warn|fail (the worst rule), 'failing': text of the first worst rule or '', 'rules': [...]}."""
    w = int(round(stats.get("window_s") or WINDOW_S))
    rules = [
        _InferenceRule(stats.get("inference_ms") or {}, stats.get("block_ms")),
        _CountRule("underruns", stats.get("underruns"), UNDERRUN_WARN, UNDERRUN_FAIL, f"output underruns in the last {w} s"),
        _E2eRule((stats.get("e2e_ms") or {}).get("p95")),
        _CountRule("input_overruns", stats.get("input_overruns"), OVERRUN_WARN, OVERRUN_FAIL, f"input overruns in the last {w} s"),
        _CountRule("convert_errors", stats.get("convert_errors"), ERROR_FAIL, ERROR_FAIL, f"convert errors in the last {w} s"),
        _MismatchRule(int(stats.get("silent_input_active") or 0), int(stats.get("silent_output_active") or 0), w),
    ]
    worst = max(rules, key=lambda r: _SEVERITY[r["status"]])
    return {"status": worst["status"], "failing": worst["message"] if worst["status"] != OK else "", "rules": rules}


def RowsFromHistory(history: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn the column lists of a History() result into one dict per second."""
    columns = {k: v for k, v in history.items() if isinstance(v, list)}
    return [{k: v[i] for k, v in columns.items()} for i in range(int(history.get("n", 0)))]


# ---- PRIVATE
_SEVERITY = {OK: 0, WARN: 1, FAIL: 2}


def _Finite(values: Sequence[float] | np.ndarray) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    return a[np.isfinite(a)]


def _Rule(name: str, status: str, value: Any, limit: Any, message: str) -> dict[str, Any]:
    return {"name": name, "status": status, "value": value, "limit": limit, "message": message}


def _InferenceRule(inference: dict[str, Any], block_ms: float | None) -> dict[str, Any]:
    p95 = inference.get("p95")
    if p95 is None or not block_ms:
        return _Rule("inference_p95", OK, p95, block_ms, "no inference measured yet")
    if p95 > block_ms * INFERENCE_FAIL_RATIO:
        return _Rule("inference_p95", FAIL, p95, block_ms * INFERENCE_FAIL_RATIO,
                     f"inference p95 {p95:.0f} ms is above the {block_ms:.0f} ms block time")
    if p95 > block_ms * INFERENCE_WARN_RATIO:
        return _Rule("inference_p95", WARN, p95, block_ms * INFERENCE_WARN_RATIO,
                     f"inference p95 {p95:.0f} ms is above {INFERENCE_WARN_RATIO * 100:.0f} percent of the {block_ms:.0f} ms block time")
    return _Rule("inference_p95", OK, p95, block_ms * INFERENCE_WARN_RATIO,
                 f"inference p95 {p95:.0f} ms is within the {block_ms:.0f} ms block time")


def _E2eRule(p95: float | None) -> dict[str, Any]:
    if p95 is None:
        return _Rule("e2e_p95", OK, None, E2E_WARN_MS, "no end-to-end latency measured yet")
    if p95 > E2E_FAIL_MS:
        return _Rule("e2e_p95", FAIL, p95, E2E_FAIL_MS, f"e2e p95 {p95:.0f} ms is above {E2E_FAIL_MS:.0f} ms")
    if p95 > E2E_WARN_MS:
        return _Rule("e2e_p95", WARN, p95, E2E_WARN_MS, f"e2e p95 {p95:.0f} ms is above {E2E_WARN_MS:.0f} ms")
    return _Rule("e2e_p95", OK, p95, E2E_WARN_MS, f"e2e p95 {p95:.0f} ms is within {E2E_WARN_MS:.0f} ms")


def _CountRule(name: str, count: Any, warn: int, fail: int, what: str) -> dict[str, Any]:
    n = int(count or 0)
    status = FAIL if n >= fail else WARN if n >= warn else OK
    return _Rule(name, status, n, fail if status == FAIL else warn, f"{n} {what}")


def _MismatchRule(silent_input: int, silent_output: int, w: int) -> dict[str, Any]:
    parts = []
    if silent_input:
        parts.append(f"{silent_input} VAD-active blocks with silent input (below {SILENT_INPUT_DB:.0f} dBFS)")
    if silent_output:
        parts.append(f"{silent_output} speech blocks with silent output")
    n = silent_input + silent_output
    status = FAIL if n >= MISMATCH_FAIL else WARN if n >= MISMATCH_WARN else OK
    text = ("; ".join(parts) if parts else "no VAD-active block with silent input or silent output") + f" in the last {w} s"
    return _Rule("vad_level_mismatch", status, n, MISMATCH_FAIL if status == FAIL else MISMATCH_WARN, text)
