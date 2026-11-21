"""Formant- and duration-preserving pitch control on the converted waveform (WORLD analysis/resynthesis).

Method: WORLD decomposes a chunk into F0, spectral envelope (formants) and aperiodicity. Only F0 is changed, the
envelope and aperiodicity are kept frame for frame, so duration is unchanged and formants do not move. This is
not resampling. The vocoder's own F0 input cannot be used for this: HiFT follows the harmonics in the mel and
ignores a rescaled excitation (measured: median F0 unchanged at +-12 semitones).
"""
from __future__ import annotations

import numpy as np
import pyworld as pw

FRAME_PERIOD_MS = 5.0
F0_FLOOR = 60.0
F0_CEIL = 600.0


# ---------------------------------------------------------------- PUBLIC
def Process(
    audio: np.ndarray,
    sample_rate: int,
    semitones: float,
    source_f0: np.ndarray | None = None,
    source_hop_s: float = 0.02,
    register_ratio: float = 1.0,
    source_start_s: float = 0.0,
) -> np.ndarray:
    """Return audio with F0 changed, same length.

    semitones: shift applied to the contour (0 keeps it).
    source_f0: if given, the live speaker's F0 contour (Hz per source_hop_s starting at source_start_s seconds
               relative to this audio's first sample, 0 = unvoiced). It replaces the converted voice's own contour, moved into the target's
               register by register_ratio (target median F0 / speaker median F0).
    With semitones == 0 and no source_f0 the input is returned untouched.
    """
    if semitones == 0 and source_f0 is None:
        return audio
    x = audio.astype(np.float64)
    f0, t = pw.dio(x, sample_rate, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL, frame_period=FRAME_PERIOD_MS)
    f0 = pw.stonemask(x, f0, t, sample_rate)
    sp = pw.cheaptrick(x, f0, t, sample_rate)
    ap = pw.d4c(x, f0, t, sample_rate)
    if source_f0 is not None:
        f0 = _ResampleContour(source_f0, source_hop_s, source_start_s, t) * register_ratio
    f0 = f0 * (2.0 ** (semitones / 12.0))
    out = pw.synthesize(f0, sp, ap, sample_rate, FRAME_PERIOD_MS)
    out = out.astype(np.float32)
    if len(out) < len(audio):
        out = np.pad(out, (0, len(audio) - len(out)))
    return out[: len(audio)]


# ---------------------------------------------------------------- PRIVATE
def _ResampleContour(contour: np.ndarray, hop_s: float, start_s: float, times: np.ndarray) -> np.ndarray:
    """Interpolate a Hz contour (0 = unvoiced) onto analysis frame times: log-linear inside voiced runs."""
    src_t = start_s + np.arange(len(contour)) * hop_s
    voiced = contour > 0
    if not voiced.any():
        return np.zeros_like(times)
    logf = np.interp(times, src_t[voiced], np.log(contour[voiced]))
    near_voiced = np.interp(times, src_t, voiced.astype(np.float64)) > 0.5
    return np.where(near_voiced, np.exp(logf), 0.0)
