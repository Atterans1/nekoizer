"""WORLD prosody stage for the Seed-VC engine (spec R1, R14, R15).

Post-conversion DSP: FCPE tracks F0, pyworld analyses the converted block and
resynthesises it with a semitone pitch shift, a spectral-envelope formant shift
and an F0 expander (deviation of log-F0 from its running median is scaled).

Prosody transfer: Seed-VC takes no source pitch (f0_condition is false and the engine passes f0=None), so the converted voice
carries the content's timing but a flattened pitch contour and loudness envelope. prosody_follow moves the converted F0 contour
toward the source's (both as deviations from their own running medians, so the target voice's register is kept) and
energy_follow moves the loudness envelope toward the source's. Both are post-hoc: the F0 transfer is a WORLD resynthesis of the
converted audio, the energy transfer a smoothed gain.
"""
import copy
import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pyworld as pw
import torch
from scipy.ndimage import maximum_filter1d
from scipy.signal import medfilt, resample_poly

import seedvc_params as P

FCPE_SR = 16000
FCPE_HOP = 160  # samples at FCPE_SR, i.e. 10 ms
WORLD_FRAME_PERIOD_MS = 5.0
WORLD_LEAD_S = 0.03  # synthesised before the returned region when only the region is analysed (transfer path)
WORLD_MARGIN_S = 0.04  # extra audio before the first analysed frame, for the pitch-adaptive analysis windows
TRANSFER_MAX_DEV_ST = 12.0  # the source's deviation from its median is clamped to +-12 semitones
TRANSFER_FADE_ST = (7.0, 10.0)  # source-vs-converted deviation gap (st) over which the transfer weight fades to zero: an F0 octave error is a 12 st gap
TRANSFER_SMOOTH_FRAMES = 5  # the weight ramps over 25 ms at voicing edges of the source
ENERGY_FLOOR_DB = -60.0  # frames below this (either signal) are silence: no gain
ENERGY_WINDOW_S = 0.02  # RMS window, as the critic's loudness track
ENERGY_GAIN = (0.5, 2.0)
ENERGY_PEAK = 0.98  # the gain never pushes a 20 ms frame (and its +-4 neighbours) above this
ENERGY_MEAN_TAU_S = 3.0  # running mean of each signal's log level: 1/count until its time constant is reached
_LOG_FLOOR = 1e-16
_LN2 = math.log(2.0)


class FcpeF0:
    """torchfcpe F0 tracker returning Hz per 10 ms frame, 0.0 where unvoiced."""

    HOP_S = 0.01
    F0_MIN = 80.0
    F0_MAX = 880.0
    THRESHOLD = 0.006
    # Voicing threshold of the tracker that reads the SOURCE for the prosody transfer. THRESHOLD calls 91 percent of 0.1 RMS white noise
    # voiced, this calls none. Picked on an independent criterion before any transfer clip was scored: against pyworld.harvest on four
    # real speech clips it keeps voicing precision at 0.99 or better (THRESHOLD: 0.81 to 0.998) for a recall cost of 0.04 to 0.09.
    # THRESHOLD itself stays: the tracker of the converted audio must run as the stage always ran it.
    SOURCE_THRESHOLD = 0.1

    # ---- PUBLIC
    def __init__(self, device=None):
        """device None picks cuda when available; self.device is the device actually used."""
        import torchfcpe

        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = self.THRESHOLD
        self._model = torchfcpe.spawn_bundled_infer_model(device=self.device)

    def WithThreshold(self, threshold):
        """A tracker that shares this one's model (no second load, no extra VRAM) and differs only in its voicing threshold."""
        other = copy.copy(self)
        other.threshold = float(threshold)
        return other

    def Infer(self, audio, sr):
        """Mono float audio at any sr -> float64 F0 array of len(audio16k)//160 + 1 frames.

        Frame i is at time i * HOP_S. Frames below F0_MIN (unvoiced) are 0.0.
        Audio shorter than one FCPE hop still yields one frame.
        """
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        sr = int(sr)
        if sr != FCPE_SR:
            g = math.gcd(FCPE_SR, sr)
            x = resample_poly(x, FCPE_SR // g, sr // g).astype(np.float32)
        n_frames = len(x) // FCPE_HOP + 1
        wav = torch.from_numpy(np.ascontiguousarray(x)).reshape(1, -1, 1).to(self.device)
        f0 = self._model.infer(
            wav,
            sr=FCPE_SR,
            decoder_mode="local_argmax",
            threshold=self.threshold,
            f0_min=self.F0_MIN,
            f0_max=self.F0_MAX,
            interp_uv=False,
            output_interp_target_length=n_frames,
        )
        out = f0.reshape(-1).detach().cpu().numpy().astype(np.float64)
        out[~np.isfinite(out)] = 0.0
        out[out < self.F0_MIN] = 0.0
        return out


@dataclass(frozen=True)
class SourceAudio:
    """The source speech a converted block is aligned to.

    x16: source audio at 16 kHz. t0: index in x16 (may be fractional) of the converted block's first sample. f0: FcpeF0.Infer of
    x16[f0_origin:] (10 ms frames), needed when prosody_follow > 0; the loudness is measured on x16 itself."""

    x16: np.ndarray
    t0: float
    f0: np.ndarray | None = None
    f0_origin: int = 0


class ProsodyStage:
    """Pitch / formant / F0-expander stage and source-prosody transfer, with running medians of the converted and source F0."""

    # ---- PUBLIC
    def __init__(self, fcpe=None, sr=P.MODEL_SR):
        self.sr = int(sr)
        self.fcpe = fcpe if fcpe is not None else FcpeF0()
        self._clock_s = 0.0          # stage clock: end time of the last applied block
        self._voiced = deque()       # (stage time s, log F0) of voiced converted frames
        self._src_voiced = deque()   # the same for the source
        self._level_mean = [None, None]  # running mean level in dB of the converted and the source speech
        self._level_count = [0, 0]
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="world")  # cheaptrick runs here while d4c runs there
        self._telemetry = {"fcpe_ms": 0.0, "world_ms": 0.0, "device": str(self.fcpe.device)}

    def Process(self, wav, new_samples, hop_samples, pitch_shift_st, formant_shift, f0_expander,
                prosody_follow=0.0, energy_follow=0.0, source=None):
        """Return (float32 array of exactly new_samples, stage_applied).

        wav is float32 mono at self.sr: context followed by the region to return; the
        output is the processed last new_samples samples. hop_samples is how much of
        wav's tail is new since the previous call (only those F0 frames feed the medians).

        Neutral controls (0.0, 0.0, 1.0, follow 0.0, 0.0) return an exact copy of the input slice with no
        FCPE/WORLD call and no state change. An exactly silent wav returns zeros and
        False, also without state change. new_samples outside 1..len(wav) or non-finite
        input in the shifted path raise ValueError.

        prosody_follow (0..1) moves the converted log-F0 contour toward the source's, energy_follow (0..1) the loudness
        envelope; both need source (SourceAudio), prosody_follow also source.f0. With follow 0.0 the output is bit-identical to
        what the stage produced before the transfer existed.
        """
        n = len(wav)
        if not 1 <= new_samples <= n:
            raise ValueError("new_samples must be within 1..len(wav)")
        if not (0.0 <= prosody_follow <= 1.0 and 0.0 <= energy_follow <= 1.0):
            raise ValueError("prosody_follow and energy_follow must be within 0..1")
        if (prosody_follow > 0.0 or energy_follow > 0.0) and source is None:
            raise ValueError("source is required when prosody_follow or energy_follow is above 0")
        if prosody_follow > 0.0 and source.f0 is None:
            raise ValueError("source.f0 is required when prosody_follow is above 0")
        neutral = (pitch_shift_st == 0.0 and formant_shift == 0.0 and f0_expander == 1.0
                   and prosody_follow == 0.0 and energy_follow == 0.0)
        if neutral:
            return wav[-new_samples:].copy(), False
        if not np.all(np.isfinite(wav)):
            raise ValueError("wav contains non-finite samples")
        if not np.any(wav):
            return np.zeros(new_samples, dtype=np.float32), False

        hop = int(min(max(hop_samples, 0), n))
        self._clock_s += hop / self.sr
        fcpe_ms = 0.0
        t1 = time.perf_counter()
        if pitch_shift_st != 0.0 or formant_shift != 0.0 or f0_expander != 1.0 or prosody_follow > 0.0:
            out, fcpe_ms, t1 = self._World(wav, new_samples, hop, pitch_shift_st, formant_shift, f0_expander, prosody_follow, source)
        else:
            out = wav[-new_samples:].copy()
        if energy_follow > 0.0:
            out = self._FollowEnergy(out, wav, hop, energy_follow, source)
        self._telemetry = {"fcpe_ms": fcpe_ms, "world_ms": (time.perf_counter() - t1) * 1000.0, "device": str(self.fcpe.device)}
        return out, True

    def MedianF0Hz(self):
        """Running median F0 (Hz) over the last F0_MEDIAN_WINDOW_S of voiced frames; 0.0 if none."""
        med = self._MedianLog(self._voiced)
        return 0.0 if med is None else float(math.exp(med))

    def Reset(self):
        """Clear the stage clock, the running medians and the running loudness means."""
        self._clock_s = 0.0
        self._voiced.clear()
        self._src_voiced.clear()
        self._level_mean = [None, None]
        self._level_count = [0, 0]

    def Telemetry(self):
        """fcpe_ms, world_ms of the last Process call that ran the stage, and the FCPE device."""
        return dict(self._telemetry)

    # ---- PRIVATE
    def _World(self, wav, new_samples, hop, pitch_shift_st, formant_shift, f0_expander, follow, source):
        """FCPE, WORLD analysis, F0 modification (shift, expander, transfer) and resynthesis. Returns (out, fcpe_ms, t_after_fcpe).

        Without the transfer the whole block is analysed and synthesised, as it always was. With it only the returned region
        plus WORLD_LEAD_S before it, because the region is all that is returned and the cost is proportional to the frames."""
        n = len(wav)
        t0 = time.perf_counter()
        f0_fcpe = self.fcpe.Infer(wav, self.sr)
        t1 = time.perf_counter()

        # One spare frame of zero padding so the synthesis is never shorter than wav.
        n_frames = int(math.ceil(n / self.sr * 1000.0 / WORLD_FRAME_PERIOD_MS)) + 1
        pad = int(math.ceil((n_frames - 1) * WORLD_FRAME_PERIOD_MS / 1000.0 * self.sr)) + 1 - n
        x = np.zeros(n + max(pad, 0), dtype=np.float64)
        x[:n] = wav
        times = np.arange(n_frames) * (WORLD_FRAME_PERIOD_MS / 1000.0)
        f0 = self._ToWorldGrid(f0_fcpe, times)
        f0_src = None
        if follow > 0.0:
            f0_src = self._ToWorldGrid(source.f0, (source.t0 + times * P.INPUT_SR - source.f0_origin) / P.INPUT_SR)

        first = self._FirstFrame(n, new_samples) if follow > 0.0 else 0
        s0 = first * self.sr // 200  # exact: _FirstFrame returns a multiple of the frames per whole sample
        if first == 0:
            xa, fa, ta = x, f0, times
        else:
            a = max(0, s0 - int(WORLD_MARGIN_S * self.sr))
            xa, fa, ta = x[a:], f0[first:], times[first:] - a / self.sr
        ap_job = self._pool.submit(pw.d4c, xa, fa, ta, self.sr)
        sp = pw.cheaptrick(xa, fa, ta, self.sr)
        ap = ap_job.result()

        voiced = f0 > 0.0
        self._Feed(self._voiced, f0, times, n, hop)
        if f0_src is not None:
            self._Feed(self._src_voiced, f0_src, times, n, hop)
        horizon = self._clock_s - P.F0_MEDIAN_WINDOW_S
        for store in (self._voiced, self._src_voiced):
            while store and store[0][0] < horizon:
                store.popleft()
        f0_mod = f0.copy()
        if voiced.any():
            logf = np.log(f0[voiced])
            med = self._MedianLog(self._voiced)
            if f0_src is not None and med is not None:
                logf = self._Transfer(f0, voiced, f0_src, med, follow)[voiced]
            if f0_expander != 1.0 and med is not None:
                logf = med + (logf - med) * f0_expander
            f0_mod[voiced] = np.exp(logf + pitch_shift_st / 12.0 * math.log(2.0))
        if formant_shift != 0.0:
            sp = self._WarpEnvelope(sp, 2.0 ** (formant_shift / 12.0))

        y = pw.synthesize(np.ascontiguousarray(f0_mod[first:]), np.ascontiguousarray(sp), np.ascontiguousarray(ap), self.sr,
                          WORLD_FRAME_PERIOD_MS)
        need = n - s0
        if len(y) < need:
            y = np.pad(y, (0, need - len(y)))
        return np.nan_to_num(y[:need][-new_samples:]).astype(np.float32), (t1 - t0) * 1000.0, t1

    def _FirstFrame(self, n, new_samples):
        """First WORLD frame to analyse when only the region is: WORLD_LEAD_S before it, on a whole-sample frame boundary."""
        whole = 200 // math.gcd(self.sr, 200)  # frames per whole number of samples: s0 = first * sr / 200 is an integer
        first = int(math.floor(((n - new_samples) / self.sr - WORLD_LEAD_S) * 1000.0 / WORLD_FRAME_PERIOD_MS))
        return max(0, first // whole * whole)

    def _Transfer(self, f0, voiced, f0_src, med, follow):
        """Log-F0 per frame (valid on voiced frames): med + (1 - w) * converted deviation + w * source deviation.

        w = follow where the source is voiced (ramped over TRANSFER_SMOOTH_FRAMES at its voicing edges) and fades to zero when the
        two deviations disagree by more than TRANSFER_FADE_ST semitones, which is what an octave error looks like."""
        lc = np.log(np.where(voiced, f0, 1.0))
        med_src = self._MedianLog(self._src_voiced)
        src_voiced = f0_src > 0.0
        if med_src is None or not src_voiced.any():
            return lc
        idx = np.arange(len(f0))
        dev_src = np.clip(np.log(np.where(src_voiced, f0_src, 1.0)) - med_src, -TRANSFER_MAX_DEV_ST / 12.0 * _LN2, TRANSFER_MAX_DEV_ST / 12.0 * _LN2)
        dev_src = medfilt(np.interp(idx, idx[src_voiced], dev_src[src_voiced]), 5)  # despike over the voiced runs, interpolated across gaps
        dev_conv = lc - med
        w = follow * np.convolve(src_voiced.astype(np.float64), np.ones(TRANSFER_SMOOTH_FRAMES) / TRANSFER_SMOOTH_FRAMES, "same")
        gap_st = np.abs(dev_src - dev_conv) * (12.0 / _LN2)
        w *= np.clip((TRANSFER_FADE_ST[1] - gap_st) / (TRANSFER_FADE_ST[1] - TRANSFER_FADE_ST[0]), 0.0, 1.0)
        return med + (1.0 - w) * dev_conv + w * dev_src

    def _FollowEnergy(self, out, wav, hop, energy, source):
        """Gain that moves the loudness envelope of out toward the source's: both are 20 ms RMS tracks (5 ms hop) in dB relative to
        their own running mean, gain_dB = energy * (source deviation - converted deviation) on frames active in both, limited to
        ENERGY_GAIN, kept below clipping, smoothed over 5 frames and interpolated to samples."""
        n, new = len(wav), len(out)
        step = self.sr * WORLD_FRAME_PERIOD_MS / 1000.0
        centers = np.arange(int(new / step) + 1) * step  # frame centres, in samples from the start of out
        win = int(round(ENERGY_WINDOW_S * self.sr))
        lo = np.round(centers - win / 2.0).astype(np.int64)
        before = wav[max(n - new - win, 0) : n - new]  # the converted block's own samples before the region, for the first frames
        ext = np.concatenate((before, out)).astype(np.float64)
        csum = np.concatenate(([0.0], np.cumsum(ext * ext)))
        a = np.clip(lo + len(before), 0, len(ext))
        b = np.clip(lo + win + len(before), 0, len(ext))
        level_conv = 10.0 * np.log10((csum[b] - csum[a]) / np.maximum(b - a, 1) + 1e-12)
        level_src = self._SourceLevel(source, (n - new + centers) * (P.INPUT_SR / self.sr))
        active = (level_conv > ENERGY_FLOOR_DB) & (level_src > ENERGY_FLOOR_DB)
        fresh = (n - new + centers) >= (n - hop)
        for k, level in enumerate((level_conv, level_src)):
            for value in level[fresh & (level > ENERGY_FLOOR_DB)]:
                self._LevelMean(k, float(value))
        if self._level_mean[0] is None or self._level_mean[1] is None:
            return out
        gain_db = np.where(active, energy * ((level_src - self._level_mean[1]) - (level_conv - self._level_mean[0])), 0.0)
        gain = np.clip(10.0 ** (gain_db / 20.0), *ENERGY_GAIN)
        peak = maximum_filter1d(np.abs(out), size=win, mode="constant")[np.clip(np.round(centers).astype(np.int64), 0, new - 1)]
        gain = np.minimum(gain, ENERGY_PEAK / np.maximum(maximum_filter1d(peak, size=9, mode="nearest"), 1e-9))
        gain = np.convolve(np.pad(gain, 2, mode="edge"), np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0, "valid")
        return (out * np.interp(np.arange(new), centers, gain)).astype(np.float32)

    @staticmethod
    def _SourceLevel(source, centers16):
        """20 ms RMS level in dB of the source around each position (16 kHz sample index into source.x16)."""
        x = source.x16
        a = np.clip(source.t0 + centers16 - 160.0, 0.0, len(x))
        b = np.clip(source.t0 + centers16 + 160.0, 0.0, len(x))
        i0, i1 = int(a.min()), min(int(math.ceil(b.max())) + 1, len(x))
        csum = np.concatenate(([0.0], np.cumsum(np.square(x[i0:i1], dtype=np.float64))))
        grid = np.arange(len(csum), dtype=np.float64)
        power = (np.interp(b - i0, grid, csum) - np.interp(a - i0, grid, csum)) / np.maximum(b - a, 1.0)
        return 10.0 * np.log10(power + 1e-12)

    def _LevelMean(self, k, value):
        """Running mean of the level of signal k (0 converted, 1 source): 1/count at first, then an exponential average."""
        self._level_count[k] += 1
        alpha = max(1.0 / self._level_count[k], 1.0 - math.exp(-WORLD_FRAME_PERIOD_MS / 1000.0 / ENERGY_MEAN_TAU_S))
        mean = self._level_mean[k]
        self._level_mean[k] = value if mean is None else mean + alpha * (value - mean)

    def _ToWorldGrid(self, f0_fcpe, times):
        """Interpolate FCPE F0 (10 ms grid) onto WORLD frame times; unvoiced if either neighbour is."""
        pos = times / FcpeF0.HOP_S
        i0 = np.clip(np.floor(pos).astype(np.int64), 0, len(f0_fcpe) - 1)
        i1 = np.clip(i0 + 1, 0, len(f0_fcpe) - 1)
        frac = np.clip(pos - i0, 0.0, 1.0)
        a, b = f0_fcpe[i0], f0_fcpe[i1]
        both = (a > 0.0) & (b > 0.0)
        out = np.where(frac < 0.5, a, b)
        la = np.log(np.where(both, a, 1.0))
        lb = np.log(np.where(both, b, 1.0))
        out = np.where(both, np.exp(la + (lb - la) * frac), out)
        return np.ascontiguousarray(out, dtype=np.float64)

    def _Feed(self, store, f0, times, n, hop):
        # A frame is new when it lies in the last hop samples of wav.
        if hop > 0:
            t_end = n / self.sr
            new = (f0 > 0.0) & (times >= t_end - hop / self.sr) & (times < t_end)
            for t, v in zip(times[new], np.log(f0[new])):
                store.append((self._clock_s - (t_end - t), float(v)))

    @staticmethod
    def _MedianLog(store):
        if not store:
            return None
        return float(np.median([v for _, v in store]))

    @staticmethod
    def _WarpEnvelope(sp, r):
        """new_sp(f) = sp(f / r) along the frequency axis in the log domain, edge-clamped."""
        k = sp.shape[1]
        src = np.clip(np.arange(k) / r, 0.0, k - 1.0)
        lo = np.floor(src).astype(np.int64)
        hi = np.minimum(lo + 1, k - 1)
        w = src - lo
        logsp = np.log(np.maximum(sp, _LOG_FLOOR))
        return np.exp(logsp[:, lo] * (1.0 - w) + logsp[:, hi] * w)
