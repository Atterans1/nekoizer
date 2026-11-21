"""Tests for engine.prosody_stage on generated voiced signals (no model-converted audio needed)."""
import math
import os
import sys
import unittest

import numpy as np
import pyworld as pw
from scipy.signal import lfilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.prosody_stage import FcpeF0, ProsodyStage

SR = 22050
CTX = int(0.5 * SR)
HOP_NEW = int(0.285 * SR)
# Identity controls that still run WORLD (the exact neutral triple is skipped by design).
IDENTITY = dict(pitch_shift_st=0.0, formant_shift=0.0, f0_expander=1.0 + 1e-9)


def MakeVoice(f0_of_t, seconds, seed=0):
    """Sawtooth source with the given F0 contour through three vowel-/a/ formant resonators."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    f0 = f0_of_t(t)
    phase = np.cumsum(f0) / SR
    x = 2.0 * (phase - np.floor(phase)) - 1.0
    x = x + 1e-3 * np.random.default_rng(seed).standard_normal(n)
    for freq, bw in ((700.0, 90.0), (1200.0, 110.0), (2600.0, 170.0)):
        rad = math.exp(-math.pi * bw / SR)
        theta = 2.0 * math.pi * freq / SR
        x = lfilter([1.0 - rad], [1.0, -2.0 * rad * math.cos(theta), rad * rad], x)
    return (0.5 * x / np.max(np.abs(x))).astype(np.float32)


def Harvest(x):
    """Independent F0 estimator (not FCPE): pyworld.harvest, 5 ms frames. Returns (f0, times)."""
    f0, t = pw.harvest(np.ascontiguousarray(x, dtype=np.float64), SR, f0_floor=50.0, f0_ceil=800.0,
                       frame_period=5.0)
    return f0, t


def MedianHz(x):
    f0, _ = Harvest(x)
    return float(np.median(f0[f0 > 0.0]))


def SemitoneStd(x, skip_s):
    f0, t = Harvest(x)
    v = (f0 > 0.0) & (t >= skip_s)
    return float(np.std(12.0 * np.log2(f0[v])))


def Centroid(x):
    spec = np.abs(np.fft.rfft(x.astype(np.float64) * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1.0 / SR)
    return float(np.sum(freqs * spec) / np.sum(spec))


class CountingFcpe:
    """Wraps a real FcpeF0 and counts Infer calls."""

    def __init__(self, inner):
        self.inner = inner
        self.device = inner.device
        self.calls = 0

    def Infer(self, audio, sr):
        self.calls += 1
        return self.inner.Infer(audio, sr)


class ProsodyStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fcpe = FcpeF0()
        print("\nFCPE device used: %s" % cls.fcpe.device)
        # Steady voice, mild 1% vibrato at 160 Hz, 1.2 s = 0.5 s context + 0.7 s region.
        cls.f0_steady = lambda t: 160.0 * (1.0 + 0.01 * np.sin(2 * np.pi * 5.0 * t))
        cls.steady = MakeVoice(cls.f0_steady, 1.2)
        cls.region = len(cls.steady) - CTX
        cls.true_median = float(np.median(cls.f0_steady(np.arange(CTX, len(cls.steady)) / SR)))
        stage = ProsodyStage(cls.fcpe, SR)
        cls.base, applied = stage.Process(cls.steady, cls.region, cls.region, **IDENTITY)
        assert applied
        cls.base_median = MedianHz(cls.base)
        cls.base_centroid = Centroid(cls.base)

    def _Run(self, **controls):
        stage = ProsodyStage(self.fcpe, SR)
        out, applied = stage.Process(self.steady, self.region, self.region, **controls)
        self.assertTrue(applied)
        return out

    def test_neutral_is_bit_exact_and_skips_fcpe(self):
        counter = CountingFcpe(self.fcpe)
        stage = ProsodyStage(counter, SR)
        out, applied = stage.Process(self.steady, self.region, self.region, 0.0, 0.0, 1.0)
        self.assertFalse(applied)
        self.assertEqual(counter.calls, 0)
        self.assertTrue(np.array_equal(out, self.steady[-self.region:]))
        self.assertEqual(out.dtype, np.float32)
        self.assertIsNot(out, self.steady)
        self.assertEqual(stage.MedianF0Hz(), 0.0)

    def test_pitch_plus_and_minus_12(self):
        up = MedianHz(self._Run(pitch_shift_st=12.0, formant_shift=0.0, f0_expander=1.0))
        down = MedianHz(self._Run(pitch_shift_st=-12.0, formant_shift=0.0, f0_expander=1.0))
        r_up, r_down = up / self.base_median, down / self.base_median
        print("pitch: base median %.2f Hz, +12 ratio %.4f (%.2f Hz), -12 ratio %.4f (%.2f Hz); true median %.2f Hz"
              % (self.base_median, r_up, up, r_down, down, self.true_median))
        self.assertAlmostEqual(r_up, 2.0, delta=0.04, msg="+12 st ratio %.4f" % r_up)
        self.assertAlmostEqual(r_down, 0.5, delta=0.01, msg="-12 st ratio %.4f" % r_down)

    def test_formant_shift_moves_centroid(self):
        c0 = self.base_centroid
        c_in = Centroid(self.steady[-self.region:])
        resynth_err = abs(c0 / c_in - 1.0)
        up = self._Run(pitch_shift_st=0.0, formant_shift=2.0, f0_expander=1.0)
        down = self._Run(pitch_shift_st=0.0, formant_shift=-2.0, f0_expander=1.0)
        c_up, c_down = Centroid(up), Centroid(down)
        print("centroid Hz: input %.1f, WORLD identity %.1f, +2 %.1f, -2 %.1f; resynthesis error vs input %.4f"
              % (c_in, c0, c_up, c_down, resynth_err))
        # A shift must exceed the WORLD resynthesis error to count as a formant effect.
        self.assertGreater(c_up / c0 - 1.0, resynth_err, "centroid +2 %.1f vs %.1f" % (c_up, c0))
        self.assertGreater(1.0 - c_down / c0, resynth_err, "centroid -2 %.1f vs %.1f" % (c_down, c0))
        for sig, name in ((up, "+2"), (down, "-2")):
            ratio = MedianHz(sig) / self.base_median
            self.assertAlmostEqual(ratio, 1.0, delta=0.02, msg="pitch under formant %s: ratio %.4f" % (name, ratio))

    def test_shifted_output_contract_median_and_reset(self):
        stage = ProsodyStage(self.fcpe, SR)
        self.assertEqual(stage.MedianF0Hz(), 0.0)
        out, applied = stage.Process(self.steady, 5000, self.region, 3.0, 0.5, 1.2)
        self.assertTrue(applied)
        self.assertEqual(out.shape, (5000,))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(out)))
        med = stage.MedianF0Hz()
        print("running median %.2f Hz vs true %.2f Hz" % (med, self.true_median))
        self.assertAlmostEqual(med / self.true_median, 1.0, delta=0.02)
        tele = stage.Telemetry()
        self.assertGreater(tele["fcpe_ms"], 0.0)
        self.assertGreater(tele["world_ms"], 0.0)
        self.assertEqual(tele["device"], str(self.fcpe.device))
        stage.Reset()
        self.assertEqual(stage.MedianF0Hz(), 0.0)

    def test_silence_is_returned_unprocessed(self):
        counter = CountingFcpe(self.fcpe)
        stage = ProsodyStage(counter, SR)
        out, applied = stage.Process(np.zeros(SR // 2, dtype=np.float32), 3000, 3000, 5.0, 1.0, 1.3)
        self.assertFalse(applied)
        self.assertEqual(out.shape, (3000,))
        self.assertFalse(out.any())
        self.assertEqual(counter.calls, 0)

    def test_streaming_call_timing(self):
        stage = ProsodyStage(self.fcpe, SR)
        ms = []
        for b in range(4):
            lo = b * 2000
            stage.Process(self.steady[lo:lo + CTX + HOP_NEW], HOP_NEW, HOP_NEW, 2.0, 1.0, 1.1)
            if b:
                ms.append((stage.Telemetry()["fcpe_ms"], stage.Telemetry()["world_ms"]))
        fcpe_ms, world_ms = np.mean(ms, axis=0)
        print("per-call ms (0.5 s ctx + 0.285 s region, %s): fcpe %.1f, world %.1f" % (self.fcpe.device, fcpe_ms, world_ms))
        self.assertGreater(fcpe_ms, 0.0)
        self.assertGreater(world_ms, 0.0)

    def test_expander_scales_logf0_deviation(self):
        # One Process call over a 3 s vibrato: the running median covers 12 whole cycles (stationary) and
        # there are no block joins whose glitches would add variance to the F0 measurement.
        center, amp_st, vib_hz, seconds = 150.0, 3.0, 4.0, 3.0
        contour = lambda t, k=1.0: center * 2.0 ** (k * amp_st * np.sin(2 * np.pi * vib_hz * t) / 12.0)
        sig = MakeVoice(contour, seconds, seed=1)
        n = len(sig)
        skip_s = 0.1  # harvest edge frames of the measured region
        grid = np.arange(0.0, (n - CTX) / SR, 0.005)
        grid = grid[grid >= skip_s]

        def TrueStd(k):
            return float(np.std(12.0 * np.log2(contour(grid + CTX / SR, k))))

        # Measuring-chain error without the stage: analyse with the TRUE F0, resynthesise with the true
        # contour whose log deviation is scaled by k, measure with harvest. Its std vs k * true std is the
        # error of "harvest on a WORLD output". The stage ratio quotients two such measurements (factor
        # and identity), so each factor's tolerance is the sum of those two errors.
        frames = np.arange(int(math.ceil(n / SR / 0.005)) + 1) * 0.005
        x = np.zeros(n + int(0.01 * SR))
        x[:n] = sig
        sp = pw.cheaptrick(x, contour(frames), frames, SR)
        ap = pw.d4c(x, contour(frames), frames, SR)

        def ChainError(k):
            y = pw.synthesize(contour(frames, k), sp, ap, SR, 5.0)[:n][CTX:]
            return abs(SemitoneStd(y, skip_s) / TrueStd(k) - 1.0)

        err = {k: ChainError(k) for k in (1.0, 1.4, 0.8)}

        def Out(f0_expander):
            out, applied = ProsodyStage(self.fcpe, SR).Process(sig, n - CTX, n, 0.0, 0.0, f0_expander)
            self.assertTrue(applied)
            return out

        s_base = SemitoneStd(Out(IDENTITY["f0_expander"]), skip_s)
        print("glide: true std %.4f st, identity-path std %.4f st, chain errors %s"
              % (TrueStd(1.0), s_base, {k: round(v, 4) for k, v in err.items()}))
        for factor in (1.4, 0.8):
            tol = err[factor] + err[1.0]
            ratio = SemitoneStd(Out(factor), skip_s) / s_base
            print("glide expander %.1f: std ratio %.4f (deviation from factor %.4f, tolerance %.4f)"
                  % (factor, ratio, ratio / factor - 1.0, tol))
            self.assertLessEqual(abs(ratio / factor - 1.0), tol,
                                 "expander %.1f: ratio %.4f, base std %.4f, tol %.4f" % (factor, ratio, s_base, tol))


if __name__ == "__main__":
    unittest.main()
