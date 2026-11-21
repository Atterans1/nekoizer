import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import seedvc_params as P
from audio.dsp import CompressorStaticCurveDb, HighPass, SoftCompressor

SR = 48000


def Sine(freq_hz, amp, seconds, sr=SR):
    t = np.arange(int(round(seconds * sr))) / sr
    return (amp * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def Rms(x):
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))


def DbToAmp(db):
    return 10.0 ** (db / 20.0)


def SplitRuns(x, sizes):
    out, pos, i = [], 0, 0
    while pos < x.size:
        n = sizes[i % len(sizes)]
        out.append(x[pos:pos + n])
        pos += n
        i += 1
    return out


class HighPassTests(unittest.TestCase):
    def GainAt(self, freq_hz):
        y = HighPass(SR).Process(Sine(freq_hz, 1.0, 3.0))
        return Rms(y[-SR:]) / Rms(Sine(freq_hz, 1.0, 1.0))

    def test_40hz_is_attenuated_to_0_2425_and_1khz_passes(self):
        self.assertAlmostEqual(self.GainAt(40.0), 0.24254, delta=0.001)
        self.assertAlmostEqual(self.GainAt(1000.0), 1.0, delta=0.001)

    def test_cutoff_is_minus_3db_at_80hz(self):
        self.assertAlmostEqual(self.GainAt(80.0), 0.70711, delta=0.001)

    def test_block_by_block_output_is_identical_to_whole_signal(self):
        x = (np.random.default_rng(1234).standard_normal(SR) * 0.1).astype(np.float32)
        whole = HighPass(SR).Process(x)
        for sizes in ([480], [1], [7, 333, 1024, 2], [1023]):
            hp = HighPass(SR)
            split = np.concatenate([hp.Process(b) for b in SplitRuns(x, sizes)])
            self.assertTrue(np.array_equal(whole, split), sizes)

    def test_dc_is_removed(self):
        y = HighPass(SR).Process(np.full(SR, 0.5, np.float32))
        self.assertLess(abs(float(y[-1])), 1e-9)
        self.assertGreater(float(y[0]), 0.49)

    def test_reset_clears_state(self):
        rng = np.random.default_rng(7)
        a = (rng.standard_normal(4800) * 0.1).astype(np.float32)
        b = (rng.standard_normal(4800) * 0.1).astype(np.float32)
        hp = HighPass(SR)
        hp.Process(a)
        hp.Reset()
        self.assertTrue(np.array_equal(hp.Process(b), HighPass(SR).Process(b)))

    def test_empty_block_returns_empty_float32(self):
        y = HighPass(SR).Process(np.zeros(0, np.float32))
        self.assertEqual((y.shape, y.dtype), ((0,), np.dtype(np.float32)))

    def test_works_at_other_rates(self):
        y = HighPass(44100).Process(Sine(40.0, 1.0, 3.0, 44100))
        self.assertAlmostEqual(Rms(y[-44100:]) / Rms(Sine(40.0, 1.0, 1.0, 44100)), 0.24254, delta=0.001)


class StaticCurveTests(unittest.TestCase):
    def test_constants_are_the_spec_choices(self):
        self.assertEqual((P.COMP_RATIO, P.COMP_THRESHOLD_DB, P.COMP_KNEE_DB), (2.5, -20.0, 6.0))

    def test_literal_curve_points(self):
        for level, expected in ((-40.0, -40.0), (-23.0, -23.0), (-20.0, -20.45), (-17.0, -18.8), (-10.0, -16.0),
                                (0.0, -12.0)):
            self.assertAlmostEqual(CompressorStaticCurveDb(level), expected, places=9, msg=str(level))

    def test_slope_above_knee_is_one_over_ratio(self):
        slope = (CompressorStaticCurveDb(-5.0) - CompressorStaticCurveDb(-10.0)) / 5.0
        self.assertAlmostEqual(slope, 0.4, places=9)

    def test_knee_is_continuous_monotonic_and_slope_between_one_and_ratio(self):
        levels = np.arange(-30.0, 0.0, 0.01)
        out = np.array([CompressorStaticCurveDb(float(v)) for v in levels])
        slope = np.diff(out) / 0.01
        self.assertTrue(np.all(slope > 0.0))
        self.assertTrue(np.all(slope <= 1.0 + 1e-9))
        self.assertTrue(np.all(slope >= 0.4 - 1e-9))
        self.assertLess(float(np.max(np.abs(np.diff(slope)))), 0.01)  # slope has no jump at either knee edge


class SoftCompressorTests(unittest.TestCase):
    def Run(self, freq, level_db, seconds, comp=None):
        comp = comp or SoftCompressor(SR)
        x = Sine(freq, DbToAmp(level_db), seconds)
        return comp, x, comp.Process(x)

    def test_steady_gain_reduction_follows_the_static_curve(self):
        for level, expected in ((-10.0, 6.0), (0.0, 12.0), (-20.0, 0.45), (-30.0, 0.0)):
            comp, _, _ = self.Run(1000.0, level, 2.0)
            self.assertAlmostEqual(comp.GetState()["gain_reduction_db"], expected, delta=0.02, msg=str(level))

    def test_ten_db_more_input_gives_four_db_more_output(self):
        lo, _, _ = self.Run(1000.0, -10.0, 2.0)
        hi, _, _ = self.Run(1000.0, 0.0, 2.0)
        gr_step = hi.GetState()["gain_reduction_db"] - lo.GetState()["gain_reduction_db"]
        self.assertAlmostEqual(10.0 - gr_step, 4.0, delta=0.03)

    def test_attack_reaches_63_percent_of_6db_after_8ms(self):
        comp = SoftCompressor(SR)
        comp.Process(Sine(1000.0, DbToAmp(-10.0), 0.008))
        self.assertAlmostEqual(comp.GetState()["gain_reduction_db"], 3.7927, delta=0.005)

    def test_release_falls_to_37_percent_after_120ms(self):
        comp = SoftCompressor(SR)
        comp.Process(Sine(1000.0, DbToAmp(-10.0), 1.0))
        comp.Process(Sine(1000.0, DbToAmp(-40.0), 0.120))
        self.assertAlmostEqual(comp.GetState()["gain_reduction_db"], 2.2073, delta=0.005)

    def test_makeup_equals_average_gain_reduction_and_level_is_restored(self):
        comp, x, y = self.Run(1000.0, -10.0, 5.0)
        self.assertAlmostEqual(comp.GetState()["makeup_db"], 6.0, delta=0.1)
        self.assertAlmostEqual(float(np.max(np.abs(y[-SR // 2:]))), DbToAmp(-10.0), delta=0.007)

    def test_makeup_is_capped_at_9db(self):
        comp, x, y = self.Run(1000.0, 0.0, 5.0)
        self.assertEqual(P.COMP_MAKEUP_MAX_DB, 9.0)
        self.assertAlmostEqual(comp.GetState()["makeup_db"], 9.0, delta=0.05)
        self.assertAlmostEqual(float(np.max(np.abs(y[-SR // 2:]))), DbToAmp(-3.0), delta=0.01)

    def test_signal_below_the_knee_passes_unchanged(self):
        comp, x, y = self.Run(1000.0, -40.0, 3.0)
        self.assertEqual(comp.GetState()["makeup_db"], 0.0)
        self.assertLess(float(np.max(np.abs(y - x))), 1e-7)

    def test_makeup_is_held_through_silence(self):
        comp = SoftCompressor(SR)
        comp.Process(Sine(1000.0, DbToAmp(-10.0), 4.0))
        before = comp.GetState()["makeup_db"]
        y = comp.Process(np.zeros(5 * SR, np.float32))
        state = comp.GetState()
        self.assertAlmostEqual(before, 6.0, delta=0.1)
        self.assertAlmostEqual(state["makeup_db"], before, delta=0.05)
        self.assertLess(state["gain_reduction_db"], 1e-6)
        self.assertTrue(np.all(y == 0.0))

    def test_makeup_follows_new_average_once_the_window_has_passed(self):
        comp = SoftCompressor(SR)
        comp.Process(Sine(1000.0, DbToAmp(-10.0), 5.0))
        comp.Process(Sine(1000.0, DbToAmp(-40.0), 6.0))
        self.assertAlmostEqual(comp.GetState()["makeup_db"], 0.0, delta=0.02)

    def test_block_by_block_output_is_identical_to_whole_signal(self):
        rng = np.random.default_rng(99)
        t = np.arange(2 * SR) / SR
        x = (rng.standard_normal(2 * SR) * 0.15 * (1.0 + 0.9 * np.sin(2 * np.pi * 3.0 * t))).astype(np.float32)
        whole = SoftCompressor(SR).Process(x)
        for sizes in ([480], [1], [7, 333, 1021, 2], [4096]):
            comp = SoftCompressor(SR)
            split = np.concatenate([comp.Process(b) for b in SplitRuns(x, sizes)])
            self.assertTrue(np.array_equal(whole, split), sizes)

    def test_input_is_not_modified_and_output_is_float32_same_length(self):
        x = Sine(1000.0, 0.5, 0.5)
        before = x.copy()
        y = SoftCompressor(SR).Process(x)
        self.assertTrue(np.array_equal(x, before))
        self.assertEqual((y.shape, y.dtype), (x.shape, np.dtype(np.float32)))

    def test_zeros_stay_exact_zeros_and_empty_block_is_ok(self):
        comp = SoftCompressor(SR)
        self.assertTrue(np.all(comp.Process(np.zeros(4800, np.float32)) == 0.0))
        self.assertEqual(comp.Process(np.zeros(0, np.float32)).shape, (0,))

    def test_reset_returns_to_unity(self):
        comp, _, _ = self.Run(1000.0, -10.0, 5.0)
        comp.Reset()
        self.assertEqual(comp.GetState(), {"gain_reduction_db": 0.0, "makeup_db": 0.0})


if __name__ == "__main__":
    unittest.main()
