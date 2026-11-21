import math
import os
import sys
import unittest

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from audio.streaming import OpenStarted, Ring, StreamResampler  # noqa: E402


def Split(x, sizes):
    pos, k = 0, 0
    while pos < x.size:
        n = sizes[k % len(sizes)]
        yield x[pos:pos + n]
        pos += n
        k += 1


class RingTests(unittest.TestCase):
    def test_put_wraps_and_get_returns_frames_in_order(self):
        ring = Ring(8)
        ring.Put(0, np.arange(6, dtype=np.float32))
        ring.Put(6, np.array([6, 7, 8, 9], dtype=np.float32))  # frames 6..9 wrap past the end
        np.testing.assert_array_equal(ring.Get(2, 8), [2, 3, 4, 5, 6, 7, 8, 9])

    def test_get_fills_the_front_of_a_longer_out_and_returns_it(self):
        ring = Ring(4)
        ring.Put(3, np.array([1, 2, 3], dtype=np.float32))  # frames 3, 4, 5 -> slots 3, 0, 1
        out = np.full(6, 9.0, np.float32)
        self.assertIs(ring.Get(3, 3, out), out)
        np.testing.assert_array_equal(out, [1, 2, 3, 9, 9, 9])

    def test_fill_zeroes_across_the_wrap_and_leaves_the_rest(self):
        ring = Ring(5)
        ring.buf[:] = 7.0
        ring.Fill(3, 4)
        np.testing.assert_array_equal(ring.buf, [0, 0, 7, 0, 0])

    def test_put_broadcasts_mono_over_channels(self):
        ring = Ring(4, (2,))
        ring.Put(3, np.array([1, 2], dtype=np.float32)[:, None])
        np.testing.assert_array_equal(ring.Get(3, 2), [[1, 1], [2, 2]])

    def test_dtype_follows_the_ring(self):
        ring = Ring(4, dtype=np.uint8)
        ring.Put(0, np.array([1, 0, 1], dtype=np.uint8))
        self.assertEqual(ring.Get(0, 3).dtype, np.uint8)


class StreamResamplerTests(unittest.TestCase):
    def Check(self, rate_in, rate_out, n_in, expected_len):
        x = (np.random.default_rng(11).standard_normal(n_in) * 0.1).astype(np.float32)
        g = math.gcd(rate_in, rate_out)
        whole = resample_poly(x, rate_out // g, rate_in // g).astype(np.float32)
        for sizes in ([480], [1], [7, 333, 1024, 2], [n_in]):
            rs = StreamResampler(rate_in, rate_out)
            out = np.concatenate([rs.Process(b) for b in Split(x, sizes)])
            self.assertEqual(out.size, expected_len, sizes)
            self.assertLess(float(np.max(np.abs(out - whole[:out.size]))), 1e-6, sizes)

    def test_48k_to_16k_equals_whole_signal_and_lags_by_10_samples(self):
        self.Check(48000, 16000, 48000, 15990)

    def test_44k1_to_16k_holds_back_only_the_filter_reach(self):
        self.Check(44100, 16000, 44100, 15990)  # ceil((44100 - 28) * 160 / 441); the old input resampler returned 15840

    def test_48k_to_24k_for_the_recorder(self):
        self.Check(48000, 24000, 48000, 23990)

    def test_22k05_to_48k_for_the_output(self):
        self.Check(22050, 48000, 29436, 64057)  # ceil((29436 - 10) * 320 / 147)

    def test_block_seams_do_not_click(self):
        t = np.arange(6 * 4851) / 22050.0
        x = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        rs = StreamResampler(22050, 48000)
        out = np.concatenate([rs.Process(b) for b in Split(x, [4851])])
        # largest step of a 0.5 amplitude 440 Hz sine at 48 kHz is 2*0.5*sin(pi*440/48000) = 0.028794
        self.assertLess(np.abs(np.diff(out[100:])).max(), 0.0295)
        self.assertGreater(np.abs(np.diff(out[100:])).max(), 0.0285)

    def test_equal_rates_pass_the_same_array_through(self):
        x = np.arange(10, dtype=np.float32)
        self.assertIs(StreamResampler(16000, 16000).Process(x), x)


class OpenStartedTests(unittest.TestCase):
    class Stream:
        def __init__(self, fail):
            self.fail, self.started, self.closed = fail, False, False

        def start(self):
            if self.fail:
                raise OSError("device busy")
            self.started = True

        def close(self):
            self.closed = True

    def test_returns_a_started_stream(self):
        s = self.Stream(False)
        self.assertIs(OpenStarted(lambda: s), s)
        self.assertEqual((s.started, s.closed), (True, False))

    def test_closes_a_stream_that_fails_to_start_and_raises(self):
        s = self.Stream(True)
        with self.assertRaisesRegex(OSError, "device busy"):
            OpenStarted(lambda: s)
        self.assertTrue(s.closed)


if __name__ == "__main__":
    unittest.main()
