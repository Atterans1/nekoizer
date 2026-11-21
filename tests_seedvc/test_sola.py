import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from audio.sola import Sola

SR = 22050
FREQ = 440.0
BLOCK, OVERLAP, SEARCH = 4851, 992, 441  # 220 ms, 45 ms, 20 ms at 22050 Hz
CHUNK = BLOCK + OVERLAP + SEARCH  # 6284
# The largest sample-to-sample step of a unit sine is 2*sin(pi*FREQ/SR) = 0.125295. The crossfade of two unit sines that
# are at most half a sample apart in phase adds under 0.002 (fade slope <= 0.0016 per sample times the sine difference),
# and the first block's fade-in from silence adds the same slope term, so 0.132 (5 percent headroom) is a safe ceiling
# while a boundary glitch of one wrong-phase sample would exceed it by a wide margin.
MAX_STEP = 0.132


def Sine(positions):
    return np.sin(2.0 * np.pi * FREQ * np.asarray(positions, dtype=np.float64) / SR)


def Chunk(start, length, frac=0.0):
    return Sine(start + frac + np.arange(length)).astype(np.float32)


def RunStream(sola, jitters, fracs, block, start=1000):
    """Feed chunks that each begin 'jitter' samples (plus a fractional phase) before the nominal timeline position."""
    out, pos = [], start
    for jit, frac in zip(jitters, fracs):
        out.append(sola.Process(Chunk(pos - jit, sola.InputSamples, frac)))
        pos += block
    return np.concatenate(out), pos


class SolaTests(unittest.TestCase):
    def test_sample_counts_at_the_spec_defaults(self):
        sola = Sola(SR, 220, 45, 20)
        self.assertEqual(
            (sola.BlockSamples, sola.OverlapSamples, sola.SearchSamples, sola.InputSamples), (4851, 992, 441, 6284)
        )

    def test_output_length_is_exactly_one_block_every_call(self):
        sola = Sola(SR, 220, 45, 20)
        rng = np.random.default_rng(7)
        for extra in (0, 0, 1, 100, 0, 5000):
            out = sola.Process(rng.standard_normal(CHUNK + extra).astype(np.float32))
            self.assertEqual(out.shape, (4851,))
            self.assertEqual(out.dtype, np.float32)

    def test_jittered_sine_has_no_discontinuity(self):
        jitters = [0, 123, 441, 7, 300, 56, 399, 211, 5, 440, 250, 90]
        fracs = [0.0, 0.31, -0.27, 0.45, -0.5, 0.12, 0.0, -0.4, 0.22, 0.5, -0.33, 0.18]
        out, _ = RunStream(Sola(SR, 220, 45, 20), jitters, fracs, BLOCK)
        steps = np.abs(np.diff(out))
        self.assertEqual(out.size, 12 * BLOCK)
        self.assertLess(steps.max(), MAX_STEP)
        self.assertGreater(steps.max(), 0.120)  # a real full-scale 440 Hz sine, not silence
        naive = np.concatenate([Chunk(1000 + k * BLOCK - j, CHUNK, f)[:BLOCK] for k, (j, f) in enumerate(zip(jitters, fracs))])
        self.assertGreater(np.abs(np.diff(naive)).max(), 0.3)  # the same blocks joined without SOLA do jump

    def test_integer_jitter_reconstructs_the_timeline_exactly(self):
        jitters = [0, 123, 441, 7, 300, 56, 399, 211]
        out, _ = RunStream(Sola(SR, 220, 45, 20), jitters, [0.0] * 8, BLOCK)
        ideal = Sine(1000 + np.arange(out.size))
        np.testing.assert_allclose(out[OVERLAP:], ideal[OVERLAP:], atol=1e-5)

    def test_crossfade_is_sin2_in_cos2_out_and_sums_to_one(self):
        sola = Sola(1000, 10, 5, 3)  # block 10, overlap 5, search 3 samples
        first = sola.Process(np.ones(18, dtype=np.float32))
        np.testing.assert_allclose(first[:5], [0.0, 0.1464466, 0.5, 0.8535534, 1.0], atol=1e-6)  # sin^2(pi/2*[0 .25 .5 .75 1])
        np.testing.assert_allclose(first[5:], np.ones(5), atol=1e-7)
        second = sola.Process(np.ones(18, dtype=np.float32))  # tail of ones fading out under ones fading in
        np.testing.assert_allclose(second, np.ones(10), atol=1e-6)

    def test_offset_search_finds_the_aligned_start_across_the_whole_window(self):
        timeline = np.random.default_rng(1234).standard_normal(200).astype(np.float32)
        for early in (0, 1, 2, 3):  # the new chunk begins 'early' samples before the nominal start (search = 3)
            sola = Sola(1000, 10, 5, 3)
            sola.Process(timeline[100:118])  # emits timeline[100:110]; tail = timeline[110:115]
            out = sola.Process(timeline[110 - early : 128 - early])
            np.testing.assert_allclose(out, timeline[110:120], atol=1e-6, err_msg=f"early={early}")

    def test_state_survives_a_rebuild(self):
        sola = Sola(SR, 220, 45, 20)
        head, pos = RunStream(sola, [0, 123, 441], [0.0] * 3, BLOCK)
        sola.Rebuild(180, 40)  # block 3969, overlap 882; the carried tail keeps its first 882 samples
        self.assertEqual((sola.BlockSamples, sola.OverlapSamples, sola.InputSamples), (3969, 882, 5292))
        tail, _ = RunStream(sola, [57, 300], [0.0, 0.0], 3969, start=pos)
        joined = np.concatenate([head, tail])
        ideal = Sine(1000 + np.arange(joined.size))
        np.testing.assert_allclose(joined[OVERLAP:], ideal[OVERLAP:], atol=1e-5)
        self.assertLess(np.abs(np.diff(joined)).max(), MAX_STEP)
        fresh = Sola(SR, 180, 40, 20)  # no carried state: fades in from silence
        first = fresh.Process(Chunk(pos - 57, 5292))
        self.assertGreater(np.abs(first[:882] - ideal[head.size : head.size + 882]).max(), 0.3)

    def test_rebuild_to_a_longer_overlap_keeps_the_part_that_exists(self):
        sola = Sola(SR, 220, 40, 20)
        head, pos = RunStream(sola, [0, 100], [0.0, 0.0], BLOCK)
        sola.Rebuild(220, 50)  # overlap 1103, only 882 samples were carried
        out = sola.Process(Chunk(pos - 9, sola.InputSamples))
        ideal = Sine(pos + np.arange(out.size))
        np.testing.assert_allclose(out[:882], ideal[:882], atol=1e-5)
        self.assertEqual(out.size, BLOCK)

    def test_reset_clears_the_tail(self):
        sola = Sola(1000, 10, 5, 3)
        sola.Process(np.ones(18, dtype=np.float32))
        sola.Process(np.ones(18, dtype=np.float32))
        sola.Reset()
        self.assertEqual(float(sola.Process(np.ones(18, dtype=np.float32))[0]), 0.0)

    def test_short_chunk_is_rejected(self):
        with self.assertRaises(ValueError):
            Sola(SR, 220, 45, 20).Process(np.zeros(CHUNK - 1, dtype=np.float32))

    def test_input_is_not_modified(self):
        chunk = np.random.default_rng(3).standard_normal(CHUNK).astype(np.float32)
        copy = chunk.copy()
        Sola(SR, 220, 45, 20).Process(chunk)
        np.testing.assert_array_equal(chunk, copy)


if __name__ == "__main__":
    unittest.main()
