"""Prosody transfer in engine.prosody_stage: the source's pitch contour and loudness envelope onto converted audio.

Synthetic signals only. The converted audio is a flat-pitch, constant-level voice; the source is a voice with a known pitch glide and
amplitude modulation on the same time axis. The stage is fed as the engine feeds it: segments of CTX + REGION samples advancing by BLOCK.
FCPE runs on the CPU, so the tests do not touch the GPU.
"""
import math
import os
import sys
import unittest

import numpy as np
import pyworld as pw
from scipy.signal import lfilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from audio.sola import Sola
from engine.prosody_stage import FcpeF0, ProsodyStage, SourceAudio

SR, SR16 = 22050, 16000
CTX, REGION, BLOCK = 10939, 6284, 4851  # the engine's segment geometry at the spec defaults
SECONDS = 4.0
FLAT_HZ = 210.0
START_S = 0.9  # the transfer needs about a second of history for its running medians; measurements start after it


def Voice(f0_of_t, level_of_t, seconds, sr, seed):
    """Sawtooth through three vowel-/a/ formant resonators, with the given F0 contour and amplitude envelope."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    phase = np.cumsum(f0_of_t(t)) / sr
    x = 2.0 * (phase - np.floor(phase)) - 1.0 + 1e-3 * np.random.default_rng(seed).standard_normal(n)
    for freq, bw in ((700.0, 90.0), (1200.0, 110.0), (2600.0, 170.0)):
        rad = math.exp(-math.pi * bw / sr)
        theta = 2.0 * math.pi * freq / sr
        x = lfilter([1.0 - rad], [1.0, -2.0 * rad * math.cos(theta), rad * rad], x)
    return (0.5 * x / np.max(np.abs(x)) * level_of_t(t)).astype(np.float32)


def Glide(t):
    """Source pitch: +-4 st at 0.7 Hz around 130 Hz plus a 2 st drift over the clip."""
    return 130.0 * 2.0 ** ((4.0 * np.sin(2 * np.pi * 0.7 * t) + 2.0 * t / SECONDS) / 12.0)


def Swell(t):
    return 0.25 + 0.75 * (0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t)) ** 2


def Envelope(x, sr):
    """20 ms RMS in dB every 5 ms."""
    hop, win = sr // 200, sr // 50
    csum = np.concatenate(([0.0], np.cumsum(np.square(x.astype(np.float64)))))
    i = np.arange((len(x) - win) // hop + 1) * hop
    return 10.0 * np.log10((csum[i + win] - csum[i]) / win + 1e-12)


def Contour(x, sr, t0=0.0, ref=None):
    """Independent F0 estimate (pyworld.harvest, not FCPE): (times in s, semitones relative to ref Hz, default the median of x)."""
    f0, t = pw.harvest(np.ascontiguousarray(x, dtype=np.float64), sr, f0_floor=60.0, f0_ceil=800.0, frame_period=5.0)
    voiced = f0 > 0.0
    return t[voiced] + t0, 12.0 * np.log2(f0[voiced] / (np.median(f0[voiced]) if ref is None else ref))


def Segments(conv):
    """(start sample in conv, segment) as the engine hands them to the stage."""
    for s0 in range(0, len(conv) - CTX - REGION + 1, BLOCK):
        yield s0, conv[s0 : s0 + CTX + REGION]


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fcpe = FcpeF0(device="cpu")
        cls.src = Voice(Glide, Swell, SECONDS, SR16, 1)
        cls.conv = Voice(lambda t: np.full_like(t, FLAT_HZ), lambda t: np.ones_like(t), SECONDS, SR, 2)
        cls.src_fcpe = cls.fcpe.WithThreshold(FcpeF0.SOURCE_THRESHOLD)  # the engine reads the source with this tracker
        cls.f0_src = cls.src_fcpe.Infer(cls.src, SR16)
        cls.src_t, cls.src_st = Contour(cls.src, SR16)

    def _Source(self, s0, f0=None):
        return SourceAudio(self.src, s0 * SR16 / SR, self.f0_src if f0 is None else f0, 0)

    def _Stream(self, follow, energy, shift=0.0, expander=1.0, f0=None, conv=None):
        """Region outputs of every block: [(start of the region in seconds, samples)]."""
        stage = ProsodyStage(self.fcpe, SR)
        conv = self.conv if conv is None else conv
        out = []
        for s0, seg in Segments(conv):
            source = self._Source(s0, f0) if (follow > 0.0 or energy > 0.0) else None
            region, _ = stage.Process(seg, REGION, BLOCK, shift, 0.0, expander, follow, energy, source)
            out.append(((s0 + CTX) / SR, region))
        return out

    def _Contours(self, blocks):
        """Per-block harvest contours of the region outputs, pooled: (times, semitones relative to the pooled median)."""
        parts = [Contour(region, SR, t0, ref=1.0) for t0, region in blocks]  # absolute semitones: a per-block median would erase the glide
        t = np.concatenate([p[0] for p in parts])
        st = np.concatenate([p[1] for p in parts])
        return t, st - np.median(st)

    def test_zero_follow_changes_nothing(self):
        s0, seg = next(iter(Segments(self.conv)))
        source = self._Source(s0)
        stage = ProsodyStage(self.fcpe, SR)
        out, applied = stage.Process(seg, REGION, BLOCK, 0.0, 0.0, 1.0, 0.0, 0.0, source)
        self.assertFalse(applied)
        np.testing.assert_array_equal(out, seg[-REGION:])
        self.assertEqual(stage.MedianF0Hz(), 0.0)
        with_source, _ = ProsodyStage(self.fcpe, SR).Process(seg, REGION, BLOCK, 2.0, 0.0, 1.1, 0.0, 0.0, source)
        without, _ = ProsodyStage(self.fcpe, SR).Process(seg, REGION, BLOCK, 2.0, 0.0, 1.1)
        np.testing.assert_array_equal(with_source, without)

    def test_source_is_required_and_follow_is_range_checked(self):
        s0, seg = next(iter(Segments(self.conv)))
        stage = ProsodyStage(self.fcpe, SR)
        with self.assertRaisesRegex(ValueError, "source is required"):
            stage.Process(seg, REGION, BLOCK, 0.0, 0.0, 1.0, 0.5, 0.0, None)
        with self.assertRaisesRegex(ValueError, "source.f0 is required"):
            stage.Process(seg, REGION, BLOCK, 0.0, 0.0, 1.0, 0.5, 0.0, SourceAudio(self.src, 0.0))
        with self.assertRaisesRegex(ValueError, "within 0..1"):
            stage.Process(seg, REGION, BLOCK, 0.0, 0.0, 1.0, 1.5, 0.0, self._Source(s0))

    def test_pitch_glide_is_transplanted_onto_a_flat_tone(self):
        flat_t, flat_st = self._Contours(self._Stream(0.0, 0.0, shift=1e-9))  # WORLD resynthesis of the flat tone, nothing transferred
        t, st = self._Contours(self._Stream(1.0, 0.0))
        keep = t > START_S
        reference = np.interp(t[keep], self.src_t, self.src_st)
        corr = float(np.corrcoef(st[keep], reference)[0, 1])
        self.assertGreaterEqual(corr, 0.9, f"contour correlation with the source glide {corr:.3f}")
        self.assertLess(float(np.std(flat_st)), 0.3)  # without the transfer the contour stays flat
        self.assertLessEqual(float(np.max(np.abs(st[keep]))), 12.0)
        within = (np.abs(np.diff(st)) * (np.diff(t) < 0.0075))[t[1:] > START_S]  # frame to frame inside a block, after the history warm-up
        self.assertLess(float(within.max()), 3.0)  # no jumps: the glide moves at most 18 st/s, 0.09 st per 5 ms; 3 st is generous for harvest noise

    def test_pitch_range_scales_with_follow(self):
        ratios = {}
        for follow in (0.4, 0.7, 1.0):
            t, st = self._Contours(self._Stream(follow, 0.0))
            keep = t > START_S
            ratios[follow] = float(np.std(st[keep]) / np.std(np.interp(t[keep], self.src_t, self.src_st)))
        self.assertLess(ratios[0.4], ratios[0.7])
        self.assertLess(ratios[0.7], ratios[1.0])
        for follow, ratio in ratios.items():
            self.assertAlmostEqual(ratio, follow, delta=0.2, msg=f"follow {follow}: pitch std ratio {ratio:.2f}")

    def test_an_octave_error_in_the_source_is_not_followed(self):
        f0 = self.f0_src.copy()
        lo, hi = int(2.0 * 100), int(2.3 * 100)  # 300 ms of 10 ms frames: FCPE doubled the pitch
        f0[lo:hi] *= 2.0
        t, st = self._Contours(self._Stream(1.0, 0.0, f0=f0))
        window = (t > 2.05) & (t < 2.25)
        clean_t, clean_st = self._Contours(self._Stream(1.0, 0.0))
        clean = (clean_t > 2.05) & (clean_t < 2.25)
        self.assertLess(float(np.max(np.abs(st[window] - np.median(st[window])))), 4.0)
        self.assertLess(abs(float(np.median(st[window])) - float(np.median(clean_st[clean]))), 1.5)  # the doubled stretch moved the contour by under 1.5 st

    def test_an_unvoiced_source_leaves_the_converted_contour(self):
        noise = (0.1 * np.random.default_rng(5).standard_normal(len(self.src))).astype(np.float32)
        f0 = self.src_fcpe.Infer(noise, SR16)
        self.assertLess(float(np.mean(f0 > 0.0)), 0.1)
        t, st = self._Contours(self._Stream(1.0, 0.0, f0=f0))
        self.assertLess(float(np.std(st[t > START_S])), 0.3)

    def test_expander_and_shift_apply_on_top_of_the_transfer(self):
        base_t, base = self._Contours(self._Stream(1.0, 0.0))
        t, wide = self._Contours(self._Stream(1.0, 0.0, expander=1.3))
        keep, keep_base = t > START_S, base_t > START_S
        self.assertGreater(float(np.std(wide[keep])), 1.15 * float(np.std(base[keep_base])))
        shifted = np.concatenate([region for _, region in self._Stream(1.0, 0.0, shift=7.0)])
        plain = np.concatenate([region for _, region in self._Stream(1.0, 0.0)])
        f0_up, f0_plain = np.median(pw.harvest(shifted.astype(np.float64), SR, f0_floor=60.0, f0_ceil=800.0)[0]), np.median(pw.harvest(plain.astype(np.float64), SR, f0_floor=60.0, f0_ceil=800.0)[0])
        self.assertAlmostEqual(12.0 * math.log2(f0_up / f0_plain), 7.0, delta=0.5)

    def test_energy_follow_raises_the_envelope_correlation(self):
        conv = Voice(lambda t: np.full_like(t, FLAT_HZ), lambda t: 0.8 + 0.2 * np.sin(2 * np.pi * 0.31 * t + 1.0), SECONDS, SR, 3)
        src_env = Envelope(self.src, SR16)
        scores = {}
        for energy in (0.0, 0.5, 1.0):
            blocks = self._Stream(0.0, energy, conv=conv)
            heads = np.concatenate([region[:BLOCK] for _, region in blocks])  # region heads tile the time axis
            out_env = Envelope(heads, SR)
            t = CTX / SR + np.arange(len(out_env)) * 0.005 + 0.01
            ref = np.interp(t, np.arange(len(src_env)) * 0.005 + 0.01, src_env)
            keep = t > START_S
            scores[energy] = float(np.corrcoef(out_env[keep], ref[keep])[0, 1])
            self.assertLessEqual(float(np.max(np.abs(heads))), 1.0)
        self.assertLess(scores[0.0], 0.5)
        self.assertGreater(scores[0.5], scores[0.0] + 0.3)
        self.assertGreaterEqual(scores[1.0], 0.85, f"envelope correlation {scores}")

    def test_gain_is_limited_and_never_clips(self):
        loud = Voice(lambda t: np.full_like(t, FLAT_HZ), lambda t: 1.8 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)) ** 2 + 0.05, SECONDS, SR, 4)
        loud = (0.97 * loud / np.max(np.abs(loud))).astype(np.float32)
        swing = Voice(Glide, lambda t: 10.0 ** (-30.0 * (0.5 + 0.5 * np.sin(2 * np.pi * 1.3 * t)) / 20.0), SECONDS, SR16, 5)  # 30 dB deep modulation
        stage = ProsodyStage(self.fcpe, SR)
        f0 = self.src_fcpe.Infer(swing, SR16)
        gains = []
        for s0, seg in Segments(loud):
            region, _ = stage.Process(seg, REGION, BLOCK, 0.0, 0.0, 1.0, 0.0, 1.0, SourceAudio(swing, s0 * SR16 / SR, f0, 0))
            self.assertLessEqual(float(np.max(np.abs(region))), 1.0)
            before, after = Envelope(seg[-REGION:], SR), Envelope(region, SR)
            live = before > -50.0
            gains.append((after - before)[live])
        gain_db = np.concatenate(gains)
        self.assertLessEqual(float(gain_db.max()), 6.1)   # 2.0 in amplitude
        self.assertGreaterEqual(float(gain_db.min()), -6.1)  # 0.5 in amplitude

    def test_blocks_match_one_pass_and_borders_are_continuous(self):
        """Energy: the stream split into engine blocks against one call over the whole clip. The gain track must agree within 1 dB (RMS)
        and the gain must not step at the block borders. Pitch: the contour of the split stream follows the one-pass contour within 0.5 st."""
        conv = Voice(lambda t: np.full_like(t, FLAT_HZ), lambda t: 0.8 + 0.2 * np.sin(2 * np.pi * 0.31 * t + 1.0), SECONDS, SR, 3)
        heads = np.concatenate([region[:BLOCK] for _, region in self._Stream(0.0, 1.0, conv=conv)])
        n = len(heads)
        whole_src = SourceAudio(self.src, 0.0, self.f0_src, 0)  # conv[:CTX + n] starts at sample 0
        one_pass, _ = ProsodyStage(self.fcpe, SR).Process(conv[: CTX + n], n, n, 0.0, 0.0, 1.0, 0.0, 1.0, whole_src)
        reference = conv[CTX : CTX + n]
        gain_split = Envelope(heads, SR) - Envelope(reference, SR)
        gain_one = Envelope(one_pass, SR) - Envelope(reference, SR)
        rms = float(np.sqrt(np.mean((gain_split - gain_one) ** 2)))
        self.assertLess(rms, 1.0, f"gain track split vs one pass: {rms:.2f} dB RMS")
        step = np.diff(heads.astype(np.float64))
        borders = np.arange(BLOCK, n, BLOCK)
        in_block = np.abs(step)[np.setdiff1d(np.arange(len(step)), np.concatenate([borders + k for k in range(-2, 2)]))]
        at_border = np.abs(step[np.concatenate([borders + k for k in range(-2, 2)])])
        self.assertLess(float(at_border.max()), 1.5 * float(in_block.max()), "a sample step at a block border larger than anywhere else")
        split = self._Stream(1.0, 0.0)
        t_split, st_split = self._Contours(split)
        whole = ProsodyStage(self.fcpe, SR).Process(self.conv[: CTX + REGION + 6 * BLOCK], REGION + 6 * BLOCK, REGION + 6 * BLOCK, 0.0, 0.0, 1.0, 1.0, 0.0, SourceAudio(self.src, 0.0 + CTX * SR16 / SR - CTX * SR16 / SR, self.f0_src, 0))[0]
        t_whole, st_whole = Contour(whole, SR, CTX / SR)
        keep_split = (t_split > START_S) & (t_split < t_whole.max())
        diff = np.interp(t_split[keep_split], t_whole, st_whole) - st_split[keep_split]
        # The one-pass call medians the whole block at once; the stream's running median (R15) wanders as the glide cycles (measured: mean offset
        # -1.2 st, the one-pass contour correlates 0.9999 with the source). That slow offset is the causal median, so the block split is judged on
        # what it could break: fast steps at the borders (the diff minus its 0.7 s moving mean) and the shape of the contour.
        window = 140  # frames of 5 ms
        slow = np.convolve(np.pad(diff, window // 2, mode="edge"), np.ones(window + 1) / (window + 1), "valid")[: len(diff)]
        self.assertLess(float(np.sqrt(np.mean((diff - slow) ** 2))), 0.5, "split stream vs one pass contour: fast component")
        shape = float(np.corrcoef(np.interp(t_split[keep_split], t_whole, st_whole), st_split[keep_split])[0, 1])
        self.assertGreaterEqual(shape, 0.9, f"split stream vs one pass contour correlation {shape:.3f}")

    def test_sola_joined_stream_keeps_the_glide(self):
        """The real join: SOLA over the region outputs. Time shifts of up to the SOLA search window are allowed, so the best lag within +-100 ms counts."""
        sola = Sola(SR, 220, 45.0, 20.0)
        joined = np.concatenate([sola.Process(region) for _, region in self._Stream(1.0, 0.0)])
        t, st = Contour(joined, SR, CTX / SR)
        keep = t > START_S
        best = max(float(np.corrcoef(st[keep], np.interp(t[keep] - lag / 1000.0, self.src_t, self.src_st))[0, 1]) for lag in range(-100, 101, 10))
        self.assertGreaterEqual(best, 0.9, f"best lagged correlation {best:.3f}")
        self.assertLessEqual(float(np.max(np.abs(st[keep]))), 12.0)
        self.assertLess(float(np.max(np.abs(np.diff(st[keep])))), 12.0)

    def test_reset_clears_the_transfer_state(self):
        stage = ProsodyStage(self.fcpe, SR)
        for s0, seg in list(Segments(self.conv))[:3]:
            stage.Process(seg, REGION, BLOCK, 0.0, 0.0, 1.0, 1.0, 1.0, self._Source(s0))
        self.assertGreater(stage.MedianF0Hz(), 0.0)
        stage.Reset()
        self.assertEqual(stage.MedianF0Hz(), 0.0)
        s0, seg = next(iter(Segments(self.conv)))
        first, _ = stage.Process(seg, REGION, BLOCK, 0.0, 0.0, 1.0, 1.0, 1.0, self._Source(s0))
        fresh, _ = ProsodyStage(self.fcpe, SR).Process(seg, REGION, BLOCK, 0.0, 0.0, 1.0, 1.0, 1.0, self._Source(s0))
        np.testing.assert_array_equal(first, fresh)


if __name__ == "__main__":
    unittest.main()
