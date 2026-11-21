import math
import os
import sys
import unittest

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import seedvc_params as P
from audio.vad import FRAME_MS, SileroVad

FRAME = P.VAD_FRAME_SAMPLES


def LoadSpeech16k(name="ref_1.wav"):
    x, sr = sf.read(os.path.join(ROOT, "testdata", name), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    g = math.gcd(sr, P.INPUT_SR)
    return resample_poly(x, P.INPUT_SR // g, sr // g).astype(np.float32)


def Frames(x):
    n = x.size // FRAME
    return x[: n * FRAME].reshape(n, FRAME)


def RunFrames(vad, x):
    probs, active, gated = [], [], []
    for frame in Frames(x):
        g, p, a = vad.Gate(frame)
        probs.append(p)
        active.append(a)
        gated.append(g)
    return np.array(probs), np.array(active), gated


class ScriptedSession:
    """Stands in for the onnxruntime session: returns scripted probabilities and records what the model was fed."""

    def __init__(self, probs):
        self.probs = list(probs)
        self.inputs = []

    def run(self, _outputs, feed):
        self.inputs.append({k: np.array(v, copy=True) for k, v in feed.items()})
        return [np.array([[self.probs.pop(0)]], np.float32), feed["state"] + 1.0]


def Scripted(probs, **kwargs):
    vad = SileroVad(**kwargs)
    vad._session = ScriptedSession(probs)
    return vad


def ActiveFlags(vad, n):
    frame = np.full(FRAME, 0.1, np.float32)
    return [vad.Process(frame)[1] for _ in range(n)]


class HangLogicTests(unittest.TestCase):
    def test_constants_are_the_spec_values(self):
        self.assertEqual((P.VAD_FRAME_SAMPLES, P.INPUT_SR, P.VAD_HANG_MS, P.VAD_THRESHOLD), (512, 16000, 120.0, 0.5))
        self.assertEqual(FRAME_MS, 32.0)

    def test_hang_is_ceil_of_120ms_in_32ms_frames(self):
        self.assertEqual(SileroVad().hang_frames, 4)
        self.assertEqual(SileroVad(hang_ms=32.0).hang_frames, 1)
        self.assertEqual(SileroVad(hang_ms=33.0).hang_frames, 2)
        self.assertEqual(SileroVad(hang_ms=0.0).hang_frames, 0)

    def test_one_speech_frame_stays_active_for_four_hang_frames(self):
        flags = ActiveFlags(Scripted([0.9] + [0.1] * 8), 9)
        self.assertEqual(flags, [True] * 5 + [False] * 4)

    def test_new_speech_restarts_the_hang(self):
        flags = ActiveFlags(Scripted([0.9, 0.1, 0.1, 0.9] + [0.1] * 6), 10)
        self.assertEqual(flags, [True] * 8 + [False] * 2)

    def test_threshold_is_inclusive(self):
        flags = ActiveFlags(Scripted([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4999]), 7)
        self.assertEqual(flags, [True] * 5 + [False] * 2)

    def test_zero_hang_ends_on_the_first_quiet_frame(self):
        flags = ActiveFlags(Scripted([0.9, 0.1, 0.1], hang_ms=0.0), 3)
        self.assertEqual(flags, [True, False, False])

    def test_model_is_fed_64_context_samples_plus_the_frame_and_carries_state(self):
        vad = Scripted([0.1, 0.1, 0.1])
        f1, f2, f3 = (np.random.default_rng(s).standard_normal(FRAME).astype(np.float32) for s in (1, 2, 3))
        for f in (f1, f2, f3):
            vad.Process(f)
        a, b, c = vad._session.inputs
        self.assertEqual(a["input"].shape, (1, 576))
        self.assertEqual(a["state"].shape, (2, 1, 128))
        self.assertEqual((a["sr"].shape, a["sr"].dtype, int(a["sr"])), ((), np.dtype(np.int64), 16000))
        self.assertTrue(np.all(a["input"][0, :64] == 0.0))
        self.assertTrue(np.array_equal(a["input"][0, 64:], f1))
        self.assertTrue(np.array_equal(b["input"][0, :64], f1[-64:]))
        self.assertTrue(np.array_equal(c["input"][0, :64], f2[-64:]))
        self.assertEqual([float(s["state"][0, 0, 0]) for s in (a, b, c)], [0.0, 1.0, 2.0])

    def test_gate_returns_the_frame_when_active_and_exact_zeros_when_not(self):
        vad = Scripted([0.9, 0.1, 0.1, 0.1, 0.1, 0.1], hang_ms=0.0)
        frame = np.random.default_rng(5).standard_normal(FRAME).astype(np.float32)
        g, p, a = vad.Gate(frame)
        self.assertTrue(a and np.array_equal(g, frame))
        g, p, a = vad.Gate(frame)
        self.assertFalse(a)
        self.assertTrue(np.all(g == 0.0))
        self.assertEqual((g.shape, g.dtype), ((FRAME,), np.dtype(np.float32)))

    def test_wrong_frame_length_is_rejected(self):
        vad = SileroVad()
        for n in (0, 256, 511, 513, 1024):
            with self.assertRaises(ValueError):
                vad.Process(np.zeros(n, np.float32))

    def test_reset_clears_hang_state_context_and_model_state(self):
        script = [0.9, 0.1, 0.9, 0.1]
        kept = Scripted(script)
        kept.Process(np.ones(FRAME, np.float32))
        self.assertEqual(ActiveFlags(kept, 3), [True, True, True])  # hang tail of frame 1, speech, hang tail
        vad = Scripted(script)
        vad.Process(np.ones(FRAME, np.float32))
        vad.Reset()
        self.assertEqual((vad.last_prob, vad.last_active), (0.0, False))
        self.assertEqual(ActiveFlags(vad, 3), [False, True, True])  # the first quiet frame after Reset has no hang tail
        first, after_reset = vad._session.inputs[0], vad._session.inputs[1]
        self.assertTrue(np.all(after_reset["input"][0, :64] == 0.0))
        self.assertEqual(float(first["state"][0, 0, 0]), 0.0)
        self.assertEqual(float(after_reset["state"][0, 0, 0]), 0.0)


class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.speech = LoadSpeech16k("ref_1.wav")

    def test_uses_the_silero_onnx_file_through_onnxruntime(self):
        vad = SileroVad()
        self.assertTrue(vad.onnx.endswith(os.path.join("silero_vad", "data", "silero_vad.onnx")))
        self.assertTrue(os.path.isfile(vad.onnx))
        self.assertEqual(type(vad._session).__module__.split(".")[0], "onnxruntime")

    def test_real_speech_is_detected_and_leading_silence_is_not(self):
        vad = SileroVad()
        probs, active, gated = RunFrames(vad, self.speech)
        frames = Frames(self.speech)
        rms_db = 20.0 * np.log10(np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)) + 1e-9)
        loud = rms_db >= -40.0
        self.assertGreater(int(loud.sum()), frames.shape[0] // 2)  # a speech clip: most frames are clearly above the noise floor
        self.assertGreaterEqual(float(active[loud].mean()), 0.97)
        self.assertFalse(active[0] or active[1])
        self.assertTrue(all(np.all(g == 0.0) for g, a in zip(gated, active) if not a))
        self.assertTrue(all(np.array_equal(g, f) for g, f, a in zip(gated, frames, active) if a))

    def test_digital_silence_is_inactive_with_exact_zero_output(self):
        probs, active, gated = RunFrames(SileroVad(), np.zeros(3 * P.INPUT_SR, np.float32))
        self.assertFalse(active.any())
        self.assertLess(float(probs.max()), 0.1)
        self.assertTrue(all(np.all(g == 0.0) for g in gated))

    def test_quiet_room_noise_is_inactive(self):
        noise = (np.random.default_rng(2024).standard_normal(5 * P.INPUT_SR) * 0.002).astype(np.float32)
        probs, active, _ = RunFrames(SileroVad(), noise)
        self.assertFalse(active.any(), f"max prob {probs.max():.3f}")

    def test_silence_after_speech_goes_inactive_after_a_hang_of_four_frames(self):
        cut = 120 * FRAME
        x = np.concatenate([self.speech[:cut], np.zeros(40 * FRAME, np.float32)])
        probs, active, _ = RunFrames(SileroVad(), x)
        last_speech = int(np.max(np.nonzero(probs >= P.VAD_THRESHOLD)[0]))
        first_off = int(np.min(np.nonzero(~active[last_speech + 1:])[0])) + last_speech + 1
        hang_frames = first_off - last_speech - 1
        self.assertEqual(hang_frames, 4)
        self.assertLessEqual(abs(hang_frames * FRAME_MS - P.VAD_HANG_MS), FRAME_MS)
        self.assertFalse(active[first_off:].any())
        self.assertFalse(active[-20:].any())


if __name__ == "__main__":
    unittest.main()
