"""RvcEngine on random-weight RVC voices (built from the vendored synthesizer) with a stand-in backbone."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import seedvc_params as P  # noqa: E402
from engine import rvc_checkpoint  # noqa: E402
from engine.rvc_engine import PlanRvc, RvcEngine  # noqa: E402
from engine.voice_engine import MakeGrid  # noqa: E402
from tests_seedvc import rvc_fixtures  # noqa: E402


class FakeBackbone:
    """Shapes of the real HuBERT (50 Hz, 256 or 768 wide) and RMVPE (10 ms, Hz) outputs, deterministic values."""

    pitch_name = "fake"

    def Features(self, wave, version):
        dim = 768 if version == "v2" else 256
        generator = torch.Generator().manual_seed(1)
        return torch.randn(1, wave.shape[1] // 320, dim, generator=generator)

    def Pitch(self, wave16):
        return np.full(len(wave16) // 160 + 1, 220.0)


class RvcEngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.params = P.ParamStore()
        self.janitor = P.Janitor()
        self.engine = RvcEngine(self.params, self.janitor, device="cpu", backbone=FakeBackbone())

    def Voice(self, version="v2", f0=True, sr=40000):
        path = rvc_fixtures.SaveCheckpoint(Path(self.tmp.name) / f"{version}{int(f0)}{sr}.pth", version, f0, sr)
        return {"name": "voice", "model_path": str(path), "has_index": False}

    def test_plan_geometry_at_spec_defaults(self):
        plan = PlanRvc(220, 45, 2.0)
        self.assertEqual(plan.win16, 37120)  # 36880 samples rounded up to whole 20 ms content frames
        self.assertEqual(plan.p_len, 232)
        self.assertEqual(plan.ret_frames, 81)  # ceil((4560 block+overlap+search + 320 lookahead + 8000 prosody context) / 160) = ceil(80.5)
        self.assertEqual(plan.skip, 151)
        self.assertEqual(plan.ret_start16, 37120 - 81 * 160)
        self.assertEqual(plan.region_start22, 11135)  # (81 * 160 - 320 - 4560) = 8080 samples at 16 kHz = 11135.25 at 22.05 kHz
        # the region ends right20 ms before the window end; the returned audio covers it with the prosody context before it
        self.assertLessEqual(plan.region_start22 + 6284, int(round(81 * 160 * 22050 / 16000)))
        self.assertGreaterEqual(plan.region_start22, 11025)

    def test_plan_covers_region_context_and_lookahead_over_the_parameter_ranges(self):
        for block_ms in (100, 220, 500):
            for overlap_ms in (40.0, 45.0, 50.0):
                for left_s in (1.5, 2.0, 2.5):
                    with self.subTest(block_ms=block_ms, overlap_ms=overlap_ms, left_s=left_s):
                        grid = MakeGrid(block_ms, overlap_ms)
                        plan = PlanRvc(block_ms, overlap_ms, left_s)
                        returned22 = -(-plan.ret_frames * 160 * 22050 // 16000)  # resample_poly output length of ret_frames 10 ms frames
                        self.assertGreaterEqual(plan.region_start22, grid.prosody_ctx22)
                        self.assertLessEqual(plan.region_start22 + grid.out22, returned22)
                        self.assertEqual(plan.win16 % 320, 0)
                        self.assertEqual(plan.skip + plan.ret_frames, plan.p_len)

    def test_v2_and_v1_voices_convert_to_exact_finite_blocks(self):
        for version, f0 in (("v2", True), ("v1", True), ("v2", False), ("v1", False)):
            with self.subTest(version=version, f0=f0):
                engine = RvcEngine(self.params, P.Janitor(), device="cpu", backbone=FakeBackbone())
                engine.LoadVoice(self.Voice(version, f0))
                block = (0.1 * np.random.RandomState(3).randn(3520)).astype(np.float32)
                results = [engine.ConvertBlock(block, True) for _ in range(3)]
                for res in results:
                    self.assertEqual(res.wav22k.shape, (6284,))
                    self.assertEqual(res.wav22k.dtype, np.float32)
                    self.assertTrue(np.isfinite(res.wav22k).all())
                    self.assertLessEqual(float(np.abs(res.wav22k).max()), 1.0)
                    self.assertFalse(res.stage_applied)

    def test_inactive_block_is_silence_and_keeps_the_stream(self):
        self.engine.LoadVoice(self.Voice())
        res = self.engine.ConvertBlock(np.zeros(3520, np.float32), False)
        self.assertEqual((res.wav22k.shape, float(np.abs(res.wav22k).max())), ((6284,), 0.0))

    def test_wrong_block_size_and_no_voice_are_refused(self):
        with self.assertRaisesRegex(ValueError, "3520"):
            self.engine.ConvertBlock(np.zeros(100, np.float32), True)
        with self.assertRaisesRegex(RuntimeError, "no RVC voice"):
            self.engine.ConvertBlock(np.zeros(3520, np.float32), True)

    def test_hostile_voice_is_refused_before_anything_is_built(self):
        marker = Path(self.tmp.name) / "ran.txt"

        class Hostile:
            def __reduce__(self):
                return (os.system, (f'echo x > "{marker}"',))

        bad = Path(self.tmp.name) / "bad.pth"
        torch.save({"config": rvc_fixtures.TinyConfig(40000), "weight": Hostile()}, bad)
        with self.assertRaises(rvc_checkpoint.UnsafeCheckpointError):
            self.engine.LoadVoice({"name": "bad", "model_path": str(bad)})
        self.assertFalse(marker.exists())
        self.assertFalse(self.engine.Loaded())

    def test_block_continuity_of_the_ring(self):
        """Consecutive blocks slide the window by exactly one block: the second call sees the first call's newest samples."""
        self.engine.LoadVoice(self.Voice("v2", False))
        blocks = [np.full(3520, float(k + 1) * 0.01, np.float32) for k in range(3)]
        for block in blocks:
            self.engine.ConvertBlock(block, True)
        ring = self.engine._ring
        self.assertTrue(np.all(ring[-3520:] == np.float32(0.03)))
        self.assertTrue(np.all(ring[-7040:-3520] == np.float32(0.02)))
        self.assertTrue(np.all(ring[-10560:-7040] == np.float32(0.01)))


if __name__ == "__main__":
    unittest.main()
