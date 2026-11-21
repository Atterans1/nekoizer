"""One slot bank with a reference clip and an RVC voice side by side: routing, engine per block, slot kinds, emotion isolation."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import seedvc_params as P  # noqa: E402
from engine.engine_host import EngineHost  # noqa: E402
from engine.rvc_engine import RvcEngine  # noqa: E402
from engine.seedvc_expressive import SeedVCExpressive  # noqa: E402
from pipeline import Pipeline  # noqa: E402
from tests_seedvc import rvc_fixtures  # noqa: E402
from tests_seedvc.test_engine_offline import FakeFcpe, FakeModel  # noqa: E402
from tests_seedvc.test_rvc_engine import FakeBackbone  # noqa: E402


def Noise(n, seed):
    return (0.1 * np.random.RandomState(seed).randn(n)).astype(np.float32)


class DualKindBankTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = mock.patch.dict(os.environ, {"S2S_DATA_DIR": self.tmp.name})
        self.env.start()
        self.addCleanup(self.env.stop)
        from engine import rvc_store
        model = rvc_fixtures.SaveCheckpoint(Path(self.tmp.name) / "tiny.pth", "v2", True, 40000)
        self.model_name = rvc_store.ImportPth(str(model), name="tiny")["name"]
        self.parts = FakeModel()
        clip = SeedVCExpressive(P.ParamStore(), P.Janitor(), ".", device="cpu")
        clip.InstallComponents(self.parts.Components(), fcpe=FakeFcpe())
        self.rvc = RvcEngine(P.ParamStore(), P.Janitor(), device="cpu", backbone=FakeBackbone())
        self.pipe = Pipeline(".", device="cpu", engines=(clip, self.rvc))
        self.pipe._stats["load"] = {}
        self.clip = clip
        self.audio = Noise(3520 * 3, 9)

    def test_bank_holds_both_kinds_and_routes_each_block_to_its_engine(self):
        self.pipe.SetSlotFromWav(0, Noise(22050 * 2, 1), 22050)
        row = self.pipe.SetSlotFromRvc(1, self.model_name)
        self.assertEqual((row["kind"], row["model"], row["version"], row["sr"]), ("rvc", self.model_name, "v2", 40000))
        kinds = [r["kind"] for r in self.pipe.SlotInfo()]
        self.assertEqual(kinds, ["clip", "rvc"] + [None] * 8)
        self.assertEqual(self.pipe.SlotInfo()[0]["seconds"], 2.0)

        self.pipe.SetManualSlot(0)
        _, stats = self.pipe.ProcessOffline(self.audio)
        self.assertEqual(stats["kinds"], ["clip"] * 3)
        clip_calls = len(self.parts.cfm_calls)
        self.assertGreater(clip_calls, 0)

        self.pipe.SetManualSlot(1)
        _, stats = self.pipe.ProcessOffline(self.audio)
        self.assertEqual(stats["kinds"], ["rvc"] * 3)
        self.assertEqual(len(self.parts.cfm_calls), clip_calls)  # no Seed-VC prompt path during RVC blocks

        self.pipe.SetManualSlot(0)
        _, stats = self.pipe.ProcessOffline(self.audio)
        self.assertEqual(stats["kinds"], ["clip"] * 3)
        self.assertEqual(self.pipe.Telemetry()["engine"]["kind"], "clip")
        self.assertGreaterEqual(self.pipe.Telemetry()["engine"]["switches"], 3)

    def test_auto_mode_never_routes_an_rvc_slot_into_the_seedvc_prompt_path(self):
        self.pipe.SetSlotFromWav(0, Noise(22050 * 2, 1), 22050)
        self.pipe.SetSlotFromRvc(1, self.model_name)
        self.pipe.SetParam("emotion_mode", "auto")
        self.pipe.SetParam("auto_balancer", False)
        _, stats = self.pipe.ProcessOffline(self.audio)
        self.assertEqual(stats["kinds"], ["clip"] * 3)
        self.assertEqual([r["loaded"] for r in self.clip.SlotInfo()], [True] + [False] * 9)
        self.pipe.ClearSlot(0)
        before = len(self.parts.cfm_calls)
        _, stats = self.pipe.ProcessOffline(self.audio)
        self.assertEqual(stats["kinds"], ["rvc"] * 3)
        self.assertEqual(len(self.parts.cfm_calls), before)
        self.assertEqual(self.pipe.Telemetry()["engine"]["unused_controls"], list(self.rvc.unused_controls))

    def test_deleting_a_voice_clears_its_slots_and_the_file(self):
        self.pipe.SetSlotFromRvc(2, self.model_name)
        self.assertEqual(self.pipe.DeleteRvcModel(self.model_name), [2])
        self.assertFalse(self.pipe.SlotInfo()[2]["loaded"])
        from engine import rvc_store
        with self.assertRaises(ValueError):
            rvc_store.Get(self.model_name)

    def test_switch_latency_is_recorded(self):
        self.pipe.SetSlotFromWav(0, Noise(22050 * 2, 1), 22050)
        self.pipe.SetSlotFromRvc(1, self.model_name)
        self.pipe.SetManualSlot(1)
        self.pipe.ProcessOffline(self.audio)
        last = self.pipe.Telemetry()["engine"]["last_switch"]
        self.assertEqual((last["from"], last["to"]), ("clip", "rvc"))
        self.assertGreaterEqual(last["ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
