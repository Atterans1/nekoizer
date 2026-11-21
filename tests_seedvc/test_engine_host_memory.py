"""EngineHost memory accounting against a scripted device: an engine's footprint is the allocation it caused, not the process total; the
idle engine is released only when the next load does not fit; the next load is planned with the measured figure."""
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import seedvc_params as P  # noqa: E402
from engine import engine_host  # noqa: E402
from engine.engine_host import EngineHost  # noqa: E402
from engine.voice_engine import KIND_CLIP, KIND_RVC, ConvertResult, VoiceEngine  # noqa: E402

TOTAL_MB = 4096.0


class Gpu:
    """The device as the host reads it: MB this process has allocated, and MB left once other programs have taken theirs."""

    def __init__(self, others_mb: float):
        self.allocated = 0.0
        self.others = others_mb

    def Free(self) -> float:
        return TOTAL_MB - self.others - self.allocated


class ScriptedEngine(VoiceEngine):
    """Allocates load_mb in Load(), voice_mb in LoadVoice() and first_mb in the first active block; Release() gives all of it back."""

    def __init__(self, gpu: Gpu, kind: str, load_mb: float, first_mb: float, voice_mb: float = 0.0, estimate: float = 0.0):
        self.kind = kind
        self.vram_estimate_mb = estimate
        self._gpu, self._load, self._first, self._voice_mb = gpu, load_mb, first_mb, voice_mb
        self._held = 0.0
        self._loaded = False
        self._warm = False
        self._voice: dict | None = None

    def _Take(self, mb: float) -> None:
        self._gpu.allocated += mb
        self._held += mb

    def Load(self) -> dict:
        self._Take(self._load)
        self._loaded, self._warm = True, False
        return {}

    def LoadVoice(self, voice: dict) -> dict:
        self._Take(self._voice_mb)
        self._voice = {"name": voice["name"]}
        return dict(self._voice)

    def Voice(self):
        return self._voice

    def UnloadVoice(self) -> None:
        self._voice = None

    def Loaded(self) -> bool:
        return self._loaded and (self.kind == KIND_CLIP or self._voice is not None)

    def Release(self) -> None:
        self._gpu.allocated -= self._held
        self._held = 0.0
        self._loaded, self._voice = False, None

    def SetSlot(self, idx, wav, sr) -> dict:
        return {"idx": idx}

    def ClearSlot(self, idx) -> None:
        pass

    def SlotInfo(self) -> list:
        return []

    def ConvertBlock(self, block16k, active, steps=None):
        if active and not self._warm:
            self._Take(self._first)
            self._warm = True
        return ConvertResult(np.zeros(1, np.float32), 0.0, 0, [0.0] * 4, 0.0, 0.0, -120.0, False)

    def ResetStream(self) -> None:
        pass

    def Telemetry(self) -> dict:
        return {}


class FootprintTests(unittest.TestCase):
    def setUp(self):
        self.gpu = Gpu(others_mb=2000.0)
        self.clip = ScriptedEngine(self.gpu, KIND_CLIP, load_mb=600.0, first_mb=100.0, estimate=1500.0)
        self.rvc = ScriptedEngine(self.gpu, KIND_RVC, load_mb=300.0, first_mb=20.0, voice_mb=100.0, estimate=700.0)
        for patcher in (mock.patch.object(EngineHost, "_FreeMb", lambda host: self.gpu.Free()),
                        mock.patch.object(engine_host, "AllocatedMb", lambda device: self.gpu.allocated),
                        mock.patch("engine.rvc_store.Get", return_value={"name": "voice"})):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.params = P.ParamStore()
        self.params.Set("emotion_mode", "manual")
        self.host = EngineHost(self.params, P.Janitor(), self.clip, self.rvc, device="cuda")
        self.host.SetClip(0, np.zeros(22050, np.float32), 22050)
        self.host.SetRvc(1, "voice")
        self.block = np.zeros(3520, np.float32)

    def Convert(self, slot: int, active: bool = True) -> dict:
        self.params.Set("manual_slot", slot)
        self.host.ConvertBlock(self.block, active)
        return self.host.Telemetry()

    def test_footprints_are_per_engine_and_the_next_load_uses_them(self):
        tel = self.Convert(1)
        self.assertEqual(tel["vram_mb"], {KIND_CLIP: 0.0, KIND_RVC: 420.0})

        tel = self.Convert(0)  # the 1500 MB estimate does not fit beside the loaded voice: the RVC engine is released first
        self.assertEqual(tel["last_switch"]["released"], [KIND_RVC])
        self.assertEqual(tel["resident"], {KIND_CLIP: True, KIND_RVC: False})
        self.assertEqual(tel["vram_mb"], {KIND_CLIP: 700.0, KIND_RVC: 0.0})

        tel = self.Convert(1)  # 4096 - 2000 - 700 leaves 1396 MB; the measured 420 MB plus the margin fits, so both stay resident
        self.assertEqual(tel["last_switch"]["released"], [])
        self.assertEqual(tel["resident"], {KIND_CLIP: True, KIND_RVC: True})
        self.assertEqual(tel["vram_mb"], {KIND_CLIP: 700.0, KIND_RVC: 420.0})
        self.assertEqual(self.gpu.allocated, 1120.0)

    def test_an_inactive_block_does_not_close_the_measurement(self):
        tel = self.Convert(0, active=False)
        self.assertEqual(tel["vram_mb"][KIND_CLIP], 600.0)
        self.assertEqual(self.host._estimate[KIND_CLIP], 1500.0)
        tel = self.Convert(0)
        self.assertEqual(tel["vram_mb"][KIND_CLIP], 700.0)
        self.assertEqual(self.host._estimate[KIND_CLIP], 700.0)

    def test_a_load_that_cannot_fit_even_alone_is_refused_and_named(self):
        self.gpu.others = 3500.0
        with self.assertLogs("seedvc.engine_host", level="ERROR"):
            tel = self.Convert(0)
        self.assertIn("not enough GPU memory for the clip engine", tel["error"])
        self.assertEqual(tel["vram_mb"], {KIND_CLIP: 0.0, KIND_RVC: 0.0})


if __name__ == "__main__":
    unittest.main()
