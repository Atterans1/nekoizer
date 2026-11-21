"""SeedVCExpressive without the real models: stand-in Seed-VC modules record what the engine passes (contract, ring, trim, live parameters)."""
import math
import types
import unittest
from pathlib import Path

import numpy as np
import torch

import engine.seedvc_expressive as module
from engine.seedvc_expressive import Components, PlanBlock, SeedVCExpressive
from seedvc_params import Janitor, ParamStore


class FakeFcpe:
    device = "fake"

    def __init__(self):
        self.calls = 0
        self.sizes = []

    def Infer(self, audio, sr):
        self.calls += 1
        self.sizes.append(len(audio))
        return np.full(int(len(audio) / sr * 100) + 1, 160.0)


class FakeModel:
    """Tiny stand-ins for the Seed-VC modules; every call is recorded so the tests can check what the engine passed."""

    def __init__(self, vocoder_kind="ramp"):
        self.semantic_inputs, self.lr_calls, self.cfm_calls, self.campplus_calls = [], [], [], 0
        self.vocoder_kind = vocoder_kind
        self.model = types.SimpleNamespace(cfm=types.SimpleNamespace(inference=self.Cfm), length_regulator=self.Lr)

    def Semantic(self, wave):
        self.semantic_inputs.append(np.array(wave, copy=True))
        return torch.zeros(1, (len(wave) - 400) // 320 + 1, 1024)

    def Lr(self, x, ylens=None, n_quantizers=None, f0=None):
        self.lr_calls.append((x.shape[1], int(ylens.max())))
        return (torch.zeros(1, int(ylens.max()), 384),)

    def Cfm(self, cat, x_lens, mel, style, f0, n_timesteps, inference_cfg_rate):
        self.cfm_calls.append({"steps": n_timesteps, "cfg": inference_cfg_rate, "style": float(style[0, 0]), "cat": cat.size(1)})
        return torch.zeros(1, 80, cat.size(1))

    def Vocoder(self, target):
        n = target.size(2) * 256
        t = torch.arange(n, dtype=torch.float32)
        wave = t if self.vocoder_kind == "ramp" else 0.3 * torch.sin(2 * math.pi * 160.0 * t / 22050.0)
        return wave.unsqueeze(0)

    def Campplus(self, feat):
        self.campplus_calls += 1
        return torch.full((1, 192), float(self.campplus_calls))

    def ToMel(self, wave):
        return torch.zeros(1, 80, wave.size(1) // 256)

    def Components(self):
        return Components(self.model, self.Semantic, self.Vocoder, self.Campplus, self.ToMel, torch.device("cpu"), 256)


def New(kind="ramp", follow=0.0):
    """An engine on stand-in models. The prosody transfer is off unless asked for: most tests check the raw vocoder region."""
    parts, params, janitor = FakeModel(kind), ParamStore(), Janitor()
    params.Set("prosody_follow", follow)
    params.Set("energy_follow", follow)
    engine = SeedVCExpressive(params, janitor, ".", device="cpu")
    engine.InstallComponents(parts.Components(), fcpe=FakeFcpe())
    return engine, parts, params, janitor


def Noise(n, seed):
    return (0.1 * np.random.RandomState(seed).randn(n)).astype(np.float32)


def Ref(engine, idx, seconds, seed):
    return engine.SetSlot(idx, Noise(int(22050 * seconds), seed), 22050)


class PlanBlockTest(unittest.TestCase):
    def test_spec_defaults(self):
        p = PlanBlock(220, 45, 2.0, 256)
        self.assertEqual((p.block16, p.ring16, p.ce_cut_frames), (3520, 36880, 75))
        self.assertEqual((p.block22, p.overlap22, p.search22, p.right22), (4851, 992, 441, 441))
        self.assertEqual((p.out22, p.prosody_ctx22, p.mel_frames), (6284, 11025, 69))
        self.assertAlmostEqual(p.dit_context_s, 0.5, places=9)

    def test_other_settings(self):
        p = PlanBlock(100, 40, 1.5, 256)
        self.assertEqual((p.block16, p.ring16, p.ce_cut_frames, p.out22, p.mel_frames), (1600, 26880, 50, 3528, 58))
        p = PlanBlock(500, 40, 2.5, 256)
        self.assertEqual((p.block16, p.ring16, p.ce_cut_frames, p.out22, p.mel_frames), (8000, 49280, 100, 12348, 93))

    def test_left_context_off_the_20ms_grid_widens_the_dit_context(self):
        p = PlanBlock(220, 45, 1.75, 256)
        self.assertEqual(p.ce_cut_frames, 62)
        self.assertAlmostEqual(p.dit_context_s, 0.51, places=9)


class ConvertBlockTest(unittest.TestCase):
    def test_ring_layout_and_trim_follow_the_gui_scheme(self):
        engine, parts, _, _ = New()
        Ref(engine, 0, 2.0, 1)
        engine.ConvertBlock(np.full(3520, 0.1, np.float32), True)
        res = engine.ConvertBlock(np.full(3520, 0.2, np.float32), True)
        ring = parts.semantic_inputs[-1]
        self.assertEqual(len(ring), 36880)
        self.assertTrue(np.all(ring[-3520:] == np.float32(0.2)))
        self.assertTrue(np.all(ring[-7040:-3520] == np.float32(0.1)))
        self.assertTrue(np.all(ring[:-7040] == 0.0))
        self.assertEqual(parts.lr_calls[-1], (40, 69))
        self.assertEqual((res.wav22k.shape, res.wav22k.dtype), ((6284,), np.float32))
        np.testing.assert_array_equal(res.wav22k, np.arange(10939, 17223, dtype=np.float32))
        self.assertFalse(res.stage_applied)

    def test_parameters_are_read_live_and_steps_argument_wins(self):
        engine, parts, params, _ = New()
        Ref(engine, 0, 2.0, 1)
        block = Noise(3520, 2)
        params.Set("diffusion_steps", 7)
        params.Set("cfg_rate", 0.3)
        r1 = engine.ConvertBlock(block, True)
        self.assertEqual((parts.cfm_calls[-1]["steps"], parts.cfm_calls[-1]["cfg"], r1.steps_used), (7, 0.3, 7))
        r2 = engine.ConvertBlock(block, True, steps=6)
        self.assertEqual((parts.cfm_calls[-1]["steps"], r2.steps_used), (6, 6))
        params.Set("left_context_s", 1.5)
        engine.ConvertBlock(block, True)
        self.assertEqual(len(parts.semantic_inputs[-1]), 28880)
        self.assertEqual(parts.lr_calls[-1], (40, 69))
        params.Set("block_ms", 100)
        with self.assertRaisesRegex(ValueError, "1600"):
            engine.ConvertBlock(block, True)
        res = engine.ConvertBlock(Noise(1600, 3), True)
        self.assertEqual(res.wav22k.shape, (3638,))
        self.assertEqual(parts.lr_calls[-1], (parts.lr_calls[-1][0], 59))

    def test_inactive_block_advances_the_ring_without_inference(self):
        engine, parts, _, _ = New()
        Ref(engine, 0, 2.0, 1)
        engine.ConvertBlock(np.full(3520, 0.2, np.float32), True)
        calls = len(parts.cfm_calls)
        res = engine.ConvertBlock(np.zeros(3520, np.float32), False)
        self.assertEqual(len(parts.cfm_calls), calls)
        self.assertEqual(res.wav22k.shape, (6284,))
        self.assertEqual((float(np.abs(res.wav22k).max()), res.inference_s, res.steps_used, res.stage_applied), (0.0, 0.0, 0, False))
        engine.ConvertBlock(np.full(3520, 0.3, np.float32), True)
        ring = parts.semantic_inputs[-1]
        self.assertTrue(np.all(ring[-3520:] == np.float32(0.3)))
        self.assertTrue(np.all(ring[-7040:-3520] == 0.0))
        self.assertTrue(np.all(ring[-10560:-7040] == np.float32(0.2)))

    def test_reset_stream_clears_the_ring(self):
        engine, parts, _, _ = New()
        Ref(engine, 0, 2.0, 1)
        engine.ConvertBlock(np.full(3520, 0.2, np.float32), True)
        engine.ResetStream()
        engine.ConvertBlock(np.full(3520, 0.3, np.float32), True)
        ring = parts.semantic_inputs[-1]
        self.assertTrue(np.all(ring[-3520:] == np.float32(0.3)))
        self.assertTrue(np.all(ring[:-3520] == 0.0))

    def test_errors(self):
        engine, parts, _, janitor = New()
        with self.assertRaisesRegex(RuntimeError, "no reference slot"):
            engine.ConvertBlock(np.zeros(3520, np.float32), True)
        Ref(engine, 0, 2.0, 1)
        with self.assertRaisesRegex(ValueError, "3520"):
            engine.ConvertBlock(np.zeros(3000, np.float32), True)
        bare = SeedVCExpressive(ParamStore(), Janitor(), ".", device="cpu")
        with self.assertRaisesRegex(RuntimeError, "not loaded"):
            bare.ConvertBlock(np.zeros(3520, np.float32), True)
        janitor.Cleanup()
        with self.assertRaisesRegex(RuntimeError, "not loaded"):
            engine.ConvertBlock(np.zeros(3520, np.float32), True)
        self.assertEqual([i["loaded"] for i in engine.SlotInfo()], [False] * 10)


class ConditioningTest(unittest.TestCase):
    def test_manual_slot_switches_prompt_latents_and_style(self):
        engine, parts, params, _ = New()
        Ref(engine, 0, 2.0, 1)  # style 1.0, 172 prompt frames
        Ref(engine, 3, 3.0, 2)  # style 2.0, 258 prompt frames
        self.assertEqual([i["loaded"] for i in engine.SlotInfo()], [True, False, False, True] + [False] * 6)
        block = Noise(3520, 3)
        res = engine.ConvertBlock(block, True)
        self.assertEqual(res.slot_weights, [1.0, 0.0, 0.0, 0.0] + [0.0] * 6)
        self.assertEqual((parts.cfm_calls[-1]["style"], parts.cfm_calls[-1]["cat"]), (1.0, 172 + 69))
        engine.SetManualSlot(3)
        self.assertEqual((params.Get("emotion_mode"), params.Get("manual_slot")), ("manual", 3))
        res = engine.ConvertBlock(block, True)
        self.assertEqual(res.slot_weights, [0.0, 0.0, 0.0, 1.0] + [0.0] * 6)
        self.assertEqual((parts.cfm_calls[-1]["style"], parts.cfm_calls[-1]["cat"]), (2.0, 258 + 69))

    def test_timbre_mix_blends_reference_and_live_style(self):
        engine, parts, params, _ = New()
        Ref(engine, 0, 2.0, 1)  # CAMPPlus call 1: style 1.0
        params.Set("timbre_mix", 0.5)
        styles = []
        for seed in range(5):
            engine.ConvertBlock(Noise(3520, 10 + seed), True)
            styles.append(parts.cfm_calls[-1]["style"])
        self.assertEqual(styles, [1.0, 1.0, 1.0, 1.0, 1.5])  # live style needs 1.0 s of active speech: first at block 5 (call 2: 2.0)
        self.assertEqual(parts.campplus_calls, 2)

    def test_full_wet_never_computes_the_live_style(self):
        engine, parts, params, _ = New()
        Ref(engine, 0, 2.0, 1)
        for seed in range(6):
            engine.ConvertBlock(Noise(3520, 20 + seed), True)
        self.assertEqual(parts.campplus_calls, 1)

    def test_prosody_stage_skipped_at_neutral_and_applied_with_pitch_shift(self):
        engine, parts, params, _ = New("sine")
        Ref(engine, 0, 2.0, 1)
        block = Noise(3520, 4)
        self.assertFalse(engine.ConvertBlock(block, True).stage_applied)
        params.Set("pitch_shift_st", 3.0)
        res = engine.ConvertBlock(block, True)
        self.assertTrue(res.stage_applied)
        self.assertEqual(res.wav22k.shape, (6284,))
        self.assertTrue(np.isfinite(res.wav22k).all())
        self.assertAlmostEqual(res.f0_median_hz, 160.0, delta=1.0)
        self.assertGreater(engine.Telemetry()["world_ms"], 0.0)

    def test_telemetry_keys(self):
        engine, _, _, _ = New()
        Ref(engine, 0, 2.0, 1)
        engine.ConvertBlock(Noise(3520, 5), True)
        tel = engine.Telemetry()
        for key in ("xlsr_ms", "length_regulator_ms", "cfm_ms", "vocoder_ms", "fcpe_ms", "world_ms", "fp16", "backend"):
            self.assertIn(key, tel)
        self.assertEqual((tel["fp16"], tel["backend"]), (False, "cpu"))


class TransferWiringTest(unittest.TestCase):
    """What the engine hands the prosody stage when the transfer is on (the stage itself is tested in test_prosody_transfer.py)."""

    @staticmethod
    def _Spy(engine):
        seen = []
        real = engine._prosody.Process

        def Spy(*args):
            seen.append(args)
            return real(*args)

        engine._prosody.Process = Spy
        return seen

    def test_defaults_leave_the_transfer_off(self):
        params = ParamStore()
        self.assertEqual((params.Get("prosody_follow"), params.Get("energy_follow")), (0.0, 0.0))

    def test_the_stage_is_handed_the_source_ring_and_its_f0(self):
        engine, parts, params, _ = New("sine", follow=0.7)
        Ref(engine, 0, 2.0, 1)
        seen = self._Spy(engine)
        res = engine.ConvertBlock(Noise(3520, 6), True)
        self.assertTrue(res.stage_applied)
        (segment, new, hop, shift, formant, expander, prosody, energy, source), = seen
        self.assertEqual((len(segment), new, hop, (shift, formant, expander), (prosody, energy)), (17223, 6284, 4851, (0.0, 0.0, 1.0), (0.7, 0.7)))
        self.assertEqual(len(source.x16), 36880)  # the whole ring: 2.0 s left context + overlap + search + block + lookahead
        # The vocoder output (17664 samples at 22050 Hz) ends where the ring ends, so its sample 0 is 12817.5 source samples before it.
        self.assertAlmostEqual(source.t0, 36880 - 17664 * 16000 / 22050, places=6)
        # FCPE saw the region, the lookahead, the stage's lead and FCPE's own left context: 8640 samples, a whole number of 10 ms frames.
        self.assertEqual(engine._fcpe.sizes, [8640, 17223])  # the engine's window, then the stage's own analysis of the converted segment
        self.assertEqual((len(source.f0), source.f0_origin), (8640 // 160 + 1, 36880 - 8640))

    def test_follow_zero_keeps_the_original_fcpe_window_and_hands_over_no_source(self):
        engine, parts, params, _ = New("sine")
        Ref(engine, 0, 2.0, 1)
        seen = self._Spy(engine)
        params.Set("pitch_shift_st", 3.0)
        engine.ConvertBlock(Noise(3520, 7), True)
        # The engine's window (the block and 0.2 s of context) is as before the transfer existed. The stage shares this tracker and
        # analyses the converted segment (17223 samples) whenever a pitch control is on: that second call predates the transfer.
        self.assertEqual(engine._fcpe.sizes, [3520 + 3200, 17223])
        self.assertIsNone(seen[0][-1])
        self.assertEqual(seen[0][6:8], (0.0, 0.0))

    def test_energy_only_needs_no_source_f0_and_no_wider_fcpe_window(self):
        engine, parts, params, _ = New("sine")
        Ref(engine, 0, 2.0, 1)
        seen = self._Spy(engine)
        params.Set("energy_follow", 0.5)
        res = engine.ConvertBlock(Noise(3520, 8), True)
        self.assertTrue(res.stage_applied)
        self.assertEqual(engine._fcpe.sizes, [3520 + 3200])
        self.assertIsNone(seen[0][-1].f0)
        self.assertEqual(len(seen[0][-1].x16), 36880)

    def test_the_transfer_runs_end_to_end_and_stays_finite_and_in_range(self):
        engine, parts, params, _ = New("sine", follow=0.7)
        Ref(engine, 0, 2.0, 1)
        for seed in range(4):
            res = engine.ConvertBlock(Noise(3520, 30 + seed), True)
            self.assertEqual((res.wav22k.shape, res.wav22k.dtype), ((6284,), np.float32))
            self.assertTrue(np.isfinite(res.wav22k).all())
            self.assertLessEqual(float(np.abs(res.wav22k).max()), 1.0)
        self.assertGreater(engine.Telemetry()["world_ms"], 0.0)

    def test_shorter_blocks_keep_the_geometry(self):
        engine, parts, params, _ = New("sine", follow=0.7)
        Ref(engine, 0, 2.0, 1)
        params.Set("block_ms", 100)
        seen = self._Spy(engine)
        engine.ConvertBlock(Noise(1600, 9), True)
        source = seen[0][-1]
        # out22 = 3638, right22 = 441: (4079 * 16000 / 22050 + 0.23 * 16000) / 160 = 41.5 frames, so 42 frames of FCPE
        self.assertEqual(engine._fcpe.sizes[0], 42 * 160)  # the engine's window; the stage's own analysis of the segment follows it
        self.assertEqual(source.f0_origin, len(source.x16) - 42 * 160)


class SourceTest(unittest.TestCase):
    def test_every_control_is_read_by_name(self):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for name in ("block_ms", "overlap_ms", "diffusion_steps", "cfg_rate", "left_context_s", "pitch_shift_st",
                     "formant_shift", "f0_expander", "prosody_follow", "energy_follow", "timbre_mix", "emotion_mode", "manual_slot"):
            self.assertIn(f'"{name}"', source, name)


if __name__ == "__main__":
    unittest.main()
