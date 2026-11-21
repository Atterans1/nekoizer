import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import seedvc_params as P


class ParamTests(unittest.TestCase):
    def test_spec_defaults(self):
        d = P.Defaults()
        self.assertEqual(d["block_ms"], 220)
        self.assertEqual(d["overlap_ms"], 45.0)
        self.assertEqual(d["diffusion_steps"], 10)
        self.assertEqual(d["cfg_rate"], 0.65)
        self.assertEqual(d["left_context_s"], 2.0)
        self.assertEqual((d["prosody_follow"], d["energy_follow"]), (0.0, 0.0))

    def test_spec_ranges(self):
        s = P.ParamSpec
        self.assertEqual((s["diffusion_steps"]["min"], s["diffusion_steps"]["max"]), (6, 16))
        self.assertEqual((s["cfg_rate"]["min"], s["cfg_rate"]["max"]), (0.0, 1.0))
        self.assertEqual((s["left_context_s"]["min"], s["left_context_s"]["max"]), (1.5, 2.5))
        self.assertEqual((s["pitch_shift_st"]["min"], s["pitch_shift_st"]["max"]), (-12.0, 12.0))
        self.assertEqual((s["formant_shift"]["min"], s["formant_shift"]["max"]), (-2.0, 2.0))
        self.assertEqual((s["f0_expander"]["min"], s["f0_expander"]["max"]), (0.8, 1.4))
        self.assertEqual((s["overlap_ms"]["min"], s["overlap_ms"]["max"]), (40.0, 50.0))
        for name in ("prosody_follow", "energy_follow"):
            self.assertEqual((s[name]["type"], s[name]["min"], s[name]["max"]), ("float", 0.0, 1.0))

    def test_validate_rejects_out_of_range_and_wrong_types(self):
        with self.assertRaises(ValueError):
            P.Validate("diffusion_steps", 17)
        with self.assertRaises(ValueError):
            P.Validate("f0_expander", 1.5)
        with self.assertRaises(ValueError):
            P.Validate("prosody_follow", 1.01)
        with self.assertRaises(ValueError):
            P.Validate("energy_follow", -0.1)
        self.assertEqual((P.Validate("prosody_follow", 0), P.Validate("energy_follow", 1)), (0.0, 1.0))
        with self.assertRaises(ValueError):
            P.Validate("monitor_enabled", 1)
        with self.assertRaises(ValueError):
            P.Validate("emotion_mode", "loud")
        with self.assertRaises(KeyError):
            P.Validate("no_such_param", 1)
        self.assertEqual(P.Validate("diffusion_steps", 9.6), 10)

    def test_overlap_must_fit_block(self):
        store = P.ParamStore()
        store.Set("block_ms", 100)
        with self.assertRaises(ValueError):
            store.Set("overlap_ms", 50.01)
        self.assertEqual(store.Get("overlap_ms"), 45.0)

    def test_presets_change_only_listed_controls_and_fourth_is_not_invented(self):
        store = P.ParamStore()
        before = store.Snapshot()
        after = store.ApplyPreset("Soft / Whisper")
        changed = {k for k in before if before[k] != after[k]}
        self.assertEqual(changed, {"emotion_mode", "manual_slot", "f0_expander"})
        self.assertEqual(len(P.PRESETS), 10)
        self.assertIsNone(P.FOURTH_PRESET_VALUES)
        self.assertEqual(P.TRUNCATED_PRESET_LABEL, "Ultra-Low-")
        with self.assertRaises(KeyError):
            store.ApplyPreset("Ultra-Low-")

    def test_subscribers_see_coerced_value(self):
        store = P.ParamStore()
        seen = []
        store.Subscribe(lambda n, v: seen.append((n, v)))
        store.Set("diffusion_steps", 7.2)
        self.assertEqual(seen, [("diffusion_steps", 7)])

    def test_janitor_runs_newest_first_once_and_survives_failures(self):
        order = []
        j = P.Janitor()
        j.Add(lambda: order.append("a"))
        j.Add(lambda: 1 / 0)
        j.Add(lambda: order.append("c"))
        j.Cleanup()
        j.Cleanup()
        self.assertEqual(order, ["c", "a"])
        self.assertEqual(j.failures, 1)


if __name__ == "__main__":
    unittest.main()
