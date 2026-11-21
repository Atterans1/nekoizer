import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import seedvc_params as P
from audio.balancer import AutoBalancer


def Fresh():
    params = P.ParamStore()
    return params, AutoBalancer(params)


def Overload(bal, t):
    """Update 0.5, then four updates of 1.0 (block 1.0 s): the EMA runs 0.5, 0.65, 0.755, 0.8285, 0.87995 and the last
    update, at time t, triggers the drop."""
    bal.Update(0.5, 1.0, now=t - 4.0)
    for k in (3.0, 2.0, 1.0, 0.0):
        bal.Update(1.0, 1.0, now=t - k)


class BalancerTests(unittest.TestCase):
    def test_ema_uses_alpha_0_3_and_triggers_at_0_85(self):
        _, bal = Fresh()
        bal.Update(0.5, 1.0, now=0.0)
        self.assertAlmostEqual(bal.State(now=0.0)["ema"], 0.5, places=9)
        for t, ema in ((1.0, 0.65), (2.0, 0.755), (3.0, 0.8285)):
            bal.Update(1.0, 1.0, now=t)
            self.assertAlmostEqual(bal.State(now=t)["ema"], ema, places=9)
            self.assertEqual(bal.EffectiveSteps(10, now=t), 10)  # still below 0.85
        bal.Update(1.0, 1.0, now=4.0)
        self.assertAlmostEqual(bal.State(now=4.0)["ema"], 0.87995, places=9)
        self.assertEqual(bal.EffectiveSteps(10, now=4.0), 8)  # requested 10 - drop 2

    def test_ratio_is_inference_over_block_time(self):
        _, bal = Fresh()
        bal.Update(0.11, 0.22, now=0.0)
        self.assertAlmostEqual(bal.State(now=0.0)["ratio"], 0.5, places=9)
        self.assertAlmostEqual(bal.State(now=0.0)["ema"], 0.5, places=9)

    def test_hold_lasts_exactly_three_seconds_then_restores(self):
        _, bal = Fresh()
        Overload(bal, 10.0)
        self.assertEqual(bal.EffectiveSteps(10, now=10.0), 8)
        self.assertEqual(bal.EffectiveSteps(10, now=12.999), 8)
        self.assertAlmostEqual(bal.State(now=12.999)["throttle_remaining_s"], 0.001, places=9)
        self.assertEqual(bal.EffectiveSteps(10, now=13.0), 10)
        self.assertFalse(bal.State(now=13.0)["throttled"])

    def test_it_can_trigger_again_after_the_restore(self):
        _, bal = Fresh()
        Overload(bal, 10.0)
        self.assertEqual(bal.EffectiveSteps(10, now=13.0), 10)
        bal.Update(1.0, 1.0, now=13.0)  # load is still high: the EMA is above 0.85
        self.assertEqual(bal.EffectiveSteps(10, now=13.0), 8)
        self.assertEqual(bal.EffectiveSteps(10, now=15.999), 8)
        self.assertEqual(bal.EffectiveSteps(10, now=16.0), 10)
        self.assertEqual(bal.State(now=16.0)["triggers"], 2)

    def test_load_under_the_threshold_never_triggers(self):
        _, bal = Fresh()
        for k in range(50):
            bal.Update(0.84, 1.0, now=float(k))
        self.assertAlmostEqual(bal.State(now=50.0)["ema"], 0.84, places=9)
        self.assertEqual(bal.EffectiveSteps(10, now=50.0), 10)
        self.assertEqual(bal.State(now=50.0)["triggers"], 0)

    def test_floor_is_six_and_the_result_never_exceeds_the_request(self):
        _, bal = Fresh()
        Overload(bal, 10.0)
        got = {r: bal.EffectiveSteps(r, now=10.0) for r in (16, 10, 8, 7, 6, 5)}
        self.assertEqual(got, {16: 14, 10: 8, 8: 6, 7: 6, 6: 6, 5: 5})

    def test_disabled_balancer_tracks_load_but_never_throttles(self):
        params, bal = Fresh()
        params.Set("auto_balancer", False)
        for k in range(5):
            bal.Update(2.0, 1.0, now=float(k))
        state = bal.State(now=5.0)
        self.assertEqual((state["enabled"], state["triggers"], state["throttled"]), (False, 0, False))
        self.assertAlmostEqual(state["ema"], 2.0, places=9)
        self.assertEqual(bal.EffectiveSteps(10, now=5.0), 10)
        params.Set("auto_balancer", True)
        bal.Update(2.0, 1.0, now=6.0)
        self.assertEqual(bal.EffectiveSteps(10, now=6.0), 8)

    def test_disabling_during_a_hold_restores_at_once(self):
        params, bal = Fresh()
        Overload(bal, 10.0)
        self.assertEqual(bal.EffectiveSteps(10, now=11.0), 8)
        params.Set("auto_balancer", False)
        self.assertEqual(bal.EffectiveSteps(10, now=11.0), 10)

    def test_block_time_must_be_positive(self):
        _, bal = Fresh()
        with self.assertRaises(ValueError):
            bal.Update(0.1, 0.0, now=0.0)

    def test_default_clock_is_monotonic_time(self):
        _, bal = Fresh()
        for _ in range(4):
            bal.Update(1.0, 1.0)
        self.assertEqual(bal.EffectiveSteps(10), 8)
        self.assertEqual(sorted(bal.State()), sorted(["enabled", "ema", "ratio", "throttled", "throttle_remaining_s", "triggers", "threshold", "step_drop", "hold_s"]))


if __name__ == "__main__":
    unittest.main()
