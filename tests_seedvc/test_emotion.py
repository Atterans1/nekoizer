"""EmotionSelector and PitchSpreadTracker with literal feature sequences (SPEC_RESOLUTIONS R11, R12)."""
import unittest

import seedvc_params as P
from engine.emotion import ANGRY, HYPED, NEUTRAL, WHISPER, EmotionSelector, PitchSpreadTracker

N = len(P.SLOT_NAMES)  # fixtures follow the engine's slot count
ALL = [True] * N
def Hot(slot):
    return [1.0 if i == slot else 0.0 for i in range(N)]


def Pad(flags):
    return list(flags) + [False] * (N - len(flags))


WARM = [(-20.0, 1.0)] * 40  # 10 s of ordinary speech at -20 dB with a calm pitch (one block per 0.25 s)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def New():
    clock = FakeClock()
    return clock, EmotionSelector(clock)


def Feed(sel, clock, seq, loaded=ALL, step=0.25, mode="auto", manual_slot=0):
    """Feed (rms_db, pitch_std_st) pairs one step apart and return the selected slot after each update."""
    out = []
    for rms, std in seq:
        sel.Update(rms, std, loaded, mode, manual_slot)
        out.append(sel.Selected())
        clock.t += step
    return out


class ConstantsTest(unittest.TestCase):
    def test_r11_r12_values(self):
        self.assertEqual(P.EMOTION_DWELL_MS, 400.0)
        self.assertEqual(P.EMOTION_RMS_WINDOW_S, 30.0)
        self.assertEqual(P.EMOTION_PITCH_WINDOW_S, 1.0)
        self.assertEqual(P.EMOTION_WHISPER_DB_BELOW_MEDIAN, 8.0)
        self.assertEqual(P.EMOTION_ANGRY_DB_ABOVE_MEDIAN, 6.0)
        self.assertEqual(P.EMOTION_ANGRY_PITCH_STD_ST, 2.5)
        self.assertEqual(P.EMOTION_HYPED_PITCH_STD_ST, 3.5)
        self.assertEqual(P.EMOTION_EMBED_TAU_S, 0.2)
        self.assertEqual(P.SLOT_NAMES, ("Neutral", "Hyped/Laughing", "Whisper/Soft", "Angry/Intense", "Sad/Low", "Excited/Fast", "Calm/Warm", "Serious/Deadpan", "Playful/Teasing", "Tense/Fearful"))


class RuleTest(unittest.TestCase):
    def Tail(self, block, count=3):
        clock, sel = New()
        return Feed(sel, clock, WARM + [block] * count)[len(WARM):]

    def test_calm_speech_stays_neutral(self):
        self.assertEqual(self.Tail((-20.0, 1.0)), [NEUTRAL, NEUTRAL, NEUTRAL])

    def test_whisper_below_median_minus_8_db(self):
        self.assertEqual(self.Tail((-28.1, 1.0)), [NEUTRAL, NEUTRAL, WHISPER])

    def test_whisper_boundary_is_strict(self):
        self.assertEqual(self.Tail((-28.0, 1.0)), [NEUTRAL, NEUTRAL, NEUTRAL])
        self.assertEqual(self.Tail((-27.9, 1.0)), [NEUTRAL, NEUTRAL, NEUTRAL])

    def test_angry_needs_loud_and_pitch_spread(self):
        self.assertEqual(self.Tail((-13.9, 2.6)), [NEUTRAL, NEUTRAL, ANGRY])
        self.assertEqual(self.Tail((-14.0, 2.6)), [NEUTRAL, NEUTRAL, NEUTRAL])  # exactly +6 dB is not above
        self.assertEqual(self.Tail((-13.9, 2.5)), [NEUTRAL, NEUTRAL, NEUTRAL])  # exactly 2.5 st is not above

    def test_hyped_on_pitch_spread_alone(self):
        self.assertEqual(self.Tail((-20.0, 3.6)), [NEUTRAL, NEUTRAL, HYPED])
        self.assertEqual(self.Tail((-20.0, 3.5)), [NEUTRAL, NEUTRAL, NEUTRAL])

    def test_rule_order(self):
        self.assertEqual(self.Tail((-13.9, 4.0)), [NEUTRAL, NEUTRAL, ANGRY])  # Angry before Hyped
        self.assertEqual(self.Tail((-30.0, 5.0)), [NEUTRAL, NEUTRAL, WHISPER])  # Whisper before Hyped

    def test_missing_pitch_is_no_spread(self):
        self.assertEqual(self.Tail((-20.0, None)), [NEUTRAL, NEUTRAL, NEUTRAL])


class DwellTest(unittest.TestCase):
    def test_flicker_never_switches(self):
        clock, sel = New()
        out = Feed(sel, clock, [(-20.0, 4.0), (-20.0, 1.0)] * 6)
        self.assertEqual(out, [NEUTRAL] * 12)

    def test_switch_needs_400_ms_of_evidence(self):
        clock, sel = New()
        out = Feed(sel, clock, [(-20.0, 4.0)] * 5, step=0.125)  # updates at 0, .125, .25, .375, .5 s
        self.assertEqual(out, [NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, HYPED])

    def test_single_opposing_block_does_not_switch_back(self):
        clock, sel = New()
        seq = [(-20.0, 4.0)] * 3 + [(-20.0, 1.0)] + [(-20.0, 4.0)] * 2
        self.assertEqual(Feed(sel, clock, seq), [NEUTRAL, NEUTRAL, HYPED, HYPED, HYPED, HYPED])

    def test_return_to_neutral_after_400_ms(self):
        clock, sel = New()
        seq = [(-20.0, 4.0)] * 3 + [(-20.0, 1.0)] * 3
        self.assertEqual(Feed(sel, clock, seq), [NEUTRAL, NEUTRAL, HYPED, HYPED, HYPED, NEUTRAL])

    def test_changing_candidate_restarts_the_timer(self):
        clock, sel = New()
        seq = WARM + [(-20.0, 4.0)] * 2 + [(-30.0, 1.0)] * 2 + [(-20.0, 4.0)] * 2
        self.assertEqual(Feed(sel, clock, seq)[len(WARM):], [NEUTRAL] * 6)


class SlotFallbackTest(unittest.TestCase):
    def test_empty_candidate_slot_falls_back_to_neutral(self):
        clock, sel = New()
        out = Feed(sel, clock, [(-20.0, 4.0)] * 5, loaded=Pad([True, False, True, True]))
        self.assertEqual(out, [NEUTRAL] * 5)
        self.assertEqual(sel.Weights(), Hot(0))

    def test_missing_neutral_falls_back_to_first_loaded(self):
        clock, sel = New()
        out = Feed(sel, clock, [(-20.0, 1.0)] * 3, loaded=Pad([False, False, True, False]))
        self.assertEqual(out, [WHISPER] * 3)
        self.assertEqual(sel.Weights(), Hot(2))

    def test_loaded_candidate_is_reached_when_neutral_is_empty(self):
        clock, sel = New()
        out = Feed(sel, clock, WARM + [(-13.9, 2.6)] * 3, loaded=Pad([False, True, False, True]))
        self.assertEqual(out[len(WARM) - 1:], [HYPED, HYPED, HYPED, ANGRY])

    def test_cleared_selected_slot_moves_at_once(self):
        clock, sel = New()
        Feed(sel, clock, WARM + [(-30.0, 1.0)] * 3)
        self.assertEqual(sel.Selected(), WHISPER)
        out = Feed(sel, clock, [(-30.0, 1.0)], loaded=Pad([True, True, False, True]))
        self.assertEqual(out, [NEUTRAL])
        self.assertEqual(sel.Weights(), Hot(0))

    def test_no_loaded_slot(self):
        clock, sel = New()
        self.assertEqual(Feed(sel, clock, [(-20.0, 1.0)] * 2, loaded=[False] * N), [-1, -1])
        self.assertEqual(sel.Weights(), [0.0] * N)


class ManualTest(unittest.TestCase):
    def test_one_hot_on_manual_slot_at_once(self):
        clock, sel = New()
        sel.Update(-20.0, 1.0, ALL, "manual", 3)
        self.assertEqual(sel.Weights(), Hot(3))
        clock.t += 0.01
        sel.Update(-20.0, 1.0, ALL, "manual", 1)
        self.assertEqual(sel.Weights(), Hot(1))
        self.assertEqual(sel.Selected(), HYPED)

    def test_manual_ignores_features(self):
        clock, sel = New()
        sel.Update(-60.0, 9.0, ALL, "manual", 0)
        self.assertEqual(sel.Weights(), Hot(0))

    def test_manual_empty_slot_falls_back(self):
        clock, sel = New()
        sel.Update(-20.0, 1.0, Pad([True, False, False, False]), "manual", 2)
        self.assertEqual(sel.Weights(), Hot(0))
        sel.Update(-20.0, 1.0, Pad([False, True, False, False]), "manual", 2)
        self.assertEqual(sel.Weights(), Hot(1))

    def test_manual_to_auto_keeps_slot_until_dwell(self):
        clock, sel = New()
        sel.Update(-20.0, 1.0, ALL, "manual", 3)
        clock.t += 0.25
        self.assertEqual(Feed(sel, clock, [(-20.0, 1.0)] * 3), [ANGRY, ANGRY, NEUTRAL])
        w = sel.Weights()
        self.assertAlmostEqual(w[NEUTRAL], 0.7134952, places=6)
        self.assertAlmostEqual(w[ANGRY], 0.2865048, places=6)

    def test_bad_arguments(self):
        clock, sel = New()
        with self.assertRaises(ValueError):
            sel.Update(-20.0, 1.0, ALL, "sideways", 0)
        with self.assertRaises(ValueError):
            sel.Update(-20.0, 1.0, [True] * 3, "auto", 0)
        with self.assertRaises(ValueError):
            sel.Update(-20.0, 1.0, ALL, "auto", N)


class SmoothingTest(unittest.TestCase):
    def test_weights_follow_tau_0_2_s(self):
        clock, sel = New()
        Feed(sel, clock, WARM)
        self.assertEqual(sel.Weights(), Hot(0))
        Feed(sel, clock, [(-30.0, 1.0)] * 2)  # candidate Whisper, dwell not reached
        self.assertEqual(sel.Weights(), Hot(0))
        Feed(sel, clock, [(-30.0, 1.0)])  # third block: switch, alpha = 1 - exp(-0.25 / 0.2)
        w = sel.Weights()
        self.assertAlmostEqual(w[NEUTRAL], 0.2865048, places=6)
        self.assertAlmostEqual(w[WHISPER], 0.7134952, places=6)
        Feed(sel, clock, [(-30.0, 1.0)])
        w = sel.Weights()
        self.assertAlmostEqual(w[NEUTRAL], 0.0820850, places=6)
        self.assertAlmostEqual(w[WHISPER], 0.9179150, places=6)
        self.assertAlmostEqual(sum(w), 1.0, places=9)


class MedianWindowTest(unittest.TestCase):
    def test_values_older_than_30_s_leave_the_median(self):
        clock, sel = New()
        Feed(sel, clock, [(0.0, 1.0)] * 10)
        clock.t = 40.0
        self.assertEqual(Feed(sel, clock, [(-30.0, 1.0)] * 3), [NEUTRAL] * 3)
        self.assertEqual(sel.State()["median_rms_db"], -30.0)

    def test_inactive_blocks_do_not_feed_features(self):
        clock, sel = New()
        Feed(sel, clock, WARM)
        for _ in range(4):
            sel.Update(-90.0, None, ALL, "auto", 0, active=False)
            clock.t += 0.25
        self.assertEqual(sel.Selected(), NEUTRAL)
        self.assertEqual(sel.State()["median_rms_db"], -20.0)

    def test_reset_forgets_history(self):
        clock, sel = New()
        Feed(sel, clock, WARM + [(-30.0, 1.0)] * 3)
        sel.Reset()
        self.assertEqual(sel.Selected(), -1)
        self.assertEqual(sel.Weights(), [0.0] * N)
        self.assertIsNone(sel.State()["median_rms_db"])


class PitchSpreadTrackerTest(unittest.TestCase):
    def test_constant_pitch_has_zero_spread(self):
        t = PitchSpreadTracker()
        t.Add([200.0] * 50)
        self.assertAlmostEqual(t.StdSemitones(), 0.0, places=9)

    def test_octave_alternation_is_six_semitones(self):
        t = PitchSpreadTracker()
        t.Add([100.0, 200.0] * 25)
        self.assertAlmostEqual(t.StdSemitones(), 6.0, places=9)

    def test_window_holds_one_second_of_frames(self):
        t = PitchSpreadTracker()
        t.Add([100.0] * 60)
        t.Add([200.0] * 100)
        self.assertAlmostEqual(t.StdSemitones(), 0.0, places=9)

    def test_unvoiced_ignored_and_few_voiced_give_none(self):
        t = PitchSpreadTracker()
        t.Add([0.0] * 80 + [150.0, 300.0] * 5)
        self.assertAlmostEqual(t.StdSemitones(), 6.0, places=9)
        t.Reset()
        t.Add([0.0] * 80 + [150.0] * 9)
        self.assertIsNone(t.StdSemitones())
        t.Reset()
        t.Add([float("nan"), float("inf"), -5.0] * 10)
        self.assertIsNone(t.StdSemitones())


if __name__ == "__main__":
    unittest.main()
