"""ReferenceBank: slot bounds, 3.0 s prompt truncation, content-hash cache, UI info dicts (spec section 2, R18)."""
import threading
import unittest

import numpy as np

from engine.reference_bank import CACHE_ENTRIES, ReferenceBank
from seedvc_params import SLOT_NAMES

N = len(SLOT_NAMES)


class FakeEncoder:
    """Stands in for the engine's CAMPPlus + XLSR + length-regulator encoder and records what the bank hands it."""

    def __init__(self):
        self.lengths = []
        self.last = None

    def __call__(self, wav22k):
        self.lengths.append(len(wav22k))
        self.last = wav22k.copy()
        frames = len(wav22k) // 256
        return (
            np.full((1, 192), float(len(self.lengths)), dtype=np.float32),
            np.zeros((1, frames, 384), dtype=np.float32),
            np.zeros((1, 80, frames), dtype=np.float32),
        )


def Tone(seconds, sr=22050, seed=0):
    return (0.1 * np.random.RandomState(seed).randn(int(seconds * sr))).astype(np.float32)


def New():
    enc = FakeEncoder()
    return enc, ReferenceBank(enc)


class TruncationTest(unittest.TestCase):
    def test_long_clip_is_cut_to_3_seconds(self):
        enc, bank = New()
        info = bank.SetSlot(0, Tone(5.0), 22050)
        self.assertEqual(enc.lengths, [66150])
        self.assertEqual(info["seconds"], 5.0)
        self.assertEqual(info["seconds_used"], 3.0)
        self.assertEqual(info["prompt_tokens"], 258)
        self.assertEqual(info["embedding_dim"], 192)
        self.assertEqual(info["name"], "Neutral")
        self.assertFalse(info["cached"])

    def test_short_clip_is_used_whole(self):
        enc, bank = New()
        info = bank.SetSlot(1, Tone(1.5), 22050)
        self.assertEqual(enc.lengths, [33075])
        self.assertEqual((info["seconds"], info["seconds_used"]), (1.5, 1.5))

    def test_minimum_length(self):
        enc, bank = New()
        with self.assertRaises(ValueError):
            bank.SetSlot(0, Tone(0.49), 22050)
        bank.SetSlot(0, Tone(0.5), 22050)
        self.assertEqual(enc.lengths, [11025])

    def test_other_sample_rate_is_resampled_to_22050(self):
        enc, bank = New()
        info = bank.SetSlot(0, Tone(2.0, sr=44100), 44100)
        self.assertEqual(enc.lengths, [44100])
        self.assertEqual(info["seconds"], 2.0)


class CacheTest(unittest.TestCase):
    def test_same_content_in_two_slots_encodes_once(self):
        enc, bank = New()
        wav = Tone(4.0)
        a = bank.SetSlot(0, wav, 22050)
        b = bank.SetSlot(1, wav.copy(), 22050)
        c = bank.SetSlot(1, wav, 22050)
        self.assertEqual(enc.lengths, [66150])
        self.assertEqual([a["cached"], b["cached"], c["cached"]], [False, True, True])
        self.assertEqual(a["content_hash"], b["content_hash"])
        self.assertIs(bank.Get(0).style, bank.Get(1).style)

    def test_different_content_encodes_again(self):
        enc, bank = New()
        bank.SetSlot(0, Tone(4.0, seed=1), 22050)
        bank.SetSlot(1, Tone(4.0, seed=2), 22050)
        self.assertEqual(len(enc.lengths), 2)
        self.assertNotEqual(bank.Get(0).content_hash, bank.Get(1).content_hash)

    def test_only_the_prompt_part_counts(self):
        enc, bank = New()
        first = Tone(5.0, seed=3)
        second = first.copy()
        second[70000:] += 0.5  # differs only after the 66150 prompt samples
        bank.SetSlot(0, first, 22050)
        info = bank.SetSlot(1, second, 22050)
        self.assertTrue(info["cached"])
        self.assertEqual(len(enc.lengths), 1)

    def test_cache_is_bounded(self):
        enc, bank = New()
        for seed in range(CACHE_ENTRIES + 5):
            bank.SetSlot(seed % 4, Tone(1.0, seed=seed), 22050)
        self.assertEqual(bank.CacheSize(), CACHE_ENTRIES)
        self.assertEqual(len(enc.lengths), CACHE_ENTRIES + 5)


class SlotTest(unittest.TestCase):
    def test_bounds(self):
        enc, bank = New()
        for bad in (-1, N, 99, 1.0, True, None, "0"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                bank.SetSlot(bad, Tone(1.0), 22050)
        with self.assertRaises(ValueError):
            bank.ClearSlot(N)
        with self.assertRaises(ValueError):
            bank.Get(-1)
        self.assertEqual(enc.lengths, [])

    def test_slot_info_is_one_ui_dict_per_slot(self):
        enc, bank = New()
        info = bank.SlotInfo()
        self.assertEqual([i["idx"] for i in info], list(range(N)))
        self.assertEqual([i["name"] for i in info], list(SLOT_NAMES))
        self.assertEqual([i["loaded"] for i in info], [False] * N)
        self.assertEqual([(i["seconds"], i["seconds_used"]) for i in info], [(0.0, 0.0)] * N)
        bank.SetSlot(2, Tone(4.0), 22050)
        info = bank.SlotInfo()
        self.assertEqual(len(info), N)
        self.assertEqual((info[2]["loaded"], info[2]["seconds"], info[2]["seconds_used"]), (True, 4.0, 3.0))
        self.assertFalse(info[0]["loaded"])
        self.assertEqual(bank.Loaded(), [False, False, True] + [False] * (N - 3))

    def test_clear_slot_keeps_cache_and_clear_drops_it(self):
        enc, bank = New()
        wav = Tone(2.0)
        bank.SetSlot(0, wav, 22050)
        bank.ClearSlot(0)
        bank.ClearSlot(0)
        self.assertEqual(bank.Loaded(), [False] * N)
        self.assertTrue(bank.SetSlot(0, wav, 22050)["cached"])
        bank.Clear()
        self.assertEqual((bank.Loaded(), bank.CacheSize()), ([False] * N, 0))
        self.assertFalse(bank.SetSlot(0, wav, 22050)["cached"])


class InputTest(unittest.TestCase):
    def test_stereo_is_averaged_to_mono(self):
        enc, bank = New()
        left = Tone(2.0, seed=5)
        bank.SetSlot(0, np.stack([left, -left], axis=1), 22050)
        self.assertEqual(len(enc.last), 44100)
        self.assertTrue(np.all(enc.last == 0.0))

    def test_int16_pcm_is_scaled_to_float(self):
        enc, bank = New()
        pcm = (Tone(2.0, seed=6) * 32768).astype(np.int16)
        bank.SetSlot(0, pcm, 22050)
        np.testing.assert_array_equal(enc.last, pcm.astype(np.float32) / 32768.0)

    def test_bad_audio_is_rejected(self):
        enc, bank = New()
        bad = Tone(2.0)
        bad[10] = np.nan
        for wav in (bad, np.zeros(0, dtype=np.float32), np.zeros((2, 2, 2), dtype=np.float32)):
            with self.assertRaises(ValueError):
                bank.SetSlot(0, wav, 22050)
        self.assertEqual(enc.lengths, [])


class ThreadTest(unittest.TestCase):
    def test_two_threads_set_different_slots(self):
        enc, bank = New()
        errors = []

        def Work(idx):
            try:
                for i in range(20):
                    bank.SetSlot(idx, Tone(1.0, seed=idx * 100 + i), 22050)
            except Exception as exc:  # noqa: BLE001 - collected and asserted below
                errors.append(exc)

        threads = [threading.Thread(target=Work, args=(i,)) for i in (0, 1)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(bank.Loaded(), [True, True] + [False] * 8)


if __name__ == "__main__":
    unittest.main()
