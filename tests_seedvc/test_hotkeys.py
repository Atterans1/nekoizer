"""Win32 hotkey registry tests (Windows only). Registration results depend on what else holds the combinations."""
from __future__ import annotations

import os
import sys
import threading
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import seedvc_params as P  # noqa: E402
from ui import hotkeys  # noqa: E402
from ui.hotkeys import HotkeyRegistry  # noqa: E402

COMBOS = [f"Ctrl+Alt+{d}" for d in "1234567890"]


def _HotkeyThreadAlive() -> bool:
    return any(t.name == hotkeys.THREAD_NAME and t.is_alive() for t in threading.enumerate())


@unittest.skipUnless(sys.platform == "win32", "RegisterHotKey is Windows only")
class HotkeyTests(unittest.TestCase):
    def test_start_reports_one_result_per_combination(self):
        reg = HotkeyRegistry(lambda slot: None)
        results = reg.Start()
        try:
            self.assertEqual([r["combo"] for r in results], COMBOS)
            for r in results:
                self.assertIsInstance(r["ok"], bool)
                if r["ok"]:
                    self.assertEqual(r["error"], 0)
                else:
                    self.assertNotEqual(r["error"], 0)
            print("\nhotkey registration on this machine:", results)
        finally:
            reg.Stop()

    def test_posted_hotkey_message_dispatches_slot_index(self):
        seen: list[int] = []
        got = threading.Event()

        def on_hotkey(slot: int) -> None:
            seen.append(slot)
            got.set()

        reg = HotkeyRegistry(on_hotkey)
        reg.Start()
        try:
            posted = hotkeys._User32().PostThreadMessageW(reg.ThreadId, hotkeys.WM_HOTKEY,
                                                          hotkeys.HOTKEY_ID_BASE + 2, 0)
            self.assertTrue(posted)
            self.assertTrue(got.wait(2.0))
            self.assertEqual(seen, [2])
        finally:
            reg.Stop()

    def test_callback_exception_does_not_end_the_loop(self):
        seen: list[int] = []
        done = threading.Event()

        def on_hotkey(slot: int) -> None:
            seen.append(slot)
            if slot == 0:
                raise RuntimeError("callback failure under test")
            done.set()

        reg = HotkeyRegistry(on_hotkey)
        reg.Start()
        try:
            user32 = hotkeys._User32()
            user32.PostThreadMessageW(reg.ThreadId, hotkeys.WM_HOTKEY, hotkeys.HOTKEY_ID_BASE + 0, 0)
            user32.PostThreadMessageW(reg.ThreadId, hotkeys.WM_HOTKEY, hotkeys.HOTKEY_ID_BASE + 1, 0)
            self.assertTrue(done.wait(2.0))
            self.assertEqual(seen, [0, 1])
        finally:
            reg.Stop()

    def test_stop_ends_the_thread_and_releases_the_combinations(self):
        reg = HotkeyRegistry(lambda slot: None)
        results = reg.Start()
        reg.Stop()
        self.assertFalse(_HotkeyThreadAlive())
        user32 = hotkeys._User32()
        for r in results:
            if not r["ok"]:
                continue
            hotkey_id = hotkeys.HOTKEY_ID_BASE + r["slot"]
            mods = hotkeys.MOD_CONTROL | hotkeys.MOD_ALT | hotkeys.MOD_NOREPEAT
            again = bool(user32.RegisterHotKey(None, hotkey_id, mods, hotkeys.VK_1 + r["slot"]))
            self.assertTrue(again, r["combo"])
            user32.UnregisterHotKey(None, hotkey_id)
        reg.Stop()  # idempotent

    def test_janitor_cleanup_stops_the_thread(self):
        janitor = P.Janitor()
        reg = HotkeyRegistry(lambda slot: None, janitor=janitor)
        reg.Start()
        self.assertTrue(_HotkeyThreadAlive())
        janitor.Cleanup()
        self.assertFalse(_HotkeyThreadAlive())
        self.assertEqual(janitor.failures, 0)


if __name__ == "__main__":
    unittest.main()
