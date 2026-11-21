"""Global Ctrl+Alt+1..0 hotkeys for manual slot selection, via Win32 RegisterHotKey (Windows only).

RegisterHotKey binds each hotkey to the thread that registered it, so registration, dispatch and unregistration all run
on one dedicated thread with its own message loop. Start() returns one result per combination: ok, and the Win32 error
code when it failed (1409 means another program already holds that combination).
"""
from __future__ import annotations

import ctypes
import logging
import threading
from ctypes import wintypes
from typing import Callable

import seedvc_params as P

LOG = logging.getLogger("seedvc.hotkeys")

# ---- PUBLIC
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_NOREPEAT = 0x4000
VK_1 = 0x31  # virtual-key code of the digit 1; slot n uses VK_1 + n
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
HOTKEY_ID_BASE = 0x5E00  # hotkey id of slot n is HOTKEY_ID_BASE + n
THREAD_NAME = "seedvc-hotkeys"


def DigitVk(slot: int) -> int:
    """Virtual-key code of the digit bound to a slot: 1..9 for slots 0..8, 0 (VK_0) for slot 9."""
    return 0x30 if slot == 9 else VK_1 + slot


def DigitLabel(slot: int) -> str:
    return "0" if slot == 9 else str(slot + 1)


class HotkeyRegistry:
    """Ctrl+Alt+1..0 presses call callback(slot). Janitor cleanup unregisters and ends the thread."""

    def __init__(self, callback: Callable[[int], None], janitor: P.Janitor | None = None, slot_count: int | None = None):
        self._callback = callback
        self._slot_count = len(P.SLOT_NAMES) if slot_count is None else slot_count
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._results: list[dict] = []
        if janitor is not None:
            janitor.Add(self.Stop)

    @property
    def ThreadId(self) -> int | None:
        return self._thread_id

    def Start(self) -> list[dict]:
        with self._lock:
            if self._thread is not None:
                return list(self._results)
            self._ready.clear()
            self._thread = threading.Thread(target=self._Loop, name=THREAD_NAME, daemon=True)
            self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("the hotkey thread did not report within 5 s")
        return list(self._results)

    def Stop(self) -> None:
        with self._lock:
            thread, tid = self._thread, self._thread_id
        if thread is None:
            return
        if tid is not None and thread.is_alive():
            _User32().PostThreadMessageW(tid, WM_QUIT, 0, 0)
        if thread is not threading.current_thread():
            thread.join(timeout=5.0)
            if thread.is_alive():
                LOG.error("hotkey thread did not stop within 5 s")
                return
        with self._lock:
            self._thread = None

    # ---- PRIVATE
    def _Loop(self) -> None:
        user32 = _User32()
        msg = wintypes.MSG()
        user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0)  # PM_NOREMOVE: creates this thread's message queue
        self._thread_id = ctypes.WinDLL("kernel32").GetCurrentThreadId()
        registered: list[int] = []
        results: list[dict] = []
        for slot in range(self._slot_count):
            hotkey_id = HOTKEY_ID_BASE + slot
            ok = bool(user32.RegisterHotKey(None, hotkey_id, MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, DigitVk(slot)))
            error = 0 if ok else ctypes.get_last_error()
            if ok:
                registered.append(hotkey_id)
            results.append({"slot": slot, "combo": f"Ctrl+Alt+{DigitLabel(slot)}", "ok": ok, "error": error})
        self._results = results
        self._ready.set()
        try:
            while True:
                got = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if got <= 0:  # 0: WM_QUIT from Stop(); -1: the queue failed; both end the loop
                    break
                if msg.message == WM_HOTKEY:
                    slot = msg.wParam - HOTKEY_ID_BASE
                    if 0 <= slot < self._slot_count:
                        self._Dispatch(slot)
        finally:
            for hotkey_id in registered:
                user32.UnregisterHotKey(None, hotkey_id)
            self._thread_id = None

    def _Dispatch(self, slot: int) -> None:
        try:
            self._callback(slot)
        except Exception:  # a failing callback must not end the hotkey loop
            LOG.exception("hotkey callback failed for slot %d", slot)


_USER32: ctypes.WinDLL | None = None


def _User32() -> ctypes.WinDLL:
    global _USER32
    if _USER32 is None:
        dll = ctypes.WinDLL("user32", use_last_error=True)
        dll.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
        dll.RegisterHotKey.restype = wintypes.BOOL
        dll.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
        dll.UnregisterHotKey.restype = wintypes.BOOL
        dll.PeekMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT,
                                     wintypes.UINT]
        dll.PeekMessageW.restype = wintypes.BOOL
        dll.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
        dll.GetMessageW.restype = ctypes.c_int
        dll.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        dll.PostThreadMessageW.restype = wintypes.BOOL
        _USER32 = dll
    return _USER32
