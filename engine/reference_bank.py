"""Ten-slot reference bank (spec section 2, R12, R18): precompute once per clip, cache by content hash.

The bank is model-agnostic. The engine passes an encoder callable that turns the (at most PROMPT_MAX_S long) 22050 Hz
prompt into the CAMPPlus style embedding, the prompt condition and the prompt mel exactly as Seed-VC's
real-time-gui.py builds them. SetSlot may be called from a UI thread while a worker reads slots with Snapshot().
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from seedvc_params import MODEL_SR, PROMPT_MAX_S, SLOT_NAMES

# ---------------------------------------------------------------- PUBLIC
MIN_REFERENCE_S = 0.5  # choice: a shorter clip cannot give a usable style embedding or prompt
CACHE_ENTRIES = 16  # choice: encoded prompts kept for re-use (a re-loaded or duplicated clip costs nothing)


@dataclass(frozen=True)
class SlotData:
    idx: int
    name: str
    content_hash: str
    seconds: float  # full length of the clip passed to SetSlot
    seconds_used: float  # length of the prompt actually encoded (at most PROMPT_MAX_S)
    style: Any  # CAMPPlus embedding (1, 192)
    prompt_condition: Any  # length-regulated XLSR features of the prompt (1, prompt_tokens, 384)
    mel: Any  # prompt mel (1, 80, prompt_tokens)
    prompt_tokens: int


Encoder = Callable[[np.ndarray], tuple]  # float32 mono at MODEL_SR -> (style, prompt_condition, mel)


class ReferenceBank:
    def __init__(self, encode: Encoder):
        self._encode = encode
        self._lock = threading.Lock()
        self._slots: list[SlotData | None] = [None] * len(SLOT_NAMES)
        self._cache: OrderedDict[str, tuple] = OrderedDict()

    def SetSlot(self, idx: int, wav: np.ndarray, sr: int) -> dict:
        """Encode (or fetch from the cache) the first PROMPT_MAX_S of wav and store it in slot idx.

        wav: mono floats, integer PCM, or (samples, channels). Returns dict(idx, name, seconds, seconds_used,
        embedding_dim, prompt_tokens, content_hash, cached)."""
        self._CheckIdx(idx)
        mono = self._ToMono22k(wav, sr)
        seconds = len(mono) / MODEL_SR
        prompt = mono[: int(MODEL_SR * PROMPT_MAX_S)]
        if len(prompt) < MIN_REFERENCE_S * MODEL_SR:
            raise ValueError(f"reference is {len(prompt) / MODEL_SR:.2f} s, need at least {MIN_REFERENCE_S} s")
        key = hashlib.sha1(prompt.tobytes() + repr(PROMPT_MAX_S).encode()).hexdigest()
        with self._lock:
            encoded = self._cache.get(key)
            if encoded is not None:
                self._cache.move_to_end(key)
        cached = encoded is not None
        if encoded is None:
            encoded = tuple(self._encode(prompt))
        style, prompt_condition, mel = encoded
        data = SlotData(idx, SLOT_NAMES[idx], key, seconds, len(prompt) / MODEL_SR, style, prompt_condition, mel,
                        int(prompt_condition.shape[1]))
        with self._lock:
            self._cache[key] = encoded
            self._cache.move_to_end(key)
            while len(self._cache) > CACHE_ENTRIES:
                self._cache.popitem(last=False)
            self._slots[idx] = data
        return self._Describe(data) | {"cached": cached}

    def ClearSlot(self, idx: int) -> None:
        self._CheckIdx(idx)
        with self._lock:
            self._slots[idx] = None

    def Clear(self) -> None:
        with self._lock:
            self._slots = [None] * len(SLOT_NAMES)
            self._cache.clear()

    def Get(self, idx: int) -> SlotData | None:
        self._CheckIdx(idx)
        with self._lock:
            return self._slots[idx]

    def Snapshot(self) -> list[SlotData | None]:
        """The slots as they are now (one consistent view for a whole block)."""
        with self._lock:
            return list(self._slots)

    def Loaded(self) -> list[bool]:
        return [s is not None for s in self.Snapshot()]

    def SlotInfo(self) -> list[dict]:
        """One dict per slot (idx 0..len(SLOT_NAMES) - 1): idx, name, loaded, seconds, seconds_used, embedding_dim, prompt_tokens, content_hash."""
        out = []
        for i, slot in enumerate(self.Snapshot()):
            if slot is None:
                out.append({"idx": i, "name": SLOT_NAMES[i], "loaded": False, "seconds": 0.0, "seconds_used": 0.0,
                            "embedding_dim": 0, "prompt_tokens": 0, "content_hash": ""})
            else:
                out.append({"loaded": True, **self._Describe(slot)})
        return out

    def CacheSize(self) -> int:
        with self._lock:
            return len(self._cache)

    # ---------------------------------------------------------------- PRIVATE
    @staticmethod
    def _CheckIdx(idx: int) -> None:
        if isinstance(idx, bool) or not isinstance(idx, (int, np.integer)) or not 0 <= idx < len(SLOT_NAMES):
            raise ValueError(f"slot index must be an int 0..{len(SLOT_NAMES) - 1}, got {idx!r}")

    @staticmethod
    def _Describe(slot: SlotData) -> dict:
        return {"idx": slot.idx, "name": slot.name, "seconds": slot.seconds, "seconds_used": slot.seconds_used,
                "embedding_dim": int(slot.style.shape[-1]), "prompt_tokens": slot.prompt_tokens,
                "content_hash": slot.content_hash}

    @staticmethod
    def _ToMono22k(wav: np.ndarray, sr: int) -> np.ndarray:
        x = np.asarray(wav)
        if np.issubdtype(x.dtype, np.integer):
            x = x.astype(np.float32) / (float(np.iinfo(x.dtype).max) + 1.0)
        x = x.astype(np.float32, copy=False)
        if x.ndim == 2:
            x = x.mean(axis=1)
        if x.ndim != 1 or x.size == 0:
            raise ValueError("wav must be a non-empty mono array or (samples, channels)")
        if not np.isfinite(x).all():
            raise ValueError("wav contains NaN or infinite samples")
        if sr != MODEL_SR:
            import librosa  # imported here: the pure tests that never resample stay fast

            x = librosa.resample(x, orig_sr=int(sr), target_sr=MODEL_SR)
        return np.ascontiguousarray(x, dtype=np.float32)
