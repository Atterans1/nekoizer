"""Silero VAD gate: 512-sample frames at 16 kHz, stateful, with a hangover."""
from __future__ import annotations

import numpy as np
import torch

FRAME = 512
SAMPLE_RATE = 16000
HANGOVER_FRAMES = 6  # ~190 ms of speech tail kept open after the last voiced frame


# ---------------------------------------------------------------- PUBLIC
class VadGate:
    def __init__(self, threshold: float = 0.5):
        from silero_vad import load_silero_vad

        self._model = load_silero_vad()
        self._threshold = self._Check(threshold)
        self._hang = 0
        self._carry = np.zeros(0, dtype=np.float32)

    def SetThreshold(self, threshold: float) -> None:
        self._threshold = self._Check(threshold)

    def Reset(self) -> None:
        self._model.reset_states()
        self._hang = 0
        self._carry = np.zeros(0, dtype=np.float32)

    def IsSpeech(self, audio16k: np.ndarray) -> bool:
        """True if any 512-sample frame in audio16k (plus leftover from earlier calls) is speech,
        or the hangover from a previous speech frame is still open."""
        buf = np.concatenate([self._carry, audio16k.astype(np.float32, copy=False)])
        n = len(buf) // FRAME
        self._carry = buf[n * FRAME:]
        speech = False
        with torch.inference_mode():
            for i in range(n):
                frame = torch.from_numpy(buf[i * FRAME:(i + 1) * FRAME])
                prob = float(self._model(frame, SAMPLE_RATE))
                if prob >= self._threshold:
                    self._hang = HANGOVER_FRAMES
                    speech = True
                elif self._hang > 0:
                    self._hang -= 1
                    speech = True
        return speech

    # ---------------------------------------------------------------- PRIVATE
    @staticmethod
    def _Check(threshold: float) -> float:
        if not 0.05 <= threshold <= 0.95:
            raise ValueError(f"vad_threshold must be between 0.05 and 0.95 (got {threshold})")
        return float(threshold)
