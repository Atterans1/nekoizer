"""Silero VAD on the silero_vad package's ONNX model, run with onnxruntime (no torch).

512-sample frames at 16 kHz (32 ms). A frame is speech when the model probability reaches the threshold; the
state then stays active for a hang tail of ceil(VAD_HANG_MS / 32 ms) further frames (R10). Inactive frames are
returned as exact digital zeros by Gate().
"""
from __future__ import annotations

import importlib.util
import math
import os

import numpy as np
import onnxruntime as ort

import seedvc_params as P

# ---- PUBLIC
FRAME_MS = 1000.0 * P.VAD_FRAME_SAMPLES / P.INPUT_SR   # 32 ms
CONTEXT_SAMPLES = 64                                   # the model input is 64 context samples + the 512-sample frame


def DefaultOnnxPath() -> str:
    """silero_vad.onnx inside the installed silero_vad package (located without importing it, so torch stays unloaded)."""
    spec = importlib.util.find_spec("silero_vad")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError("silero_vad is not installed, cannot locate silero_vad.onnx")
    path = os.path.join(list(spec.submodule_search_locations)[0], "data", "silero_vad.onnx")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


class SileroVad:
    # ---- PUBLIC
    def __init__(self, onnx: str | None = None, threshold: float = P.VAD_THRESHOLD, hang_ms: float = P.VAD_HANG_MS):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.onnx = onnx or DefaultOnnxPath()
        self._session = ort.InferenceSession(self.onnx, sess_options=opts, providers=["CPUExecutionProvider"])
        self._sr = np.array(P.INPUT_SR, dtype=np.int64)
        self.threshold = threshold
        self.hang_frames = math.ceil(hang_ms / FRAME_MS)
        self.Reset()

    def Process(self, frame: np.ndarray) -> tuple[float, bool]:
        """One 512-sample 16 kHz frame -> (speech probability, active flag including the hang tail)."""
        if frame.shape != (P.VAD_FRAME_SAMPLES,):
            raise ValueError(f"expected a frame of {P.VAD_FRAME_SAMPLES} samples, got shape {frame.shape}")
        frame = frame.astype(np.float32, copy=False)
        model_input = np.concatenate((self._context, frame))[np.newaxis]
        out, self._state = self._session.run(None, {"input": model_input, "state": self._state, "sr": self._sr})
        self._context = frame[-CONTEXT_SAMPLES:].copy()
        prob = float(out[0, 0])
        if prob >= self.threshold:
            self._hang_left = self.hang_frames
            active = True
        elif self._hang_left > 0:
            self._hang_left -= 1
            active = True
        else:
            active = False
        self.last_prob = prob
        self.last_active = active
        return prob, active

    def Gate(self, frame: np.ndarray) -> tuple[np.ndarray, float, bool]:
        """Process() plus gating: the frame itself when active, exact digital zeros when not."""
        prob, active = self.Process(frame)
        return (frame if active else np.zeros(P.VAD_FRAME_SAMPLES, np.float32)), prob, active

    def Reset(self) -> None:
        self._state = np.zeros((2, 1, 128), np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, np.float32)
        self._hang_left = 0
        self.last_prob = 0.0
        self.last_active = False
