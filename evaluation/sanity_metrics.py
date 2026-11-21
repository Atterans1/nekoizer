"""Known-answer checks for metrics.py (CPU): identical, pitch-shifted (mean removal should keep corr high), noise, and time-reversed."""
import os
import sys
from math import gcd

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import metrics as M  # noqa: E402


def Rs(x, src, dst):
    g = gcd(src, dst)
    return resample_poly(x, dst // g, src // g).astype(np.float32)


x24, sr = sf.read(os.path.join(ROOT, "testdata", "ref_0.wav"), dtype="float32")
x16 = Rs(x24, sr, 16000)
rng = np.random.default_rng(0)
shifted = librosa.effects.pitch_shift(x16, sr=16000, n_steps=4.0).astype(np.float32)
cases = {
    "identical": x16,
    "pitch_shift_+4st": shifted,
    "white_noise": (0.1 * rng.standard_normal(len(x16))).astype(np.float32),
    "time_reversed": x16[::-1].copy(),
}
for name, conv in cases.items():
    p = M.Prosody(x16, conv)
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()}, flush=True)
print("cos self", round(M.Cos(np.arange(5.0), np.arange(5.0)), 6), "cos orth", round(M.Cos([1, 0], [0, 1]), 6))
