"""Plays testdata/ref_1.wav on a loop at low volume through the speakers once the live run has loaded its reference slot."""
import sys, time
from pathlib import Path
import numpy as np, sounddevice as sd, soundfile as sf
ROOT = Path(__file__).resolve().parent.parent
log = ROOT / "live_run" / "err.txt"
seconds, gain = float(sys.argv[1]), float(sys.argv[2])
t0 = time.time()
while time.time() - t0 < 180:
    if log.exists() and "reference slot 0" in log.read_text(encoding="utf-8", errors="ignore"):
        break
    time.sleep(0.25)
else:
    print("reference never loaded"); sys.exit(2)
x, sr = sf.read(ROOT / "testdata" / "ref_1.wav", dtype="float32")
x = x.mean(axis=1) if x.ndim > 1 else x
x = x / max(1e-6, float(np.abs(x).max())) * gain
reps = int(np.ceil(seconds * sr / len(x))) + 1
y = np.tile(x, reps)[: int(seconds * sr)]
wasapi = next(i for i, h in enumerate(sd.query_hostapis()) if h["name"] == "Windows WASAPI")
dev = next(i for i, d in enumerate(sd.query_devices()) if d["hostapi"] == wasapi and d["max_output_channels"] > 0 and "Speaker" in d["name"])
rate = int(sd.query_devices(dev)["default_samplerate"])
from scipy.signal import resample_poly
from math import gcd
g = gcd(rate, sr)
y = resample_poly(y, rate // g, sr // g).astype(np.float32)
ch = sd.query_devices(dev)["max_output_channels"]
y = np.repeat(y[:, None], min(ch, 2), axis=1)
sr = rate
print("playing on", sd.query_devices(dev)["name"], "peak", float(np.abs(y).max()), flush=True)
sd.play(y, sr, device=dev, blocking=True)
print("playback done", flush=True)
