"""Equivalence and timing harness for the audio chain.

  python tools/chain_equiv.py run --root <tree> --out a.npz      run every scenario on the audio/engine code under <tree>
  python tools/chain_equiv.py compare a.npz b.npz                 diff outputs, print timings and speed-up

<tree> is the repo root (live code) or a snapshot that holds audio/ and engine/. Scenarios use a real speech clip, so the
resamplers, rings and emotion selector are exercised on the signals they see in the live app.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "testdata" / "ref_0.wav"


def Api(tree: Path) -> dict:
    sys.path[:0] = [str(tree), str(ROOT)]
    from audio import input_chain as ic, output_chain as oc
    from engine import emotion
    if (tree / "audio" / "streaming.py").exists():
        from audio.streaming import StreamResampler
        return {"in_rs": StreamResampler, "out_rs": StreamResampler, "mon": ic.MonitorRing, "fifo": oc._Fifo, "emo": emotion}
    return {"in_rs": ic.StreamResampler, "out_rs": oc._StreamResampler, "mon": ic.MonitorRing, "fifo": oc._Ring, "emo": emotion}


def Speech(rate: int) -> np.ndarray:
    x, sr = sf.read(str(CLIP), dtype="float32")
    x = x.mean(axis=1) if x.ndim == 2 else x
    g = math.gcd(sr, rate)
    return resample_poly(x, rate // g, sr // g).astype(np.float32)


def Chunks(x: np.ndarray, n: int):
    for i in range(0, x.size, n):
        yield x[i:i + n]


def Stream(cls, x, rate_in, rate_out, n):
    rs = cls(rate_in, rate_out)
    return np.concatenate([rs.Process(c) for c in Chunks(x, n)])


def ScMicResample(a):   # 48 kHz mic to 16 kHz in 50 ms chunks
    return Stream(a["in_rs"], Speech(48000), 48000, 16000, 2400)


def ScMic441(a):        # 44.1 kHz mic to 16 kHz
    return Stream(a["in_rs"], Speech(44100), 44100, 16000, 2205)


def ScRecorder(a):      # 48 kHz to the 24 kHz recorder
    return Stream(a["in_rs"], Speech(48000), 48000, 24000, 2400)


def ScOutResample(a):   # 22.05 kHz model blocks to the 48 kHz device
    return Stream(a["out_rs"], Speech(22050), 22050, 48000, 4851)


def ScMonitorRing(a):
    rng = np.random.default_rng(3)
    ring, outs = a["mon"](48000), []
    for _ in range(600):
        ring.Write(rng.standard_normal(int(rng.integers(100, 3000))).astype(np.float32))
        outs.append(ring.Read(480))
    return np.concatenate(outs + [np.array([ring.underruns, ring.skips], np.float32)])


def ScOutputFifo(a):
    rng = np.random.default_rng(4)
    fifo, outs = a["fifo"](9600, 2), []
    for _ in range(400):
        free = fifo.cap - (fifo.w - fifo.r)
        n = int(rng.integers(200, 1500))
        pad = int(rng.integers(0, 300)) if rng.random() < 0.3 else 0
        if pad + n <= free:
            fifo.Write(rng.standard_normal(n).astype(np.float32), pad)
        take = min(int(rng.integers(100, 1200)), fifo.w - fifo.r)
        out = np.zeros((take, 2), np.float32)
        fifo.Read(out, take)
        outs.append(out.ravel())
    return np.concatenate(outs)


def ScEmotion(a):
    rng, t = np.random.default_rng(5), [0.0]
    sel = a["emo"].EmotionSelector(clock=lambda: t[0])
    rows = []
    for i in range(900):
        t[0] += 0.05
        rms = -30.0 + 8.0 * math.sin(i / 40.0) + float(rng.normal(0, 3))
        spread = None if i % 17 == 0 else abs(float(rng.normal(2.0, 2.0)))
        loaded = [True, i > 60, i > 120, i % 300 < 250]
        w = sel.Update(rms, spread, loaded, "auto" if i % 400 < 300 else "manual", i % 4, active=i % 11 != 0)
        rows.append(w + [sel.Selected()])
    return np.asarray(rows, np.float64)


SCENARIOS = {"mic_48k_to_16k": ScMicResample, "mic_44k1_to_16k": ScMic441, "recorder_48k_to_24k": ScRecorder,
             "out_22k_to_48k": ScOutResample, "monitor_ring": ScMonitorRing, "output_fifo": ScOutputFifo, "emotion": ScEmotion}


def Run(tree: Path, out: Path, reps: int) -> None:
    api = Api(tree)
    arrays, timings = {}, {}
    for name, fn in SCENARIOS.items():
        best = float("inf")
        for _ in range(reps):
            t = time.perf_counter()
            arrays[name] = fn(api)
            best = min(best, time.perf_counter() - t)
        timings[name] = best
    np.savez(out, **arrays, _timings=json.dumps(timings), _tree=str(tree))
    print(json.dumps({k: round(v * 1000, 2) for k, v in timings.items()}))


def Compare(a_path: Path, b_path: Path) -> int:
    a, b = np.load(a_path), np.load(b_path)
    ta, tb = json.loads(str(a["_timings"])), json.loads(str(b["_timings"]))
    bad = 0
    print(f"{'scenario':22} {'len a/b':>16} {'max|diff| on common prefix':>28} {'ms a':>9} {'ms b':>9} {'a/b':>6}")
    for name in SCENARIOS:
        x, y = a[name], b[name]
        n = min(x.size, y.size)
        d = float(np.max(np.abs(x.ravel()[:n] - y.ravel()[:n]))) if n else float("nan")
        exact = name in ("monitor_ring", "output_fifo", "emotion")
        ok = d == 0.0 if exact else d < 1e-6
        bad += not ok
        print(f"{name:22} {x.size:>7}/{y.size:<8} {d:>28.3g} {ta[name]*1000:>9.2f} {tb[name]*1000:>9.2f} {ta[name]/tb[name]:>6.2f} {'ok' if ok else 'MISMATCH'}")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--root", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--reps", type=int, default=5)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    if args.cmd == "run":
        Run(Path(args.root).resolve(), Path(args.out), args.reps)
    else:
        sys.exit(1 if Compare(Path(args.a), Path(args.b)) else 0)
