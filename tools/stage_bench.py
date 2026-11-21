"""Median stage cost per block: python tools/stage_bench.py [runs]. Real FCPE on the GPU (or CPU if absent), synthetic voiced audio at the engine's geometry."""
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from engine.prosody_stage import FcpeF0, ProsodyStage, SourceAudio

SR, SR16, CTX, REGION, BLOCK = 22050, 16000, 10939, 6284, 4851


def Main(runs=40):
    fcpe = FcpeF0()
    src_fcpe = fcpe.WithThreshold(FcpeF0.SOURCE_THRESHOLD)
    t = np.arange(SR * 8) / SR
    conv = (0.3 * np.sin(2 * np.pi * np.cumsum(210.0 * 2.0 ** (np.sin(t) / 12.0)) / SR)).astype(np.float32)
    src = (0.3 * np.sin(2 * np.pi * np.cumsum(130.0 * 2.0 ** (np.sin(1.3 * t) / 12.0)) / 16000 * 1.0)).astype(np.float32)
    f0 = src_fcpe.Infer(src, SR16)
    rows = {}
    for follow in (0.0, 0.7):
        stage = ProsodyStage(fcpe, SR)
        wall, fc, wd = [], [], []
        for i in range(runs):
            s0 = (i % 6) * BLOCK
            seg = conv[s0 : s0 + CTX + REGION]
            source = SourceAudio(src, s0 * SR16 / SR, f0, 0) if follow else None
            t0 = time.perf_counter()
            stage.Process(seg, REGION, BLOCK, 0.0 if follow else 1.0, 0.0, 1.0, follow, 0.0, source)
            wall.append((time.perf_counter() - t0) * 1000.0)
            tel = stage.Telemetry()
            fc.append(tel["fcpe_ms"])
            wd.append(tel["world_ms"])
        wall, fc, wd = wall[5:], fc[5:], wd[5:]
        rows[follow] = (statistics.median(wall), min(wall), max(wall), statistics.median(fc), statistics.median(wd))
        print(f"follow {follow}: wall median {rows[follow][0]:.1f} ms (min {rows[follow][1]:.1f}, max {rows[follow][2]:.1f}), fcpe {rows[follow][3]:.1f}, world+rest {rows[follow][4]:.1f}, device {fcpe.device}")


if __name__ == "__main__":
    Main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
