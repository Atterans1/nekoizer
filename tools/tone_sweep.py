"""Convert source clips to target voices under different prosody settings and save the WAVs
(live_run/tone2/<target>__<source>__<config>.wav, 22.05 kHz float, SOLA-joined 220 ms blocks).

  python tools/tone_sweep.py --configs follow0,follow07              every target x source pair
  python tools/tone_sweep.py --targets ref_0 --sources ref_1 --configs follow0
Targets named self_* are the same-speaker ceiling: they are only paired with their own source.
Scoring is a separate step (live_run/tone2_score.py) so this process holds the GPU only while generating.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import seedvc_params as P  # noqa: E402
import tools.bench_engine as B  # noqa: E402
from engine.seedvc_expressive import SeedVCExpressive  # noqa: E402

# ---------------------------------------------------------------- PUBLIC
OUT = ROOT / "live_run" / "tone2"
CLIPS = ROOT / "evaluation" / "corpus" / "clips"
SOURCE_S = 8.0  # seconds of each source that are converted
BLOCK_MS = 220
TARGETS = {
    "ref_0": ROOT / "testdata" / "ref_0.wav",
    "5661cb5f98bf": CLIPS / "5661cb5f98bf_target.wav",
    "90cbdbdbf782": CLIPS / "90cbdbdbf782_target.wav",
    "self_ref_1": ROOT / "testdata" / "ref_1.wav",
    "self_0df3b6d3074a": CLIPS / "0df3b6d3074a_target.wav",
}
SOURCES = {"ref_1": ROOT / "testdata" / "ref_1.wav", "0df3b6d3074a": CLIPS / "0df3b6d3074a_source.wav"}
SOURCES.update({k: CLIPS / f"{k}_source.wav" for k in ("0e2227d44886", "1c43ae5daafe", "451bfe759fdd", "48ad8d245374", "4fca7d79fd72", "508e46b4e153")})  # PICK_RULE_V2 confirmatory sources
OWN_VOICE = {"self_ref_1": "ref_1", "self_0df3b6d3074a": "0df3b6d3074a"}
CONFIGS = {
    "follow0": {"prosody_follow": 0.0, "energy_follow": 0.0},
    "follow04": {"prosody_follow": 0.4, "energy_follow": 0.4},
    "follow07": {"prosody_follow": 0.7, "energy_follow": 0.6},
    "follow10": {"prosody_follow": 1.0, "energy_follow": 1.0},
}


def Pairs(targets, sources) -> list:
    return [(t, s) for t in targets for s in sources if t not in OWN_VOICE or OWN_VOICE[t] == s]


def Main(argv=None) -> int:
    args = _ParseArgs(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = Pairs(args.targets, args.sources)
    B.AcquireGpuLock()
    params, janitor = P.ParamStore(), P.Janitor()
    try:
        engine = SeedVCExpressive(params, janitor, ROOT / "assets" / "seedvc", device="cuda")
        engine.Load()
        loaded = None
        for target, source in pairs:
            B.RefreshGpuLock()
            if loaded != target:
                voice, voice_sr = B._Mono(TARGETS[target])
                engine.SetSlot(0, voice, voice_sr)
                loaded = target
            clip, sr = B._Mono(SOURCES[source])
            src16 = resample_poly(clip, P.INPUT_SR, sr).astype(np.float32)[: int(SOURCE_S * P.INPUT_SR)]
            B.RunPass(engine, params, src16, 10, BLOCK_MS, B.SEED)  # warm-up: caches, autotune, graph capture
            for name in args.configs:
                overrides = CONFIGS[name]
                for key, value in overrides.items():
                    params.Set(key, value)
                rows, audio = B.RunPass(engine, params, src16, 10, BLOCK_MS, B.SEED, True)
                path = out / f"{target}__{source}__{name}.wav"
                sf.write(path, audio, P.MODEL_SR, subtype="FLOAT")
                ms = lambda key: [r[key] for r in rows if key in r]  # noqa: E731
                line = {"wav": path.name, "blocks": len(rows), "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "inference_ms_median": float(np.median(ms("inference_ms"))), "inference_ms_p95": float(np.percentile(ms("inference_ms"), 95)),
                        "inference_ms_p25": float(np.percentile(ms("inference_ms"), 25)), "inference_ms_p75": float(np.percentile(ms("inference_ms"), 75)),
                        "inference_ms_max": float(np.max(ms("inference_ms"))),
                        "fcpe_ms_median": float(np.median(ms("fcpe_ms"))), "fcpe_ms_p95": float(np.percentile(ms("fcpe_ms"), 95)),
                        "world_ms_median": float(np.median(ms("world_ms"))), "world_ms_p95": float(np.percentile(ms("world_ms"), 95))}
                print(json.dumps(line), flush=True)
                with open(out / "sweep_log.jsonl", "a", encoding="utf-8") as log:
                    log.write(json.dumps(line) + "\n")
                for key in overrides:
                    params.Set(key, P.ParamSpec[key]["default"])
        print("TONE_DONE", flush=True)
        return 0
    finally:
        janitor.Cleanup()
        B.ReleaseGpuLock()


# ---------------------------------------------------------------- PRIVATE
def _ParseArgs(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", type=lambda v: v.split(","), default=list(TARGETS))
    ap.add_argument("--sources", type=lambda v: v.split(","), default=list(SOURCES))
    ap.add_argument("--configs", type=lambda v: v.split(","), default=list(CONFIGS))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    for names, table, what in ((args.targets, TARGETS, "target"), (args.sources, SOURCES, "source"), (args.configs, CONFIGS, "config")):
        for name in names:
            if name not in table:
                ap.error(f"unknown {what} {name!r}; choose from {sorted(table)}")
    return args


if __name__ == "__main__":
    sys.exit(Main())
