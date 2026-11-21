"""Real-model benchmark of SeedVCExpressive (docs/seedvc/ENGINE_NOTES.md).

Loads the engine once, streams testdata/ref_1.wav (6.5 s) to the testdata/ref_0.wav voice in block_ms blocks and records, per
configuration, per-stage ms (median / p95), inference ms (median / p95 / max), blocks finished inside real time and VRAM.
The shared GPU lock (live_run/gpu.lock) is taken before the model is loaded and released on every exit path.

  python tools/bench_engine.py                  steps 6/10/16 at 220 ms and block 220/320/440 at 10 steps -> live_run/bench.json
  python tools/bench_engine.py --baseline       the same, written to live_run/bench_baseline.json, plus engine/examples/baseline_defaults.wav
  python tools/bench_engine.py --grid 260,300   extra block sizes at 10 steps (smallest real-time block search)
  python tools/bench_engine.py --profile        per-stage CPU issue time vs GPU busy time, kernel launches, sync census -> live_run/profile.json

Output equivalence (mel L1 and CAMPPlus cosine) is measured on the default configuration: two passes with the same noise seeds
(determinism floor D), one pass with other seeds (seed floor S), and, when engine/examples/baseline_defaults.wav exists, the new
output against it.
"""
from __future__ import annotations

import argparse
import atexit
import datetime as dt
import json
import os
import subprocess
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import yaml
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import seedvc_params as P  # noqa: E402
from audio.sola import Sola  # noqa: E402
from engine import compat  # noqa: E402
from engine.seedvc_expressive import SEED_VC_DIR, LoadComponents, SeedVCExpressive  # noqa: E402

# ---------------------------------------------------------------- PUBLIC
GPU_LOCK = ROOT / "live_run" / "gpu.lock"
LOCK_OWNER = "PerfAgent"
LOCK_STALE_S = 15 * 60.0
LOCK_POLL_S = 20.0
LOCK_MAX_WAIT_S = 10 * 60.0
SEED = 1234
OTHER_SEED = 4321
MEASURED_PASSES = 3
STAGES = ("xlsr", "length_regulator", "cfm", "vocoder", "fcpe", "world", "campplus_live", "other")
STEP_CONFIGS = ((6, 220), (10, 220), (16, 220), (10, 320), (10, 440))
BASELINE_WAV = ROOT / "engine" / "examples" / "baseline_defaults.wav"
OPTIMISED_WAV = ROOT / "engine" / "examples" / "optimised_defaults.wav"


def AcquireGpuLock(path=GPU_LOCK, owner=LOCK_OWNER, poll_s=LOCK_POLL_S, max_wait_s=LOCK_MAX_WAIT_S, stale_s=LOCK_STALE_S,
                   clock=time.time, sleep=time.sleep) -> float:
    """Create the lock file holding owner and an ISO time; return the seconds waited. A lock younger than stale_s is respected
    (poll every poll_s, TimeoutError after max_wait_s); an older one is stale and replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = clock()
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = clock() - path.stat().st_mtime
                holder = path.read_text(encoding="utf-8", errors="replace").strip()
            except FileNotFoundError:
                continue
            if age > stale_s:
                print(f"gpu.lock held by '{holder}' is {age:.0f} s old (stale after {stale_s:.0f} s): replacing it", flush=True)
                path.unlink(missing_ok=True)
                continue
            if clock() - started + poll_s > max_wait_s:
                raise TimeoutError(f"gpu.lock still held by '{holder}' after {clock() - started:.0f} s")
            print(f"gpu.lock held by '{holder}' ({age:.0f} s old); waiting {poll_s:.0f} s", flush=True)
            sleep(poll_s)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"{owner} {dt.datetime.now().isoformat(timespec='seconds')}\n")
        return clock() - started


def RefreshGpuLock(path=GPU_LOCK, owner=LOCK_OWNER) -> None:
    """Touch the lock so a long run is not mistaken for a stale one."""
    try:
        if Path(path).read_text(encoding="utf-8", errors="replace").startswith(owner):
            os.utime(path)
    except FileNotFoundError:
        pass


def ReleaseGpuLock(path=GPU_LOCK, owner=LOCK_OWNER) -> bool:
    """Delete the lock only when this owner wrote it."""
    try:
        if not Path(path).read_text(encoding="utf-8", errors="replace").startswith(owner):
            return False
        Path(path).unlink()
        return True
    except FileNotFoundError:
        return False


def Summary(values) -> dict:
    a = np.asarray(list(values), dtype=np.float64)
    return {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max()), "mean": float(a.mean())}


def Verdict(runs: list) -> dict:
    """Smallest block_ms at 10 steps whose inference p95 is under the block, and whether the spec defaults (220 ms, 10 steps) are real time."""
    at10 = sorted((r for r in runs if r["steps"] == 10), key=lambda r: r["block_ms"])
    real_time = [r["block_ms"] for r in at10 if r["inference_ms"]["p95"] < r["block_ms"]]
    default = next((r for r in at10 if r["block_ms"] == 220), None)
    return {
        "smallest_block_ms_realtime_at_10_steps": real_time[0] if real_time else None,
        "defaults_realtime": None if default is None else bool(default["inference_ms"]["p95"] < 220),
        "defaults_inference_p95_ms": None if default is None else default["inference_ms"]["p95"],
        "criterion": "inference_ms p95 < block_ms, other parameters at the spec defaults",
    }


def GpuClocks() -> dict:
    try:
        fields = "clocks.sm,clocks.max.sm,power.draw,temperature.gpu,memory.used"
        text = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True).stdout
        return dict(zip(("sm_mhz", "sm_max_mhz", "power_w", "temp_c", "memory_used_mb"), (float(v) for v in text.strip().split(","))))
    except (OSError, subprocess.SubprocessError, ValueError) as err:
        return {"error": f"{type(err).__name__}: {err}"}


class OutputMetrics:
    """Mel L1 and CAMPPlus cosine between two 22050 Hz waves, with the mel settings and CAMPPlus weights the engine uses."""

    def __init__(self, assets_dir, device: torch.device):
        compat.PrepareSeedVc(SEED_VC_DIR)
        from modules.audio import mel_spectrogram
        from modules.campplus.DTDNN import CAMPPlus

        assets = Path(assets_dir)
        with open(assets / "config_dit_mel_seed_uvit_xlsr_tiny.yml", "r") as handle:
            config = yaml.safe_load(handle)
        spect = config["preprocess_params"]["spect_params"]
        self._mel_args = {
            "n_fft": spect["n_fft"], "win_size": spect["win_length"], "hop_size": spect["hop_length"], "num_mels": spect["n_mels"],
            "sampling_rate": config["preprocess_params"]["sr"], "fmin": spect.get("fmin", 0),
            "fmax": None if spect.get("fmax", "None") == "None" else 8000, "center": False,
        }
        self._mel = mel_spectrogram
        self._device = device
        self._campplus = CAMPPlus(feat_dim=80, embedding_size=192)
        self._campplus.load_state_dict(torch.load(assets / "campplus_cn_common.bin", map_location="cpu"))
        self._campplus.eval().to(device)

    def MelL1(self, a: np.ndarray, b: np.ndarray) -> float:
        """Mean absolute difference of the natural-log mel (clamp 1e-5, as Seed-VC's mel function) over the shared length."""
        n = min(len(a), len(b))
        with torch.no_grad():
            ma, mb = (self._mel(torch.from_numpy(np.ascontiguousarray(x[:n], dtype=np.float32)).unsqueeze(0).to(self._device), **self._mel_args) for x in (a, b))
        return float((ma - mb).abs().mean())

    def Cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity of the CAMPPlus embeddings of the two whole waves."""
        ea, eb = (self._Embed(x) for x in (a, b))
        return float(torch.nn.functional.cosine_similarity(ea, eb, dim=-1).mean())

    # ---- PRIVATE
    def _Embed(self, wave22k: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            wave = torch.from_numpy(np.ascontiguousarray(wave22k, dtype=np.float32)).to(self._device)
            wave16 = torchaudio.functional.resample(wave, P.MODEL_SR, P.INPUT_SR)
            feat = kaldi.fbank(wave16.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=P.INPUT_SR)
            return self._campplus((feat - feat.mean(dim=0, keepdim=True)).unsqueeze(0))


def RunPass(engine, params, src16: np.ndarray, steps: int, block_ms: int, seed: int, keep_audio: bool = False):
    """Stream src16 once in block_ms blocks. Returns (per-block rows, SOLA output or None). The torch RNG is reseeded before every
    block, so equal code gives equal noise. inference_ms is the wall time of ConvertBlock (the engine's own stage clock synchronises)."""
    params.Set("block_ms", block_ms)
    params.Set("diffusion_steps", steps)
    engine.ResetStream()
    n = int(round(block_ms * P.INPUT_SR / 1000.0))
    sola = Sola(P.MODEL_SR, block_ms, params.Get("overlap_ms"), P.SOLA_SEARCH_MS) if keep_audio else None
    rows, outs = [], []
    for i, start in enumerate(range(0, len(src16) - n + 1, n)):
        torch.manual_seed(seed + i)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = engine.ConvertBlock(src16[start : start + n], True)
        wall = (time.perf_counter() - t0) * 1000.0
        tel = engine.Telemetry()
        rows.append({"inference_ms": wall, "engine_ms": result.inference_s * 1000.0, **{f"{s}_ms": tel[f"{s}_ms"] for s in STAGES if f"{s}_ms" in tel}})
        if sola is not None:
            outs.append(sola.Process(result.wav22k))
    return rows, (np.concatenate(outs) if outs else None)


def RunConfig(engine, params, src16, steps: int, block_ms: int, seeds, keep_audio: bool = False):
    """One warm-up pass (cuDNN autotune, GPU clocks, caches; discarded), then one measured pass per seed. Returns (summary, audios)."""
    RunPass(engine, params, src16, steps, block_ms, SEED)
    torch.cuda.reset_peak_memory_stats()
    rows, audios = [], []
    for seed in seeds:
        got, audio = RunPass(engine, params, src16, steps, block_ms, seed, keep_audio)
        rows += got
        audios.append(audio)
    inference = [r["inference_ms"] for r in rows]
    summary = {
        "steps": steps,
        "block_ms": block_ms,
        "blocks": len(rows),
        "inference_ms": Summary(inference),
        "engine_ms_median": float(np.median([r["engine_ms"] for r in rows])),
        "within_block": int(sum(v < block_ms for v in inference)),
        "stage_ms": {s: Summary([r[f"{s}_ms"] for r in rows]) for s in STAGES if f"{s}_ms" in rows[0]},
        "vram_mb": {"allocated_peak": round(torch.cuda.max_memory_allocated() / 2**20, 1), "reserved_peak": round(torch.cuda.max_memory_reserved() / 2**20, 1)},
    }
    return summary, audios


def FormatTable(runs: list) -> str:
    lines = [f"{'steps':>5} {'block':>5} {'blocks':>6} {'infer med':>9} {'p95':>7} {'max':>7} {'in-block':>8}  stage medians (ms)"]
    for r in runs:
        stages = " ".join(f"{s}={v['median']:.1f}" for s, v in r["stage_ms"].items())
        i = r["inference_ms"]
        lines.append(f"{r['steps']:>5} {r['block_ms']:>5} {r['blocks']:>6} {i['median']:>9.1f} {i['p95']:>7.1f} {i['max']:>7.1f} {r['within_block']:>3}/{r['blocks']:<4}  {stages}")
    return "\n".join(lines)


def Main(argv=None) -> int:
    args = _ParseArgs(argv)
    out_path = Path(args.out) if args.out else ROOT / "live_run" / ("bench_baseline.json" if args.baseline else "profile.json" if args.profile else "bench.json")
    atexit.register(ReleaseGpuLock)
    waited = AcquireGpuLock()
    janitor = P.Janitor()
    janitor.Add(ReleaseGpuLock)
    try:
        report = _Run(args, janitor, waited)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        print(f"wrote {out_path}", flush=True)
        return 0
    finally:
        janitor.Cleanup()


# ---------------------------------------------------------------- PRIVATE
def _ParseArgs(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", action="store_true", help="write bench_baseline.json and engine/examples/baseline_defaults.wav")
    ap.add_argument("--profile", action="store_true", help="per-stage CPU issue vs GPU busy time and sync census")
    ap.add_argument("--grid", default="", help="comma-separated extra block_ms values, run at 10 steps")
    ap.add_argument("--passes", type=int, default=MEASURED_PASSES, help="measured passes per configuration (default 3)")
    ap.add_argument("--no-graph", action="store_true", help="run the estimator eagerly (no CUDA graph), for a same-conditions comparison")
    ap.add_argument("--out", default="", help="output JSON path")
    return ap.parse_args(argv)


def _Mono(path: Path):
    wav, sr = sf.read(path, dtype="float32")
    return (wav.mean(axis=1) if wav.ndim > 1 else wav), sr


def _Prepare(engine):
    """Load the models, set the target voice (ref_0) and return the source clip (ref_1) at 16 kHz."""
    load = engine.Load()
    voice, voice_sr = _Mono(ROOT / "testdata" / "ref_0.wav")
    engine.SetSlot(0, voice, voice_sr)
    clip, clip_sr = _Mono(ROOT / "testdata" / "ref_1.wav")
    return load, resample_poly(clip, P.INPUT_SR, clip_sr).astype(np.float32)


def _Run(args, janitor, waited: float) -> dict:
    assets = ROOT / "assets" / "seedvc"
    params = P.ParamStore()
    engine = SeedVCExpressive(params, janitor, assets, device="cuda", cuda_graph=not args.no_graph)
    report = {"meta": {"date": dt.datetime.now().isoformat(timespec="seconds"), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                       "lock_waited_s": round(waited, 1), "gpu_clocks_start": GpuClocks()}}
    if args.profile:
        report["profile"] = _Profile(engine, janitor, assets, params)
        return report
    report["meta"]["load"], src16 = _Prepare(engine)
    report["meta"].update({"clip": "testdata/ref_1.wav", "voice": "testdata/ref_0.wav", "clip_s": round(len(src16) / P.INPUT_SR, 2), "passes": args.passes})
    configs = list(STEP_CONFIGS) + [(10, int(b)) for b in args.grid.split(",") if b.strip() and (10, int(b)) not in STEP_CONFIGS]
    runs, audios = [], None
    for steps, block_ms in configs:
        RefreshGpuLock()
        default = (steps, block_ms) == (10, 220)
        seeds = (SEED, SEED, OTHER_SEED) if default else (SEED,) * args.passes
        summary, got = RunConfig(engine, params, src16, steps, block_ms, seeds, keep_audio=default)
        runs.append(summary)
        audios = got if default else audios
        print(FormatTable([summary]).split("\n")[-1], flush=True)
    report["runs"] = runs
    report["verdict"] = Verdict(runs)
    report["equivalence"] = _Equivalence(assets, audios, args.baseline)
    report["meta"]["gpu_clocks_end"] = GpuClocks()
    print(FormatTable(runs), flush=True)
    print(json.dumps({"verdict": report["verdict"], "equivalence": report["equivalence"]}, indent=1), flush=True)
    return report


def _Equivalence(assets, audios, baseline: bool) -> dict:
    """audios = default-configuration outputs of seeds (SEED, SEED, OTHER_SEED)."""
    first, again, other = audios
    metrics = OutputMetrics(assets, torch.device("cuda"))
    out = {
        "determinism_floor_D": {"mel_l1": metrics.MelL1(first, again), "campplus_cos": metrics.Cosine(first, again)},
        "seed_floor_S": {"mel_l1": metrics.MelL1(first, other), "campplus_cos": metrics.Cosine(first, other)},
        "samples": int(len(first)),
        "vs_baseline": None,
    }
    if baseline:
        BASELINE_WAV.parent.mkdir(parents=True, exist_ok=True)
        sf.write(BASELINE_WAV, first, P.MODEL_SR, subtype="FLOAT")
        out["baseline_wav"] = str(BASELINE_WAV)
    else:
        OPTIMISED_WAV.parent.mkdir(parents=True, exist_ok=True)
        sf.write(OPTIMISED_WAV, first, P.MODEL_SR, subtype="FLOAT")
        if BASELINE_WAV.exists():
            ref, _ = sf.read(BASELINE_WAV, dtype="float32")
            out["vs_baseline"] = {"mel_l1": metrics.MelL1(first, ref), "campplus_cos": metrics.Cosine(first, ref), "samples_baseline": int(len(ref))}
    return out


def _Profile(engine, janitor, assets, params) -> dict:
    """Is the diffusion estimator CPU-launch-bound or GPU-bound? One real estimator call, run as bursts of 10 with no sync inside.

    burst_issue_ms: time to queue 10 calls (the CPU's launch work). burst_total_ms: until the GPU has finished them all. If issue is
    close to total the step is launch-bound (a CUDA graph helps); if issue is far below total the GPU is the limit. sync_warnings:
    GPU to CPU syncs inside one call. graph: the call captured once (mask built from the host-known length, so nothing inside
    syncs) and replayed: capture_ok, burst times, and max |diff| against the eager output."""
    import warnings
    _, src16 = _Prepare(engine)
    import modules.diffusion_transformer as dit  # importable only once the engine has loaded (compat.PrepareSeedVc sets the path)
    RunPass(engine, params, src16, 10, 220, SEED)  # warm-up: caches, cuDNN autotune, GPU clocks
    estimator = engine._Components().model.cfm.estimator
    eager, seen = estimator.forward, {}

    def Capture(*args, **kwargs):
        seen.setdefault("args", (args, kwargs))
        return eager(*args, **kwargs)

    estimator.forward = Capture
    RunPass(engine, params, src16, 10, 220, SEED)
    estimator.forward = eager
    args, kwargs = seen["args"]
    out = {"estimator_arg_shapes": [tuple(a.shape) if torch.is_tensor(a) else a for a in args], "x_lens_device": str(args[2].device),
           "gpu_clocks_before": GpuClocks()}

    def Burst(fn, n=10, rounds=7):
        issue, total = [], []
        for _ in range(rounds):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            issue.append((time.perf_counter() - t0) * 1000.0 / n)
            torch.cuda.synchronize()
            total.append((time.perf_counter() - t0) * 1000.0 / n)
        return {"issue_ms_per_call": float(np.median(issue)), "total_ms_per_call": float(np.median(total))}

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        call = lambda: eager(*args, **kwargs)  # noqa: E731
        reference = call().float().clone()
        out["eager"] = Burst(call)
        torch.cuda.set_sync_debug_mode("warn")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call()
        torch.cuda.set_sync_debug_mode("default")
        out["eager"]["sync_warnings"] = len(caught)
        out["eager"]["sync_sites"] = sorted({f"{Path(w.filename).name}:{w.lineno}" for w in caught})[:6]
        graph = {}
        original_mask = dit.sequence_mask
        try:
            length = int(args[2].item()) + 2  # x_lens + style_as_token + time_as_token, known on the host
            gpu_args = (args[0], args[1], args[2].to(args[0].device), *args[3:])
            dit.sequence_mask = lambda lengths, max_length=None: torch.arange(length, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
            gcall = lambda: eager(*gpu_args, **kwargs)  # noqa: E731
            graph_ref = gcall().float().clone()
            graph["mask_patch_max_abs_diff_vs_eager"] = float((graph_ref - reference).abs().max())
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    gcall()
            torch.cuda.current_stream().wait_stream(stream)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_out = gcall()
            g.replay()
            torch.cuda.synchronize()
            graph["capture_ok"] = True
            graph["max_abs_diff_vs_eager"] = float((static_out.float() - reference).abs().max())
            graph["max_abs_reference"] = float(reference.abs().max())
            graph.update(Burst(g.replay))
        except Exception as exc:  # noqa: BLE001 - the failure is the finding
            graph["capture_ok"] = False
            graph["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        finally:
            dit.sequence_mask = original_mask
        out["graph"] = graph
    out["gpu_clocks_after"] = GpuClocks()
    return out


if __name__ == "__main__":
    sys.exit(Main())
