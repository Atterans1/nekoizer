"""Real-engine measurements of the RVC engine and of one slot bank that holds reference clips and RVC voices.

  python tools/rvc_bench.py switch [--budget-mb N] [--out PATH]
      clip and RVC voices in one bank, switched through Pipeline: engine per block, swap latency, VRAM per switch, auto-mode isolation;
      --budget-mb makes the memory policy see a card with only N MB for this process, so it has to release the idle engine
  python tools/rvc_bench.py bench [--voice NAME] [--wav PATH] [--out PATH]
      per-block latency of the RVC engine at the default block size (median, p95, max, per stage) and the converted testdata/ref_1.wav;
      NAME is an imported voice, a random voice of the real architecture when omitted

The shared GPU lock (live_run/gpu.lock) is taken before any model is loaded and released on every exit path.
"""
from __future__ import annotations

import argparse
import atexit
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

import seedvc_params as P  # noqa: E402
from audio.sola import Sola  # noqa: E402
from engine import rvc_store, rvc_synth, rvc_weights  # noqa: E402
from engine.rvc_engine import RvcEngine  # noqa: E402
from engine.seedvc_expressive import SeedVCExpressive  # noqa: E402
from pipeline import Pipeline  # noqa: E402
from tools.bench_engine import AcquireGpuLock, GpuClocks, RefreshGpuLock, ReleaseGpuLock, Summary  # noqa: E402

# ---------------------------------------------------------------- PUBLIC
LOCK_OWNER = "RvcFinish"
MEASURED_PASSES = 3
PHASE_BLOCKS = 8  # blocks converted after the first, switching, block of a phase
BANK = {0: "clip ref_0", 1: "rvc random v2 40k", 2: "clip evaluation target", 3: "rvc random v1 48k"}
PHASES = (0, 1, 2, 3, 1, 0, 2)  # the first is the warm-up on the clip engine (excluded from the switch table)
# upsample rates and kernel sizes of the standard 192-channel RVC voice, per output sample rate
ARCHITECTURE = {40000: ([10, 10, 2, 2], [16, 16, 4, 4]), 48000: ([12, 10, 2, 2], [24, 20, 4, 4])}


def SaveRandomVoice(path: Path, version: str = "v2", sr: int = 40000, seed: int = 0) -> Path:
    """An exported .pth with the architecture and size of a real RVC voice and random weights (timing and memory do not depend on the values)."""
    rates, kernels = ARCHITECTURE[sr]
    config = [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], rates, 512, kernels, 109, 256, sr]
    torch.manual_seed(seed)
    net = rvc_synth.NewSynthesizer(config, version, True)
    weight = {k: v.half() for k, v in net.state_dict().items() if not k.startswith("enc_q.")}
    torch.save({"config": config, "weight": weight, "info": "random weights, real architecture", "sr": f"{sr // 1000}k", "f0": 1, "version": version}, path)
    return path


def RunSwitch(report: dict, budget_mb: float) -> None:
    """Slots 0 and 2 hold reference clips, slots 1 (v2 40k) and 3 (v1 48k) random voices. Every block goes through Pipeline.ProcessOffline."""
    if budget_mb:
        _LimitRoom(budget_mb)
    real_weights = rvc_weights.WeightsDir()
    work = Path(tempfile.mkdtemp(prefix="rvc_switch_"))
    os.environ["S2S_DATA_DIR"] = str(work)  # a hermetic voice store: the user's imported voices stay untouched
    params, janitor = P.ParamStore(), P.Janitor()
    assets = ROOT / "assets" / "seedvc"
    clip = SeedVCExpressive(params, janitor, assets, device="cuda")
    rvc = RvcEngine(params, janitor, device="cuda", weights_dir=real_weights)
    pipe = Pipeline(assets, params=params, device="cuda", engines=(clip, rvc))
    calls = _CountCalls({"clip": clip, "rvc": rvc})
    try:
        src16 = np.tile(_Clip16(ROOT / "testdata" / "ref_1.wav"), 12)
        voice_a = rvc_store.ImportPth(SaveRandomVoice(work / "a.pth", "v2", 40000, 1), name="RandomV2")["name"]
        voice_b = rvc_store.ImportPth(SaveRandomVoice(work / "b.pth", "v1", 48000, 2), name="RandomV1")["name"]
        report["voices"] = rvc_store.List()
        report["rvc_prepare"] = _WaitReady(rvc)
        started = time.perf_counter()
        pipe._host.LoadClip()  # what Pipeline.Start() does first
        report["app_start_clip_load"] = {"seconds": round(time.perf_counter() - started, 2), "vram": _Vram(), "engine": pipe.Telemetry()["engine"]}
        clip_a, sr_a = _Mono(ROOT / "testdata" / "ref_0.wav")
        clip_b, sr_b = _Mono(ROOT / "evaluation" / "corpus" / "clips" / "5661cb5f98bf_target.wav")
        report["slots"] = [pipe.SetSlotFromWav(0, clip_a, sr_a), pipe.SetSlotFromRvc(1, voice_a), pipe.SetSlotFromWav(2, clip_b, sr_b), pipe.SetSlotFromRvc(3, voice_b)]
        report["slot_info"] = pipe.SlotInfo()
        report["phases"] = []
        for index, slot in enumerate(PHASES):
            RefreshGpuLock(owner=LOCK_OWNER)
            row = _Phase(pipe, calls, slot, src16, index * (PHASE_BLOCKS + 1) * _BlockSamples(params))
            row["label"] = "warm-up" if index == 0 else BANK[slot]
            report["phases"].append(row)
            if row["slot_kind"] == "rvc":
                report.setdefault("rvc_pitch_tracker", rvc.Telemetry()["pitch"])
            print(json.dumps({k: row[k] for k in ("label", "slot", "first_block", "switch", "resident", "engine_vram_mb", "vram_after_switch")}, default=str), flush=True)
        report["auto_mode"] = _AutoMode(pipe, params, calls, clip, rvc, src16, (clip_a, sr_a))
        print(json.dumps(report["auto_mode"], default=str), flush=True)
    finally:
        pipe.janitor.Cleanup()
        janitor.Cleanup()
        shutil.rmtree(work, ignore_errors=True)


def RunBench(report: dict, voice_name: str | None, wav_out: Path | None) -> None:
    """Per-block latency of the RVC engine alone at the default block size, and the converted clip."""
    real_weights = rvc_weights.WeightsDir()
    work = None
    if voice_name is None:
        work = Path(tempfile.mkdtemp(prefix="rvc_bench_"))
        os.environ["S2S_DATA_DIR"] = str(work)
        voice_name = rvc_store.ImportPth(SaveRandomVoice(work / "random.pth", "v2", 40000, 0), name="RandomV2")["name"]
    params, janitor = P.ParamStore(), P.Janitor()
    engine = RvcEngine(params, janitor, device="cuda", weights_dir=real_weights)
    try:
        report["meta"]["load"] = engine.Load()
        report["meta"]["voice"] = engine.LoadVoice(rvc_store.Get(voice_name) | {"name": voice_name})
        report["meta"]["voice_load_s"] = engine.Telemetry()["voice_load_s"]
        report["meta"]["pitch"] = engine.Telemetry()["pitch"]
        src16 = _Clip16(ROOT / "testdata" / "ref_1.wav")
        block_ms = params.Get("block_ms")
        _RunPass(engine, params, src16, keep_audio=False)  # warm-up: cudnn autotune, GPU clocks, caches
        torch.cuda.reset_peak_memory_stats()
        rows, audio = [], None
        for pass_index in range(MEASURED_PASSES):
            RefreshGpuLock(owner=LOCK_OWNER)
            got, kept = _RunPass(engine, params, src16, keep_audio=pass_index == 0)
            rows += got
            audio = kept if kept is not None else audio
        inference = [r["inference_ms"] for r in rows]
        stats = Summary(inference)
        report["bench"] = {
            "block_ms": block_ms, "blocks": len(rows), "clip_s": round(len(src16) / P.INPUT_SR, 2), "inference_ms": stats,
            "p95_over_block": stats["p95"] / block_ms, "within_block": int(sum(v < block_ms for v in inference)),
            "realtime_by_the_0_9_rule": bool(stats["p95"] <= 0.9 * block_ms),
            "stage_ms": {s: Summary([r[f"{s}_ms"] for r in rows]) for s in ("features", "synth", "post")},
            "vram_mb": {"allocated_peak": round(torch.cuda.max_memory_allocated() / 2**20, 1), "reserved_peak": round(torch.cuda.max_memory_reserved() / 2**20, 1)},
            "output_samples": int(len(audio)), "output_rms_dbfs": float(20 * np.log10(np.sqrt(np.mean(audio.astype(np.float64) ** 2)) + 1e-12)),
            "output_finite": bool(np.isfinite(audio).all()),
        }
        if wav_out is not None:
            wav_out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(wav_out, audio, P.MODEL_SR, subtype="FLOAT")
            report["bench"]["wav"] = str(wav_out)
        print(json.dumps(report["bench"], indent=1), flush=True)
    finally:
        janitor.Cleanup()
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)


def Main(argv=None) -> int:
    args = _ParseArgs(argv)
    atexit.register(ReleaseGpuLock, owner=LOCK_OWNER)
    waited = AcquireGpuLock(owner=LOCK_OWNER)
    report: dict = {"meta": _Meta(waited)}
    out = Path(args.out) if args.out else ROOT / "live_run" / f"rvc_{args.mode}.json"
    try:
        report["meta"]["budget_mb"] = args.budget_mb or None
        if args.mode == "switch":
            RunSwitch(report, args.budget_mb)
        else:
            RunBench(report, args.voice, Path(args.wav) if args.wav else None)
        return 0
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["meta"]["gpu_clocks_end"] = GpuClocks()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        print(f"wrote {out}", flush=True)
        ReleaseGpuLock(owner=LOCK_OWNER)


# ---------------------------------------------------------------- PRIVATE
def _ParseArgs(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("switch", "bench"))
    ap.add_argument("--budget-mb", type=float, default=0.0, help="switch: room the memory policy believes this process has (0: the driver's real free-memory reading)")
    ap.add_argument("--voice", default=None, help="bench: the imported voice to run (default: a random voice of the real architecture)")
    ap.add_argument("--wav", default="", help="bench: write the converted testdata/ref_1.wav here (22.05 kHz float)")
    ap.add_argument("--out", default="", help="output JSON path (default live_run/rvc_<mode>.json)")
    return ap.parse_args(argv)


def _Meta(waited: float) -> dict:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10).stdout.strip()
    return {"date": dt.datetime.now().isoformat(timespec="seconds"), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            "lock_waited_s": round(waited, 1), "gpu_clocks_start": GpuClocks(), "gpu_utilization_memory_used_total": smi,
            "note": "the GPU is shared with whatever else runs on it: memory.used above includes other processes"}


def _Mono(path: Path):
    wav, sr = sf.read(path, dtype="float32")
    return (wav.mean(axis=1) if wav.ndim > 1 else wav), sr


def _Clip16(path: Path) -> np.ndarray:
    wav, sr = _Mono(path)
    return resample_poly(wav, P.INPUT_SR, sr).astype(np.float32)


def _BlockSamples(params) -> int:
    return int(round(params.Get("block_ms") * P.INPUT_SR / 1000.0))


def _Vram() -> dict:
    """This process's allocator counters, and what the driver says about the whole card (other programs included)."""
    free, total = torch.cuda.mem_get_info()
    smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10).stdout.strip().split(",")
    return {"allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 1), "reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 1),
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1), "driver_free_mb": round(free / 2**20),
            "total_mb": round(total / 2**20), "card_used_mb": int(smi[0]), "card_utilization_pct": int(smi[1])}


def _WaitReady(rvc: RvcEngine, limit_s: float = 300.0) -> dict:
    """The first Prepare() checks the pinned HuBERT and RMVPE files in the background: wait for it so no phase measures that check."""
    started = time.perf_counter()
    state = rvc.Prepare()
    while state["state"] != "ready" and time.perf_counter() - started < limit_s:
        time.sleep(0.25)
        state = rvc.Prepare()
    return {"state": state["state"], "seconds": round(time.perf_counter() - started, 2)}


def _LimitRoom(budget_mb: float) -> None:
    """Make the host's free-memory reading budget minus what this process has reserved, as on a card with only budget_mb for it."""
    from engine.engine_host import EngineHost
    EngineHost._FreeMb = lambda self: budget_mb - torch.cuda.memory_reserved() / 2**20


def _CountCalls(engines: dict) -> Counter:
    """Count the ConvertBlock calls each engine receives, so every phase can show which engine converted its blocks."""
    counts: Counter = Counter()
    for key, engine in engines.items():
        original = engine.ConvertBlock

        def counted(*args, _original=original, _key=key, **kwargs):
            counts[_key] += 1
            return _original(*args, **kwargs)

        engine.ConvertBlock = counted
    return counts


def _Phase(pipe, calls: Counter, slot: int, src16: np.ndarray, offset: int) -> dict:
    block = _BlockSamples(pipe.params)
    switches_before, calls_before = pipe.Telemetry()["engine"]["switches"], dict(calls)
    torch.cuda.reset_peak_memory_stats()
    pipe.SetManualSlot(slot)
    started = time.perf_counter()
    _, first = pipe.ProcessOffline(src16[offset: offset + block])  # the engine switch happens inside this block
    first_ms = (time.perf_counter() - started) * 1000.0
    engine = pipe.Telemetry()["engine"]
    vram = _Vram()
    _, steady = pipe.ProcessOffline(src16[offset + block: offset + block * (1 + PHASE_BLOCKS)])
    return {
        "slot": slot, "slot_kind": pipe.SlotInfo()[slot]["kind"],
        "first_block": {"kinds": first["kinds"], "wall_ms": round(first_ms, 1), "inference_ms": round(first["inference_s_median"] * 1000.0, 1)},
        "switch": engine["last_switch"] if engine["switches"] > switches_before else None,
        "resident": engine["resident"], "engine_vram_mb": {k: round(v, 1) for k, v in engine["vram_mb"].items()}, "vram_after_switch": vram,
        "steady": {"kinds": dict(Counter(steady["kinds"])), "inference_ms_median": round(steady["inference_s_median"] * 1000.0, 1),
                   "inference_ms_p95": round(steady["inference_s_p95"] * 1000.0, 1)},
        "convert_calls": {k: calls[k] - calls_before.get(k, 0) for k in ("clip", "rvc")},
    }


def _AutoMode(pipe, params, calls: Counter, clip, rvc, src16: np.ndarray, clip_a) -> list:
    """Auto (emotion) mode: reference clips only. With no clip left the RVC voice answers; a clip back in a slot takes over again."""
    params.Set("emotion_mode", "auto")
    params.Set("auto_balancer", False)
    block = _BlockSamples(params)
    rows = []

    def Run(label: str) -> None:
        before = dict(calls)
        _, stats = pipe.ProcessOffline(src16[: block * 4])
        voice = rvc.Voice()
        rows.append({"step": label, "kinds": dict(Counter(stats["kinds"])), "convert_calls": {k: calls[k] - before.get(k, 0) for k in ("clip", "rvc")},
                     "seed_vc_bank_loaded_slots": [r["loaded"] for r in clip.SlotInfo()], "rvc_voice": None if voice is None else voice["name"],
                     "slots": [r["kind"] for r in pipe.SlotInfo()]})

    Run("auto, clips in slots 0 and 2, RVC voices in 1 and 3")
    pipe.ClearSlot(0)
    pipe.ClearSlot(2)
    Run("auto, both clips cleared")
    pipe.SetSlotFromWav(0, clip_a[0], clip_a[1])
    Run("auto, a clip back in slot 0")
    return rows


def _RunPass(engine, params, src16: np.ndarray, keep_audio: bool):
    block_ms = params.Get("block_ms")
    n = int(round(block_ms * P.INPUT_SR / 1000.0))
    engine.ResetStream()
    sola = Sola(P.MODEL_SR, block_ms, params.Get("overlap_ms"), P.SOLA_SEARCH_MS) if keep_audio else None
    rows, outs = [], []
    for start in range(0, len(src16) - n + 1, n):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = engine.ConvertBlock(src16[start: start + n], True)
        wall = (time.perf_counter() - started) * 1000.0
        tel = engine.Telemetry()
        rows.append({"inference_ms": wall, "features_ms": tel["features_ms"], "synth_ms": tel["synth_ms"], "post_ms": tel["post_ms"]})
        if sola is not None:
            outs.append(sola.Process(result.wav22k))
    return rows, (np.concatenate(outs) if outs else None)


if __name__ == "__main__":
    sys.exit(Main())
