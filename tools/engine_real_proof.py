"""Real-model proof: streams R/testdata/ref_1.wav to the ref_0.wav voice through SeedVCExpressive + Sola at the spec defaults.
Writes live_run/engine_proof.json and engine/examples/*.wav. No mocks."""
import json, os, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
import seedvc_params as P
from audio.sola import Sola
from engine.seedvc_expressive import SeedVCExpressive

out = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
def Load16k(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1: x = x.mean(axis=1)
    return x, sr
ref, ref_sr = Load16k(ROOT / "testdata" / "ref_0.wav")
src, src_sr = Load16k(ROOT / "testdata" / "ref_1.wav")
src16 = resample_poly(src, 16000, src_sr).astype(np.float32) if src_sr != 16000 else src

params = P.ParamStore(); jan = P.Janitor()
t0 = time.perf_counter()
eng = SeedVCExpressive(params, jan, ROOT / "assets" / "seedvc", device="cuda")
out["load"] = eng.Load(); out["load_wall_s"] = round(time.perf_counter() - t0, 1)
out["slot"] = eng.SetSlot(0, ref, ref_sr)
out["defaults"] = {k: params.Get(k) for k in ("block_ms", "overlap_ms", "diffusion_steps", "cfg_rate", "left_context_s")}

def Stream(steps=None, max_blocks=None, save=None):
    eng.ResetStream()
    bm, om = params.Get("block_ms"), params.Get("overlap_ms")
    n = int(round(bm * P.INPUT_SR / 1000))
    sola = Sola(P.MODEL_SR, bm, om, P.SOLA_SEARCH_MS)
    times, stage, outs, lens, nan = [], [], [], set(), False
    for i, s in enumerate(range(0, len(src16) - n + 1, n)):
        if max_blocks and i >= max_blocks: break
        r = eng.ConvertBlock(src16[s:s + n], True, steps=steps)
        lens.add(int(r.wav22k.size)); nan = nan or bool(np.isnan(r.wav22k).any())
        times.append(r.inference_s); stage.append(dict(eng.Telemetry().get("stage_ms", {})))
        outs.append(sola.Process(r.wav22k))
    audio = np.concatenate(outs)
    if save: sf.write(save, audio, P.MODEL_SR)
    t = np.array(times); w = t[1:] if len(t) > 1 else t   # block 0 includes CUDA warm-up
    return {"blocks": len(t), "steps_used": r.steps_used, "first_block_s": round(float(t[0]), 3),
            "median_s": round(float(np.median(w)), 3), "p95_s": round(float(np.percentile(w, 95)), 3), "max_s": round(float(w.max()), 3),
            "real_time_factor_median": round(float(np.median(w)) / (bm / 1000), 2), "within_220ms": int((w < bm / 1000).sum()), "of": int(len(w)),
            "wav22k_lengths": sorted(lens), "has_nan": nan, "out_seconds": round(len(audio) / P.MODEL_SR, 2),
            "out_peak": round(float(np.abs(audio).max()), 3), "out_rms": round(float(np.sqrt((audio ** 2).mean())), 4),
            "stage_ms_last": stage[-1], "vram_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20)}

out["default_10_steps"] = Stream(None, None, str(ROOT / "engine" / "examples" / "streamed_defaults.wav"))
Path(ROOT / "live_run" / "engine_proof.json").write_text(json.dumps(out, indent=1, default=str))
out["steps_6"] = Stream(6, 14); out["steps_16"] = Stream(16, 14)
params.Set("pitch_shift_st", 3.0)
r = Stream(None, 6); out["pitch_plus3_applied"] = r
params.Set("pitch_shift_st", 0.0)
Path(ROOT / "live_run" / "engine_proof.json").write_text(json.dumps(out, indent=1, default=str))
jan.Cleanup()
print("PROOF DONE")
