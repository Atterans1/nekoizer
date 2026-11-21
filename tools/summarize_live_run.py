"""Reads live_run/report.json (from app_seedvc.py --headless-test) and prints the numbers the task asks for."""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
rep = json.loads((root / "live_run" / "report.json").read_text(encoding="utf-8"))
f = rep["final"]
pl, out, inp, eng = f.get("pipeline") or {}, f.get("output") or {}, f.get("input") or {}, f.get("engine") or {}
rows = {
    "run_seconds": rep["seconds"],
    "input_mode": inp.get("mode"), "input_device": inp.get("device"), "input_samplerate": inp.get("samplerate"),
    "enhancements_bypassed": inp.get("enhancements_bypassed"), "input_overruns": inp.get("overruns"),
    "output_device": out.get("device"), "output_samplerate": out.get("samplerate"),
    "blocks_total": pl.get("blocks"), "blocks_active": pl.get("active_blocks"), "blocks_silent": pl.get("silent_blocks"),
    "convert_errors": pl.get("convert_errors"),
    "inference_s_mean": pl.get("inference_s_mean"), "last_inference_s": pl.get("last_inference_s"),
    "e2e_ms_last": out.get("e2e_ms_last"), "e2e_ms_mean": out.get("e2e_ms_mean"), "e2e_ms_max": out.get("e2e_ms_max"),
    "output_underruns": out.get("underruns"), "blocks_out": out.get("blocks_out"),
    "balancer": out.get("balancer"), "last_error": f.get("error"),
    "params": {k: f["params"][k] for k in ("block_ms", "overlap_ms", "diffusion_steps", "cfg_rate", "left_context_s")},
}
print(json.dumps(rows, indent=1, default=str))
sys.exit(0)
