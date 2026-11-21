"""HELD-OUT experiment (criteria in preregistered_thresholds_heldout.json, hash-locked before running).

Convert every target/source pair with the shipped defaults, measure, apply the pre-registered criteria.

Refuses to run if preregistered_thresholds.json differs from the hash recorded before any conversion.
"""
import hashlib
import json
import os
import sys
import time
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import params as PARAMS  # noqa: E402
from audio_engine import AudioEngine  # noqa: E402
from vc_inference import VCEngine  # noqa: E402
from resemblyzer import VoiceEncoder  # noqa: E402
from resemblyzer.audio import preprocess_wav  # noqa: E402
import metrics as M  # noqa: E402

EV = os.path.join(ROOT, "evaluation")
TH_PATH = os.path.join(EV, "preregistered_thresholds_heldout.json")
TH_SHA256 = "9cfdf9a7ddedf4ca522ae6c5afd938de140523b07abf4f8fbf512c3f8d16bf1e"
MANIFEST = os.path.join(EV, "corpus", "manifest.json")
CONV = os.path.join(EV, "converted_heldout")
RESULTS_JSON = os.path.join(EV, "results_heldout.json")
os.makedirs(CONV, exist_ok=True)


def Log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def Rs(x, src, dst):
    if src == dst:
        return np.asarray(x, dtype=np.float32)
    g = gcd(src, dst)
    return resample_poly(x, dst // g, src // g).astype(np.float32)


def Sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def Clean(v):
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: Clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [Clean(x) for x in v]
    if isinstance(v, (np.floating,)):
        return Clean(float(v))
    return v


def Main():
    if Sha256(TH_PATH) != TH_SHA256:
        raise SystemExit("preregistered_thresholds.json changed after registration; refusing to run")
    with open(TH_PATH, encoding="utf-8") as fh:
        th = json.load(fh)
    vs, pr = th["voice_similarity"], th["prosody"]
    lo, hi = pr["duration_ratio_converted_over_source"]

    with open(MANIFEST, encoding="utf-8") as fh:
        allc = json.load(fh)["clips"]
    used = {c["rec_id"] for c in allc if c["role"] != "unused"}
    fresh = [c for c in allc if c["role"] == "unused" and c["rec_id"] not in used and not c["dropped_reason"]]
    targets = sorted([c for c in fresh if c["id"].endswith("_target")], key=lambda c: c["id"])
    sources = sorted([c for c in fresh if c["id"].endswith("_source")], key=lambda c: c["id"])
    assert len(targets) == len(sources) == 5, (len(targets), len(sources))
    Log("targets", len(targets), "sources", len(sources))

    pairs = []
    skipped_same_speaker = []
    for ti, t in enumerate(targets):
        for k in (1, 2):
            s = sources[(ti + k) % 5]
            if s["rec_id"] == t["rec_id"]:
                skipped_same_speaker.append((t["id"], s["id"]))
                continue
            pairs.append((t, s))
    Log("pairs", len(pairs), "skipped same speaker", skipped_same_speaker)

    vc = VCEngine()
    vc.Load()
    engine = AudioEngine(vc, PARAMS.Defaults())
    enc = VoiceEncoder("cpu")
    import torch

    def CamEmb(x16):
        with torch.inference_mode():
            t = torch.from_numpy(np.ascontiguousarray(x16)).unsqueeze(0).to(vc._device)
            emb = vc._s3gen.speaker_encoder.inference(t.to(vc._s3gen.dtype))
        return emb.float().reshape(-1).cpu().numpy()

    def Emb(x16):
        return {
            "campplus": CamEmb(x16),
            "resemblyzer": enc.embed_utterance(preprocess_wav(x16, source_sr=16000)),
        }

    cache = {}

    def ClipEmb(c):
        if c["id"] not in cache:
            x16 = Rs(sf.read(os.path.join(ROOT, c["path"]), dtype="float32")[0], 24000, 16000)
            cache[c["id"]] = (x16, Emb(x16))
        return cache[c["id"]]

    rows = []
    for ARM_NAME, ARM_PARAMS in (("A_default", None), ("B_f0_preserve", {"f0_preserve": True})):
     current = None
     for n, (t, s) in enumerate(pairs, 1):
        row = {"arm": ARM_NAME, "n": n, "target": t["id"], "source": s["id"], "target_gender": t["gender_est"], "source_gender": s["gender_est"]}
        try:
            if current != t["id"]:
                info = vc.LoadReference(os.path.join(ROOT, t["path"]))
                current = t["id"]
                Log("reference", t["id"], info)
            tgt16, E_tgt = ClipEmb(t)
            src16, E_src = ClipEmb(s)
            t0 = time.time()
            out24, stats = engine.ProcessOffline(src16.copy(), ARM_PARAMS)
            out_path = os.path.join(CONV, f"{ARM_NAME}__{t['id']}__{s['id']}.wav")
            sf.write(out_path, np.clip(np.asarray(out24, np.float32), -1, 1), 24000, subtype="PCM_16")
            conv24 = sf.read(out_path, dtype="float32")[0]
            conv16 = Rs(conv24, 24000, 16000)
            E_conv = Emb(conv16)

            cos = M.Cos
            sim = {
                "resemblyzer_conv_tgt": cos(E_conv["resemblyzer"], E_tgt["resemblyzer"]),
                "resemblyzer_src_tgt": cos(E_src["resemblyzer"], E_tgt["resemblyzer"]),
                "campplus_conv_tgt": cos(E_conv["campplus"], E_tgt["campplus"]),
                "campplus_src_tgt": cos(E_src["campplus"], E_tgt["campplus"]),
            }
            sim["resemblyzer_gain"] = sim["resemblyzer_conv_tgt"] - sim["resemblyzer_src_tgt"]
            sim["campplus_gain"] = sim["campplus_conv_tgt"] - sim["campplus_src_tgt"]
            pro = M.Prosody(src16, conv16)
            checks = {
                "indep_gain>=0.10": sim["resemblyzer_gain"] >= vs["independent_gain_min"],
                "campplus_conv_tgt>=0.60": sim["campplus_conv_tgt"] >= vs["campplus_cos_converted_target_min"],
                "campplus_gain>=0.15": sim["campplus_gain"] >= vs["campplus_gain_over_source_min"],
                "f0_logcorr>=0.55": pro["f0_logcorr"] >= pr["logf0_pearson_mean_removed_min"],
                "joint_frames>=40": pro["joint_voiced_frames"] >= pr["min_jointly_voiced_frames"],
                "energy_corr>=0.60": pro["energy_logcorr"] >= pr["log_rms_envelope_corr_min"],
                "duration_ratio_in_0.95_1.05": lo <= pro["duration_ratio"] <= hi,
            }
            row.update({
                "passed": all(checks.values()), "checks": checks, "similarity": sim, "prosody": pro,
                "stats": {k: stats[k] for k in stats}, "wall_s": round(time.time() - t0, 2), "output": os.path.relpath(out_path, ROOT).replace(os.sep, "/"),
                "error": None,
            })
        except Exception as exc:
            row.update({"passed": False, "checks": None, "similarity": None, "prosody": None, "error": f"{type(exc).__name__}: {exc}"})
        rows.append(row)
        Log("row", n, t["id"], "->", s["id"], "PASS" if row["passed"] else "FAIL", row.get("error") or "")
        with open(RESULTS_JSON, "w", encoding="utf-8") as fh:
            json.dump(Clean({"thresholds_sha256": TH_SHA256, "skipped_same_speaker": skipped_same_speaker, "rows": rows}), fh, indent=1)
    Log("done", len(rows), "conversions, passed", sum(r["passed"] for r in rows))


if __name__ == "__main__":
    Main()
