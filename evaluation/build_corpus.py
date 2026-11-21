"""Corpus build for the voice-conversion evaluation.

fetch  (CPU): search public-domain audio (candidates cached in corpus/candidates.json), range-fetch the first
       PARTIAL_BYTES of each item (one reader per item), cut one target clip (7-13.5 s) and one source clip
       (5-9 s) from disjoint parts of each recording.
select (GPU): embed every clip (CAMPPlus + resemblyzer), drop a clip whose CAMPPlus cosine to a clip from
       another recording exceeds 0.80, pick 4 targets and 8 sources, write manifest.json + cosine matrices.
"""
import glob
import hashlib
import json
import os
import sys
import time
from math import gcd
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import web_audio as W  # noqa: E402

EV = os.path.join(ROOT, "evaluation")
CORP = os.path.join(EV, "corpus")
CLIPS = os.path.join(CORP, "clips")
RAW = os.path.join(CORP, "raw")
for d in (CLIPS, RAW):
    os.makedirs(d, exist_ok=True)

NEW_RECORDINGS = 16
ATTEMPT_CAP = 40
PARTIAL_BYTES = 6 * 1024 * 1024
DIVERSITY_COS = 0.80
N_TARGETS = 4
N_SOURCES = 8
TARGET_WIN = (7.0, 13.5, 10.0)  # lo, hi, wanted seconds
SOURCE_WIN = (5.0, 9.0, 7.0)
PAD_S = 0.15
MIN_SPEECH_RATIO = 0.6
PHASE = sys.argv[1] if len(sys.argv) > 1 else ""

QUERIES = [
    ("narration", "librivox male narration audiobook chapter"),
    ("narration", "librivox female narration audiobook chapter"),
    ("poetry", "librivox poetry reading poems"),
    ("dialogue", "librivox novel dialogue reading"),
    ("children", "librivox children stories reading"),
    ("essay", "librivox essays reading"),
    ("narration", "librivox short stories reading"),
    ("dialogue", "librivox fiction characters reading"),
    ("poetry", "librivox sonnets verse"),
    ("narration", "librivox history reading chapter"),
]


def Log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def Rs(x, src, dst):
    if src == dst:
        return x.astype(np.float32, copy=False)
    g = gcd(src, dst)
    return resample_poly(x, dst // g, src // g).astype(np.float32)


def ItemKey(cand):
    cid = cand["id"]
    return cid.split("/")[0] if cid.startswith("archive:") else cid


def RecId(cand):
    return hashlib.sha1(ItemKey(cand).encode("utf-8")).hexdigest()[:12]


def Stem(cand):
    return hashlib.sha1(cand["id"].encode("utf-8")).hexdigest()[:16]


def SearchOrder():
    batches = []
    for label, query in QUERIES:
        try:
            found = W.SearchAudio(query, limit=10)
        except Exception as exc:
            Log("search failed", query, type(exc).__name__, str(exc)[:160])
            continue
        Log("search", label, "|", query, "->", len(found))
        batches.append([dict(c, style=label, query=query) for c in found])
    order, seen = [], set()
    for k in range(10):
        for batch in batches:
            if k < len(batch) and ItemKey(batch[k]) not in seen:
                seen.add(ItemKey(batch[k]))
                order.append(batch[k])
    return order


def FetchPartial(cand, dest_dir, max_bytes):
    """Range-request the first max_bytes of a candidate (a whole LibriVox book is 30-60 MB; a few minutes is enough)."""
    import requests

    ext = os.path.splitext(urlparse(cand["url"]).path)[1].lower() or ".mp3"
    path = os.path.join(dest_dir, Stem(cand) + "_partial" + ext)
    got = 0
    headers = {"User-Agent": W.USER_AGENT, "Range": f"bytes=0-{max_bytes - 1}"}
    with requests.get(cand["url"], headers=headers, stream=True, timeout=W.HTTP_TIMEOUT_S) as resp:
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if not ("audio" in ctype or "octet" in ctype or "ogg" in ctype):
            raise ValueError(f"not audio: {ctype}")
        with open(path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=256 * 1024):
                if not chunk:
                    continue
                room = max_bytes - got
                fh.write(chunk[:room])
                got += min(len(chunk), room)
                if got >= max_bytes:
                    break
    if got == 0:
        raise ValueError("empty download")
    return path, got


def Decode(path):
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    except RuntimeError:
        return W._DecodeMono(path)
    return data.mean(axis=1).astype(np.float32), sr


def PickWindow(segs, lo, hi, want, dur, first, stop):
    """Contiguous run of VAD segments in [first, stop) with padded span in [lo, hi], speech ratio >= MIN_SPEECH_RATIO, closest to want."""
    best = None
    for i in range(first, stop):
        speech = 0.0
        for j in range(i, stop):
            speech += segs[j][1] - segs[j][0]
            raw_span = segs[j][1] - segs[i][0]
            if raw_span > hi + 2 * PAD_S:
                break
            start = max(0.0, segs[i][0] - PAD_S)
            end = min(dur, segs[j][1] + PAD_S)
            span = end - start
            if not lo <= span <= hi or speech / raw_span < MIN_SPEECH_RATIO:
                continue
            score = abs(span - want)
            if best is None or score < best[0]:
                best = (score, start, end)
    return None if best is None else (best[1], best[2])


def Cut(x24, start, end):
    a = int(round(start * 24000))
    b = int(round(end * 24000))
    clip = x24[a:b].copy()
    fade = int(0.01 * 24000)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    clip[:fade] *= ramp
    clip[-fade:] *= ramp[::-1]
    return clip


def CheckClip(name, clip):
    frac = float(np.mean(np.abs(clip) >= 0.99))
    if frac > 0.001:
        raise ValueError(f"{name} clipping fraction {frac:.4f}")
    if float(np.sqrt(np.mean(clip ** 2))) < 1e-3:
        raise ValueError(f"{name} is silent")


def Fetch():
    cache = os.path.join(CORP, "candidates.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as fh:
            order = json.load(fh)
        Log("candidates from cache", len(order))
    else:
        order = SearchOrder()
        with open(cache, "w", encoding="utf-8") as fh:
            json.dump(order, fh, indent=1)
    Log("candidate items", len(order))

    done = attempts = 0
    spent = 0
    for cand in order:
        if done >= NEW_RECORDINGS or attempts >= ATTEMPT_CAP:
            break
        rid = RecId(cand)
        meta_path = os.path.join(CORP, f"rec_{rid}.json")
        if os.path.exists(meta_path):
            done += 1
            continue
        attempts += 1
        try:
            path, got = FetchPartial(cand, RAW, PARTIAL_BYTES)
            spent += got
            mono, sr = Decode(path)
            dur = mono.size / sr
            if dur < 60:
                Log("skip short", rid, round(dur, 1))
                continue
            x16 = Rs(mono, sr, 16000)
            x24 = Rs(mono, sr, 24000)
            segs = W._SpeechSegments(x16)
            n = len(segs)
            if n == 0:
                Log("skip no speech", rid)
                continue
            t = PickWindow(segs, *TARGET_WIN, dur, 0, n)
            if t is None:
                Log("skip no target window", rid)
                continue
            ts, te = t
            after = next((k for k, s in enumerate(segs) if s[0] >= te), n)
            before = max((k for k, s in enumerate(segs) if s[1] <= ts), default=-1) + 1
            s_win = PickWindow(segs, *SOURCE_WIN, dur, after, n) if after < n else None
            if s_win is None:
                s_win = PickWindow(segs, *SOURCE_WIN, dur, 0, before)
            if s_win is None:
                Log("skip no source window", rid)
                continue
            ss, se = s_win
            tc, sc = Cut(x24, ts, te), Cut(x24, ss, se)
            CheckClip("target", tc)
            CheckClip("source", sc)
            tpath = os.path.join(CLIPS, f"{rid}_target.wav")
            spath = os.path.join(CLIPS, f"{rid}_source.wav")
            sf.write(tpath, np.clip(tc, -1, 1), 24000, subtype="PCM_16")
            sf.write(spath, np.clip(sc, -1, 1), 24000, subtype="PCM_16")
            meta = {
                "rec_id": rid, "fetch_index": done, "title": cand.get("title"), "item": ItemKey(cand),
                "candidate_id": cand["id"], "source_url": cand["url"], "page_url": cand.get("page_url"),
                "license": cand.get("license"), "style_query": cand["style"], "query": cand["query"],
                "fetched_bytes": got, "fetched_audio_s": round(dur, 2), "speech_s": round(sum(e - s for s, e in segs), 2),
                "target": {"path": tpath, "start_s": round(ts, 2), "end_s": round(te, 2)},
                "source": {"path": spath, "start_s": round(ss, 2), "end_s": round(se, 2)},
            }
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=1)
            done += 1
            Log("ok", rid, cand["style"], "target", round(te - ts, 2), "source", round(se - ss, 2), "audio_s", round(dur, 1))
        except Exception as exc:
            Log("FAIL", cand["id"][:90], type(exc).__name__, str(exc)[:200])
    Log("fetch done: recordings", done, "attempts", attempts, "MB", round(spent / 2 ** 20, 1))


def Cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def Select():
    import torch
    from resemblyzer import VoiceEncoder
    from resemblyzer.audio import preprocess_wav
    from vc_inference import VCEngine

    clips = []
    for name in ("ref_0", "ref_1"):
        clips.append({
            "clip_id": name, "rec_id": name, "role_pool": "reference", "path": os.path.join(ROOT, "testdata", name + ".wav"),
            "source_url": "testdata (earlier web_audio extraction; mp3 to URL mapping not recorded)",
            "page_url": None, "license": "Internet Archive item, licence not recorded (LibriVox/public-domain search)",
            "title": name, "style_query": "unrecorded", "window_s": None, "fetch_index": -1,
        })
    metas = []
    for path in glob.glob(os.path.join(CORP, "rec_*.json")):
        with open(path, encoding="utf-8") as fh:
            metas.append(json.load(fh))
    metas.sort(key=lambda m: m["fetch_index"])
    for meta in metas:
        base = {k: meta[k] for k in ("rec_id", "title", "source_url", "page_url", "license", "style_query", "fetch_index")}
        for role in ("target", "source"):
            clips.append({**base, "clip_id": f"{meta['rec_id']}_{role}", "role_pool": role,
                          "path": meta[role]["path"], "window_s": [meta[role]["start_s"], meta[role]["end_s"]]})
    Log("clips to embed", len(clips))

    vc = VCEngine()
    vc.Load()
    enc = VoiceEncoder("cpu")
    for c in clips:
        wav24, sr = sf.read(c["path"], dtype="float32")
        x16 = Rs(wav24, sr, 16000)
        c["duration_s"] = round(len(x16) / 16000, 2)
        with torch.inference_mode():
            t = torch.from_numpy(x16).unsqueeze(0).to(vc._device)
            emb = vc._s3gen.speaker_encoder.inference(t.to(vc._s3gen.dtype))
        c["campplus"] = emb.float().reshape(-1).cpu().numpy()
        c["resemblyzer"] = enc.embed_utterance(preprocess_wav(x16, source_sr=16000))
        f0, vflag, _ = librosa.pyin(x16, fmin=60, fmax=500, sr=16000, hop_length=320)
        voiced = f0[vflag & np.isfinite(f0)]
        if voiced.size:
            c["median_f0_hz"] = round(float(np.median(voiced)), 1)
            c["gender_est"] = "male-range" if np.median(voiced) < 165 else "female-range"
        else:
            c["median_f0_hz"] = None
            c["gender_est"] = "unknown"
        Log("embedded", c["clip_id"], c["duration_s"], "s f0", c["median_f0_hz"], c["gender_est"])

    ids = [c["clip_id"] for c in clips]
    mat_cp = [[Cos(a["campplus"], b["campplus"]) for b in clips] for a in clips]
    mat_rz = [[Cos(a["resemblyzer"], b["resemblyzer"]) for b in clips] for a in clips]
    with open(os.path.join(CORP, "cosine_matrix_all.json"), "w", encoding="utf-8") as fh:
        json.dump({"ids": ids, "campplus": mat_cp, "resemblyzer": mat_rz}, fh, indent=1)

    kept_idx = []
    for i, c in enumerate(clips):
        clash = [(mat_cp[i][j], clips[j]["clip_id"]) for j in kept_idx
                 if clips[j]["rec_id"] != c["rec_id"] and mat_cp[i][j] > DIVERSITY_COS]
        if clash:
            c["dropped_reason"] = f"CAMPPlus cos {max(clash)[0]:.3f} to {max(clash)[1]} (different recording, > {DIVERSITY_COS})"
        else:
            c["dropped_reason"] = None
            kept_idx.append(i)
    kept = [clips[i] for i in kept_idx]
    for i, c in enumerate(clips):
        others = [mat_cp[i][j] for j in range(len(clips)) if clips[j]["rec_id"] != c["rec_id"]]
        c["max_campplus_cos_other_recording"] = round(max(others), 4) if others else None
    Log("kept", len(kept), "dropped", [c["clip_id"] for c in clips if c["dropped_reason"]])

    targets = [c for c in kept if c["role_pool"] == "reference"]
    tpool = [c for c in kept if c["role_pool"] == "target"]
    while len(targets) < N_TARGETS and tpool:
        pick = min(tpool, key=lambda c: (sum(1 for t in targets if t["gender_est"] == c["gender_est"]),
                                         max([Cos(c["campplus"], t["campplus"]) for t in targets] or [0.0])))
        targets.append(pick)
        tpool = [c for c in tpool if c is not pick]
    target_recs = {t["rec_id"] for t in targets}
    sources = []
    spool = [c for c in kept if c["role_pool"] == "source" and c["rec_id"] not in target_recs]
    while len(sources) < N_SOURCES and spool:
        pick = min(spool, key=lambda c: (sum(1 for s in sources if s["gender_est"] == c["gender_est"]),
                                         sum(1 for s in sources if s["style_query"] == c["style_query"])))
        sources.append(pick)
        spool = [c for c in spool if c is not pick]

    chosen = {c["clip_id"]: "target" for c in targets}
    chosen.update({c["clip_id"]: "source" for c in sources})
    out = []
    for c in clips:
        out.append({
            "id": c["clip_id"], "role": chosen.get(c["clip_id"], "unused"), "rec_id": c["rec_id"],
            "path": os.path.relpath(c["path"], ROOT).replace(os.sep, "/"),
            "duration_s": c["duration_s"], "gender_est": c["gender_est"], "median_f0_hz": c["median_f0_hz"],
            "style": c["style_query"], "title": c.get("title"), "source_url": c.get("source_url"), "page_url": c.get("page_url"),
            "licence": c.get("license"), "window_s": c.get("window_s"),
            "max_campplus_cos_other_recording": c["max_campplus_cos_other_recording"],
            "dropped_reason": c["dropped_reason"], "fetch_index": c.get("fetch_index"),
        })
    with open(os.path.join(CORP, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"diversity_campplus_cos_max": DIVERSITY_COS, "n_targets": len(targets),
                   "n_sources": len(sources), "clips": out}, fh, indent=1)
    Log("targets", [c["clip_id"] for c in targets], "sources", [c["clip_id"] for c in sources])
    Log("target genders", [c["gender_est"] for c in targets], "source genders", [c["gender_est"] for c in sources])


if __name__ == "__main__":
    if PHASE == "fetch":
        Fetch()
    elif PHASE == "select":
        Select()
    else:
        raise SystemExit("usage: build_corpus.py fetch|select")
