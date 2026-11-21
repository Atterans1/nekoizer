import sys, time, json, importlib.util
t0 = time.time()
R = r"C:/Users/jeffr/s2s-voicechanger"
out = {}
for name in ["numpy", "scipy", "librosa", "soundfile", "pyworld", "resemblyzer", "webrtcvad", "transformers", "torch", "torchaudio", "torchfcpe", "onnxruntime"]:
    try:
        m = __import__(name)
        out[name] = getattr(m, "__version__", "ok")
    except Exception as e:
        out[name] = "FAIL " + repr(e)[:200]
print(json.dumps(out), flush=True)
import numpy as np, soundfile as sf, torch
print("cuda", torch.cuda.is_available(), "threads", torch.get_num_threads(), flush=True)
# legacy metrics by path
spec = importlib.util.spec_from_file_location("legacy_eval_metrics", R + "/evaluation/metrics.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
print("legacy metrics funcs", [n for n in dir(M) if n[0].isupper()], flush=True)
# resemblyzer
from resemblyzer import VoiceEncoder
from resemblyzer.audio import preprocess_wav
enc = VoiceEncoder("cpu")
x, sr = sf.read(R + "/testdata/ref_0.wav", dtype="float32")
print("ref_0 sr", sr, "len", len(x), flush=True)
from scipy.signal import resample_poly
x16 = resample_poly(x, 2, 3).astype(np.float32) if sr == 24000 else x
e = enc.embed_utterance(preprocess_wav(x16, source_sr=16000))
print("resemblyzer emb", e.shape, float(np.linalg.norm(e)), flush=True)
# CAMPPlus (CPU)
sys.path.insert(0, R + "/third_party/seed-vc")
import torchaudio.compliance.kaldi as kaldi
from modules.campplus.DTDNN import CAMPPlus
cam = CAMPPlus(feat_dim=80, embedding_size=192)
cam.load_state_dict(torch.load(R + "/assets/seedvc/campplus_cn_common.bin", map_location="cpu"))
cam.eval()
with torch.no_grad():
    w = torch.from_numpy(x16).unsqueeze(0)
    feat = kaldi.fbank(w, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    emb = cam(feat.unsqueeze(0))
print("campplus emb", tuple(emb.shape), flush=True)
# corpus clip format
import json as _j
mf = _j.load(open(R + "/evaluation/corpus/manifest.json", encoding="utf-8"))
c = [c for c in mf["clips"] if c["role"] == "source"][0]
y, sr2 = sf.read(R + "/" + c["path"], dtype="float32")
print("source clip", c["id"], "sr", sr2, "shape", y.shape, "peak", float(np.abs(y).max()), flush=True)
print("elapsed", round(time.time() - t0, 1), flush=True)
