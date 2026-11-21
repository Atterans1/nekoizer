"""Prosody and similarity metrics for the voice-conversion evaluation (16 kHz inputs)."""
import numpy as np
import librosa

F0_FMIN, F0_FMAX, F0_HOP = 60.0, 500.0, 320
MIN_JOINT_FRAMES = 40
RMS_FRAME, RMS_HOP = 640, 320


def Pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or a.size != b.size:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    if den <= 0:
        return float("nan")
    return float((a * b).sum() / den)


def Cos(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def F0(x16):
    f0, voiced, _ = librosa.pyin(x16, fmin=F0_FMIN, fmax=F0_FMAX, sr=16000, hop_length=F0_HOP)
    return f0, voiced & np.isfinite(f0)


def Prosody(src16, conv16):
    """Log-F0 Pearson on jointly voiced frames (mean-removed per contour), log-RMS envelope Pearson at 20 ms hop, duration ratio."""
    fs, vs = F0(src16)
    fc, vc = F0(conv16)
    n = min(len(fs), len(fc))
    joint = vs[:n] & vc[:n]
    n_joint = int(joint.sum())
    f0_corr = Pearson(np.log(fs[:n][joint]), np.log(fc[:n][joint])) if n_joint >= 2 else float("nan")
    rs = librosa.feature.rms(y=src16, frame_length=RMS_FRAME, hop_length=RMS_HOP, center=True)[0]
    rc = librosa.feature.rms(y=conv16, frame_length=RMS_FRAME, hop_length=RMS_HOP, center=True)[0]
    m = min(len(rs), len(rc))
    energy_corr = Pearson(np.log10(rs[:m] + 1e-8), np.log10(rc[:m] + 1e-8))
    return {
        "f0_logcorr": f0_corr,
        "joint_voiced_frames": n_joint,
        "f0_frames_compared": int(n),
        "src_voiced_frac": float(vs.mean()) if vs.size else 0.0,
        "conv_voiced_frac": float(vc.mean()) if vc.size else 0.0,
        "energy_logcorr": energy_corr,
        "duration_ratio": float(len(conv16) / len(src16)),
    }
