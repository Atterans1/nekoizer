"""The shared RVC models: HuBERT (content features) and the pitch tracker (RMVPE, or FCPE when the RMVPE file is absent)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from engine import rvc_weights
from engine.rvc_checkpoint import LoadStateDict
from third_party.rvc import rmvpe

# ---- PUBLIC
RMVPE_THRESHOLD = 0.03


class RvcHubert:
    """HuBERT base in the transformers layout, with the RVC final projection (768 -> 256) used by v1 voices."""

    def __init__(self, config_path: Path, weights_path: Path, device: torch.device, half: bool):
        from transformers import HubertConfig, HubertModel

        config = HubertConfig(**json.loads(Path(config_path).read_text(encoding="utf-8")))
        state = LoadStateDict(weights_path)
        final = {k[len("final_proj."):]: v for k, v in state.items() if k.startswith("final_proj.")}
        body = {k: v for k, v in state.items() if not k.startswith("final_proj.")}
        model = HubertModel(config)
        missing, unexpected = model.load_state_dict(body, strict=False)
        if unexpected or [k for k in missing if not k.endswith("masked_spec_embed")]:
            raise RuntimeError(f"the HuBERT weights do not match its configuration (missing {missing[:3]}, unexpected {unexpected[:3]})")
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.final_proj.load_state_dict(final)
        dtype = torch.float16 if half else torch.float32
        self.model = model.eval().to(device=device, dtype=dtype)
        self.final_proj = self.final_proj.eval().to(device=device, dtype=dtype)
        self._half = half

    def Features(self, wave: torch.Tensor, version: str) -> torch.Tensor:
        wave = wave.half() if self._half else wave.float()
        if version == "v1":
            hidden = self.model(wave, output_hidden_states=True, return_dict=True).hidden_states[9]
            return self.final_proj(hidden)
        return self.model(wave, return_dict=True).last_hidden_state


class RmvpeF0:
    """RMVPE pitch tracker (vendored network from RVC, MIT): Pitch(wave16) -> Hz per 10 ms frame, 0 where unvoiced."""

    pitch_name = "rmvpe"

    def __init__(self, weights_path: Path, device: torch.device, half: bool):
        self._device = device
        self._half = half
        self._net = rmvpe.E2E(4, 1, (2, 2))
        self._net.load_state_dict(LoadStateDict(weights_path))
        self._net = self._net.eval().to(device=device, dtype=torch.float16 if half else torch.float32)
        self._mel = rmvpe.MelSpectrogram(half, 128, 16000, 1024, 160, None, 30, 8000).to(device)
        self._cents = np.pad(20 * np.arange(360) + 1997.3794084376191, (4, 4))

    def Pitch(self, wave16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(wave16, dtype=np.float32)).to(self._device).unsqueeze(0)
            mel = self._mel(x, center=True)
            frames = mel.shape[-1]
            pad = 32 * ((frames - 1) // 32 + 1) - frames
            if pad > 0:
                mel = F.pad(mel, (0, pad))
            mel = mel.half() if self._half else mel.float()
            hidden = self._net(mel)[0, :frames].float().cpu().numpy()
        return self._Decode(hidden)

    def _Decode(self, salience: np.ndarray) -> np.ndarray:
        padded = np.pad(salience, ((0, 0), (4, 4)))
        center = np.argmax(padded, axis=1) + 4
        index = center[:, None] + np.arange(-4, 5)[None, :]
        sal = np.take_along_axis(padded, index, axis=1)
        cents = (sal * self._cents[index]).sum(axis=1) / sal.sum(axis=1)
        cents[salience.max(axis=1) <= RMVPE_THRESHOLD] = 0.0
        f0 = 10.0 * 2.0 ** (cents / 1200.0)
        f0[f0 == 10.0] = 0.0
        f0[~np.isfinite(f0)] = 0.0
        return _FillUnvoiced(f0)


class FcpeF0:
    """Fallback pitch: the FCPE tracker of engine.prosody_stage, same frame grid (10 ms, Hz, 0 where unvoiced)."""

    pitch_name = "fcpe"

    def __init__(self, device: torch.device):
        from engine.prosody_stage import FcpeF0 as Tracker
        self._tracker = Tracker(str(device))

    def Pitch(self, wave16: np.ndarray) -> np.ndarray:
        return _FillUnvoiced(self._tracker.Infer(wave16, 16000))


class Backbone:
    def __init__(self, hubert, pitch):
        self._hubert = hubert
        self._pitch = pitch
        self.pitch_name = pitch.pitch_name

    def Features(self, wave: torch.Tensor, version: str) -> torch.Tensor:
        return self._hubert.Features(wave, version)

    def Pitch(self, wave16: np.ndarray) -> np.ndarray:
        return self._pitch.Pitch(wave16)


def LoadBackbone(weights_dir: Path | None, device: torch.device, half: bool) -> Backbone:
    """HuBERT plus pitch from the weights folder. The pitch is FCPE when rmvpe.pt is absent."""
    folder = Path(weights_dir) if weights_dir is not None else rvc_weights.WeightsDir()
    hubert = RvcHubert(rvc_weights.PathOf("hubert_base/config.json", folder),
                       rvc_weights.PathOf("hubert_base/pytorch_model.bin", folder), device, half)
    rmvpe_path = rvc_weights.PathOf("rmvpe.pt", folder)
    pitch = RmvpeF0(rmvpe_path, device, half) if rmvpe_path.exists() else FcpeF0(device)
    return Backbone(hubert, pitch)


# ---- PRIVATE
def _FillUnvoiced(f0: np.ndarray) -> np.ndarray:
    out = np.asarray(f0, dtype=np.float64).copy()
    voiced = out > 0.0
    if voiced.any() and not voiced.all():
        idx = np.arange(len(out))
        out[~voiced] = np.interp(idx[~voiced], idx[voiced], out[voiced])
    return out
