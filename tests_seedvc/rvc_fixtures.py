"""Tiny random-weight RVC voices for tests: built from the vendored synthesizer and saved in the exported .pth layout."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

from engine import rvc_synth  # noqa: E402

# upsample_rates, upsample_kernel_sizes per output rate; the product of the rates is the samples per 10 ms frame
RATES = {
    32000: ([10, 4, 2, 2, 2], [16, 16, 4, 4, 4]),
    40000: ([10, 10, 2, 2], [16, 16, 4, 4]),
    48000: ([12, 10, 2, 2], [24, 20, 4, 4]),
}
SR_NAMES = {32000: "32k", 40000: "40k", 48000: "48k"}


def TinyConfig(sr: int = 40000, speakers: int = 1) -> list:
    rates, kernels = RATES[sr]
    return [1025, 32, 16, 16, 32, 2, 2, 3, 0, "1", [3, 7], [[1, 3, 5], [1, 3, 5]], rates, 32, kernels, speakers, 16, sr]


def MakeNet(version: str = "v2", f0: bool = True, sr: int = 40000, seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    return rvc_synth.NewSynthesizer(TinyConfig(sr), version, f0)


def SaveCheckpoint(path, version: str = "v2", f0: bool = True, sr: int = 40000, seed: int = 0) -> Path:
    """Write a random tiny voice as an exported RVC .pth (half-precision weights, no posterior encoder), like the webui exports."""
    net = MakeNet(version, f0, sr, seed)
    weight = {k: v.half() for k, v in net.state_dict().items() if not k.startswith("enc_q.")}
    torch.save({"config": TinyConfig(sr), "weight": weight, "info": "tiny random", "sr": SR_NAMES[sr], "f0": int(f0), "version": version}, path)
    return Path(path)
