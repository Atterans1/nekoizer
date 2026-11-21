"""Build the RVC synthesizer (vendored in third_party/rvc) from a validated checkpoint."""
from __future__ import annotations

import warnings

import torch

# torch deprecates torch.nn.utils.weight_norm, which the vendored RVC code and the .pth key names (weight_g, weight_v) still use
warnings.filterwarnings("ignore", message=r".*torch\.nn\.utils\.weight_norm.*", category=FutureWarning)

from engine.rvc_checkpoint import RvcCheckpoint, UnsafeCheckpointError  # noqa: E402
from third_party.rvc import models  # noqa: E402

# ---------------------------------------------------------------- PUBLIC
_CLASSES = {
    ("v1", True): models.SynthesizerTrnMs256NSFsid,
    ("v1", False): models.SynthesizerTrnMs256NSFsid_nono,
    ("v2", True): models.SynthesizerTrnMs768NSFsid,
    ("v2", False): models.SynthesizerTrnMs768NSFsid_nono,
}


def NewSynthesizer(config: list, version: str, f0: bool) -> torch.nn.Module:
    """An untrained synthesizer of the given layout (random weights, posterior encoder still attached)."""
    return _CLASSES[(version, bool(f0))](*config, is_half=False)


def BuildSynthesizer(checkpoint: RvcCheckpoint, device, half: bool) -> torch.nn.Module:
    """The inference synthesizer of a checkpoint: weights loaded, weight norm folded, on device, in half precision if asked."""
    net = NewSynthesizer(checkpoint.config, checkpoint.version, checkpoint.f0)
    del net.enc_q
    try:
        missing, unexpected = net.load_state_dict(checkpoint.weight, strict=False)
    except RuntimeError as exc:
        raise UnsafeCheckpointError(f"the model weights do not fit the model configuration ({str(exc).splitlines()[0][:160]})") from exc
    if missing:
        raise UnsafeCheckpointError(f"the model is missing {len(missing)} parameters (first: {missing[0]})")
    if [k for k in unexpected if not k.startswith("enc_q.")]:
        raise UnsafeCheckpointError(f"the model has {len(unexpected)} parameters the configuration does not use (first: {unexpected[0]})")
    net.eval()
    net.remove_weight_norm()
    net.to(device)
    return net.half() if half else net.float()
