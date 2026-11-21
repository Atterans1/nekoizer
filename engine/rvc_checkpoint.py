"""Safe loading of RVC voice models and of the weight files the RVC engine reads.

An RVC voice model is a pickle, and a pickle can run code. Every weights file of the RVC engine is read through SafeTorchLoad,
the only torch.load call in the RVC code. weights_only=True makes the unpickler accept tensors, plain containers and numbers and
nothing else: any other global (os.system included) is refused before it runs. PyTorch before 2.6 had a bypass of that mode
(CVE-2025-32434), so an older PyTorch refuses to load at all.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------- PUBLIC
MAX_MODEL_BYTES = 400 * 1024 * 1024
MAX_PARAMETERS = 200_000_000
MIN_TORCH = (2, 6)
CONFIG_LENGTH = 18
FEATURE_WIDTH = {"v1": 256, "v2": 768}
SAMPLE_RATE_NAMES = {"32k": 32000, "40k": 40000, "48k": 48000}


class UnsafeCheckpointError(ValueError):
    """A weights file was refused. The message is written for the user."""


@dataclass(frozen=True)
class RvcCheckpoint:
    config: list  # the 18 constructor arguments of the RVC synthesizer: sr as an int, spk_embed_dim read from emb_g
    weight: dict  # parameter name -> CPU tensor
    version: str  # "v1" (256-wide content features) or "v2" (768-wide)
    f0: bool  # True for a pitch-conditioned (NSF) model
    sr: int  # output sample rate
    spk_count: int
    info: str


def SafeTorchLoad(path, max_bytes: int | None = None, label: str | None = None) -> Any:
    """torch.load(weights_only=True) with a size cap. Raises UnsafeCheckpointError for anything that is not plain weights.
    label is the file name to show in messages when path is a scratch copy of the user's file."""
    _RequireSafeTorch()
    path = Path(path)
    shown = label or path.name
    limit = MAX_MODEL_BYTES if max_bytes is None else max_bytes
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise UnsafeCheckpointError(f"{shown} cannot be read ({exc.strerror or exc}).") from exc
    if size > limit:
        raise UnsafeCheckpointError(f"{shown} is {size / 2**20:.0f} MB, larger than the {limit // 2**20} MB limit.")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # hostile or damaged files raise UnpicklingError, RuntimeError, EOFError, zip and value errors
        raise UnsafeCheckpointError(f"{shown} was refused: {_Reason(exc)}. Nothing in it was run.") from exc


def LoadStateDict(path, max_bytes: int | None = None, key: str | None = None) -> dict:
    """A tensor-only weights file (the HuBERT, RMVPE and pretrained-generator files). key selects a nested state dict."""
    obj = SafeTorchLoad(path, max_bytes)
    if key is not None and isinstance(obj, dict) and key in obj:
        obj = obj[key]
    return _TensorDict(obj, Path(path).name)


def LoadCheckpoint(path, max_bytes: int | None = None, label: str | None = None) -> RvcCheckpoint:
    """Load and validate an exported RVC voice model. Nothing is built or run; the caller constructs the synthesizer."""
    name = label or Path(path).name
    obj = SafeTorchLoad(path, max_bytes, name)
    if not isinstance(obj, dict):
        raise UnsafeCheckpointError(f"{name} is not an RVC voice model.")
    if "weight" not in obj or "config" not in obj:
        hint = " It looks like a training checkpoint: export the small voice model from RVC first." if "model" in obj or "optimizer" in obj else ""
        raise UnsafeCheckpointError(f"{name} is not an exported RVC voice model (it has no 'config' and 'weight').{hint}")
    weight = _TensorDict(obj["weight"], name)
    config = obj["config"]
    if not isinstance(config, (list, tuple)) or len(config) != CONFIG_LENGTH:
        raise UnsafeCheckpointError(f"{name} has an unexpected model configuration (need {CONFIG_LENGTH} entries).")
    config = list(config)
    version = obj.get("version", "v1")
    if version not in FEATURE_WIDTH:
        raise UnsafeCheckpointError(f"{name} declares version {_Short(version)!r}; only v1 and v2 models are supported.")
    f0 = obj.get("f0", 1)
    if isinstance(f0, bool) or f0 not in (0, 1):
        raise UnsafeCheckpointError(f"{name} declares an unsupported pitch flag.")
    _CheckConfig(config, name)
    sr = SAMPLE_RATE_NAMES.get(config[-1]) if isinstance(config[-1], str) else config[-1]
    if not isinstance(sr, int) or math.prod(config[12]) * 100 != sr:
        raise UnsafeCheckpointError(f"{name} has a sample rate that does not match its upsampling layers.")
    embed, phone, pre = (weight.get(k) for k in ("emb_g.weight", "enc_p.emb_phone.weight", "dec.conv_pre.weight"))
    if embed is None or phone is None or pre is None:
        raise UnsafeCheckpointError(f"{name} is missing core RVC parameters, so it is not a complete voice model.")
    if embed.ndim != 2 or phone.ndim != 2 or phone.shape[1] != FEATURE_WIDTH[version]:
        raise UnsafeCheckpointError(f"{name} says it is {version} but its content layer is {phone.shape[1] if phone.ndim == 2 else '?'} wide.")
    if ("enc_p.emb_pitch.weight" in weight) != bool(f0):
        raise UnsafeCheckpointError(f"{name} pitch flag does not match its parameters.")
    config[15], config[17] = int(embed.shape[0]), int(sr)
    return RvcCheckpoint(config, weight, version, bool(f0), int(sr), int(embed.shape[0]), str(obj.get("info", ""))[:120])


# ---------------------------------------------------------------- PRIVATE
_CHANNEL_LIMIT = 4096
_COUNT_LIMIT = 64


def _Short(value: Any) -> str:
    text = str(value).strip().splitlines()[0] if str(value).strip() else type(value).__name__
    return text[:200]


def _Reason(exc: Exception) -> str:
    """Why a weights file was refused, in words that do not repeat torch's advice to turn weights_only off."""
    found = re.search(r"(?i)unsupported (?:global: )?global ([\w.]+)", str(exc))  # torch 2.6 and 2.14 word it differently
    if found:
        return f"it contains a code reference ({found.group(1)}) that a weights file never has"
    if "weights_only" in str(exc):
        return "it is not a plain weights file"
    return f"it is not a plain weights file, or it is damaged ({_Short(exc)})"


def _RequireSafeTorch() -> None:
    match = re.match(r"(\d+)\.(\d+)", torch.__version__)
    if match is None or tuple(int(g) for g in match.groups()) < MIN_TORCH:
        raise UnsafeCheckpointError(f"PyTorch {torch.__version__} is too old to open voice files safely (2.6 or newer is needed).")


def _TensorDict(obj: Any, name: str) -> dict:
    if not isinstance(obj, dict) or not obj:
        raise UnsafeCheckpointError(f"{name} does not hold a dictionary of weights.")
    total = 0
    for key, value in obj.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise UnsafeCheckpointError(f"{name} holds something other than named tensors.")
        if value.is_sparse or not (value.dtype.is_floating_point or value.dtype == torch.int64):
            raise UnsafeCheckpointError(f"{name} holds a tensor type that weights never use ({value.dtype}).")
        total += value.numel()
    if total > MAX_PARAMETERS:
        raise UnsafeCheckpointError(f"{name} has {total / 1e6:.0f} million parameters, more than the {MAX_PARAMETERS // 10**6} million limit.")
    return obj


def _CheckConfig(config: list, name: str) -> None:
    def Ints(value: Any, low: int, high: int) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high

    def IntList(value: Any) -> bool:
        return isinstance(value, (list, tuple)) and 0 < len(value) <= 8 and all(Ints(v, 1, _COUNT_LIMIT) for v in value)

    channel_slots = (0, 2, 3, 4, 13, 16)
    count_slots = (5, 6, 7)
    ok = (all(Ints(config[i], 1, _CHANNEL_LIMIT) for i in channel_slots) and all(Ints(config[i], 1, _COUNT_LIMIT) for i in count_slots)
          and Ints(config[1], 1, 10**6) and isinstance(config[8], (int, float)) and not isinstance(config[8], bool) and 0 <= config[8] < 1
          and config[9] in ("1", "2") and IntList(config[10]) and IntList(config[12]) and IntList(config[14])
          and isinstance(config[11], (list, tuple)) and all(IntList(row) for row in config[11])
          and len(config[10]) == len(config[11]) and len(config[12]) == len(config[14]) and Ints(config[15], 0, 10**6)
          and (Ints(config[17], 1, 10**6) or config[17] in SAMPLE_RATE_NAMES))
    if not ok:
        raise UnsafeCheckpointError(f"{name} has a model configuration outside the ranges RVC models use.")
