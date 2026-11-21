"""Environment adaptation for running third_party/seed-vc under torch 2.14 and transformers 5.2 (SPEC_RESOLUTIONS R3).

Seed-VC is pinned to torch 2.4 and transformers 4.46. Nothing in third_party is edited: every adaptation lives here and
is listed in docs/seedvc/ENGINE_NOTES.md.
"""
from __future__ import annotations

import contextlib
import os
import sys
import warnings
from pathlib import Path

import torch

# ---------------------------------------------------------------- PUBLIC


def PrepareSeedVc(seed_vc_dir) -> None:
    """Make Seed-VC's top-level 'modules' package importable and keep every Hugging Face lookup local."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    path = str(Path(seed_vc_dir).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
    # torch 2.14 deprecates torch.nn.utils.weight_norm, which Seed-VC's HiFT generator and WaveNet still use; behaviour is unchanged
    warnings.filterwarnings("ignore", message=r".*torch\.nn\.utils\.weight_norm.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.*", category=FutureWarning)


@contextlib.contextmanager
def HostKnownSequenceMask(width: int):
    """Inside the block, the DiT builds its attention mask with a width the host already knows.

    Seed-VC's sequence_mask(length) reads length.max() and calls arange on it, which synchronises the GPU with the CPU on every call
    and cannot be captured in a CUDA graph. For one length the two masks are equal: arange(width) < length."""
    import modules.diffusion_transformer as dit

    original = dit.sequence_mask
    dit.sequence_mask = lambda lengths, max_length=None: torch.arange(width, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
    try:
        yield
    finally:
        dit.sequence_mask = original
