"""GraphedDiT: the Seed-VC DiT estimator replayed as a CUDA graph.

The estimator is about 600 small kernels per call. Launched eagerly, the CPU spends longer queueing them than the GPU spends running
them, and the DiT's attention mask (sequence_mask on a GPU length) also forces a GPU to CPU sync on every call. A captured graph
launches the whole call at once. The output is bit-identical to the eager call (docs/seedvc/ENGINE_NOTES.md has the measurements).

A graph is tied to its input shapes and to the mask length, so graphs are kept per (shapes, length, dtype, autocast) in a small LRU.
Capture needs a host-known mask length: the caller passes x_lens on the CPU, and capture_scope makes the DiT build its mask from it.

There is no eager fallback after a failed capture: a capture that fails part-way can leave the CUDA stream invalidated, so the next eager
call in the same process fails too. The graph-or-eager choice is made once, at load (SeedVCExpressive(cuda_graph=...)), and a capture
failure raises.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

# ---- PUBLIC
MAX_GRAPHS = 4          # a block-size or reference-length change adds a graph; the oldest is dropped past this
WARMUP_CALLS = 3        # eager calls on a side stream before capture (cuDNN autotune, allocator, lazy init)


class GraphedDiT(nn.Module):
    def __init__(self, dit: nn.Module, capture_scope: Callable[[int], AbstractContextManager], max_graphs: int = MAX_GRAPHS):
        """capture_scope(width) is entered around warm-up and capture; it must make the DiT build its attention mask of the given
        width without reading a GPU tensor (compat.HostKnownSequenceMask)."""
        super().__init__()
        self.dit = dit
        self._scope = capture_scope
        self._max = max_graphs
        self._graphs: OrderedDict[tuple, _Entry] = OrderedDict()
        self._captures = self._replays = 0

    def forward(self, x, prompt_x, x_lens, t, style, cond):
        key = (tuple(x.shape), tuple(cond.shape), int(x_lens[0]), x.dtype, torch.is_autocast_enabled())
        entry = self._graphs.get(key)
        if entry is None:
            entry = self._Capture(key, x, prompt_x, x_lens, t, style, cond)
        self._graphs.move_to_end(key)
        entry.inputs[0].copy_(x)
        entry.inputs[1].copy_(prompt_x)
        entry.inputs[2].copy_(t)
        entry.inputs[3].copy_(style)
        entry.inputs[4].copy_(cond)
        entry.graph.replay()
        self._replays += 1
        return entry.output.clone()

    def Stats(self) -> dict:
        return {"graphs": len(self._graphs), "captures": self._captures, "replays": self._replays}

    # ---- PRIVATE
    def _Capture(self, key, x, prompt_x, x_lens, t, style, cond) -> "_Entry":
        with torch.inference_mode(False):  # ordinary tensors, so replay can refill them whatever mode the caller (or a library it uses) is in
            inputs = [torch.empty_like(a).copy_(a) for a in (x, prompt_x, t, style, cond)]
        lens = x_lens.to(x.device)
        width = key[2] + int(self.dit.style_as_token) + int(self.dit.time_as_token)  # the mask width the DiT builds
        call = lambda: self.dit(inputs[0], inputs[1], lens, inputs[2], inputs[3], inputs[4])  # noqa: E731
        with self._scope(width):
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(WARMUP_CALLS):
                    call()
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = call()
        self._graphs[key] = _Entry(graph, inputs, lens, output)
        self._captures += 1
        while len(self._graphs) > self._max:
            self._graphs.popitem(last=False)
        return self._graphs[key]


@dataclass
class _Entry:
    graph: torch.cuda.CUDAGraph
    inputs: list          # static x, prompt_x, t, style, cond the graph reads
    lens: torch.Tensor    # static x_lens on the GPU: the graph reads it, so it must outlive capture
    output: torch.Tensor  # static result the graph writes
