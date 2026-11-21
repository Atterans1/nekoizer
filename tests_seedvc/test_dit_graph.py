import contextlib
import os
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.dit_graph import GraphedDiT  # noqa: E402

CUDA = torch.cuda.is_available()


class ToyDiT(nn.Module):
    """Same call signature and mask attributes as the Seed-VC DiT, with a few real kernels."""

    style_as_token = True
    time_as_token = True

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x, prompt_x, x_lens, t, style, cond):
        valid = torch.arange(x.shape[1], device=x.device)[None, :, None] < x_lens.to(x.device)[:, None, None]  # reads x_lens like the DiT mask
        return self.lin(x + prompt_x + cond) * t.view(-1, 1, 1) * valid + style.mean()


def Scope(width):
    return contextlib.nullcontext()


def Inputs(seed, n=6, b=2):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *shape: torch.randn(*shape, device="cuda", generator=g)  # noqa: E731
    return r(b, n, 4), r(b, n, 4), torch.tensor([n]), r(b), r(b, 3), r(b, n, 4)


@unittest.skipUnless(CUDA, "needs a CUDA device")
class GraphedDiTTests(unittest.TestCase):
    def test_replay_matches_eager_for_new_inputs_on_every_call(self):
        dit = ToyDiT().cuda()
        graphed = GraphedDiT(dit, Scope)
        with torch.no_grad():
            for seed in (1, 2, 3):
                a = Inputs(seed)
                torch.testing.assert_close(graphed(*a), dit(*a), rtol=0, atol=0)
        self.assertEqual(graphed.Stats(), {"graphs": 1, "captures": 1, "replays": 3})

    def test_a_new_shape_captures_again_and_the_oldest_graph_is_dropped(self):
        dit = ToyDiT().cuda()
        graphed = GraphedDiT(dit, Scope, max_graphs=1)
        with torch.no_grad():
            graphed(*Inputs(1, n=6))
            graphed(*Inputs(1, n=8))
            self.assertEqual((graphed.Stats()["captures"], graphed.Stats()["graphs"]), (2, 1))
            graphed(*Inputs(1, n=6))
        self.assertEqual(graphed.Stats()["captures"], 3)

    def test_replay_does_not_read_a_freed_length_tensor(self):
        """The graph reads the GPU copy of x_lens it was captured with. If nothing keeps that copy alive, later allocations overwrite it."""
        dit = ToyDiT().cuda()
        graphed = GraphedDiT(dit, Scope)
        a = Inputs(1, n=6)
        with torch.no_grad():
            graphed(*a)
            junk = [torch.zeros(1, dtype=torch.long, device="cuda") for _ in range(512)]  # recycle any freed small block
            torch.testing.assert_close(graphed(*a), dit(*a), rtol=0, atol=0)
        del junk

    def test_a_capture_failure_raises_and_caches_nothing(self):
        def Failing(width):
            raise RuntimeError("mask scope unavailable")

        graphed = GraphedDiT(ToyDiT().cuda(), Failing)
        with torch.no_grad(), self.assertRaisesRegex(RuntimeError, "mask scope unavailable"):
            graphed(*Inputs(4))
        self.assertEqual(graphed.Stats(), {"graphs": 0, "captures": 0, "replays": 0})


if __name__ == "__main__":
    unittest.main()
