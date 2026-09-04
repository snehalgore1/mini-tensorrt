"""Shared model definition.

Both export_models.py and gen_goldens.py import from here so the exported
weights and the golden reference outputs come from the exact same, seeded model.
There is no other source of truth for the MLP.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 0


class MLP(nn.Module):
    """Tiny MLP: Linear(8->16) -> GELU -> Linear(16->4) -> Softmax.

    Deliberately small so the exported weights and .npy fixtures stay under a
    few KB. Batch size is fixed at 1 (static shapes, DESIGN D1); with batch=1 the
    bias add is an identical-shape elementwise Add, so the runtime needs no
    broadcasting yet.
    """

    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(8, 16)
        self.fc1 = nn.Linear(16, 4)

    def forward(self, x):
        x = self.fc0(x)
        x = F.gelu(x)  # exact, erf-based (approximate='none') -- must match kernel
        x = self.fc1(x)
        x = torch.softmax(x, dim=-1)
        return x


def build_mlp() -> MLP:
    """Return the MLP with deterministic weights."""
    torch.manual_seed(SEED)
    model = MLP().eval()
    return model


def sample_input() -> torch.Tensor:
    """Deterministic [1, 8] input for the whole-model golden."""
    g = torch.Generator().manual_seed(SEED + 1)
    return torch.randn(1, 8, generator=g, dtype=torch.float32)


def to_f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().numpy().astype(np.float32)
