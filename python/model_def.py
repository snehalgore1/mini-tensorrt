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


# --- Multi-head transformer block -------------------------------------------
# Pre-LN block: LayerNorm -> multi-head self-attention -> residual ->
# LayerNorm -> MLP(Gelu) -> residual. Kept small so committed fixtures stay tiny;
# the perf work uses large standalone GEMMs, not this block.

TF = {"S": 8, "D": 32, "H": 2, "hidden": 128, "eps": 1e-5}
TF["d"] = TF["D"] // TF["H"]


def build_transformer_params():
    """Deterministic weights as float32 numpy arrays, in [in, out] layout so
    each maps directly to a MatMul(x, W) in the runtime graph."""
    g = torch.Generator().manual_seed(SEED + 7)
    D, H, hidden = TF["D"], TF["H"], TF["hidden"]

    def r(*shape, s=0.1):
        return (torch.randn(*shape, generator=g) * s).float()

    p = {
        "ln1_g": (1.0 + r(D)),
        "ln1_b": r(D),
        "Wq": r(D, D),
        "Wk": r(D, D),
        "Wv": r(D, D),
        "Wo": r(D, D),
        "ln2_g": (1.0 + r(D)),
        "ln2_b": r(D),
        "W1": r(D, hidden),
        "b1": r(1, hidden),
        "W2": r(hidden, D),
        "b2": r(1, D),
    }
    return {k: to_f32(v) for k, v in p.items()}


def sample_transformer_input():
    g = torch.Generator().manual_seed(SEED + 8)
    return to_f32(torch.randn(TF["S"], TF["D"], generator=g))


def transformer_forward(params, x_np):
    """Reference forward mirroring the runtime graph exactly."""
    S, D, H, d = TF["S"], TF["D"], TF["H"], TF["d"]
    eps = TF["eps"]
    t = {k: torch.from_numpy(v) for k, v in params.items()}
    x = torch.from_numpy(x_np)

    h = F.layer_norm(x, [D], t["ln1_g"], t["ln1_b"], eps)
    Q, K, V = h @ t["Wq"], h @ t["Wk"], h @ t["Wv"]  # [S, D]
    Qh = Q.reshape(S, H, d).permute(1, 0, 2)          # [H, S, d]
    Kh = K.reshape(S, H, d).permute(1, 0, 2)
    Vh = V.reshape(S, H, d).permute(1, 0, 2)
    scores = (Qh @ Kh.transpose(1, 2)) * (1.0 / (d ** 0.5))  # [H, S, S]
    attn = F.softmax(scores, dim=-1)
    ctx = attn @ Vh                                   # [H, S, d]
    ctxM = ctx.permute(1, 0, 2).reshape(S, D)         # [S, D]
    x1 = x + ctxM @ t["Wo"]
    h2 = F.layer_norm(x1, [D], t["ln2_g"], t["ln2_b"], eps)
    m = F.gelu(h2 @ t["W1"] + t["b1"])                # [S, hidden]
    out = x1 + m @ t["W2"] + t["b2"]
    return to_f32(out)
