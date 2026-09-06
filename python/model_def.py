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


# --- GPT-2-small-scale stacked transformer ----------------------------------
# Same pre-LN block as above but at realistic GPT-2-small dimensions and stacked
# over N_LAYERS. This is the "real model" the Week 3/4 optimizations are measured
# on: at these sizes memory planning saves megabytes and latency is milliseconds,
# so the ablation numbers are meaningful (the tiny [S,D]=[8,32] block above gives
# byte-scale numbers). The FFN uses tanh-approx GELU ("gelu_new"), GPT-2's real
# activation, matching the GeluTanh kernel. Weights are large (~28 MB/layer f32),
# so the exported .bin and goldens are regenerated, not committed (see .gitignore).
GPT2 = {"S": 128, "D": 768, "H": 12, "hidden": 3072, "eps": 1e-5, "n_layers": 2}
GPT2["d"] = GPT2["D"] // GPT2["H"]  # 64

# Per-layer weight names, in blob order. Callers prefix with "l{i}_".
GPT2_LAYER_WEIGHTS = ["ln1_g", "ln1_b", "Wq", "Wk", "Wv", "Wo",
                      "ln2_g", "ln2_b", "W1", "b1", "W2", "b2"]


def build_gpt2_params():
    """Deterministic per-layer weights for the stacked GPT-2 block, in [in, out]
    layout. Keys are prefixed by layer, e.g. 'l0_Wq'. Small init (~0.02) keeps
    activations well-conditioned, as in real GPT-2."""
    g = torch.Generator().manual_seed(SEED + 17)
    D, hidden, n_layers = GPT2["D"], GPT2["hidden"], GPT2["n_layers"]

    def r(*shape, s=0.02):
        return (torch.randn(*shape, generator=g) * s).float()

    params = {}
    for i in range(n_layers):
        layer = {
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
        for k, v in layer.items():
            params[f"l{i}_{k}"] = to_f32(v)
    return params


def sample_gpt2_input():
    g = torch.Generator().manual_seed(SEED + 18)
    return to_f32(torch.randn(GPT2["S"], GPT2["D"], generator=g))


def gpt2_forward(params, x_np):
    """Reference forward for the stacked GPT-2 block. Mirrors the exported graph
    exactly (pre-LN block per layer, tanh-approx GELU in the FFN)."""
    S, D, H, d = GPT2["S"], GPT2["D"], GPT2["H"], GPT2["d"]
    eps, n_layers = GPT2["eps"], GPT2["n_layers"]
    t = {k: torch.from_numpy(v) for k, v in params.items()}
    x = torch.from_numpy(x_np)

    for i in range(n_layers):
        p = lambda name: t[f"l{i}_{name}"]  # noqa: E731
        h = F.layer_norm(x, [D], p("ln1_g"), p("ln1_b"), eps)
        Q, K, V = h @ p("Wq"), h @ p("Wk"), h @ p("Wv")
        Qh = Q.reshape(S, H, d).permute(1, 0, 2)
        Kh = K.reshape(S, H, d).permute(1, 0, 2)
        Vh = V.reshape(S, H, d).permute(1, 0, 2)
        scores = (Qh @ Kh.transpose(1, 2)) * (1.0 / (d ** 0.5))
        attn = F.softmax(scores, dim=-1)
        ctx = attn @ Vh
        ctxM = ctx.permute(1, 0, 2).reshape(S, D)
        x1 = x + ctxM @ p("Wo")
        h2 = F.layer_norm(x1, [D], p("ln2_g"), p("ln2_b"), eps)
        m = F.gelu(h2 @ p("W1") + p("b1"), approximate="tanh")  # gelu_new
        x = x1 + m @ p("W2") + p("b2")
    return to_f32(x)
