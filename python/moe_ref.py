"""Faithful reference for the GPT-OSS MoE block (MLPBlock), written as explicit stages.

The official implementation is `openai/gpt-oss@2770201`, `gpt_oss/torch/model.py:249-336`.
It computes the expert FFN by *gathering each token's 4 expert weight matrices*
(`mlp1_weight[expert_indices]`, a [T, k, 2I, H] copy). That is the right way to write
a readable reference, but no inference runtime executes it that way.

This file writes the same math in the shape the MiniTensorRT kernels will use:

    x[T,H] -> RMSNorm -> t[T,H]
           -> router: logits = t @ Wg^T + bg; top-k (sorted); softmax over the k
           -> dispatch: stable counting sort of the T*k (token, slot) assignments by expert
           -> experts: per-expert segment GEMMs   h = t_e @ W1[e]^T + b1[e]
                                                   a = swiglu(h)
                                                   y = a @ W2[e]^T + b2[e]
           -> combine: out[t] = sum_slot w[t,slot] * y_sorted[inv[t*k+slot]]
           -> residual: x + out

`python/check_official_moe.py` proves this staged formulation equals the official
`MLPBlock.forward`. Every C++/CUDA kernel is later golden-tested against these stages.

Weight layouts are kept exactly as in the official checkpoint (out_features, in_features):
    norm_scale [H]      gate_w [E,H]      gate_b [E]
    w1 [E, 2I, H]       b1 [E, 2I]        w2 [E, H, I]       b2 [E, H]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MoEConfig:
    hidden: int = 2880          # H   (gpt-oss-20b config.json: hidden_size)
    intermediate: int = 2880    # I   (intermediate_size)
    num_experts: int = 32       # E   (num_local_experts)
    top_k: int = 4              # k   (num_experts_per_tok / experts_per_token)
    swiglu_limit: float = 7.0   # model.py:20
    swiglu_alpha: float = 1.702  # model.py:249
    rms_eps: float = 1e-5       # model.py:34, config.json rms_norm_eps


GPT_OSS_20B = MoEConfig()


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def round_to_bf16(t: torch.Tensor) -> torch.Tensor:
    """Round to the nearest bf16-representable value, keep the original dtype.

    The real checkpoint stores these tensors in bf16 (MoE weights in MXFP4, which
    upcast to bf16), so values that are exactly bf16-representable let the same
    FP32 kernels later consume real weights without a second rounding step.
    """
    return t.to(torch.bfloat16).to(t.dtype)


def make_params(cfg: MoEConfig, seed: int = 0, dtype=torch.float32,
                weight_scale: float = 1.0, bf16_round: bool = True,
                gate_bias_boost: dict[int, float] | None = None) -> dict:
    """Random parameters with fan-in scaled init (std = weight_scale / sqrt(fan_in)).

    `gate_bias_boost` adds a constant to chosen router biases. Boosting 4 experts by a
    large value forces every token to select exactly those 4 (the "same-four-experts"
    case: top-4 semantics are kept and the other E-4 experts receive zero tokens).
    """
    g = torch.Generator().manual_seed(seed)
    H, I, E = cfg.hidden, cfg.intermediate, cfg.num_experts

    def rnd(*shape, fan_in):
        t = torch.empty(*shape, dtype=dtype)
        # Fill expert-by-expert so a real-dims run does not need a second full copy.
        if len(shape) == 3:
            for e in range(shape[0]):
                t[e] = torch.randn(*shape[1:], generator=g, dtype=dtype) * (weight_scale / fan_in ** 0.5)
        else:
            t.copy_(torch.randn(*shape, generator=g, dtype=dtype) * (weight_scale / fan_in ** 0.5))
        return t

    p = {
        "norm_scale": 1.0 + 0.1 * torch.randn(H, generator=g, dtype=dtype),
        "gate_w": rnd(E, H, fan_in=H),
        "gate_b": 0.1 * torch.randn(E, generator=g, dtype=dtype),
        "w1": rnd(E, 2 * I, H, fan_in=H),
        "b1": 0.1 * torch.randn(E, 2 * I, generator=g, dtype=dtype),
        "w2": rnd(E, H, I, fan_in=I),
        "b2": 0.1 * torch.randn(E, H, generator=g, dtype=dtype),
    }
    if gate_bias_boost:
        for e, v in gate_bias_boost.items():
            p["gate_b"][e] += v
    if bf16_round:
        for name, t in p.items():
            if t.dim() == 3:  # round in place per expert (memory at real dims)
                for e in range(t.shape[0]):
                    t[e] = round_to_bf16(t[e])
            else:
                p[name] = round_to_bf16(t)
    # norm_scale is fp32 in the checkpoint (model.py:39-41); round anyway -- harmless,
    # and keeps "all weights are bf16-representable" a single simple statement.
    return p


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def rmsnorm(x: torch.Tensor, scale: torch.Tensor, eps: float) -> torch.Tensor:
    """model.py:43-47. The normalization is computed in fp32 regardless of input dtype.

        y = x / sqrt(mean(x^2) + eps) * scale

    Unlike LayerNorm there is no mean subtraction and no bias. `scale` is NOT cast:
    exactly like the official code, `t * scale` follows torch type promotion (scale is
    an fp32 parameter in the real model, so this is fp32 there).
    """
    t = x.float()
    t = t * torch.rsqrt(torch.mean(t ** 2, dim=-1, keepdim=True) + eps)
    return (t * scale).to(x.dtype)


def topk_stable(logits: torch.Tensor, k: int):
    """Top-k along the last dim, sorted descending, ties broken by LOWER expert index.

    torch.topk leaves tie order unspecified; a runtime needs a defined rule so CPU and
    GPU agree. A stable descending sort keeps equal keys in original (index) order.
    """
    vals, idx = torch.sort(logits, dim=-1, descending=True, stable=True)
    return vals[..., :k], idx[..., :k]


def router(t: torch.Tensor, gate_w, gate_b, k: int):
    """model.py:314-317.

    logits = t @ gate_w^T + gate_b                      [T, E]
    (vals, idx) = top-k(logits), sorted descending      [T, k]
    w = softmax(vals) over the k selected                [T, k]

    Softmax AFTER top-k: the weights are a distribution over only the selected experts
    (they sum to 1). Softmax-then-top-k would give weights that sum to < 1.
    """
    logits = t @ gate_w.T + gate_b
    vals, idx = topk_stable(logits, k)
    w = torch.softmax(vals, dim=-1)
    return logits, idx, w


def swiglu(h: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """model.py:249-256. INTERLEAVED halves: even columns gate, odd columns linear.

        glu = clamp(h[..., 0::2], max=limit)          (upper clamp only)
        lin = clamp(h[..., 1::2], -limit, limit)
        a   = glu * sigmoid(alpha * glu) * (lin + 1)
    """
    glu = h[..., 0::2].clamp(max=limit)
    lin = h[..., 1::2].clamp(min=-limit, max=limit)
    return glu * torch.sigmoid(alpha * glu) * (lin + 1)


def dispatch(t: torch.Tensor, idx: torch.Tensor, num_experts: int):
    """Group the T*k (token, slot) assignments by expert with a stable counting sort.

    Assignment id a = token*k + slot (row-major flatten of idx[T,k]).
      counts[e]   = #assignments to expert e                (histogram)
      offsets     = exclusive prefix sum, length E+1; offsets[E] == T*k
      src[j]      = assignment id stored at sorted row j     (stable: ascending a within e)
      inv[a]      = sorted row holding assignment a          (inverse permutation)
      t_sorted[j] = t[src[j] // k]                           [T*k, H]

    T*k is fixed for a given T, so every buffer here has a static shape; only the
    split points (offsets) depend on the routing.
    """
    T, k = idx.shape
    flat = idx.reshape(-1)                                        # [T*k], value = expert
    counts = torch.bincount(flat, minlength=num_experts)
    offsets = torch.zeros(num_experts + 1, dtype=torch.long)
    offsets[1:] = torch.cumsum(counts, 0)
    # Stable counting sort, written out (this is the algorithm the CPU kernel uses).
    cursor = offsets[:-1].clone()
    src = torch.empty(T * k, dtype=torch.long)
    for a in range(T * k):
        e = int(flat[a])
        src[cursor[e]] = a
        cursor[e] += 1
    inv = torch.empty_like(src)
    inv[src] = torch.arange(T * k)
    t_sorted = t[src // k]
    return t_sorted, src, inv, offsets


def experts_grouped(t_sorted, offsets, w1, b1, w2, b2, alpha, limit):
    """One GEMM pair per expert over its contiguous segment of sorted rows.

    For expert e and rows r in [offsets[e], offsets[e+1]):
        h = t_sorted[r] @ w1[e]^T + b1[e]      [m_e, 2I]
        a = swiglu(h)                          [m_e, I]
        y = a @ w2[e]^T + b2[e]                [m_e, H]
    Empty experts (m_e == 0) do nothing and never read their weights.
    """
    E = w1.shape[0]
    y_sorted = torch.empty(t_sorted.shape[0], w2.shape[1], dtype=t_sorted.dtype)
    for e in range(E):
        lo, hi = int(offsets[e]), int(offsets[e + 1])
        if lo == hi:
            continue
        h = t_sorted[lo:hi] @ w1[e].T + b1[e]
        a = swiglu(h, alpha, limit)
        y_sorted[lo:hi] = a @ w2[e].T + b2[e]
    return y_sorted


def combine(y_sorted, inv, w):
    """out[t] = sum over slot s (in sorted top-k order) of w[t,s] * y_sorted[inv[t*k+s]].

    A GATHER through the inverse permutation, not a scatter-add: each output row is
    written by exactly one owner, so no atomics/races, and the summation order is fixed.
    """
    T, k = w.shape
    out = torch.zeros(T, y_sorted.shape[1], dtype=y_sorted.dtype)
    for s in range(k):
        out += w[:, s:s + 1] * y_sorted[inv.view(T, k)[:, s]]
    return out


def moe_forward(x: torch.Tensor, p: dict, cfg: MoEConfig, return_stages: bool = False):
    """Full MLPBlock: x + MoE(RMSNorm(x)). x is [T, H]."""
    t = rmsnorm(x, p["norm_scale"], cfg.rms_eps)
    logits, idx, w = router(t, p["gate_w"], p["gate_b"], cfg.top_k)
    t_sorted, src, inv, offsets = dispatch(t, idx, cfg.num_experts)
    y_sorted = experts_grouped(t_sorted, offsets, p["w1"], p["b1"], p["w2"], p["b2"],
                               cfg.swiglu_alpha, cfg.swiglu_limit)
    moe_out = combine(y_sorted, inv, w)
    out = x + moe_out
    if not return_stages:
        return out
    return out, dict(t=t, logits=logits, idx=idx, w=w, t_sorted=t_sorted, src=src,
                     inv=inv, offsets=offsets, y_sorted=y_sorted, moe_out=moe_out)


def moe_forward_naive(x: torch.Tensor, p: dict, cfg: MoEConfig) -> torch.Tensor:
    """Per-(token, slot) loop: the ablation baseline the C++ `moe_naive` mirrors."""
    t = rmsnorm(x, p["norm_scale"], cfg.rms_eps)
    _, idx, w = router(t, p["gate_w"], p["gate_b"], cfg.top_k)
    out = torch.zeros_like(x)
    for i in range(x.shape[0]):
        for s in range(cfg.top_k):
            e = int(idx[i, s])
            h = p["w1"][e] @ t[i] + p["b1"][e]
            a = swiglu(h, cfg.swiglu_alpha, cfg.swiglu_limit)
            out[i] += w[i, s] * (p["w2"][e] @ a + p["b2"][e])
    return x + out
