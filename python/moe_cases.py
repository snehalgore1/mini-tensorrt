"""The MoE golden cases, shared by gen_goldens.py (fixtures) and export_models.py (the
moe_layer graph). Keeping them in one place guarantees the exported graph's weights are
exactly the weights the goldens were computed with.

Runtime weight layout (D-MoE-3). The CPU GEMM computes C = A @ B with B stored [K, N], so
the [out, in] matrices are transposed ONCE here:
    gate_wT [H, E]     w1t [E, H, 2I]     w2t [E, I, H]
moe_ref keeps the official [out, in] layout. The transpose is checked to be exact.
"""

from __future__ import annotations

import torch

import moe_ref as ref

CASES = {
    # name: (cfg, T, make_params kwargs)
    "small": (ref.MoEConfig(hidden=8, intermediate=8, num_experts=8, top_k=4), 6,
              dict(seed=101)),
    "four": (ref.MoEConfig(hidden=8, intermediate=8, num_experts=8, top_k=4), 6,
             dict(seed=102, gate_bias_boost={1: 20.0, 3: 20.0, 4: 20.0, 6: 20.0})),
    "clamp": (ref.MoEConfig(hidden=8, intermediate=8, num_experts=8, top_k=4), 32,
              dict(seed=103, weight_scale=12.0)),
    "e32": (ref.MoEConfig(hidden=16, intermediate=16, num_experts=32, top_k=4), 32,
            dict(seed=104)),
    # Decode-like: 3 tokens x 4 slots = 12 assignments over 32 experts, so at least 20
    # experts are empty from ordinary random routing (no forcing).
    "sparse": (ref.MoEConfig(hidden=16, intermediate=16, num_experts=32, top_k=4), 3,
               dict(seed=105)),
}

TOPK_MARGIN = 1e-4  # goldens must be tie-free: k-th vs (k+1)-th logit gap at least this


def runtime_weights(p: dict) -> dict:
    """Official-layout params -> the runtime (transposed) layout, all f32 contiguous."""
    rw = {
        "norm_scale": p["norm_scale"],
        "gate_wT": p["gate_w"].T.contiguous(),
        "gate_b": p["gate_b"],
        "w1t": p["w1"].transpose(1, 2).contiguous(),
        "b1": p["b1"],
        "w2t": p["w2"].transpose(1, 2).contiguous(),
        "b2": p["b2"],
    }
    for e in range(p["w1"].shape[0]):  # transpose is exact (a pure permutation of values)
        assert torch.equal(rw["w1t"][e], p["w1"][e].T)
        assert torch.equal(rw["w2t"][e], p["w2"][e].T)
    return rw


def build_case(name: str) -> dict:
    """Inputs, runtime weights and every intermediate stage for one case."""
    cfg, T, kw = CASES[name]
    p = ref.make_params(cfg, **kw)
    g = torch.Generator().manual_seed(kw["seed"] + 1000)
    x = ref.round_to_bf16(torch.randn(T, cfg.hidden, generator=g))
    out, st = ref.moe_forward(x, p, cfg, return_stages=True)

    sv = torch.sort(st["logits"], dim=-1, descending=True).values
    margin = (sv[:, cfg.top_k - 1] - sv[:, cfg.top_k]).min().item()
    assert margin >= TOPK_MARGIN, f"case {name}: near-tie in top-k (margin {margin:.2e})"

    # GEMM1 output h and SwiGLU output a, per sorted row (not returned by moe_forward).
    E = cfg.num_experts
    h = torch.zeros(T * cfg.top_k, 2 * cfg.intermediate)
    for e in range(E):
        lo, hi = int(st["offsets"][e]), int(st["offsets"][e + 1])
        if hi > lo:
            h[lo:hi] = st["t_sorted"][lo:hi] @ p["w1"][e].T + p["b1"][e]
    a = ref.swiglu(h, cfg.swiglu_alpha, cfg.swiglu_limit)

    return dict(cfg=cfg, T=T, x=x, weights=runtime_weights(p), margin=margin,
                stages=dict(t=st["t"], logits=st["logits"], w=st["w"], idx=st["idx"],
                            t_sorted=st["t_sorted"], src=st["src"], inv=st["inv"],
                            offsets=st["offsets"], h=h, a=a, y_sorted=st["y_sorted"],
                            moe=st["moe_out"], out=out))
