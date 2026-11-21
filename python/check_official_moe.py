"""Checkpoint 1: prove python/moe_ref.py == the official GPT-OSS MLPBlock.

The oracle is a verbatim copy of `MLPBlock` from openai/gpt-oss@2770201
(python/third_party/gpt_oss_mlpblock.py). Both implementations receive identical
parameters and inputs. The comparisons, in order:

  1. FP64  : both computed in float64 (RMSNorm is fp32 in both, by design of the
             official code). Separates "same math" from "same rounding". Any mismatch
             here would be a SEMANTIC bug, not floating-point noise.
  2. FP32  : the precision the MiniTensorRT kernels run in. Differences come only from
             summation order (einsum over a gathered [T,k,2I,H] tensor vs our per-expert
             segment GEMMs).
  3. Cases : random seeds, real E=32/k=4 routing width, all tokens on the same four
             experts (E-4 experts empty), clamp boundaries hit on both sides, input
             magnitudes 1e-3 .. 1e3, plus our grouped path == our naive path.
  4. BF16  : official run in its shipped dtype (bf16 weights/activations) vs our FP32
             path -- informational only: the size of the error bf16 itself introduces.
  5. --real: one run at true gpt-oss-20b MoE dims (H=I=2880, E=32, k=4), T=4.
             Needs ~6 GB RAM.

Run:  .venv/bin/python python/check_official_moe.py [--real]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import moe_ref as ref  # noqa: E402
from third_party.gpt_oss_mlpblock import MLPBlock, ModelConfig  # noqa: E402

FAILS: list[str] = []


def official_block(cfg: ref.MoEConfig, p: dict, dtype) -> MLPBlock:
    """Build the official module and load our parameters into it (no copies)."""
    oc = ModelConfig(num_experts=cfg.num_experts, experts_per_token=cfg.top_k,
                     hidden_size=cfg.hidden, intermediate_size=cfg.intermediate,
                     swiglu_limit=cfg.swiglu_limit)
    with torch.device("meta"):  # skip the official torch.empty allocation
        m = MLPBlock(oc)
    P = torch.nn.Parameter
    # Official RMSNorm.scale is fp32 in every mode (model.py:39-41).
    m.norm.scale = P(p["norm_scale"].float() if dtype != torch.float64 else p["norm_scale"],
                     requires_grad=False)
    m.norm.eps = cfg.rms_eps
    m.gate.weight = P(p["gate_w"].to(dtype), requires_grad=False)
    m.gate.bias = P(p["gate_b"].to(dtype), requires_grad=False)
    m.mlp1_weight = P(p["w1"].to(dtype), requires_grad=False)
    m.mlp1_bias = P(p["b1"].to(dtype), requires_grad=False)
    m.mlp2_weight = P(p["w2"].to(dtype), requires_grad=False)
    m.mlp2_bias = P(p["b2"].to(dtype), requires_grad=False)
    return m


def official_forward(m: MLPBlock, x: torch.Tensor, k: int):
    """Run the official forward; capture the gate logits with a hook so routing can be
    compared directly (re-applying the identical torch.topk call from model.py:315)."""
    cap = {}
    h = m.gate.register_forward_hook(lambda _m, _i, o: cap.__setitem__("g", o))
    with torch.inference_mode():
        out = m(x)
    h.remove()
    top = torch.topk(cap["g"], k=k, dim=-1, sorted=True)
    return out, top.indices, torch.softmax(top.values, dim=1)


def errs(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """(max abs error, normwise rel error = max|a-b| / max|b|)."""
    d = (a.double() - b.double()).abs().max().item()
    return d, d / max(b.double().abs().max().item(), 1e-300)


RTOL, ATOL = 1e-5, 1e-6   # repo standard for FP32 (CLAUDE.md "Reference baselines")


def term_scale(x, st, p, k) -> torch.Tensor:
    """Per-element size of what was SUMMED to make each output, not the size of the result.

        out[t] = x[t] + sum_s w[t,s] * (W2[e] @ a + b2[e])
        scale[t] = |x[t]| + sum_s w[t,s] * (|W2[e]| @ |a| + |b2[e]|)

    Floating-point summation error is bounded by (number of terms * eps) * sum |term|, and
    that bound does not shrink when the terms cancel. An output element that ends up near
    zero after a 2880-term dot product still carries an error proportional to its terms.
    Comparing against |ref| alone would wrongly fail it; comparing against `scale` does not.
    """
    xd = x.double()
    T = x.shape[0]
    out = xd.abs().clone()
    for t in range(T):
        for s in range(k):
            j = int(st["inv"][t * k + s])
            e = int(st["idx"][t, s])
            h = st["t_sorted"][j].double() @ p["w1"][e].double().T + p["b1"][e].double()
            act = ref.swiglu(h, ref.GPT_OSS_20B.swiglu_alpha, ref.GPT_OSS_20B.swiglu_limit)
            out[t] += st["w"][t, s].double() * (p["w2"][e].double().abs() @ act.abs()
                                                 + p["b2"][e].double().abs())
    return out


def close_stats(got: torch.Tensor, ref: torch.Tensor, rtol=RTOL, atol=ATOL, scale=None) -> dict:
    """Elementwise combined tolerance, as numpy.allclose: |got-ref| <= atol + rtol*|ref|.

    Reported SEPARATELY:
      max_abs  = max |got-ref|
      max_rel  = max |got-ref|/|ref| over elements with |ref| >= atol/rtol only. Below that
                 floor the atol term dominates the tolerance, and a relative error against a
                 near-zero reference is meaningless (it blows up as ref -> 0).
      near0    = how many elements fell under the floor (they are judged by atol)
      worst    = max |got-ref| / (atol + rtol*|ref|); the gate is worst <= 1
    """
    g, r = got.double(), ref.double()
    d = (g - r).abs()
    floor = atol / rtol
    big = r.abs() >= floor
    mag = r.abs() if scale is None else scale.double()
    return dict(max_abs=d.max().item(),
                max_rel=(d[big] / r[big].abs()).max().item() if big.any() else 0.0,
                near0=int((~big).sum()),
                worst=(d / (atol + rtol * mag)).max().item())


def report(name, ok, **kv):
    tag = "PASS" if ok else "FAIL"
    if not ok:
        FAILS.append(name)
    fields = "  ".join(f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}" for k, v in kv.items())
    print(f"[{tag}] {name:<34} {fields}")


def compare(name, cfg, p, x, dtype, tol_rel=1e-6, check_naive=False, clamp_stats=False,
            term_gate=False):
    """Gates are normwise-RELATIVE (max|a-b| / max|b|): an absolute tolerance is
    meaningless when the clamp case produces outputs ~1e3. The MoE-only contribution of
    the official block is recovered as (out - x), which itself carries a rounding error
    of ~eps32*max|x| -- so its allowance includes that term (dominant when |x| >> |moe|)."""
    xd = x.to(dtype)
    pd = {k: v.to(dtype) for k, v in p.items()}
    m = official_block(cfg, p, dtype)
    o_out, o_idx, o_w = official_forward(m, xd, cfg.top_k)
    r_out, st = ref.moe_forward(xd, pd, cfg, return_stages=True)

    idx_equal = bool(torch.equal(o_idx, st["idx"]))
    w_abs, _ = errs(st["w"], o_w)
    cs = close_stats(r_out, o_out, scale=term_scale(xd, st, pd, cfg.top_k) if term_gate else None)
    if term_gate:  # also show what the plain |ref|-relative check would have said
        kv0 = close_stats(r_out, o_out)["worst"]
    _, out_normrel = errs(r_out, o_out)
    moe_abs, moe_rel = errs(st["moe_out"], o_out - xd)  # MoE contribution, residual removed
    kv = dict(idx_equal=idx_equal, w_abs=w_abs, out_abs=cs["max_abs"], out_rel=cs["max_rel"],
              near0=cs["near0"], tol_used=cs["worst"], out_normrel=out_normrel,
              moe_abs=moe_abs, empty_experts=int((st["offsets"][1:] == st["offsets"][:-1]).sum()))
    if term_gate:
        kv["tol_used_vs_|ref|"] = kv0
    eps = torch.finfo(dtype).eps
    moe_allow = tol_rel * st["moe_out"].abs().max().item() + 4 * eps * xd.abs().max().item()
    ok = idx_equal and cs["worst"] <= 1.0 and moe_abs <= moe_allow
    if check_naive:
        n_abs, n_rel = errs(ref.moe_forward_naive(xd, pd, cfg), r_out)
        kv["naive_vs_grouped_rel"] = n_rel
        ok = ok and n_rel <= tol_rel
    if clamp_stats:
        h = torch.cat([st["t_sorted"][lo:hi] @ pd["w1"][e].T + pd["b1"][e]
                       for e in range(cfg.num_experts)
                       for lo, hi in [(int(st["offsets"][e]), int(st["offsets"][e + 1]))]
                       if hi > lo])
        kv["glu>7"] = int((h[:, 0::2] > cfg.swiglu_limit).sum())
        kv["lin<-7"] = int((h[:, 1::2] < -cfg.swiglu_limit).sum())
        kv["lin>7"] = int((h[:, 1::2] > cfg.swiglu_limit).sum())
        ok = ok and kv["glu>7"] > 0 and kv["lin<-7"] > 0 and kv["lin>7"] > 0
    report(name, ok, **kv)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="also run true 20B MoE dims (T=4)")
    ap.add_argument("--seeds", type=int, default=50)
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 1)
    print(f"torch {torch.__version__}  oracle: openai/gpt-oss@2770201 MLPBlock (verbatim)\n")

    small = ref.MoEConfig(hidden=8, intermediate=8, num_experts=8, top_k=4)
    medium = ref.MoEConfig(hidden=64, intermediate=64, num_experts=32, top_k=4)
    g = torch.Generator().manual_seed(1234)

    print("== 1. FP64: same math? (semantic check) ==")
    worst = (0.0, 0.0)
    for s in range(args.seeds):
        p = ref.make_params(small, seed=s, dtype=torch.float64)
        x = torch.randn(6, small.hidden, generator=g, dtype=torch.float64)
        m = official_block(small, p, torch.float64)
        o, oi, _ = official_forward(m, x, small.top_k)
        r, st = ref.moe_forward(x, p, small, return_stages=True)
        assert torch.equal(oi, st["idx"]), f"routing mismatch seed {s}"
        worst = max(worst, errs(r, o))
    report(f"fp64 small x{args.seeds} seeds", worst[1] <= 1e-12, out_abs=worst[0], out_rel=worst[1])

    print(f"\n== 2/3. FP32: kernel precision, all cases (gate: |d| <= {ATOL:g} + {RTOL:g}*|ref|) ==")
    agg = dict(max_abs=0.0, max_rel=0.0, near0=0, worst=0.0)
    for s in range(args.seeds):
        p = ref.make_params(small, seed=s)
        x = torch.randn(6, small.hidden, generator=g)
        m = official_block(small, p, torch.float32)
        o, oi, _ = official_forward(m, x, small.top_k)
        r, st = ref.moe_forward(x, p, small, return_stages=True)
        assert torch.equal(oi, st["idx"]), f"routing mismatch seed {s}"
        cs = close_stats(r, o)
        agg = {k: (agg[k] + cs[k]) if k == "near0" else max(agg[k], cs[k]) for k in agg}
    report(f"fp32 small x{args.seeds} seeds", agg["worst"] <= 1.0, out_abs=agg["max_abs"],
           out_rel=agg["max_rel"], near0=agg["near0"], tol_used=agg["worst"])

    p = ref.make_params(small, seed=7)
    compare("fp32 small T=6 (+naive)", small, p, torch.randn(6, 8, generator=g),
            torch.float32, check_naive=True)

    p = ref.make_params(medium, seed=8)
    compare("fp32 E=32 k=4 T=64 (+naive)", medium, p, torch.randn(64, 64, generator=g),
            torch.float32, check_naive=True)

    four = {1: 20.0, 3: 20.0, 4: 20.0, 6: 20.0}
    p = ref.make_params(small, seed=9, gate_bias_boost=four)
    st = compare("fp32 same-four-experts T=6", small, p, torch.randn(6, 8, generator=g),
                 torch.float32, check_naive=True)
    sel = sorted(set(st["idx"].flatten().tolist()))
    report("  -> every token picked {1,3,4,6}", sel == [1, 3, 4, 6], selected=str(sel))

    p = ref.make_params(small, seed=10, weight_scale=12.0)
    compare("fp32 clamp boundaries T=32", small, p, torch.randn(32, 8, generator=g),
            torch.float32, clamp_stats=True)

    for mag in (1e-3, 1.0, 1e3):
        p = ref.make_params(small, seed=11)
        x = mag * torch.randn(6, 8, generator=g)
        compare(f"fp32 input magnitude {mag:g}", small, p, x, torch.float32)

    print("\n== tie rule (ours only; torch.topk leaves ties unspecified) ==")
    logits = torch.tensor([[1.0, 3.0, 3.0, 0.5, 3.0, 2.0]])
    _, idx = ref.topk_stable(logits, 4)
    report("ties -> lower index first", idx.tolist() == [[1, 2, 4, 5]], idx=str(idx.tolist()))

    print("\n== 4. BF16 shipped precision vs our FP32 (informational, not a gate) ==")
    p = ref.make_params(medium, seed=12)
    x = torch.randn(64, 64, generator=g).to(torch.bfloat16).float()
    m = official_block(medium, p, torch.bfloat16)
    o, oi, _ = official_forward(m, x.to(torch.bfloat16), medium.top_k)
    r, st = ref.moe_forward(x, p, medium, return_stages=True)
    a, rel = errs(r, o.float())
    tok_diff = int((oi.sort(-1).values != st["idx"].sort(-1).values).any(-1).sum())
    same = (oi.sort(-1).values == st["idx"].sort(-1).values).all(-1)
    a_same, rel_same = errs(r[same], o.float()[same])
    moe_a, moe_rel = errs(st["moe_out"], o.float() - x)
    print(f"[INFO] bf16-official vs fp32-ours     out_abs={a:.3e} out_rel={rel:.3e}  "
          f"moe_abs={moe_a:.3e} moe_rel={moe_rel:.3e}  "
          f"tokens_with_different_expert_set={tok_diff}/64")
    print(f"[INFO]   tokens whose expert set matched  out_abs={a_same:.3e} out_rel={rel_same:.3e}")

    if args.real:
        print("\n== 5. Real gpt-oss-20b MoE dims, FP32, T=4 ==")
        cfg = ref.GPT_OSS_20B
        p = ref.make_params(cfg, seed=13)
        # K = 2880-term dot products: gate against the summed-term magnitude (term_scale),
        # because near-zero outputs keep an error proportional to their terms.
        compare("fp32 H=I=2880 E=32 k=4 T=4", cfg, p,
                torch.randn(4, cfg.hidden, generator=g), torch.float32, term_gate=True)

    print(f"\n{'ALL PASSED' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
