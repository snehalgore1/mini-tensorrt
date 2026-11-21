# GPT-OSS MoE block — verified semantics (Checkpoint 1)

This document pins down exactly what the GPT-OSS MoE block computes, before any kernel in this
repo implements it. A fast implementation of the wrong function is not an optimization.

- **Oracle:** `openai/gpt-oss` @ [`2770201`](https://github.com/openai/gpt-oss/tree/277020131d524f06bed5f66b694a92ed7f6e5224),
  `gpt_oss/torch/model.py`. It is vendored verbatim, with attribution, in
  `python/third_party/gpt_oss_mlpblock.py`.
- **Dimensions:** Hugging Face [`openai/gpt-oss-20b/config.json`](https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json).
- **Our reference:** `python/moe_ref.py`. It computes the same math in the staged
  router → dispatch → grouped-experts → combine form that the C++ and CUDA kernels will use.
- **Proof of equivalence:** `python/check_official_moe.py` (results below).

Notation: T = tokens, H = hidden size, I = expert intermediate size, E = experts, k = experts per
token.

## 1. Dimensions (gpt-oss-20b)

| Symbol | Value | config.json key |
|---|---|---|
| H | 2880 | `hidden_size` |
| I | 2880 | `intermediate_size` |
| E | 32 | `num_local_experts` |
| k | 4 | `num_experts_per_tok` / `experts_per_token` |
| RMSNorm eps | 1e-5 | `rms_norm_eps` |
| SwiGLU limit, alpha | 7.0, 1.702 | `ModelConfig.swiglu_limit` ([L20](https://github.com/openai/gpt-oss/blob/277020131d524f06bed5f66b694a92ed7f6e5224/gpt_oss/torch/model.py#L20)); alpha is the `swiglu` default ([L249](https://github.com/openai/gpt-oss/blob/277020131d524f06bed5f66b694a92ed7f6e5224/gpt_oss/torch/model.py#L249)) |

Note that `ModelConfig`'s *defaults* are the 120B model (E=128, 36 layers). The 20B values come
from its own `config.json`.

## 2. The computation, stage by stage

The whole block is `MLPBlock.forward`
([L312–336](https://github.com/openai/gpt-oss/blob/277020131d524f06bed5f66b694a92ed7f6e5224/gpt_oss/torch/model.py#L312-L336)).
For an input x of shape [T, H]:

### 2.1 RMSNorm, inside the block ([L43–47](https://github.com/openai/gpt-oss/blob/277020131d524f06bed5f66b694a92ed7f6e5224/gpt_oss/torch/model.py#L43-L47))

```
t = x / sqrt( mean_j(x_j²) + eps ) * scale          computed in fp32
```

- There is no mean subtraction and no bias, unlike GPT-2's LayerNorm. The MoE block owns its
  norm, and the residual add at the end uses the *un-normalized* x.
- The normalization runs in fp32 whatever the input dtype (`x.float()`). It is then multiplied by
  `scale` under normal torch type promotion. `scale` is an fp32 parameter
  ([L39–41](https://github.com/openai/gpt-oss/blob/277020131d524f06bed5f66b694a92ed7f6e5224/gpt_oss/torch/model.py#L39-L41)).
- **Found during verification:** our first draft cast `scale` to fp32. That matches in FP32 but
  rounds differently whenever `scale` has a wider dtype. The FP64 check caught it (error 1.1e-7
  instead of ~1e-16). The fix is to follow promotion exactly, as the official code does.
- **Not scale-invariant at small magnitudes:** RMSNorm(c·x) = RMSNorm(x) only while
  mean(x²) ≫ eps. With |x| ~ 1e-3, mean(x²) ~ 1e-6 < eps = 1e-5, so eps dominates the
  denominator.

### 2.2 Router ([L271–273](https://github.com/openai/gpt-oss/blob/277020131d524f06bed5f66b694a92ed7f6e5224/gpt_oss/torch/model.py#L271-L273), [L314–317](https://github.com/openai/gpt-oss/blob/277020131d524f06bed5f66b694a92ed7f6e5224/gpt_oss/torch/model.py#L314-L317))

```
logits = t @ Wgᵀ + bg                         Wg [E, H], bg [E]   (nn.Linear has bias=True by default)
(vals, idx) = topk(logits, k=4, sorted=True)  [T, 4], descending
w = softmax(vals)                             [T, 4], over the 4 selected logits only
```

- **Softmax comes after top-k.** The 4 weights form a probability distribution over the
  selected experts only, so they sum to exactly 1. If softmax came first over all E experts and
  then top-k was taken, the weights would sum to less than 1. Taking the softmax after top-k is
  equivalent to "softmax over everything, then renormalize the top 4". That is why the official
  code needs no separate renormalization step.
- The router is **not quantized** in the shipped checkpoint (HF `quantization_config.modules_to_not_convert`
  lists `model.layers.*.mlp.router`). Only the expert weights are MXFP4.
- **Ties:** `torch.topk` does not specify which index wins a tie. A runtime needs a defined rule
  so that CPU and GPU agree. **Our rule: the lower expert index wins**, implemented as a stable
  descending sort (`moe_ref.topk_stable`). Golden inputs are tie-free, so this choice never
  affects the comparison with the official code.

### 2.3 Expert MLP1 + SwiGLU ([L275–292](https://github.com/openai/gpt-oss/blob/277020131d524f06bed5f66b694a92ed7f6e5224/gpt_oss/torch/model.py#L275-L292), [L320–323](https://github.com/openai/gpt-oss/blob/277020131d524f06bed5f66b694a92ed7f6e5224/gpt_oss/torch/model.py#L320-L323), [L249–256](https://github.com/openai/gpt-oss/blob/277020131d524f06bed5f66b694a92ed7f6e5224/gpt_oss/torch/model.py#L249-L256))

For each token and each selected expert e:

```
h   = W1[e] @ t + b1[e]                 W1 [E, 2I, H]  (out, in),  b1 [E, 2I]   → h ∈ R^{2I}
glu = clamp(h[0::2], max = 7)           even columns; UPPER clamp only
lin = clamp(h[1::2], -7, 7)             odd columns; clamped on both sides
a   = glu · σ(1.702 · glu) · (lin + 1)  a ∈ R^{I}
```

- **The two halves are interleaved**, not stacked. Column 2j is the gate for output j and column
  2j+1 is its linear partner. A kernel that splits h into `h[:I]` and `h[I:]` silently computes
  the wrong function. This matters for the fused-epilogue GEMM in Milestone 1B: the pairing is
  (2j, 2j+1).
- `x·σ(1.702x)` is the sigmoid approximation of GELU (a "SiLU with alpha"). The `+1` on the
  linear half is part of the trained model (the comment at L255 says so).
- The clamps are asymmetric. A very negative gate value is already driven to ~0 by the sigmoid,
  so only its top is clamped.

### 2.4 Expert MLP2 ([L293–310](https://github.com/openai/gpt-oss/blob/277020131d524f06bed5f66b694a92ed7f6e5224/gpt_oss/torch/model.py#L293-L310), [L326–331](https://github.com/openai/gpt-oss/blob/277020131d524f06bed5f66b694a92ed7f6e5224/gpt_oss/torch/model.py#L326-L331))

```
y = W2[e] @ a + b2[e]                   W2 [E, H, I] (out, in),  b2 [E, H]
```

In the official code, b2 is added *after* a tensor-parallel `all_reduce` (L329–331), so it is
added once rather than once per rank. On one device this is just `+ b2`.

### 2.5 Combine + residual ([L333–336](https://github.com/openai/gpt-oss/blob/277020131d524f06bed5f66b694a92ed7f6e5224/gpt_oss/torch/model.py#L333-L336))

```
out[t] = x[t] + Σ_{s=0..3} w[t,s] · y[t,s]
```

## 3. Gathered (official) vs grouped (ours) formulation

The official code is written for readability. It **gathers every token's 4 weight matrices**
(`mlp1_weight[expert_indices]` → a [T, 4, 2I, H] copy) and runs a batched einsum. At 20B dims
that is T × 4 × 5760 × 2880 values. Each token re-reads its experts' full weights, and two tokens
that pick the same expert read that expert's weights twice.

`moe_ref.py` computes the same function grouped by expert, which is how the kernels will run it:

| Step | Definition | Shape (static for a given T) |
|---|---|---|
| assignment id | a = token·k + slot | T·k |
| `offsets` | exclusive prefix sum of per-expert counts; `offsets[E] = T·k` | E+1 |
| `src[j]` | assignment id at sorted row j (stable counting sort by expert) | T·k |
| `inv[a]` | sorted row of assignment a; `inv[src[j]] = j` | T·k |
| `t_sorted[j]` | `t[src[j] // k]` | [T·k, H] |
| experts | for each e: rows `offsets[e]..offsets[e+1]` → MLP1 → SwiGLU → MLP2 | [T·k, H] |
| combine | `out[t] = Σ_s w[t,s] · y_sorted[inv[t·k+s]]` | [T, H] |

Design facts that later decisions depend on:

- **Every token has exactly k assignments, so T·k is fixed.** All dispatch buffers have static
  shapes; only the split points (`offsets`) depend on the data. This is how dynamic routing fits
  MiniTensorRT's static-shape, plan-ahead memory model. It will be the first entry in
  `docs/ENGINEERING_DECISIONS.md`.
- **The mapping is unambiguous.** `src[j]` gives both the token (`// k`) and the slot (`% k`) of
  each sorted row. `inv` turns the combine into a **gather**: each output row has exactly one
  writer, so there are no atomics or races, and the slot summation order is fixed.
- The counting sort is **stable**: within an expert, rows stay in assignment order. That makes
  the result deterministic, independent of thread scheduling.
- An empty expert (count 0) never touches its weights. In the grouped form, each expert's
  weights are read once per GEMM no matter how many tokens chose it.

## 4. Verification results

Command: `.venv/bin/python python/check_official_moe.py --real` (torch 2.15.0, CPU, Apple M1 Pro).
Both implementations get identical parameters and inputs. Parameters are random, fan-in-scaled
and rounded to bf16-representable values, so the same kernels can later take real checkpoint
weights.

Metrics:
- `out_rel` = max|ours − official| / max|official|, i.e. normwise relative error. FP32 gates
  use **out_rel ≤ 1e-6**. An absolute tolerance is meaningless when the clamp case produces
  outputs around 10³.
- `moe_*` compares only the MoE contribution, with the residual removed. The official block only
  returns x + MoE(x), so its contribution has to be recovered as (out − x). That subtraction
  adds error of about eps₃₂·max|x|, and the allowance includes it.

| Check | Routing identical | out_abs | out_rel | Notes |
|---|---|---|---|---|
| **FP64, 50 random seeds** (T=6, H=I=8, E=8, k=4) | yes, all | 8.9e-16 | **3.2e-16** | Same math. Only fp64 rounding remains. |
| FP32, 50 random seeds | yes, all | 7.2e-7 | 2.4e-7 | |
| FP32 T=6 + naive path | yes | 2.4e-7 | 9.5e-8 | grouped vs naive rel 4.8e-8 |
| FP32 E=32, k=4, T=64 + naive | yes | 7.2e-7 | 1.8e-7 | grouped vs naive rel 1.8e-7 |
| FP32 same-four-experts (router bias +20 on {1,3,4,6}) | yes | 2.4e-7 | 8.8e-8 | All tokens chose {1,3,4,6}. **4 empty experts.** |
| FP32 clamp boundaries (weights ×12, T=32) | yes | 1.8e-4 | 2.0e-7 | All three clamps fired: glu>7: 317, lin<−7: 324, lin>7: 248 |
| FP32 input magnitude 1e-3 | yes | 8.9e-8 | 1.6e-7 | |
| FP32 input magnitude 1 | yes | 2.4e-7 | 6.5e-8 | |
| FP32 input magnitude 1e3 | yes | 0 | 0 | moe_rel 4.2e-5 is entirely the (out − x) recovery: eps₃₂·10³ ≈ 1.2e-4 |
| Tie rule (ours only) | — | — | — | logits [1,3,3,0.5,3,2] → [1,2,4,5] |
| **FP32 real 20B MoE dims** (H=I=2880, E=32, k=4, T=4) | yes | 3.1e-6 | **7.5e-7** | 19 of 32 experts empty (4 tokens × 4 = 16 assignments, 13 distinct experts) |

**Informational: bf16, the precision the model ships in, vs our FP32** (E=32, k=4, T=64):
out_rel 4.5e-3 and MoE-contribution rel 8.5e-3. All 64 tokens picked the same expert *set*. In
some tokens the order inside the top 4 changed, because near-tied logits round differently in
bf16. So bf16 arithmetic costs roughly 10⁻² relative error against FP32, about 4 orders of
magnitude more than the summation-order differences above.

### What the numbers mean
- **FP64 at 3e-16 means the semantics are identical.** If routing, interleaving, clamping,
  softmax placement or bias handling differed, the error would be order 1, not machine epsilon.
- **FP32 differences of 1e-7 to 1e-6 are summation order.** The official einsum and our segment
  GEMMs add the same products in a different order. The error grows slightly with the reduction
  length: 7.5e-7 at K=2880 vs about 1e-7 at K=8. That is the expected behavior of fp32
  accumulation.
- **Tolerance rule (revised at Checkpoint 2, replaces the earlier normwise plan):**
  - Every float comparison is **elementwise and combined**: `|got − ref| ≤ atol + rtol·mag`.
    The repo standard is `rtol = 1e-5, atol = 1e-6`.
  - **Max abs error and max rel error are reported separately.** Max rel is taken only over
    elements with `|ref| ≥ atol/rtol = 0.1`. Below that floor the atol term dominates, and a
    relative error against a near-zero value is meaningless, so those elements are only
    counted.
  - **Per-stage checks** use `mag = |ref|`.
  - **Whole-layer checks** use `mag` = the summed magnitude of the terms that produced each
    output: `|x| + Σ_s w·(|a|·|W2| + |b2|)`.
    - FP32 summation error scales with Σ|term|, and it does not shrink when the terms cancel.
    - Measured case 1: the clamp case has outputs ≈ 6 built from terms ≈ 585.
    - Measured case 2: the 20B-dim check has 2880-term dot products.
    - In both cases `mag = |ref|` reports pure rounding noise as an 8× (clamp case) or 2×
      (20B dims) violation, while the term-magnitude rule uses 12% and 1.4% of its budget.
    - A real bug (a wrong expert or a wrong column) gives errors on the order of `mag`, which is
      ~10⁵× the tolerance, so this rule does not hide bugs.
  - **Routing indices** (`idx`, `src`, `inv`, `offsets`) must match exactly. Goldens are
    tie-free by construction: the k-th vs (k+1)-th logit gap is ≥ 1e-4.
  - Any bf16 or FP16 path later gets its own documented tolerance. The measurement above
    suggests about 1e-2.

## 5. What this does *not* verify
- **Real checkpoint weights.** The parameters are random (bf16-representable). The MXFP4
  dequantization of real expert weights (`gpt_oss/torch/weights.py`) is out of scope for
  Milestones 1A and 1B.
- **Other official backends.** Only the PyTorch reference implementation was used, not the
  Triton/MXFP4 kernels (`gpt_oss/triton/moe.py`) or the Hugging Face `transformers` port.
- **Attention, RoPE, sinks, the embedding and the LM head.** These are not part of the MoE
  block.

## 6. Open question for review
- The golden fixtures (next step, 1A) will use **T=6, H=I=8, E=8, k=4**, with extra fixtures for
  same-four-experts, clamp boundaries and input magnitudes. E=8 is smaller than the real E=32,
  so that each fixture stays a few hundred bytes. The E=32/k=4 routing width is already covered
  above and will get its own fixture as well.
