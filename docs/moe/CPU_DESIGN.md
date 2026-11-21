# Milestone 1A — CPU operator split and dispatch representation (Checkpoint 2)

Status: **APPROVED (Checkpoint 2) and IMPLEMENTED** in `src/ops/kernels_moe.cpp`. The decisions and what was learned while implementing them are in [`docs/ENGINEERING_DECISIONS.md`](../ENGINEERING_DECISIONS.md). The router uses `MatMul + Add + MoETopK`, as proposed.

This document fixes three things before any kernel is written:

1. how the MoE block becomes graph nodes,
2. what the dispatch buffers look like, and
3. how both fit the runtime's invariants:
   - static shapes,
   - the executor never allocates,
   - kernels are free functions keyed on `(op_type, dtype of outputs[0])`.

The math is the one already verified against the official code in [SEMANTICS.md](SEMANTICS.md).

---

## 1. Constraints from the existing runtime (read from the code, not assumed)

| Fact | Where | Consequence for MoE |
|---|---|---|
| All tensor shapes are fixed in `TensorInfo` before `Executor` is built. | `include/mtrt/node.h`, CLAUDE.md invariant 3 | Every MoE buffer needs a shape that does not depend on routing. |
| `run()` must not allocate. Intermediates are views into one planned arena. | `src/runtime/executor.cpp`, invariant 4 | No `std::vector` scratch inside kernels, including the counting sort. |
| A kernel is looked up by the **dtype of `outputs[0]`**. | `executor.cpp` (`registry.lookup(op, dtype of outputs[0])`) | An op with mixed outputs (f32 + i32) must list its f32 output first. |
| `gemm_auto(A,B,C,M,N,K)` computes C = A·B, row-major, with `B` stored as `[K,N]`. It **overwrites** C. | `backends/cpu/gemm.h`, `gemm.cpp` | Official weights are `[out,in]`, so they must be transposed once (see D3). Bias is added after the GEMM. |
| `Add` broadcasts a last-dim bias. `MatMul` exists. | `executor.cpp` validate, `kernels_matmul.cpp` | The router's linear layer and the residual can reuse existing ops. |
| The JSON loader parses `f32` and `i32` tensors and int/double/int-list attrs. | `src/frontend/json/json_loader.cpp` | Index tensors can be `i32`. No loader change is needed. |
| The test `.npy` reader is **f32 only**. | `tests/support/npy.h:56` | Index goldens (`idx`, `src`, `inv`, `offsets`) need a small `<i4` extension. |

---

## 2. Proposed graph (one MoE layer)

```
x[T,H] ──► RMSNorm(eps) ──► t[T,H]
t, gate_wT[H,E] ──► MatMul ──► z[T,E];  z, gate_b[E] ──► Add ──► logits[T,E]
logits ──► MoETopK(k) ──► w[T,k] f32,  idx[T,k] i32
t, idx ──► MoEDispatch(E) ──► t_sorted[T·k,H] f32, src[T·k] i32, inv[T·k] i32, offsets[E+1] i32
t_sorted, offsets, W1T[E,H,2I], b1[E,2I] ──► MoEGroupedMatMul ──► h[T·k,2I]
h ──► SwiGLUInterleaved(alpha, limit) ──► a[T·k,I]
a, offsets, W2T[E,I,H], b2[E,H] ──► MoEGroupedMatMul ──► y_sorted[T·k,H]
y_sorted, inv, w ──► MoECombine ──► m[T,H]
x, m ──► Add ──► out[T,H]
```

**New ops: 6** (`RMSNorm`, `MoETopK`, `MoEDispatch`, `MoEGroupedMatMul`, `SwiGLUInterleaved`,
`MoECombine`). **Reused ops: 2** (`MatMul`, `Add`). `MoEGroupedMatMul` is one kernel used twice.

Every shape above depends only on T, H, I, E and k. The routing decides only the **values** in
`offsets`, `src` and `inv`.

### Op contracts

| Op | Inputs | Outputs (outputs[0] first) | Attrs | What it computes |
|---|---|---|---|---|
| `RMSNorm` | x [T,H], scale [H] | t [T,H] | `eps` | `x / sqrt(mean(x²)+eps) * scale` |
| `MoETopK` | logits [T,E] | **w [T,k] f32**, idx [T,k] i32 | `k` | top-k sorted descending, ties go to the lower index, then softmax over the k values |
| `MoEDispatch` | t [T,H], idx [T,k] | **t_sorted [Tk,H] f32**, src, inv [Tk] i32, offsets [E+1] i32 | `num_experts` | stable counting sort of the assignments `a = token·k + slot` by expert, then a row gather |
| `MoEGroupedMatMul` | A [Tk,K], offsets [E+1], W [E,K,N], b [E,N] | C [Tk,N] | — | for each expert e with rows `[off[e], off[e+1])`: `C_seg = A_seg·W[e] + b[e]`. Empty segments are skipped. |
| `SwiGLUInterleaved` | h [Tk,2I] | a [Tk,I] | `alpha`, `limit` | `g=min(h[2j],L)`, `l=clamp(h[2j+1],-L,L)`, `a[j]=g·σ(αg)·(l+1)` |
| `MoECombine` | y_sorted [Tk,H], inv [Tk], w [T,k] | m [T,H] | — | `m[t] = Σ_s w[t,s]·y_sorted[inv[t·k+s]]`, summed in slot order s=0..k−1 |

---

## 3. Decisions, with the alternatives considered

Each decision gets an ID. They move into `docs/ENGINEERING_DECISIONS.md` once approved.

### D-MoE-1 — Split into stage ops, not one monolithic `MoE` op

| Option | Pro | Con |
|---|---|---|
| A. One `MoE` op | Fewest nodes. Can fuse freely. | One opaque golden test. The profiler sees one bar. Can't reuse MatMul/Add. |
| **B. Stage ops (proposed)** | Each stage has its own golden test from `moe_ref` stages. The profiler gives a per-stage time breakdown, which H5 (Amdahl) needs. Maps 1:1 onto the GPU kernels in 1B. | More nodes, plus the intermediates `h` and `a` (see the memory note). |
| C. Per-expert MatMul nodes | Reuses MatMul entirely. | **Impossible under static shapes.** Each expert's row count is data-dependent. |

Option C shows why a grouped op must exist at all. A static graph cannot hold a
data-dependent number of rows per expert, so a single op must read `offsets` **at run time** and
loop over the segments.

### D-MoE-2 — Separate `SwiGLUInterleaved` instead of fusing it into GEMM1

The unfused version is the correctness baseline. It is also exactly V0's structure in 1B.
Fusing SwiGLU into GEMM1's epilogue would be a later optimization, landed as its own commit
with a measured before/after, per the repo rule of one optimization per commit. The cost is
`h[Tk,2I]`, the largest intermediate: 377.5 MB at T=4096, but only 0.18 MB at T=4. The memory
planner frees it after SwiGLU.

### D-MoE-3 — Transpose the expert and router weights once, at export

The official layout is `[E, out, in]`. `gemm_auto` wants `B` as `[K=in, N=out]`.

| Option | Pro | Con |
|---|---|---|
| **A. Store `W1T [E,H,2I]`, `W2T [E,I,H]`, `gate_wT [H,E]` (proposed)** | Reuses the tuned, threaded NEON GEMM unchanged. Zero runtime cost. | The on-disk layout differs from the checkpoint, so it must be documented. |
| B. Add a GEMM that takes transposed B (dot products of rows) | Keeps the checkpoint layout. Natural for decode (m small). | A new, untuned kernel. The GEMM ladder doesn't support it. A big change to a component that isn't the subject of this study. |

Transposing does **not** break the interleaving. Column `j` of `h` is still row `j` of the
original `W1[e]`, so even columns stay gate and odd columns stay linear. The exporter will assert
`W1T[e] == W1[e].T` exactly, and `moe_ref.py` keeps the official layout.

### D-MoE-4 — Dispatch representation: sorted-contiguous rows

| Option | Pro | Con |
|---|---|---|
| **A. Sorted contiguous (proposed)**: `t_sorted[Tk,H]` + `offsets[E+1]` + `src`/`inv` | Each expert's rows are one contiguous block, so `gemm_auto` runs on it directly. Static size Tk·H. No tokens dropped. | Pays one gather copy (Tk·H floats). |
| B. Capacity buffers `[E, C, H]` (Switch/GShard style) | Fixed per-expert shape, so a plain batched GEMM works. | **Changes the semantics**: tokens beyond capacity C are dropped, or C = T, which wastes E·T·H memory. GPT-OSS does not drop tokens. |
| C. No physical gather: the GEMM reads A rows through `src` | Saves the Tk·H copy. | Needs a GEMM that can read gathered rows, i.e. a new kernel. Kept as a **candidate 1B GPU optimization**. |

**Why keep both `src` and `inv`?**
- `src[j]` = which assignment sits in sorted row j. Dispatch uses it, and it is how you read the
  sort.
- `inv[a]` = which sorted row holds assignment a. Combine uses it so that each output row
  **reads** its k results (a gather). It never **scatter-adds** into a shared row.
- One rule ties them together: `inv[src[j]] = j`. The cost is Tk int32s (64 KB at T=4096).

**Why `int32`?** Tk ≤ 2³¹ by a huge margin. It also matches the existing `i32` dtype, and it is
half the bandwidth of int64.

### D-MoE-5 — Counting sort with no scratch memory

A counting sort normally needs a `cursor[E]` scratch array, but `run()` must not allocate. The
trick is to reuse `offsets` itself:

```
1. offsets[0..E] = 0;  for a in 0..Tk-1:  offsets[idx[a] + 1]++          // histogram
2. prefix sum:         offsets[e+1] += offsets[e]                          // offsets[e] = start of e
3. for a in 0..Tk-1 (ascending → stable):
       e = idx[a];  j = offsets[e]++;  src[j] = a;  inv[a] = j;  copy t[a / k] → t_sorted[j]
   // now offsets[e] == old offsets[e+1] (the END of e), so shift right by one:
4. for e = E down to 1: offsets[e] = offsets[e-1];  offsets[0] = 0
```

- Total cost is O(Tk + E). There is no comparison sort.
- **Stability** (ascending `a` within each expert) comes from step 3's ascending loop. That
  makes the layout deterministic, so it can be golden-tested exactly.
- The gather is fused into the scatter loop.

### D-MoE-6 — Top-k selection: a single pass with strict `>`

k = 4 and E = 32, so a full sort per token is wasteful. The plan is to keep a sorted array of 4
and insert logit `e` only when it is **strictly greater** than the current 4th value. Experts are
visited in ascending index order, so on a tie the earlier (lower-index) expert stays. That
implements the tie rule from SEMANTICS.md without a stable sort. The softmax over the 4 values
subtracts the max first, as `torch.softmax` does.

### D-MoE-7 — Combine is a gather. The residual reuses `Add`.

- Each output row `m[t]` is written by exactly one loop iteration. There are no races, and the
  summation order is fixed (slot 0..k−1), so the result is bitwise-reproducible run to run.
- Folding `+x` into `MoECombine` saves one pass over [T,H]. That is a later, measured fusion,
  not part of the baseline.

### D-MoE-8 — The naive baseline is a benchmark function, not a graph op

- `moe_naive` loops over tokens and slots doing GEMVs against the expert weights. It exists only
  so `bench_moe` can measure the grouped speedup.
- Registering it as an op would violate the rule "narrow beats broad: no ops for completeness".
- It lives in `benchmarks/` and a test checks it against the same golden.

### D-MoE-9 — Expert loop is serial; parallelism comes from `gemm_auto`

- **Baseline:** `MoEGroupedMatMul` calls `gemm_auto` once per non-empty expert, in a serial loop.
- **Known weakness:** when T is small, each segment has m ≈ Tk/E rows. That is below
  `gemm_auto`'s threading threshold (M ≥ 32), so decode-sized runs would use a single core.
- **Possible fix:** parallelize across experts instead of within one GEMM.
- That fix is a candidate optimization. It needs a profile first and a measured number in its
  commit. It is noted here so nobody mistakes it for an oversight.

---

## 4. Invariant checklist

| Invariant | How the design satisfies it |
|---|---|
| 3. Static shapes | Every tensor's shape is a function of (T, H, I, E, k). `offsets[E] == T·k` is asserted in Debug. |
| 4. No allocation in `run()` | The sort uses `offsets` as its cursor (D-MoE-5). Top-k uses a fixed `k ≤ 8` stack array (`k` is asserted). `alloc_invariant_test` gets an MoE graph. |
| 5. Golden test for every op | Per-stage `.npy` from `moe_ref.moe_forward(..., return_stages=True)`, plus the whole layer through `Executor`. |
| 6. Passes keep numerics | No new passes in 1A. |
| 7. Debug asserts | Index ranges, `0 ≤ idx < E`, offsets monotone, `inv[src[j]] == j`. |

## 5. Golden fixtures and tolerances

Generated by `gen_moe_goldens()` from `moe_ref.py`, and therefore traceable to the official code.

| Fixture set | Dims | Purpose |
|---|---|---|
| `moe_small_*` | T=6, H=I=8, E=8, k=4 | Every per-stage tensor. Random weights. |
| `moe_four_*` | same, router bias on {1,3,4,6} | All tokens select the same 4 experts, so 4 experts are empty. |
| `moe_clamp_*` | T=32, weights ×12 | All three clamp sides fire. |
| `moe_e32_*` | T=64, H=I=64, E=32, k=4 | The real routing width, with naturally empty experts. |

**Tolerances:**
- Indices (`idx`, `src`, `inv`, `offsets`) must match **exactly**.
- Float stages use the repo's standard `rtol=1e-5, atol=1e-6`.
- Any loosening needs a written reason.

Small fixtures keep the repo light. The real-dims check (H=I=2880) runs in `bench_moe` against
`moe_ref` output computed on the fly, not stored as a 1 GB fixture.

**`npy.h` change:** accept `'<i4'` and return `int32` data. It is ~15 lines, test-support only.

## 6. File plan (after approval)

| File | Change |
|---|---|
| `src/ops/kernels_moe.cpp` | new: the 6 kernels + `register_moe_kernels` |
| `src/ops/kernels.h`, `registry.cpp` | declare and call `register_moe_kernels` |
| `tests/support/npy.h` | `<i4` support |
| `tests/moe_test.cpp` | stage goldens, the dispatch invariants from the plan, repeated `run()` with different routings |
| `python/gen_goldens.py` | `gen_moe_goldens()` |
| `python/export_models.py` | `export_moe()` → `models/moe_layer.json/.bin` (transposed weights, D-MoE-3) |
| `tests/alloc_invariant_test.cpp` | add the MoE graph |
| `benchmarks/bench_moe.cpp` | naive vs grouped at real dims |
| `docs/ENGINEERING_DECISIONS.md` | D-MoE-1..9 |
| `docs/moe/DEEP_DIVE_1A.md` | questions only |

The implementation order follows the repo rule "golden test before kernel": for each op, fixture
→ failing test → kernel.

## 7. Open questions for the reviewer

1. Approve the **stage split** (D-MoE-1) and an unfused SwiGLU for the baseline (D-MoE-2)?
2. Approve **transposing weights at export** (D-MoE-3), rather than writing a transposed-B GEMM?
3. Approve the **sorted-contiguous** dispatch with `src` + `inv` + `offsets` (D-MoE-4)?
4. Should the router be `MatMul + Add + MoETopK` (reuses 2 tested ops, proposed), or a single
   `MoERouter` op (one node, logits+bias fused)?
