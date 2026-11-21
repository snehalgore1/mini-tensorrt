# Engineering decisions log: MoE track

Each entry records the problem, the options considered, the choice, its cost, and how it is
verified.
- Architectural decisions for the base runtime stay in [DESIGN.md](DESIGN.md).
- The MoE semantics are in [moe/SEMANTICS.md](moe/SEMANTICS.md).
- The CPU op design is in [moe/CPU_DESIGN.md](moe/CPU_DESIGN.md).

Numbers quoted here are measured unless marked *derived* (computed from a measured time and a
formula).

---

## Milestone 1A: CPU correctness and runtime integration

### D-MoE-1 · The MoE block is six stage ops plus reused MatMul/Add, not one monolithic op
- **Problem.** The runtime needs a graph form for a layer whose per-expert row counts depend
  on the data.
- **Options.**
  - (A) One `MoE` op.
  - (B) Stage ops.
  - (C) Per-expert `MatMul` nodes.
- **Choice: B.**
  - Graph: `RMSNorm`, `MatMul`+`Add` (router), `MoETopK`, `MoEDispatch`,
    `MoEGroupedMatMul` ×2, `SwiGLUInterleaved`, `MoECombine`, `Add` (residual).
  - Each stage has its own golden test.
  - The profiler gives per-stage time, which H5/Amdahl needs.
  - The stages map 1:1 onto the GPU kernels in 1B.
- **Why not C.** Per-expert nodes are impossible under static shapes. How many rows an
  expert gets is known only at run time, so exactly one op has to read `offsets` and loop.
- **Cost.** Ten nodes instead of one, plus the `h` and `a` intermediates (sized below).
- **Verified by:** `tests/moe_test.cpp`, which runs 7 tests per golden case × 5 cases.

### D-MoE-2 · SwiGLU stays unfused in the baseline
- **Choice.** Fusing SwiGLU into GEMM1's epilogue would remove `h[T·k, 2I]`, the largest
  intermediate. That is a separate, measured optimization commit.
- **Cost.** Extra memory traffic for `h`. The planner frees `h` after SwiGLU.

### D-MoE-3 · Expert and router weights are transposed once, at export
- **Problem.** The checkpoint stores weights `[out, in]`. The tuned `cpu::gemm_auto` wants a
  row-major `B[K = in, N = out]`.
- **Options.**
  - (A) Store `w1t[E,H,2I]`, `w2t[E,I,H]`, `gate_wT[H,E]`.
  - (B) Write a transposed-B GEMM.
- **Choice: A.** The tuned, threaded GEMM is reused unchanged, at zero runtime cost.
- **Cost.** The on-disk layout differs from the checkpoint. `python/moe_cases.py` asserts
  the transpose is exact.
- **Why the SwiGLU interleave survives.** Column `j` of `h` is still row `j` of the original
  `W1[e]`.

### D-MoE-4 · Dispatch layout: contiguous rows sorted by expert
- **Buffers.** `t_sorted[T·k,H]` + `offsets[E+1]` + `src[T·k]` + `inv[T·k]`, all `int32`
  for the index tensors.
- **Rejected options.**
  - Capacity buffers `[E,C,H]`: these drop tokens or waste E·T·H memory, which changes
    GPT-OSS semantics.
  - Gather-inside-the-GEMM: needs a new kernel. It is kept as a 1B candidate.
- **Invariants.** `offsets` is non-decreasing, `offsets[E] = T·k`, `src` is a permutation,
  `inv[src[j]] = j`, the order inside each expert is stable, and every token appears exactly
  k times.
- **Verified by:**
  - Every golden case.
  - 500 random routings in C++ (`MoE.DispatchInvariantsRandomRoutings`).
  - 41,888 routings in `python/verify_counting_sort.py`: 39,588 exhaustive plus 2,000
    random, plus 300 compared against `moe_ref.dispatch`.

### D-MoE-5 · The counting sort uses `offsets` as its own cursor, so it needs no scratch
- **Problem.** `run()` may not allocate, but a counting sort normally needs `cursor[E]`.
- **Algorithm.**
  1. Histogram into `off[e+1]`.
  2. Prefix sum.
  3. Scatter with ascending `a`. Afterwards `off[e]` is the **end** of segment e.
  4. Shift right by one, **downward**, so each slot is read before it is overwritten.
- **Why nothing is overwritten.** During step 3, `off[e]` only moves through
  `[start_e, end_e)`. Those intervals are disjoint and together cover `[0, T·k)`. So each
  slot is written exactly once. The Python checker instruments the write count of every
  slot.
- **Empty experts.** `start_e = end_e`, so the cursor never moves. After the shift,
  `off[e] = off[e+1]`.
- **Cost.** O(T·k + E) with no comparisons. The row gather is fused into the scatter loop.

### D-MoE-6 · Top-k is one pass with strict `>` insertion
- **Choice.** k ≤ 8 candidates live in a stack array. A logit is inserted only where it is
  strictly greater than an entry.
- **Why it implements the tie rule.** Experts are visited in ascending order, so equal
  logits keep the lower index first.
- **Cost.** O(E·k) per token, with no sort and no allocation.
- **Verified by:** `MoE.TopKTiesPreferLowerIndex`, which covers a three-way tie and an
  all-equal row.

### D-MoE-7 · Combine gathers through `inv`; the residual is a separate `Add`
- **Why.** Each output row has a single writer, summing in a fixed slot order. There are no
  atomics, and runs are bitwise reproducible.
- **Later option.** Folding `+x` into `MoECombine` saves one [T,H] pass. That would be a
  measured change.

### D-MoE-8 · The naive baseline is a benchmark function, not a registered op
- **What it is.** `benchmarks/bench_moe.cpp` loops over (token, slot) and calls the **same**
  `gemm_auto` with M = 1.
- **What the ablation measures.** Grouped vs naive changes exactly one variable: batching an
  expert's rows into one GEMM.
- **Why not an op.** Registering it as an op would break the rule "narrow beats broad".

### D-MoE-9 · Experts run serially; parallelism lives inside `gemm_auto`
- **Known weakness.** Small T gives a few rows per expert. That is below `gemm_auto`'s
  threading threshold (M ≥ 32), and an M = 1 call still packs the whole weight panel.
- **Measured consequence.** See D-MoE-14.

### D-MoE-10 · Tolerance: combined abs + rel, reported separately, near-zero handled
- **Rule.** `|got − ref| ≤ atol + rtol·mag`, with the repo standard
  `rtol = 1e-5, atol = 1e-6`.
- **Reported separately:**
  - max abs error
  - max rel error, only over `|ref| ≥ atol/rtol = 0.1`
  - how many elements are near zero
  - the fraction of the tolerance used
- **Per-stage checks** use `mag = |ref|`.
- **Whole-layer checks** use `mag` = the summed term magnitude
  `|x| + Σ_s w·(|a|·|W2| + |b2|)`.
  - **Why.** FP32 summation error grows with Σ|term|, not with |result|.
  - **Measured.** In the clamp golden case, element 237 is 6.07 but its terms sum to 585 in
    magnitude. The error of 8.2e-4 is 1.4e-6 of that, about 12 FP32 ulps. A `|ref|`-relative
    gate calls that an 8.2× violation; the term-magnitude gate uses 11.7% of its budget.
  - The same effect appears at real 20B dims in the Python oracle: 2.17× vs 1.4%.
- **Does this hide bugs?** A wrong expert or a wrong column produces errors on the order of
  `mag`, about 10⁵× the tolerance.
- **Where.** `tests/support/golden.h` (`close_stats`) and `python/check_official_moe.py`
  (`close_stats`, `term_scale`).

### D-MoE-11 · GEMM scratch for every worker thread is created when the pool is created
- **Audit of `backends/cpu/gemm.cpp`.**
  - Inputs are row-major and contiguous, with leading dimensions K (A) and N (B and C).
    There are no stride parameters, which is why each expert segment must be contiguous.
  - No alignment is required. NEON `vld1q_f32` is unaligned-safe on arm64. Loads come from
    the packed panels, and all edges are zero-padded.
  - C is **overwritten**: it is zeroed, then accumulated.
  - Threading splits **output rows** across the pool, `parallel_for(M)`, and `gemm_auto`
    threads only when M ≥ 32 and 2MNK ≥ 2²¹.
  - Scratch is `static thread_local` grow-only `packA` and `packB`. They are sized by the
    cache-block constants (MC×KC and KC×NC), not by M, N or K, so each thread allocates them
    once, on its first GEMM.
- **The gap this exposed.** Workers claim ranges dynamically. Which workers have touched
  their scratch therefore depends on scheduling and on the largest M seen so far. In MoE,
  segment sizes change with the routing between runs, so a cold worker could allocate
  inside a later `run()`. The `ranges_` vector also grew lazily, up to the thread count.
- **Reproduced.** `alloc_invariant_test gemm-first-touch` runs one M = 1 GEMM, then an
  M = 256 threaded GEMM: **17 heap allocations**. That is 7 cold workers × 2 panels, plus 3
  growth steps of `ranges_`.
- **Fix.**
  - `ThreadPool` reserves `ranges_` to its thread count in the constructor. `parallel_for`
    never makes more ranges than threads.
  - `gemm_auto`'s shared pool is created by `shared_warm_pool()`. It runs one tiny GEMM on
    the calling thread, then `size()` ranges that each wait at a barrier until all of them
    are claimed. That forces one range onto each distinct worker, and each worker runs a
    tiny GEMM.
- **Result.** **0 allocations**, measured 3 times out of 3. The MoE check, warm-up on
  1-row segments then measurement on 8-row segments, allows 3 allocations; all 3 are for
  the returned outputs, the same as the dense graph.
- **Cost.** About 1.25 MB of scratch per thread is created up front, on the first GEMM call,
  instead of lazily. It is the same memory, just allocated earlier.

### D-MoE-12 · The arena hands out typed views (the int32 index intermediates)
- **Bug found by the Debug build.** `Arena::view_at` returned views typed as the f32 buffer
  and asserted "FP32 only". MoE's `idx`, `src`, `inv` and `offsets` are the first non-f32
  **intermediates**.
  - In Release they were silently tagged `kF32`. They worked only because both types are 4
    bytes.
  - In Debug, every planned MoE run aborted.
- **Fix.** `Tensor::view_bytes(dtype, shape, byte_offset)` returns an explicit, non-allocating
  view that reinterprets bytes. `Arena::view_at` uses it and asserts the view is in bounds.
  The planner already aligns every tensor to 16 bytes.
- **Verified by:** `Arena.TypedViewsCarryTheirDtype`. The MoE kernels now Debug-assert `kI32`
  on every index tensor.

### D-MoE-13 · RMSNorm accumulates the sum of squares in double
- **Why.** Over H = 2880 terms, a running fp32 sum adds about sqrt(H)·eps32 of relative error
  to every output. A double accumulator costs H multiply-adds per token, which is
  negligible. The rest of the formula follows the official operation order,
  `(x·inv)·scale`.

### D-MoE-14 · Measured: what limits the CPU MoE layer (from `bench_moe`, RESULTS.md MoE section)
- **The expert GEMMs are the layer.** `MoEGroupedMatMul` is ≥ 96% of the time at every T.
- **Time follows experts touched (U), not tokens.**
  - Uniform routing: T = 16 → 64 is 4× more tokens and FLOPs, but the time goes from
    701 → 752 ms, with U going from 30 → 32.
  - Same4 routing at T = 64 (U = 4) takes 142 ms, against 752 ms for uniform (U = 32).
  - This is the CPU preview of H3 (weight-bound at small T) and H4 (skew changes the cost).
- **Small-M GEMMs are far below memory bandwidth.**
  - At T = 1 the derived weight stream is 5.6 GB/s (4 experts, 71 ms). The M1 Pro's memory
    bandwidth spec is about 200 GB/s.
  - **Likely cause, read from the code and not yet profiled:** `gemm_auto` with M = 1 takes
    the single-threaded packed NEON path (M < 32). It packs every weight panel and runs 8-row
    micro-tiles for one real row.
  - **Candidate fix:** a small-M / GEMV path that streams the weights once, threaded over N,
    or threading across experts (D-MoE-9).
  - This is not implemented. It needs a profile first and lands as its own measured commit.
- **Batching is the whole win of dispatch.**
  - Grouped vs naive is 1.0× at T = 1, where there is nothing to batch.
  - It rises to 18.6× (uniform, about 32 rows per expert) and 79× (same4, 256 rows per
    expert) at T = 256.
  - The two paths are bitwise identical.
