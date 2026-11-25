# Milestone 1B — GPU routing and dispatch design (Checkpoint 3)

Status: **APPROVED (Checkpoint 3).** The decisions are logged as D-MoE-15..16 in [ENGINEERING_DECISIONS.md](../ENGINEERING_DECISIONS.md). Kernels: `backends/cuda/moe.cu`. The target is a Kaggle T4
(sm_75), GPU 0 only, FP32.

This checkpoint covers only the **data-movement half** of the MoE layer on the GPU: top-k,
dispatch, row gather and combine. These kernels are **shared by every variant**. V0, V0-oracle,
V1 and V2 differ only in how they run the expert GEMMs, and each of those gets its own
checkpoint (4). The dispatch must therefore be:

1. **Correct and testable exactly.** Index outputs equal the CPU kernels bit for bit.
2. **Free of host synchronization.** V1 and V2 read `offsets` on the device. Only V0 copies it
   to the host, deliberately, because that copy is what H1 measures.
3. **Deterministic.** The same input gives the same layout on every run.
4. **Workspace-free and graph-capturable.** All buffers come from the static plan, which makes
   optional microbench (b), CUDA-Graph capture of V1, possible.

Raw speed is a secondary criterion. At every T, dispatch is a tiny fraction of the expert GEMMs.
On CPU those were ≥ 96% of the layer; on GPU this is **expected, not yet measured**, and the
stage-event breakdown for H5 will measure it.

---

## 1. Pipeline on the GPU (one stream, no host round-trip in V1/V2)

```
x ──RMSNorm──► t ──router GEMM (cuBLAS) + bias──► logits
logits ──TopK (warp/token)──► w[T,k], idx[T,k]
idx ──Dispatch (block/expert)──► src[Tk], inv[Tk], offsets[E+1]
t, src ──GatherRows──► t_sorted[Tk,H]
   ... expert GEMMs: variant-specific, Checkpoint 4 ...
y_sorted, inv, w ──Combine (gather)──► moe[T,H] ──Add(x)──► out
```

- **The router GEMM is the same in every variant** (cuBLAS SGEMM on the layer's stream), so it
  is not a variable in any comparison.
- **The residual stays a separate `Add`**, as on CPU, so stage goldens line up. Fusing it is a
  later, measured change.

## 2. Dispatch: the options

| Option | How | Stable / deterministic | Host sync | Workspace | Launches | Work |
|---|---|---|---|---|---|---|
| A. Host sort | D2H `idx`, CPU counting sort, H2D | yes | **yes** | host buffers | — | O(Tk) on the CPU, plus 2 copies |
| B. Atomic histogram + scan + atomic scatter | `atomicAdd` per expert to get a slot | **no**: order within an expert depends on the warp schedule | no | counters | 3 | O(Tk), but contention on hot experts (same-4 sends every atomic to 4 counters) |
| C. `cub::DeviceRadixSort::SortPairs` (keys = expert id, bits 0..log2 E) + offsets via search | library radix sort | yes (CUB radix sort is stable) | no | **temp storage** (sized by a query call) | ~3–6 | O(Tk) |
| **D. Rank by counting (proposed)** | one block per expert: count smaller ids → start; stable ballot compaction | **yes** | no | **none** | **1** | O(E·Tk) compares |

**Why D.**
- **It has the smallest correctness surface.** No atomics, no inter-block communication and no
  library internals.
- **Each block writes a disjoint set:**
  - `src[start_e .. start_e+count_e)`
  - `inv[a]` only for its own assignments
  - `offsets[e]`

  So the kernel cannot race by construction, and `compute-sanitizer racecheck` can confirm it.
- **The result is unique, so the CPU is an exact oracle.** A *stable* sort by key has exactly
  one valid output for a given `idx`. The GPU must therefore produce the CPU kernel's
  `src`/`inv`/`offsets` **exactly**, and tests can compare indices with `==`, not tolerances.
  Option B could not be tested that way.
- **Its O(E·Tk) cost is negligible here (estimated, not measured).**
  - At T = 4096: Tk = 16,384 and `idx` is 64 KB.
  - Each of the 32 blocks reads it twice, about 4 MB, almost all from L2.
  - That is microseconds, against expert GEMMs of about 815 GFLOP.
- **When to use C instead.** D scales with E. For E = 128 (gpt-oss-120b) it is still small. At
  much larger E·Tk, option C wins, and it stays a **drop-in fallback** because its output is
  identical.

### The proposed kernel (D), precisely

Launch: grid = E blocks, 256 threads each. Block `e`:

```
pass 1  start = 0
        for each chunk c of 256 assignments:
            pred = idx[a] < e;  start += block_reduce(popc(ballot(pred)))
        offsets[e] = start;  if e == E-1: offsets[E] = Tk

pass 2  cursor = 0
        for each chunk c of 256 assignments (a ascending):
            pred   = idx[a] == e
            b      = ballot(pred)                     // 32-bit mask per warp
            mine   = popc(b & lanes_below_me)         // rank inside my warp
            before = exclusive_scan(per-warp popc)    // smem, 8 warps
            if pred: j = start + cursor + before[warp] + mine
                     src[j] = a;  inv[a] = j
            cursor += total for the chunk;  __syncthreads()
```

- **Stability.** Chunks go in ascending `a`, warps within a chunk are ordered, and lanes within
  a warp are ordered by `a`. So equal keys keep their order.
- **Empty experts.** `count_e = 0`, so block `e` writes only `offsets[e]` and returns. That is
  correct with no special case.
- **Prototype.** [`python/sim_gpu_dispatch.py`](../../python/sim_gpu_dispatch.py) simulates
  exactly this at warp granularity, including ballots, per-warp prefix, chunking and a partial
  last chunk.
  - It matches the CPU counting sort on **26,568 routings**: exhaustive small grids, plus random
    cases with E up to 128, same-k routing and T = 1.
  - It proves every `src`/`inv` slot is written exactly once.

### Row gather (separate kernel)

- **Kernel.** `t_sorted[j] = t[src[j] / k]`: one block per sorted row, `float4` copies
  (H = 2880 is divisible by 4).
- **Why separate from dispatch.** Inside dispatch, the copy work would follow the routing
  skew: with same-4 routing, 4 blocks would copy everything. A separate kernel spreads the
  Tk·H copy evenly.
- **Later option.** Folding the gather into the GEMM's A-tile loads (D-MoE-4 option C) saves
  Tk·H·4 bytes of write+read. That is decided per variant at Checkpoint 4, not here.

## 3. Top-k: one warp per token

- **Layout.** Lane `e` holds `logits[t, e]`. For E > 32, each lane holds ⌈E/32⌉ values and
  reduces them locally first.
- **Selection.** There are k rounds of a butterfly (`__shfl_xor_sync`) argmax under the key
  **(value descending, index ascending)**. The winner's lane drops out, so this is exactly the
  CPU tie rule.
- **Softmax.** It runs over the k winners in lane 0, max subtracted first, then writes `w` and
  `idx`.
- **Prototype check.** The simulation matches the CPU strict-`>` insertion on **20,000
  tie-heavy rows** (integer logits).

## 4. Combine and RMSNorm

- **Combine.** Grid = T blocks; threads stride over H. Each output element is
  `Σ_{s=0..k-1} w[t,s]·y_sorted[inv[t·k+s]]` in slot order, with a single writer. No atomics,
  and repeated runs are bitwise identical.
- **RMSNorm.** One block per token. The sum of squares is a block tree reduction in **fp32**.
  The CPU uses a double accumulator, but double runs at 1/32 rate on the T4. A tree reduction's
  error grows like log2(H)·eps, not H·eps, so it stays well inside the tolerance.
- **Consequence.** These float stages are **not bitwise equal** to the CPU, and they are
  compared with the D-MoE-10 tolerance rule. Index stages are compared exactly.

## 5. Stream and synchronization rules (the audit list for H1 and sync-free claims)

- Every MoE launch takes an explicit `cudaStream_t`.
- The router's cuBLAS call uses `cublasSetStream` on that stream.
- **Nothing runs on the legacy default stream.** The existing GPT-2 kernels in
  `backends/cuda/kernels.cu` do launch on it, so the MoE launchers are new functions with a
  stream parameter, not reuses.
- **Forbidden in the V1/V2 run path:**
  - `cudaMemcpy` (synchronous)
  - any D2H `cudaMemcpyAsync`
  - `cuda*Synchronize`
  - `cudaStreamQuery` / `cudaEventSynchronize`
  - `cudaMalloc` / `cudaFree`
  - default-stream work
- **V0's one intended sync:** `cudaMemcpyAsync(offsets → pinned host)` + `cudaStreamSynchronize`.
  It is the only entry on V0's list, and it is what V0 vs V0-oracle measures.
- **Evidence, strongest first:**
  1. a code audit of each run path, recorded in the decisions log
  2. nsys, if the platform permits profiling
  3. CUDA-Graph capture of V1 (optional (b)); capture fails on any synchronizing call
- `cudaStreamQuery` is not treated as evidence.

## 6. Memory (no new buffers beyond the 1B plan)

| Buffer | Size at T = 4096 | Notes |
|---|---|---|
| `idx` (i32), `w` (f32) | 64 KB each | |
| `src`, `inv` (i32) | 64 KB each | |
| `offsets` (i32) | 132 B | V0: plus a 132 B pinned host mirror, allocated at setup |
| `t_sorted` | 188.7 MB | as in the 1B buffer-lifetime table |
| dispatch workspace | **0** | option C would need CUB temp storage |

All of these are allocated once, before any timed region.

## 7. Code placement and the runner

- **Kernels.** New file `backends/cuda/moe.cu`, with launchers in `backends/cuda/moe.h`, all
  stream-taking:
  - `moe_rmsnorm`
  - `moe_topk`
  - `moe_dispatch`
  - `moe_gather_rows`
  - `moe_combine`
  - `moe_add`
- **Runner: a dedicated `MoELayerCuda`** that owns the device buffers and the uploaded weights
  (one copy shared by all variants), with `run(variant)`.
- **Why not `CudaExecutor`.**
  - It launches per node on the default stream.
  - V0 needs a host round-trip in the middle of the layer.
  - Adding MoE nodes there is not required for the 1B study.
- **This is a scope decision** (question 4 below).

## 8. Verification plan (before any variant is timed)

1. **Goldens.** `cuda_test` runs each GPU stage on the 5 MoE golden cases with golden stage
   inputs:
   - indices (`idx`, `src`, `inv`, `offsets`) and `t_sorted` must be **exactly equal**
   - float stages use the D-MoE-10 rule (combined tolerance, max abs and max rel reported)
2. **Random routings, GPU vs CPU kernel, exact.**
   - 500 cases of the CPU test's kind: same-4, T = 1, sparse.
   - Plus E = 32 at T = 4096 and a partial last chunk (Tk not a multiple of 256).
3. **Poisoned outputs.** Fill `src`/`inv`/`offsets` with `0xFF` before every test run, so any
   slot left unwritten is caught.
4. **Determinism.** Run 10 times and require bitwise-identical `out`.
5. **Sanitizers.** `compute-sanitizer --tool memcheck`, `racecheck` and `synccheck` on the
   golden cases.
6. **Timing (CUDA events only, outside correctness runs).** Stage times for TopK / Dispatch /
   Gather / Combine per T. These feed the H5 breakdown.
   - Not a hypothesis by itself.
   - Option C is timed only if D shows up as a measurable fraction of the layer.

These run on a T4 (Colab or Kaggle); I don't have a GPU here. I'll prepare the notebook steps, you run them and
paste the outputs, and only those outputs go into the docs.

## 9. Open questions for the reviewer

1. Approve **D (rank by counting)** as the dispatch kernel, with CUB radix sort (C) as the
   documented, output-identical fallback?
2. Approve a **separate balanced row-gather kernel**, deferring "gather inside the GEMM" to the
   variant checkpoint?
3. **Non-finite logits.** The CPU top-k currently treats NaN inconsistently: it can keep a NaN
   if one appears among the first k experts. `torch.topk` ranks NaN first. Proposed:
   - declare finite logits a precondition of the MoE layer
   - Debug-assert it on CPU
   - document GPU behaviour on NaN as unspecified, rather than invent a third rule

   The alternative is to define NaN = −∞ on both sides now.
4. Approve a dedicated **`MoELayerCuda` runner** for 1B, instead of wiring the MoE nodes into
   `CudaExecutor`?
5. Accept **fp32 tree reductions on the GPU** (RMSNorm), and therefore tolerance-based rather
   than bitwise agreement with the CPU for float stages?
