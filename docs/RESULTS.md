# Results

**Measured numbers only.** Nothing in this file may be an estimate, a projection, or a
target. If a measurement has not been taken, the cell stays `TBD`.

Machine, compiler, and flags are recorded with every table because numbers without them
are not reproducible.

---

## Environment

| Field | Value |
|---|---|
| Machine | MacBook (arm64), macOS 26.6.2 |
| CPU | Apple M1 Pro |
| Compiler | Apple clang 21.0.0 (clang-2100.1.1.101) |
| Flags | -O3 -DNDEBUG |
| Build type | Release |

---

## Correctness

| Model | Reference | Max abs error | Max rel error | Tolerance | Pass |
|---|---|---|---|---|---|
| MLP | PyTorch eager | 2.98e-08 | 1.16e-07 | rtol 1e-5, atol 1e-5 | yes |
| MLP | ONNX Runtime CPU | TBD | TBD | TBD | TBD |

Measured by the `Model.MlpMatchesPyTorch` golden test (Release build, environment
above). The MLP ends in Softmax, so atol is 1e-5 per CLAUDE.md; the observed error is
at float32 rounding level. The ONNX Runtime row is TBD until `python/reference.py`
wires that comparison.

---

## Memory planning (Week 3)

| Configuration | Peak intermediate bytes | Allocation count | Bytes reused |
|---|---|---|---|
| Naive per-op allocator | 224 | 5 | 0 |
| Greedy arena planner | 128 | 1 | 96 |

MLP intermediates (t0..t4): three [1,16] (64 B) and two [1,4] (16 B). The naive
allocator gives each its own buffer (224 B, 5 allocations). The greedy-by-size
planner reuses space between tensors whose lifetimes don't overlap, packing them
into a single 128 B arena (1 allocation) -- a 43% reduction, 96 B reused.
Measured by `MemoryPlanner.ReportStats` (16-byte aligned slots). Outputs, inputs,
and weights are excluded (identical under both allocators).

---

## Fusion (Week 4)

| Configuration | p50 latency | p95 latency | Peak intermediate bytes |
|---|---|---|---|
| Unfused | ~330 ns | ~375 ns | 128 |
| MatMul+Bias+Gelu fused | ~290 ns | ~334 ns | 80 |

Measured by `benchmarks/bench_model --iters 5000` (Release, Apple M1 Pro), median
of several runs. **Honest reading:** the memory result is exact and deterministic
every run -- fusion eliminates the two intermediates (t0, t1) of the first Linear,
cutting planned peak from 128 B to 80 B (37.5%). The *latency* gain is real but
small and close to the measurement noise floor at this model size: p50 improves by
0-40 ns run to run (mean drops ~15%), and the fused path is never slower. This is
expected -- with only six ops on a [1,x] MLP there is little arithmetic to save;
fusion's latency payoff grows with model size, where eliminating intermediate
writes/reads and per-op overhead matters more. The fusion here is validated
primarily as a *memory and IR-rewrite* win, with latency as a modest bonus.

Per-operator flame chart: generate with `bench_model --trace mlp.trace.json` and
open in chrome://tracing or Perfetto (trace files are gitignored, regenerable).

---

## GEMM optimization ladder (Week 5)

Problem size: **N = 1024** (square, FP32). Measured ceilings on this machine
(`bench_gemm`): peak 1-core NEON FMA ≈ **98 GFLOP/s**, 8 P-cores ≈ **786 GFLOP/s**,
triad memory bandwidth ≈ **67 GB/s**. (Peaks vary ±~15% run to run from thermal
throttling on a fanless-class laptop; values below are one representative run.)

| Step | Description | GFLOP/s | Speedup vs naive | % of NEON peak |
|---|---|---|---|---|
| 0 | Naive triple loop (ijk) | 1.7 | 1.0x | 2% (1-core) |
| 1 | Loop reorder (ikj) | 25.8 | 15x | 26% (1-core) |
| 2 | Register blocking (4x4) | 25.5 | 15x | 26% (1-core) |
| 3 | Cache tiling (M/N/K) | 26.8 | 16x | 27% (1-core) |
| 4 | Panel packing | 26.2 | 15x | 27% (1-core) |
| 5 | NEON microkernel (8x8) | 77.7 | 46x | **79% (1-core)** |
| 6 | Multithreaded (8 cores) | 310.4 | 183x | **39% (8-core)** |
| ref | Apple Accelerate (cblas_sgemm) | 2319.4 | 1364x | 295% of 8-core NEON |

**Per-step attribution.** The two dominant wins are (1) **naive → reorder, ~15x**:
switching `ijk` to `ikj` makes the inner loop unit-stride over B and C, which fixes the
cache behavior and lets Apple Clang auto-vectorize it. (2) **packed → NEON, ~3x**: an
explicit 8x8 microkernel (16 `float32x4` accumulators) exposes enough independent FMA
chains to saturate the four NEON pipes — the compiler's auto-vectorization of the scalar
loop only reached ~26 GFLOP/s. The scalar register/tiling/packing steps land near the
auto-vectorized reorder on this compiler; their real payoff shows at **N=2048, where
register blocking alone collapses to ~10 GFLOP/s (B no longer fits cache) while cache
tiling holds at ~26** — i.e. tiling earns its keep exactly when the working set stops
fitting. Threading adds ~4x over single-core NEON (not 8x — see below).

**Remaining gap to Accelerate, explained.** Threaded reaches ~310 GFLOP/s vs Accelerate's
~2319 — about **13% of Accelerate**. The dominant reason is **AMX**: Accelerate dispatches
to Apple's on-die matrix coprocessor, which is not a public instruction set and is
unreachable from portable NEON. Accelerate's 2319 GFLOP/s is **~3x above the entire 8-core
NEON roofline (786)** — no NEON kernel, however tuned, can match it. Measured against the
ceiling we *can* target: single-core NEON hits **79% of the 1-core peak** (packing +
8x8 microkernel are near-optimal for NEON); multicore reaches only 39% of the 8-core peak
because GEMM at this size becomes **memory-bandwidth-bound** (67 GB/s triad) and the P-cores
share L2/bandwidth and thermal headroom, so throughput scales ~4x rather than 8x. Secondary
gaps: no software prefetch, a fixed (untuned) block size, and no packing of the C tile.

Reproduce: `./build/benchmarks/bench_gemm --sizes 256,512,1024,2048 && python python/plot_gemm.py`.

---

## Full benchmark matrix (Week 8)

| System | p50 (ms) | p95 (ms) | Peak mem | Notes |
|---|---|---|---|---|
| PyTorch eager | TBD | TBD | TBD | high-level baseline |
| ONNX Runtime CPU | TBD | TBD | TBD | optimized reference |
| MiniTensorRT naive | TBD | TBD | TBD | correctness baseline |
| + memory planner | TBD | TBD | TBD | |
| + fusion | TBD | TBD | TBD | |
| + SIMD GEMM | TBD | TBD | TBD | |
| + threading | TBD | TBD | TBD | |

All latencies measured after warm-up, steady state, setup and parse time excluded.
