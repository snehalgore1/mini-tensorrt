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

Problem size: TBD. Peak theoretical FP32 throughput: TBD GFLOP/s.

| Step | Description | GFLOP/s | Speedup vs naive | % of peak |
|---|---|---|---|---|
| 0 | Naive triple loop | TBD | 1.00x | TBD |
| 1 | Loop reorder | TBD | TBD | TBD |
| 2 | Register blocking | TBD | TBD | TBD |
| 3 | Cache tiling | TBD | TBD | TBD |
| 4 | Panel packing | TBD | TBD | TBD |
| 5 | NEON microkernel | TBD | TBD | TBD |
| 6 | Multithreaded | TBD | TBD | TBD |
| ref | Apple Accelerate | TBD | TBD | TBD |

Remaining gap to Accelerate, explained: TBD

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
