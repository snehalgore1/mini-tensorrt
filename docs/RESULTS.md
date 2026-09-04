# Results

**Measured numbers only.** Nothing in this file may be an estimate, a projection, or a
target. If a measurement has not been taken, the cell stays `TBD`.

Machine, compiler, and flags are recorded with every table because numbers without them
are not reproducible.

---

## Environment

| Field | Value |
|---|---|
| Machine | TBD |
| CPU | TBD |
| Compiler | TBD |
| Flags | TBD |
| Build type | Release |

---

## Correctness

| Model | Reference | Max abs error | Max rel error | Tolerance | Pass |
|---|---|---|---|---|---|
| MLP | PyTorch eager | TBD | TBD | atol 1e-6 | TBD |
| MLP | ONNX Runtime CPU | TBD | TBD | atol 1e-6 | TBD |

---

## Memory planning (Week 3)

| Configuration | Peak intermediate bytes | Allocation count | Bytes reused |
|---|---|---|---|
| Naive per-op allocator | TBD | TBD | 0 |
| Greedy arena planner | TBD | TBD | TBD |

---

## Fusion (Week 4)

| Configuration | p50 latency | p95 latency | Peak intermediate bytes |
|---|---|---|---|
| Unfused | TBD | TBD | TBD |
| MatMul+Bias+Gelu fused | TBD | TBD | TBD |

Per-operator profile before fusion: TBD
Per-operator profile after fusion: TBD

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
