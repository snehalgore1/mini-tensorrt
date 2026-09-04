# Roadmap

**Current week: 3**

Each week has a hard "definition of done." Do not start the next week until the current
week's DoD is met and its results are recorded. Weeks 1 through 4 are the core; the
project is a legitimate portfolio piece if it stops after Week 4 or 5.

---

## Week 1: Skeleton, Tensor, IR

Build the foundation and prove it with a hand-constructed graph.

- CMake project, GoogleTest wired via `FetchContent`, `ctest` passing
- `Tensor`: dtype, shape, strides, ownership, views, `MTRT_ASSERT` bounds checks
- `Graph` IR: `Node` (op type, input/output tensor ids, attributes), tensor table,
  topological sort with cycle detection
- Naive `Executor` and op registry with exactly two kernels: `Add`, `Relu`
- Unit tests for tensor construction, views, strides, topo sort, and both kernels

**DoD.** A hand-built graph (`Add` then `Relu`) executes and returns correct values in a
test. Clean-checkout `cmake && ctest` passes.

**Interview value.** Ownership model, why an IR exists, topological scheduling.

---

## Week 2: Naive kernels, exporter, golden tests

Get a real model running end to end against a real reference.

- `python/export_models.py`: PyTorch MLP to JSON topology + `.bin` weights
- `python/gen_goldens.py`: per-op and whole-model reference outputs as `.npy`
- `.npy` reader in C++ tests
- JSON frontend producing IR
- Kernels: `Gemm`/`MatMul`, `Add`, `Relu`, `Gelu`, `Softmax`
- Golden test per kernel plus a whole-model golden test

**DoD.** The exported MLP runs in MiniTensorRT and matches PyTorch within tolerance. This is
the first checkpoint that is genuinely worth talking about.

---

## Week 3: Memory planner

- Live-interval analysis over topological order
- Greedy-by-size offset assignment into a single arena (see DESIGN D4)
- Naive per-op allocator retained behind `--allocator=naive`
- Instrumentation: peak arena bytes, allocation count, bytes reused
- Test asserting no live-range overlap in the produced plan
- Correctness test: identical outputs under both allocators

**DoD.** A table in `docs/RESULTS.md` showing peak bytes and allocation count, naive
versus planned, on the MLP.

**Interview value.** Lifetime analysis, arena allocation, why static shapes enable it.

---

## Week 4: Profiler and first fusion pass

- Per-node timing, Chrome Trace Event JSON output
- Pass infrastructure operating on IR
- Fusion: `MatMul + Add(bias) + Gelu` into one fused node with its own kernel
- Numerical equivalence test with the pass on and off
- Constant folding as a second, simpler pass if time allows

**DoD.** A flame chart screenshot in the README, plus measured fusion effect on latency
and on peak intermediate memory. If fusion does not help, `docs/RESULTS.md` says so and
explains why. A negative result honestly analyzed is a good result here.

**This is the "shippable if life explodes" milestone.** Correct runtime, memory
optimization, graph optimization, profiler, measured ablations. Resume-ready.

---

## Week 5: The GEMM week

The primary performance investment. Measure after every single step.

1. Naive triple loop (baseline GFLOP/s)
2. Loop reorder for cache-friendly access
3. Register blocking (accumulate a small output tile in registers)
4. Cache tiling / blocking on M, N, K
5. Packing A and B panels into contiguous buffers
6. NEON intrinsics microkernel (FMA over `float32x4_t`)
7. Multithreading over output tiles via the thread pool

- Compare against Apple Accelerate BLAS
- Roofline analysis: arithmetic intensity, achieved versus peak

**DoD.** A GFLOP/s-per-step plot, an Accelerate comparison, and a written explanation of
the remaining gap. This is the flagship artifact of the project.

---

## Week 6: ONNX frontend and CNN operators

- ONNX C++ library dependency, model to IR conversion
- Prove the IR was frontend-agnostic (no changes needed below `src/frontend/`)
- `Conv2D` (im2col plus GEMM, reusing Week 5 work), `MaxPool`, `Reshape`, `Transpose`
- Export a small CNN, validate against ONNX Runtime
- Strict subset: reject unsupported ops with a clear error, do not silently misbehave

**DoD.** A CNN loaded from a real `.onnx` file matches ONNX Runtime output.

---

## Week 7: Pick one, CUDA or quantization

Do not attempt both.

**Option A, CUDA** (rented GPU: Colab T4, or vast.ai / Lambda for a real dev loop)
- Tiled shared-memory GEMM kernel
- One elementwise kernel
- Host/device transfer accounting, CPU versus GPU comparison at several sizes

**Option B, INT8 quantization**
- Symmetric per-tensor quantization, calibration script in Python
- Quantized GEMM with INT32 accumulation
- Accuracy versus speed tradeoff table

**DoD.** Either path produces a measured comparison table and an honest writeup.

---

## Week 8: Report, docs, demo

- `docs/RESULTS.md` completed: full benchmark matrix, every ablation, p50/p95 latency
- Architecture doc with a graph diagram before and after fusion
- Memory lifetime diagram showing arena reuse
- README with the flame chart, the GEMM plot, and a clear build guide
- "What I would build next in a production runtime" section
- Verify clean-checkout build on a fresh machine or container

**DoD.** Someone who has never seen the repo can clone it, build it, run the benchmarks,
and understand what was measured and why.

---

## Scope tripwires

Stop and reconsider if any of these happen:

- More than two days on protobuf or ONNX parsing
- Implementing an operator no model in `models/` uses
- Optimizing a kernel the profiler has not flagged as hot
- Week 5 running long. It is allowed to. Take time from Week 7, not from Week 8.
