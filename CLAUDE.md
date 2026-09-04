# MiniTensorRT

A lightweight C++ neural network inference runtime. Loads a constrained model graph,
executes a small operator set on CPU, applies graph and memory optimizations, and
benchmarks correctness and speed against PyTorch and ONNX Runtime.


---

## Prime directive

This is a **portfolio and interview project targeting systems and inference roles**
(NVIDIA, Qualcomm, inference infrastructure teams). Every decision is judged by one
question: *does this produce a defensible, measured technical artifact?*

That means:

- **Measured beats implemented.** An optimization with no before/after number is worth
  nothing here. Every optimization lands with a benchmark result in `docs/RESULTS.md`.
- **Correct beats fast.** No kernel is optimized until a golden test proves it correct.
- **Narrow beats broad.** We support the smallest operator set that runs our target
  models. Adding an operator "for completeness" is a scope violation.

## Non-goals

Do not implement these unless the roadmap explicitly reaches them:

- Dynamic shapes. All shapes resolve at load time.
- Training, autograd, or any backward pass.
- Operators not required by a model in `models/`.
- A general-purpose protobuf parser written by hand. Use the `onnx` C++ library.
- Broad dtype support. FP32 only until Week 7.
- Distributed or multi-device execution.

---

## Architecture

The **internal IR is the contract.** Frontends produce IR. Passes rewrite IR. The
executor consumes IR. Nothing downstream of the frontend may reference ONNX types.

```
include/mtrt/        public headers
src/
  tensor/               Tensor, dtype, shape, strides, ownership
  graph/                Node, Graph, topological ordering, shape inference
  frontend/
    json/               simple JSON+bin loader (built first)
    onnx/               ONNX loader (added Week 6)
  ops/                  operator registry + CPU kernels
  memory/               arena, lifetime analysis, greedy planner
  optimizer/            IR rewrite passes (fusion, constant folding)
  profiler/             per-node timing, Chrome trace JSON emitter
  runtime/              Executor, session setup
backends/cpu/           SIMD and threaded kernels
backends/cuda/          optional, Week 7
tests/                  GoogleTest unit + golden tests
benchmarks/             benchmark harness and ablation drivers
python/                 model export, reference output generation
models/                 exported models + golden .npy fixtures
docs/                   DESIGN.md, ROADMAP.md, RESULTS.md
```

## Core invariants

Breaking any of these is a bug even if tests pass.

1. **Tensor owns or views, never ambiguously.** `Tensor` holds a `std::shared_ptr`
   to buffer storage plus shape/strides/dtype. A tensor that views arena memory is
   constructed explicitly as a view. No implicit deep copies anywhere in the hot path.
2. **The IR is frontend-agnostic.** `src/graph/` must not `#include` anything from
   `src/frontend/`.
3. **All shapes are static and resolved before execution.** Shape inference runs once
   at load. The executor never allocates.
4. **The executor never allocates.** All intermediate memory comes from the arena,
   planned ahead of time. Allocation during `Executor::run()` is a bug.
5. **Every operator has a golden test** comparing against a PyTorch-generated `.npy`
   fixture within an explicit tolerance. No exceptions.
6. **Optimization passes preserve numerics within tolerance.** Every pass has a test
   that runs the same graph with the pass on and off and compares outputs.
7. **Debug builds assert shapes and bounds.** Release builds do not. Use the
   `MTRT_ASSERT` macro, not raw `assert`.

## Build and test

Dependencies come in through CMake `FetchContent`. Use these exact pins; this
combination is verified working.

```cmake
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2)
```

Minimum CMake 3.20, C++17.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure

# debug build with assertions and sanitizers
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug -DMTRT_SANITIZERS=ON
cmake --build build-debug -j && ctest --test-dir build-debug --output-on-failure

# regenerate golden fixtures (requires python venv, see python/README.md)
python python/export_models.py && python python/gen_goldens.py

# benchmarks
./build/benchmarks/bench_gemm
./build/benchmarks/bench_model --model models/mlp.json --iters 200
```

Always build **Release** for any timing claim. Debug numbers are meaningless.

## Workflow rules

- **Plan before large changes.** For anything touching more than two files, produce a
  plan first and wait for approval.
- **Golden test before kernel.** Write the failing test, generate the fixture from
  PyTorch, then implement.
- **Profile before optimizing.** Never optimize a kernel that the profiler has not
  shown to be hot. Record the profile in the commit message.
- **One optimization per commit**, with its measured effect in the message. This
  history *is* the ablation study and feeds `docs/RESULTS.md` directly.
- **Do not touch `docs/RESULTS.md` with estimated numbers.** Real measurements only.

## C++ conventions

- C++17. Apple Clang on arm64 macOS is the primary target.
- `#pragma once`. Namespace `mtrt`. Nested namespaces for subsystems.
- `snake_case` for functions and variables, `PascalCase` for types,
  `kConstantCase` for constants, trailing `_` on private members.
- Prefer value semantics and `std::unique_ptr`. `shared_ptr` only for tensor storage.
- No raw `new`/`delete` outside the arena allocator.
- No exceptions in the execution hot path. Setup and parsing may throw.
- `const` and `noexcept` where they are true, not decoratively.
- Headers in `include/mtrt/` are the public surface and stay minimal.

## Reference baselines

- `python/reference.py` runs PyTorch eager and ONNX Runtime CPU on the same inputs.
- Correctness tolerance: `rtol=1e-5, atol=1e-6` for FP32 elementwise and GEMM.
  LayerNorm and Softmax may need `atol=1e-5`. Document any loosened tolerance and why.

## Current phase

See `docs/ROADMAP.md`. Update the "Current week" line there when a week closes.
Read `docs/DESIGN.md` before proposing any architectural change; the decisions there
were made deliberately and have written rationale.
