# MiniTensorRT

A lightweight C++ neural network inference runtime.

MiniTensorRT loads a model graph, executes it on CPU with a small hand-written operator set,
and applies ahead-of-time memory planning and graph fusion. It exists to answer a
specific question: what actually happens below PyTorch, and how much of the performance
gap between a naive executor and a production runtime can be closed by hand.

> Status: in development. See `docs/ROADMAP.md`.

## What it does

- Loads a static-shape model graph into a frontend-agnostic internal IR
- Executes a constrained FP32 operator set on CPU
- Plans tensor lifetimes and packs intermediates into a single reused arena
- Rewrites the graph with fusion passes, verified numerically equivalent
- Profiles per-operator cost and emits Chrome Trace Event output
- Benchmarks against PyTorch eager and ONNX Runtime CPU

## What it does not do

MiniTensorRT is not a TensorRT replacement and does not try to be. It is a narrow,
end-to-end runtime built to be understood completely rather than to be complete. Dynamic
shapes, training, and broad operator coverage are explicitly out of scope. See
`docs/DESIGN.md` for the reasoning behind each of these boundaries.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Results

Full benchmark matrix and per-optimization ablations: `docs/RESULTS.md`.

<!-- Week 8: add the GEMM optimization plot, the flame chart screenshot,
     and the before/after fusion graph diagram here. -->

## Design

Every significant decision is recorded with its alternatives and rationale in
`docs/DESIGN.md`.

## What I would build next in a production runtime

<!-- Week 8: this section is a large part of why this project is worth reading.
     Be specific and be honest about the gap. -->
