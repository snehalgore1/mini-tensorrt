# Design decisions

Each entry states the decision, the alternatives considered, and why. These double as
interview answers, so the rationale matters more than the decision.

---

## D1. Static shapes only

**Decision.** All tensor shapes are resolved during graph load. The executor sees fully
concrete shapes.

**Alternatives.** Dynamic shapes with runtime shape propagation, as PyTorch and ONNX
Runtime support.

**Why.** Dynamic shapes force runtime allocation, defeat ahead-of-time memory planning,
and roughly triple the complexity of every pass. TensorRT itself builds shape-specialized
engines for exactly this reason. Static shapes are what make the memory planner and the
fusion passes tractable, and they are the enabling assumption for most of the performance
work in this project.

**Cost.** Batch size changes require a rebuild of the plan. Acceptable, and honest to
state in the README.

---

## D2. FP32 only until Week 7

**Decision.** One dtype through the core build-out. Quantization is a late milestone,
not a foundation.

**Why.** Multi-dtype dispatch doubles the kernel surface before there is anything worth
dispatching. The op registry is keyed on `(op_type, dtype)` from day one so adding INT8
later is additive, not a rewrite.

---

## D3. Frontend is pluggable; the JSON loader comes first

**Decision.** The first frontend reads a trivial format: a JSON topology file plus a
raw binary weights blob, both emitted by `python/export_models.py`. The ONNX frontend
is added in Week 6, after the engine already runs.

**Alternatives.** Start with ONNX parsing, as most tutorials do.

**Why.** Two reasons. First, risk isolation: protobuf schema wrangling is a time sink
with near-zero technical signal, and putting it on the critical path in Week 2 means a
parser bug blocks all downstream work. Second, architecture: building a second frontend
later *proves* the IR is frontend-agnostic rather than merely asserting it. That is a
stronger claim in a design discussion.

---

## D4. Memory planner uses greedy-by-size offset assignment

**Decision.** Compute each intermediate tensor's live interval (first write to last
read) over the topological order, sort tensors by size descending, and assign each the
lowest arena offset that does not overlap a live conflicting tensor. One contiguous
arena allocation for the whole graph.

**Alternatives.** Naive per-op malloc/free; reference counting with a free list;
optimal interval-graph coloring.

**Why.** This is the algorithm TFLite and TVM actually use, so it is citable and
defensible. Optimal packing is NP-hard in general; greedy-by-size gets most of the
benefit and is explainable on a whiteboard in two minutes. The naive allocator stays in
the tree behind a flag as the ablation baseline.

**Measured claim this enables.** Peak intermediate bytes and allocation count, planned
versus naive, on the same graph.

---

## D5. Operator dispatch via a registry of function objects

**Decision.** A registry maps `(op_type, dtype)` to a kernel function. Kernels are free
functions taking a small `OpContext` (input tensor views, output tensor views, attributes).

**Alternatives.** Virtual base class `Op` with a subclass per operator; templates
specialized per op.

**Why.** A virtual call per node per inference is real overhead at small graph sizes,
and an inheritance hierarchy makes fusion awkward because a fused node is not naturally
a subclass of anything. Free functions plus a registry keep the executor loop tight,
make kernels trivially unit-testable in isolation, and let a fused kernel register the
same way any other kernel does.

---

## D6. Own thread pool, not OpenMP

**Decision.** A small `std::thread` pool with a parallel-for helper.

**Why.** Apple Clang does not ship OpenMP by default, so requiring it adds a setup tax
for anyone building the project, including an interviewer who clones it. Writing the
pool is roughly 150 lines, is itself a reasonable systems artifact, and removes a
dependency.

---

## D7. Profiler emits Chrome Trace Event JSON

**Decision.** Per-node begin/end timings written in the Chrome Trace Event format,
viewable in `chrome://tracing` or Perfetto.

**Why.** Near-zero implementation cost for a real flame chart, which is the single best
visual artifact in the README. Custom ASCII profilers look homemade; a Perfetto trace
looks like tooling.

---

## D8. Golden fixtures generated from PyTorch, stored as `.npy`

**Decision.** `python/gen_goldens.py` runs each operator and each whole model through
PyTorch, saving inputs and expected outputs as `.npy`. C++ tests load them with a small
npy reader.

**Why.** Decouples the test suite from having Python in the build. Fixtures are
regenerable and diffable. `.npy` is a trivial format to read in C++ (about 80 lines) and
avoids a serialization dependency.

**Cost.** Fixtures are binary files in git. Keep them tiny (small shapes) and they stay
under a few hundred KB total.

---

## D9. GEMM is the primary performance investment

**Decision.** Week 5 is dedicated to taking the matrix multiply from a naive triple loop
to a blocked, packed, NEON-vectorized, multithreaded kernel, measuring each step
independently.

**Why.** GEMM dominates runtime for both MLP and Transformer workloads, so it is where
optimization is legitimate rather than premature. More importantly it is the part of this
project that produces a genuinely distinguishing artifact: a step-by-step GFLOP/s
progression with a roofline analysis. Many people build a toy runtime. Very few can
explain why their GEMM went from 1 to 30 GFLOP/s and where the remaining gap to BLAS is.

**Deliverable.** A plot of GFLOP/s per optimization step, plus an honest comparison
against Apple Accelerate with the remaining gap explained.

---

## D10. ARM NEON before CUDA

**Decision.** Hand-vectorize with ARM NEON intrinsics on the development machine. CUDA
is Week 7, on a rented GPU, scoped to one tiled GEMM and one elementwise kernel.

**Why.** The development machine is Apple Silicon, so NEON is the SIMD ISA that can
actually be profiled and iterated on locally. It is also directly on target for Qualcomm,
whose silicon is ARM. CUDA remains worth a week because it is a recognized checkbox for
NVIDIA-track roles, but it is additive and the project is complete without it.

---

## D11. The naive path is never deleted

**Decision.** Naive allocator, unfused graph, and scalar kernels all stay in the tree
behind runtime flags.

**Why.** They are the ablation baseline. The benchmark matrix in `docs/RESULTS.md`
requires running the same model with each optimization independently disabled. Deleting
the naive path destroys the ability to make measured claims, which is the entire point
of the project.
