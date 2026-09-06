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

## Real GPT-2 124M — end-to-end real workload (#2)

The runtime loads **real GPT-2-small (124M)** weights exported from HuggingFace
(12 layers, 12 heads, d=768, vocab 50257; ~622 MB FP32; 365-node graph) and both
reproduces its logits and generates identical text.

| Check | Result |
|---|---|
| Logit parity vs HuggingFace (seq_len 64) | **0 / 64** next-token argmax mismatches; max abs logit err **4.3e-4** |
| Greedy generation vs HF `generate(do_sample=False)` | **identical token ids** |

Example (`python/gpt2_generate.py --prompt "The quick brown fox" --max-new 20 --check`):

> The quick brown foxes are a great way to get a little bit of a kick out of your dog.

matches HuggingFace greedy decoding exactly. Measured by `Gpt2Real.LogitsMatchHuggingFace`
(logit parity) and the `--check` path of `python/gpt2_generate.py` (generation parity).
The runtime does the full inference; only tokenization/detokenization is Python. Weights
are regenerated (`python/export_gpt2_hf.py`), not committed; the C++ test skips when absent.

Generation uses a fixed-max-context static graph (fill positions 0..t, read `logits[t]`;
causal masking makes later positions irrelevant), recomputing the graph per token —
respecting the static-shape invariant.

**KV-cache (N3).** A dedicated incremental-decode path caches each layer's K/V for past
positions, so per token it computes Q/K/V only for the new token and attends over the
cache (O(t) work) instead of recomputing the whole graph (O(S)):

| Decode path | 20 tokens (prompt 4) | Output |
|---|---|---|
| Full-recompute static graph (N2) | 5415 ms | — |
| KV-cache incremental (N3) | 2890 ms | **byte-identical ids** |

**1.87× faster, identical tokens** (`run_gpt2 --mode bench` asserts id-for-id equality).
This is a deliberate, bounded extension of the static-shape model — K/V buffers are
pre-allocated to a fixed max context and a runtime position tracks the valid length. The
speedup understates the asymptotic win: the recompute baseline does O(S) work per token
regardless of position, so the gap widens with context length. Measured on Apple M1 Pro,
Release; the incremental path reuses the tuned GEMM (`gemm_auto`) for its matvecs.

---

## GPU backend — real GPT-2 on a CUDA GPU (#1)

The runtime has a CUDA backend that runs a whole model on the GPU: weights uploaded
once, one device buffer per tensor, each graph node dispatched to a CUDA kernel
(elementwise/LayerNorm/CausalSoftmax/Transpose/Gather hand-written; MatMul and
BatchedMatMul via cuBLAS). It reuses the frontend-agnostic `Graph` IR unchanged — the
*same* graph runs on CPU or GPU. All CUDA is gated behind `-DMTRT_CUDA=ON`; the
macOS/CPU build is untouched.

| Check (Tesla T4, Colab, CUDA 12.8) | Result |
|---|---|
| Per-op golden tests on GPU (11 ops vs PyTorch) | all PASS (max err ≤ 4.8e-7) |
| Tiny transformer block: GPU vs CPU | PASS (max err 7.2e-7) |
| **Real GPT-2 124M: GPU vs CPU oracle** | **PASS — 0 argmax mismatches, max logit err 3.97e-4** |

So real GPT-2 produces the same next-token predictions on the GPU as on the CPU (which
matches HuggingFace). Reproduce on Colab: see `colab/README.md`.

### GPU GEMM roofline (own kernel vs cuBLAS, Tesla T4)

The GPU analogue of the CPU NEON ladder: a naive one-thread-per-output kernel, a
shared-memory tiled kernel, and cuBLAS, in GFLOP/s (FP32, square).

| N | naive | tiled | cuBLAS | tiled / cuBLAS |
|---|---|---|---|---|
| 512 | 365 | 598 | 3209 | 18.6% |
| 1024 | 384 | 898 | 5763 | 15.6% |
| 2048 | 450 | 896 | 5948 | 15.1% |

Shared-memory tiling gives **~2.3× over naive** (each global element is reused `TILE`
times); the tiled kernel then reaches **~15% of cuBLAS / ~11% of the T4's ~8.1 TFLOP/s FP32
peak**, correct to 1.6e-4 vs cuBLAS. The remaining gap is the honest one: cuBLAS adds
register/warp blocking, vectorized loads, and double-buffering that a single 32×32
one-element-per-thread tile does not — the same "textbook kernel vs tuned library" story as
NEON vs Accelerate. `bench_gemm_cuda`.

### CPU vs GPU, whole model

| System (same Colab host) | GPT-2 124M forward, S=64 |
|---|---|
| CPU executor | 1499.8 ms |
| CUDA executor (T4) | **21.7 ms** |

Real GPT-2 runs in **~22 ms on the T4**, a **~60–70×** speedup here. Honest caveat: the CPU
baseline is Colab's x86 CPU running the *portable scalar-fallback* GEMM — the tuned NEON
path is arm64-only — so this ratio flatters the GPU; on an arm64 SIMD CPU the gap narrows.
The GPU latency itself is the headline number. Measured by `cuda_model_test`.

### GPU memory arena

The device analogue of the CPU memory planner: instead of one `cudaMalloc` per
intermediate, the `CudaExecutor` packs all intermediates into a single device arena at
planned offsets, reusing space between tensors with disjoint lifetimes (safe — kernels run
in topo order on one stream).

| Configuration (real GPT-2 124M, S=64) | Peak intermediate mem | Allocations |
|---|---|---|
| Naive (one `cudaMalloc` per intermediate) | 88.5 MB | 364 |
| Greedy device arena | **1.7 MB** | **1** |

**86.8 MB reused, a 98% reduction** — and the whole model still produces identical tokens
(`argmax_mism=0`), so the reuse preserves numerics on the GPU. Measured by `cuda_model_test`.

---

## Memory planning (Week 3)

| Model | Configuration | Peak intermediate bytes | Alloc count | Bytes reused | Reduction |
|---|---|---|---|---|---|
| MLP | Naive per-op allocator | 224 | 5 | 0 | — |
| MLP | Greedy arena planner | 128 | 1 | 96 | 43% |
| GPT-2 block (2 layers) | Naive per-op allocator | 29,491,200 (28.1 MB) | 51 | 0 | — |
| GPT-2 block (2 layers) | Greedy arena planner | 3,538,944 (3.4 MB) | 1 | 25,952,256 (24.8 MB) | **88%** |

MLP intermediates (t0..t4): three [1,16] (64 B) and two [1,4] (16 B). The naive
allocator gives each its own buffer (224 B, 5 allocations). The greedy-by-size
planner reuses space between tensors whose lifetimes don't overlap, packing them
into a single 128 B arena (1 allocation) -- a 43% reduction, 96 B reused.

**The MLP is too small for the absolute numbers to mean anything (96 B).** The
GPT-2-small block (S=128, D=768, H=12, FFN=3072, 2 layers) is where the planner
earns its keep: 51 intermediates totalling **28.1 MB** under the naive allocator
collapse into a single **3.4 MB** arena -- an **88% reduction, 24.8 MB reused**.
The deeper win vs the MLP is structural: a transformer is a long chain of
mostly single-consumer intermediates (projections, attention scratch, the
[128,3072] FFN activations), so most lifetimes are disjoint and the arena packs
them tightly. Measured by `MemoryPlanner.ReportStats` / `ReportStatsGpt2`
(16-byte aligned slots). Outputs, inputs, and weights are excluded (identical
under both allocators). The GPT-2 model is regenerated, not committed
(`python/export_models.py`); the test skips when it is absent.

---

## Fusion (Week 4)

| Model | Configuration | p50 latency | p95 latency | Peak intermediate bytes |
|---|---|---|---|---|
| MLP | Unfused | ~330 ns | ~375 ns | 128 |
| MLP | MatMul+Bias+Gelu fused | ~290 ns | ~334 ns | 80 |
| GPT-2 block | Unfused | ~73.3 ms | ~77.9 ms | 3,538,944 (3.4 MB) |
| GPT-2 block | +FFN fused (MatMul+Bias+GeluTanh) | ~72.0 ms | ~75.3 ms | 2,359,296 (2.25 MB) |

MLP measured by `bench_model --iters 5000`; GPT-2 by `bench_model --model
models/gpt2_block.json --warmup 20 --iters 200` (Release, Apple M1 Pro), median of
several runs. **Honest reading:** fusion's win here is **memory, not latency**, at
both sizes. On the MLP it cuts planned peak 128 B -> 80 B (37.5%); on the GPT-2
block the per-layer FFN Linear->Bias->GeluTanh collapses into one node, cutting
peak intermediate memory **3.4 MB -> 2.25 MB (33%)**. The latency delta is small
and near the noise floor (MLP: ~15% of nanoseconds; GPT-2: ~1-2% of ~73 ms) and
the fused path is never slower. This is expected once the MatMul runs on the tuned
GEMM (below): the fused epilogue saves an intermediate write/read of the FFN
activation, but the matmul FLOPs dominate, so the memory saving shows up far more
clearly than the latency one. Fusion is validated primarily as a *memory and
IR-rewrite* win. (The GPT-2 FFN uses tanh-approx GELU, GPT-2's "gelu_new", hence
the `FusedMatMulBiasGeluTanh` variant.)

---

## GEMM in the model + per-op profile (Week 4/5 tie-in)

The Week-5 GEMM ladder was previously reachable only from `bench_gemm`; the model
executor's `MatMul` ran the naive triple loop. Wiring the tuned ladder
(`gemm_auto`: packed 8x8 NEON microkernel, multithreaded above a size threshold)
into the `MatMul` (and fused) kernels makes it accelerate a real model.
`MTRT_MATMUL=naive|neon|threaded` overrides the dispatch (naive = the ablation
baseline).

| GPT-2 block, MatMul path | p50 latency (unfused) | Speedup |
|---|---|---|
| `MTRT_MATMUL=naive` (triple loop) | ~2,289 ms | 1.0x |
| default (NEON microkernel + threaded) | ~73.3 ms | **~31x** |

Per-operator attribution of one optimized run (`bench_model --trace`, ~75.7 ms):

| Op | Time | Share | Note |
|---|---|---|---|
| BatchedMatMul | 34.8 ms | 45.9% | attention QK^T and attn·V -- **still the naive kernel** |
| MatMul | 31.6 ms | 41.7% | projections + FFN, now on the tuned GEMM |
| Transpose | 3.3 ms | 4.4% | head-split / merge copies |
| GeluTanh | 3.1 ms | 4.1% | |
| Softmax / LayerNorm / Add / Reshape / Scale | <2 ms each | <5% total | |

**Reading (profile before optimizing).** Speeding up `MatMul` gave ~31x on the
full model and, as expected, shifted the bottleneck: attention's **naive
`BatchedMatMul` is now the single hottest op (45.9%)** because it was never
optimized. That is the clear next step -- route each batch slice of
`BatchedMatMul` through the same `gemm_auto` -- and would roughly halve the
remaining runtime. Recorded here rather than done silently (one optimization per
commit; the profiler now justifies it).

Per-operator flame chart: generate with `bench_model --model
models/gpt2_block.json --trace gpt2.trace.json` and open in chrome://tracing or
Perfetto (trace files are gitignored, regenerable).

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

### Cache-block autotuning (depth over novelty)

Rather than assert the block sizes, sweep them: `bench_autotune` measures the NEON kernel
across **36 (MC, NC, KC) configurations** (MC∈{64,128,256}, NC∈{64,128,256,512},
KC∈{128,256,512}) at N=1024.

| Configuration | GFLOP/s |
|---|---|
| Worst in sweep | 65.9 |
| Ladder default (128 / 128 / 256) | 75.5 |
| **Autotuned optimum (128 / 512 / 512)** | **~85** |

**~12% over the hand-picked default**, and a 66→85 spread across the grid. The optimum has
*larger* N and K blocks: the packed B panel (NC×KC×4 ≈ 1 MB) fits comfortably in the M1
Pro's large L2, so bigger panels amortize packing and B-reload without spilling cache —
the kernel is compute-bound, not L2-capacity-bound, at these sizes. Applying 128/512/512 to
`gemm_neon` lifts single-core NEON to **~85 GFLOP/s ≈ 87% of the 1-core peak** (from ~79%),
the one honest lever over the AMX-bound Accelerate gap. Reproduce: `./build/benchmarks/bench_autotune`.

---

## INT8 weight quantization (#5)

Symmetric INT8 quantization of the matmul weight matrices (`Wq/Wk/Wv/Wo/Wfc/Wproj` +
tied `lm_head`); embeddings, LayerNorm, and biases stay FP32. The runtime's `MatMulQ`
kernel dequantizes the int8 weight in the inner loop (f32 accumulate) — golden-tested
(`Golden.MatMulQ`).

| | Size |
|---|---|
| Quantized weights (FP32 → INT8) | 494 MB → **124 MB** (4.0×) |
| Full model | 652 MB → 282 MB (2.31×) |

**Per-tensor vs per-channel** (max abs logit error vs FP32, real GPT-2 124M):

| Scheme | mean logit err | max logit err |
|---|---|---|
| Per-tensor (one scale per matrix) | 8.98 | 22.3 |
| **Per-channel (one scale per output column)** | **0.59** | **2.53** |

The headline finding: **per-tensor INT8 wrecks GPT-2** (max logit error 22) because a single
per-matrix scale is dominated by weight outliers, quantizing everything else near zero.
**Per-channel scaling recovers it — 8.8× lower error** at the *same* 4× compression — which is
why production transformer quantization is per-channel. Weight-only INT8 still perturbs logits
(residual ~0.6 mean), motivating calibration / mixed precision / GPTQ as next steps. This is
the accuracy/size half; a *speed* win needs INT8 SIMD (NEON `SDOT`) with activation
quantization — future work. Reproduce: `python python/quantize_gpt2.py`.

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
