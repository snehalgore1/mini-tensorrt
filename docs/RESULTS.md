# Results

**Measured numbers only.** Nothing in this file may be an estimate, a projection, or a
target. If a measurement has not been taken, the cell stays `TBD`.

Machine, compiler, and flags are recorded with every table because numbers without them
are not reproducible.

---

## Headline results

The full detail (methodology, caveats, reproduce commands) is in the sections below; these
are the results worth opening the repo for — all measured on the hardware in **Environment**,
all reproducible:

| Result | Measured | |
|---|---|---|
| Real GPT-2 124M vs HuggingFace | **0 / 64** next-token mismatches, max logit err 4.3e-4 | [details](#real-gpt-2-124m--end-to-end-real-workload-2) |
| GEMM ladder (FP32, M1 Pro) | **1.7 → 310 GFLOP/s (~180×)**, ~87% of 1-core NEON peak; Accelerate gap is the AMX ceiling | [details](#gemm-optimization-ladder-week-5) |
| GPT-2 block, full ladder | **56× faster, 12.5× smaller** vs naive (2281 → 40.7 ms; 28 → 2.25 MB) | [details](#full-benchmark-matrix-week-8) |
| INT8 quantization of GPT-2 | **per-channel recovers it — 8.8× lower error than per-tensor** at 4× weight compression | [details](#int8-weight-quantization-5) |
| Real GPT-2 on a Tesla T4 (GPU) | **~22 ms** end to end; FP16 tensor cores ~8× over FP32 | [details](#gpu-backend--real-gpt-2-on-a-cuda-gpu-1) |

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
| MLP (loaded from `.onnx`) | ONNX Runtime CPU | 0 | 0 | rtol 1e-5, atol 1e-5 | yes |

Measured by the `Model.MlpMatchesPyTorch` and `OnnxFrontend.MatchesOnnxRuntime` golden
tests (Release build, environment above). The MLP ends in Softmax, so atol is 1e-5 per
CLAUDE.md; the PyTorch error is at float32 rounding level. The ONNX row is now measured:
the *same* `models/mlp.onnx` runs in both ONNX Runtime and MiniTensorRT (via the new ONNX
frontend) and the outputs are **bit-identical** (max abs err 0) — see the ONNX frontend
section below.

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

### GPU FlashAttention (F-GPU) — measured on a Colab T4

**F-GPU — CUDA FlashAttention** (`backends/cuda/flash_attention.cu`): the online-softmax fused
causal attention kernel, a GPU port of the CPU kernel — one thread per (head, query), O(d)
register scratch, no `[H,S,S]` materialization. Correctness is a real test: `cuda_test` runs it
against the *same* PyTorch golden as the CPU kernel (`op_flashattn_*`).

| Check (Tesla T4, CUDA 12.8) | Result |
|---|---|
| FlashAttn GPU vs PyTorch golden (`op_flashattn_*`) | **PASS**, max abs err **1.19e-07** |

So the fused online-softmax attention produces the same result on the GPU as PyTorch's
reference, at float32 rounding level. Measured by `./build/backends/cuda/cuda_test`.

### FP16 tensor cores (H) — measured on a Colab T4

**H — FP16 tensor-core GEMM** (`backends/cuda/gemm_fp16.cu`): a hand-written WMMA (16³) kernel
with FP32 accumulate, plus a cuBLAS FP16 tensor-op reference — the FP16 analogue of the FP32 GPU
roofline above. GFLOP/s, FP16 in / FP32 accumulate, square, row-major (Tesla T4, CUDA 12.8):

| N | wmma-f16 | cuBLAS-f16 | cuBLAS-f32 | wmma / cuBLAS-f16 | err vs f32 |
|---|---|---|---|---|---|
| 512  | 1807 | 12096 | 3369 | 14.9% | 9.5e-3 |
| 1024 | 2135 | 23853 | 4294 |  9.0% | 1.5e-2 |
| 2048 | 2725 | 34131 | 4156 |  8.0% | 2.1e-2 |

**Correctness first:** our WMMA output matches cuBLAS FP16 to **0–3.8e-5** (bit-identical at
N≥1024), and the FP16-vs-FP32 gap is **~0.01–0.02** — genuine reduced-mantissa rounding, tiny
next to the output magnitudes (sums of thousands of terms). So the kernel is correct; the
interesting story is the performance.

**The tensor-core win, and why our kernel doesn't capture it.** cuBLAS FP16 reaches **34.1
TFLOP/s ≈ 52% of the T4's ~65 TFLOP/s tensor-core peak — up to 8.2× over cuBLAS FP32** (34131 vs
4156 at N=2048), which is the whole reason FP16 tensor cores are the GPU-inference fast path. Our
WMMA kernel reaches only **~2.7 TFLOP/s (~8% of cuBLAS FP16)** — and, tellingly, **below cuBLAS
FP32 (4.2 TFLOP/s)**. That is the honest and instructive result: a naive one-warp-per-16×16-tile
WMMA kernel with **no shared-memory staging and no double-buffering** re-streams A/B tiles from
global memory every K-step, so it is **memory-bandwidth bound — the tensor cores starve waiting
for data**. Faster math makes the memory bottleneck bite *harder*, which is exactly why our WMMA
captures a *smaller* fraction of its library (8%) than the FP32 tiled kernel did of cuBLAS FP32
(~15%): the same "textbook kernel vs tuned library" gap as NEON-vs-Accelerate, amplified. The
lesson — tensor cores are useless without feeding them (shared-memory tiling + double-buffering)
— is the point of the experiment. Reproduce: `bench_gemm_fp16 --sizes 512,1024,2048` (build with
`-DCMAKE_CUDA_ARCHITECTURES=75`; on the wrong arch the `__CUDA_ARCH__>=700` WMMA path stubs out).

A full FP16 *model* path (running GPT-2 end to end in half precision through the executor) is the
larger next step beyond this GEMM-level study.

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

**These per-step numbers are compiler-specific, and that is itself the point.** The scalar
register/tiling/packing steps depend entirely on what the compiler auto-vectorizes. On
GCC 13 / x86 the same ladder measures reorder ~9.6, register **~2.2**, tiled ~5.5 GFLOP/s —
steps 2–4 go *backwards*, because the hand-written 4x4 scalar microkernel actively **defeats
the auto-vectorization that the plain `ikj` reorder enables**. That is the real lesson of the
scalar tier: a "smarter" scalar kernel can lose to a naive loop the compiler vectorizes for
free, and the only portable way to guarantee the SIMD win is to write the vector kernel
explicitly (the NEON step), which is compiler-independent. (Numbers above are Apple Clang /
M1 Pro; the ladder is regenerated per machine.)

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

**Profile-driven step: tuning `BatchedMatMul`.** With `MatMul` tuned, the profiler
flagged attention's `BatchedMatMul` (QKᵀ and attn·V) as the single hottest op --
it was still the naive triple loop. Routing each batch slice through the same
`gemm_auto` (both attention products are plain `[M,K]@[K,N]` per head, since the
operand is pre-transposed) collapses it. Before/after, same machine, one
representative run each (`bench_model --model models/gpt2_block.json --warmup 20
--iters 200 --trace`):

| Op | Before (naive BMM) | After (tuned BMM) |
|---|---|---|
| BatchedMatMul | 33.7 ms (47.5%) | **3.9 ms (9.8%)** — ~8.6× |
| MatMul | 29.1 ms (41.0%) | 27.3 ms (**69.0%**) |
| Transpose | 2.9 ms (4.0%) | 2.8 ms (7.1%) |
| GeluTanh | 2.5 ms (3.4%) | 2.7 ms (6.8%) |
| Softmax / LayerNorm / Add / Reshape / Scale | <1.3 ms each | <1.3 ms each |
| **Whole block, p50 (unfused)** | **70.3 ms** | **40.7 ms** |

**Reading (profile before optimizing).** The profiler predicted this would "roughly
halve the remaining runtime," and it did: the block drops **70.3 → 40.7 ms p50
(1.73×)** off a single op change, because `BatchedMatMul` itself goes **8.6× faster**
(33.7 → 3.9 ms). The bottleneck then shifts back to `MatMul` (now 69% — projections +
FFN, the largest FLOP contributor), which is the expected steady state: the two dense
matmul families dominate, and both now run the tuned NEON ladder. Whole-model
correctness is unchanged — `Gpt2Real.LogitsMatchHuggingFace` still passes (0 argmax
mismatches vs HuggingFace), so the reassociation from the blocked kernel stays within
tolerance. `MTRT_MATMUL=naive` still forces both back to the triple loop for the ablation.

Per-operator flame chart: generate with `bench_model --model
models/gpt2_block.json --trace gpt2.trace.json` and open in chrome://tracing or
Perfetto (trace files are gitignored, regenerable).

---

## Full benchmark matrix (Week 8)

The cumulative optimization ladder on the **GPT-2-small block** (2 layers, S=128, D=768,
H=12, FFN=3072 — the same model as the memory/fusion tables above), plus the PyTorch-eager
reference. Each MiniTensorRT row adds **one** optimization to the row above it. Latency is
p50/p95 wall time per forward (`bench_model --model models/gpt2_block.json`, warm, steady
state, setup/parse excluded); the GEMM variant is selected with `MTRT_MATMUL`.

| System | p50 (ms) | p95 (ms) | Peak mem | What changed |
|---|---|---|---|---|
| PyTorch eager (Accelerate, 6 thr) | **6.7** | 7.8 | — (framework allocator) | high-level reference |
| ONNX Runtime CPU (MLAS) | **11.9** | 12.5 | — (framework allocator) | optimized graph-opt reference |
| MiniTensorRT naive | 2281 | 2343 | 28.1 MB | triple-loop GEMM, naive per-op allocator |
| + memory planner | 2281 | 2343 | **3.4 MB** | greedy arena — memory only, latency unchanged |
| + fusion | 2229 | 2292 | **2.25 MB** | FFN MatMul+Bias+GeluTanh fused (memory again) |
| + SIMD GEMM (NEON 8×8) | **63.8** | 66.9 | 2.25 MB | packed NEON microkernel — **36× latency** |
| + threading (8 cores) | **40.9** | 45.2 | 2.25 MB | multithreaded over tiles — 1.6× more |

The `MiniTensorRT naive` and `+ memory planner` rows show identical latency **by
construction, not coincidence**: the arena planner changes only *where* intermediates live,
never the compute path, so both rows run the same kernels — only peak memory differs. (The
executor is also allocation-free in that compute path; a warm `run()` heap-allocates only the
returned output vector, enforced by the `AllocInvariant` test.)

**Reading.** The ladder isolates each optimization on the axis it actually moves: the
memory planner and fusion cut peak memory **28.1 → 3.4 → 2.25 MB (12.5×)** with latency flat
(they are memory/IR-rewrite wins — see the Fusion section), while the NEON microkernel
and threading cut latency **2281 → 40.9 ms (56×)** with memory flat. End to end MiniTensorRT
goes from a correctness baseline to **56× faster and 12.5× smaller**. Both production runtimes
still win: PyTorch eager by **~6×** (6.7 ms) and ONNX Runtime by **~3.4×** (11.9 ms). This is
the same AMX story as the GEMM ladder — PyTorch dispatches to Accelerate, which uses Apple's
on-die matrix coprocessor (~3× above the entire NEON roofline, unreachable from portable NEON);
ORT's MLAS CPU kernels don't hit AMX, so it lands between PyTorch and us, its edge coming from
graph-level fusion and hand-tuned kernels. The gap we *can* close, we did; the rest is AMX and
years of kernel engineering.

All latencies measured after warm-up, steady state, setup and parse time excluded.
Reproduce: PyTorch row `python python/bench_torch_block.py`; ONNX Runtime row `python
python/bench_ort_block.py`; MiniTensorRT rows `MTRT_MATMUL={naive,neon,threaded}
./build/benchmarks/bench_model --model models/gpt2_block.json` (naive-allocator peak from
`MemoryPlanner.ReportStatsGpt2`). ORT and PyTorch time the *same* block; ORT via a
torch.onnx export, run through ONNX Runtime (a reference baseline — distinct from
MiniTensorRT's own ONNX frontend, proven on the MLP in the ONNX frontend section).

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

## FlashAttention — fused attention with online softmax (F)

A single fused kernel computes `softmax(scale·QKᵀ + causal_mask)·V` with **online
(streaming) softmax**, never materializing the `[H,S,S]` score matrix — it keeps a running
max, denominator, and weighted output per query, rescaling on the fly. It replaces the
BatchedMatMul → Scale → CausalSoftmax → BatchedMatMul chain (and the K transpose) with one
op. Golden-tested against PyTorch `scaled_dot_product_attention` (`Golden.FlashAttention`).

On real GPT-2 the fused model produces **identical tokens** (argmax matches the standard
model at every position, max logit diff ~1e-3 from online-softmax reassociation). The memory
win is O(S²) scratch removed, so it **grows with sequence length**:

| Seq len S | standard peak | FlashAttention peak | reduction |
|---|---|---|---|
| 64 | 1.69 MB | 1.69 MB | 0% |
| 256 | 7.5 MB | 6.75 MB | 10% |
| 512 | 27 MB | **13.5 MB** | **50%** |

The honest reading: at short S the `[S,3072]` FFN activations set the peak, so removing the
smaller `[H,S,S]` attention scratch changes nothing; once S is large enough that the O(S²)
scratch dominates (long context), FlashAttention roughly halves peak memory — which is
exactly the regime it exists for. Enable with `export_gpt2_hf.py --flash`; measured by
`Gpt2Real.FlashAttentionMatchesAndSavesMemory`. A GPU FlashAttention kernel is the next step.

---

## Throughput — prefill vs decode (T)

Latency (per `run()`) is only half the picture; the number an inference runtime is judged
on is **tokens/second**. LLM serving reports it in two regimes, and this runtime already
has both as distinct execution paths:

- **Prefill** — the static graph processes **all S positions in one forward**, so the
  position dimension folds into the batched GEMMs (the tuned NEON ladder). S tokens land
  per forward.
- **Decode** — the KV-cache steps **one token at a time**; each step is O(context) work
  but reloads the full weight set.

Real GPT-2 124M, Apple M1 Pro, Release, S=64 (median of 10 warm forwards for prefill; KV
decode steady-state; `run_gpt2 --mode throughput`):

| Regime | Throughput | Per-token cost |
|---|---|---|
| **Prefill** (static graph, 64 positions/forward) | **287 tok/s** | 3.5 ms/token (222.7 ms ÷ 64) |
| **Decode** (KV-cache, 1 token/step) | **8.8 tok/s** | 113.6 ms/token |

**Prefill is ~32× the decode throughput per token** — the defining asymmetry of LLM
inference, and the numbers show exactly why. Decode is **memory-bandwidth bound**: to
produce one token it must stream all ~622 MB of weights through the ALUs for a single row
of work (arithmetic intensity ≈ 1), so per-token time is set by DRAM bandwidth, not FLOPs.
Prefill is **compute bound**: the same weight load is amortized across 64 positions in one
batched GEMM (intensity ≈ 64×), so it runs near the tuned-GEMM roofline. This is why
production stacks batch aggressively and why decode, not prefill, is the serving
bottleneck. Both paths here run **batch=1** — the executor is single-input by design
(static shapes, one sequence); batching independent sequences to raise decode utilization
is the natural next step and would need a batch dimension threaded through the executor.
The prefill path benefits directly from the tuned `MatMul`/`BatchedMatMul` above.

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

## ONNX frontend — a second real frontend (Week 6)

The runtime now has a **second frontend** that ingests real `.onnx` files, alongside the
JSON loader. This is the payoff of DESIGN D3: the internal IR was built frontend-agnostic,
and a second real frontend *proves* it rather than asserting it. The ONNX loader parses the
protobuf (vendored `onnx.proto` → `protoc`, gated behind `-DMTRT_ONNX=ON` so the core build
needs no protobuf) into the exact same `Graph`/`Tensor` IR; nothing ONNX-specific escapes
`onnx_loader.cpp`, and the *same* `Executor` and kernels run the result.

| Check (`models/mlp.onnx`, MLP) | Result |
|---|---|
| MiniTensorRT (loaded from `.onnx`) vs **ONNX Runtime CPU** | **bit-identical**, max abs err **0** |
| ONNX-loaded graph vs JSON-loaded graph, same executor | identical output (frontend-agnostic) |
| Topology recovered from `.onnx` | 11 tensors, 6 nodes, 4 weights — matches the JSON MLP |

The ONNX graph is emitted (`python/export_onnx.py`) with ops that map 1:1 onto our kernels
(`MatMul`, `Add`, `Gelu`, `Softmax`), from the *same* seeded weights as the JSON export, so
`mlp.onnx` and `mlp.json` are the identical function — which is why the two frontends produce
identical output. The loader is a **strict subset**: an unmapped op or a dynamic/symbolic
dimension throws with a clear message rather than silently misbehaving (shapes come from the
file's `value_info`, populated by `onnx.shape_inference.infer_shapes` at export, honoring the
static-shape invariant). Measured by `OnnxFrontend.*` (built only when `MTRT_ONNX=ON`).
Conv/pooling operators and a CNN remain out of scope — the frontend-agnostic claim is proven
without them.
