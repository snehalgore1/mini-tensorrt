# Running the CUDA backend on Google Colab

MiniTensorRT's GPU backend is developed on a CUDA machine (a free/Pro Colab **T4** is
plenty). The macOS/CPU build never touches CUDA — it is all gated behind `-DMTRT_CUDA=ON`.

## One-time per Colab session

1. **Set the runtime to GPU:** *Runtime → Change runtime type → T4 GPU*.
2. Paste each block below into a cell.

```bash
# Confirm a GPU is attached and CUDA is available.
!nvidia-smi
!nvcc --version
```

```bash
# Clone (Colab VMs are ephemeral, so re-clone each session).
%cd /content
!rm -rf mini-tensorrt
!git clone https://github.com/snehalgore1/mini-tensorrt.git
%cd mini-tensorrt
```

```bash
# Configure + build the CUDA backend, then run the M0 smoke test.
# -DCMAKE_CUDA_ARCHITECTURES=75 targets the T4's tensor cores: without it, CUDA 12
# nvcc defaults to sm_52 and the FP16 WMMA kernel (needs sm_70+) compiles to a stub.
!cmake -B build -DCMAKE_BUILD_TYPE=Release -DMTRT_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
!cmake --build build -j --target cuda_smoke
!./build/backends/cuda/cuda_smoke
```

If you configured `build/` earlier without the arch flag, the value is cached — pass
`-DCMAKE_CUDA_ARCHITECTURES=75` again (it overrides the cache) or `rm -rf build` and
reconfigure, otherwise the FP16 kernel stays stubbed.

Expected output (M0):

```
[CUDA] device 0: Tesla T4 (sm_75, 15.8 GB)
[CUDA] smoke OK: roundtrip + kernel launch verified on Tesla T4
```

That confirms the toolchain end to end (nvcc, cudart, kernel launch, host<->device
copies). Later milestones add the device tensor, cuBLAS/attention kernels, and a
`CudaExecutor` that runs the whole GPT-2 block on the GPU — validated against the CPU
reference from this repo. Paste the `[CUDA] ...` / `[RESULTS] ...` output back so the
numbers can be recorded in `docs/RESULTS.md`.

## Correctness + kernel tests (all GPU kernels vs PyTorch goldens)

```bash
!cmake --build build -j --target cuda_test cuda_model_test
!./build/backends/cuda/cuda_test          # per-op goldens, incl. FlashAttention (F-GPU)
!./build/backends/cuda/cuda_model_test    # whole GPT-2 block: GPU vs CPU
```

`cuda_test` should end with `[CUDA-TEST] all passed`, and the `FlashAttn` line confirms
the fused online-softmax GPU kernel matches the PyTorch golden.

## F-GPU + H benchmarks (paste output back to record in RESULTS.md)

These two rows in `docs/RESULTS.md` are marked "pending Colab measurement" — they need a
real GPU. Run the benches and paste the output back:

```bash
# H: FP16 tensor-core GEMM (our WMMA kernel vs cuBLAS FP16 vs FP32), GFLOP/s.
!cmake --build build -j --target bench_gemm_fp16
!./build/backends/cuda/bench_gemm_fp16 --sizes 512,1024,2048

# F-GPU FlashAttention correctness is covered by cuda_test above; the FP32 GEMM
# roofline (context for the FP16 numbers) is:
!cmake --build build -j --target bench_gemm_cuda
!./build/backends/cuda/bench_gemm_cuda --sizes 512,1024,2048
```

The FP16 bench prints `wmma-f16`, `cuBLAS-f16`, and `cuBLAS-f32` GFLOP/s per size plus the
FP16-vs-FP32 accuracy delta. Expected shape of the result: cuBLAS FP16 (tensor cores) is a
large multiple of cuBLAS FP32 on the T4 (~65 vs ~8 TFLOP/s peak), and our WMMA kernel lands
some fraction of cuBLAS FP16 — the "textbook tensor-core kernel vs tuned library" story,
the FP16 analogue of the NEON-vs-Accelerate and tiled-vs-cuBLAS gaps already in the repo.
