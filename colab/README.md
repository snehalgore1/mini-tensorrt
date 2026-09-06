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
!cmake -B build -DCMAKE_BUILD_TYPE=Release -DMTRT_CUDA=ON
!cmake --build build -j --target cuda_smoke
!./build/backends/cuda/cuda_smoke
```

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
