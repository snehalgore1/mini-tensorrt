# Running the MoE GPU study on Kaggle (Milestone 1B)

> The same steps for **Google Colab** are in [`colab/README.md`](../colab/README.md) under
> "MoE study". Both platforms use a T4; use whichever is available.

The GPU work is developed without a local GPU. The CUDA sources are type-checked on the Mac
(`tools/cuda_syntax_check/check.sh`), but they are **built, tested and timed only here**, on a
Kaggle T4. Paste every output block back; only pasted outputs go into the docs.

## 0. Notebook settings (once)

- **Settings → Accelerator → GPU T4 ×2.** The study uses **GPU 0 only**; every command below sets
  `CUDA_VISIBLE_DEVICES=0`.
- **Settings → Internet → On.** It is needed for `git clone` and CMake's GoogleTest fetch.
- Paste each block into its own cell.

## 1. Environment record

```bash
!nvidia-smi
!nvidia-smi --query-gpu=index,name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,power.limit --format=csv
!nvcc --version | tail -2
!cmake --version | head -1    # need >= 3.20; if older:  !pip install -q "cmake>=3.20"
!nproc; free -g | head -2
```

## 2. Clone and build

```bash
%cd /kaggle/working
!rm -rf mini-tensorrt
!git clone --branch feature/moe https://github.com/snehalgore1/mini-tensorrt.git
%cd mini-tensorrt
!git log --oneline -1
!cmake -B build -DCMAKE_BUILD_TYPE=Release -DMTRT_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 > /dev/null
!cmake --build build -j4 --target moe_cuda_test bench_moe_cuda 2>&1 | tail -5
```

## 3. Correctness (must pass before any timing)

```bash
!CUDA_VISIBLE_DEVICES=0 ./build/backends/cuda/moe_cuda_test
```

The output should end with `[MOE-CUDA] N passed, 0 failed -> all passed`. It covers:
- per-kernel goldens for 5 cases (index outputs exact)
- dispatch and top-k checked exactly against the CPU kernels
- V0 and V0-oracle end to end
- V0 == V0-oracle bitwise
- 10 repeated runs bitwise identical
- forced routing

```bash
# Sanitizers (slower; run each and paste the last lines).
!CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool memcheck  ./build/backends/cuda/moe_cuda_test | tail -3
!CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool racecheck ./build/backends/cuda/moe_cuda_test | tail -3
!CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool synccheck ./build/backends/cuda/moe_cuda_test | tail -3
```

If `compute-sanitizer` is not on `PATH`, use `/usr/local/cuda/bin/compute-sanitizer`.

## 4. Benchmark: V0 vs V0-oracle (H1)

The first command is a quick smoke run (a few seconds) to confirm everything works end to end.

```bash
!CUDA_VISIBLE_DEVICES=0 ./build/backends/cuda/bench_moe_cuda --T 1,64 --iters 20 --warmup 5
```

Then the full sweep, with the stage breakdown and a clock record before and after (several
minutes):

```bash
!nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv,noheader
!CUDA_VISIBLE_DEVICES=0 ./build/backends/cuda/bench_moe_cuda --stages --sha $(git rev-parse --short HEAD) --csv /kaggle/working/bench_moe_cuda_v0.csv
!nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv,noheader
!cat /kaggle/working/bench_moe_cuda_v0.csv
```

The plan calls for **two sessions**: run section 4 again in a fresh Kaggle session on another
day, and paste both. Compare timings only within a session; across sessions, compare the
conclusions.

## Notes

- **Clocks.** Kaggle does not allow locking clocks, so the header records the max clocks and
  the `nvidia-smi` lines record the observed ones. The CV column shows the run-to-run spread.
- **Profilers.** Nsight Systems and Nsight Compute are optional. If `nsys` is available,
  `nsys profile --trace=cuda ./build/backends/cuda/bench_moe_cuda --T 64 --routing uniform --iters 20`
  gives the API trace for the sync audit. If it is unavailable, the writeup says so.
