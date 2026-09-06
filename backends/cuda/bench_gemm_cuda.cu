// GPU GEMM roofline study: our hand-written naive and tiled shared-memory kernels
// vs cuBLAS, in GFLOP/s, against the T4's ceilings. The GPU analogue of the CPU
// bench_gemm (NEON ladder vs Accelerate). FP32, square, row-major.
//
//   ./build/backends/cuda/bench_gemm_cuda [--sizes 512,1024,2048]

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "backends/cuda/kernels.h"

namespace {
#define BK_CK(call)                                                          \
  do {                                                                       \
    const cudaError_t e__ = (call);                                          \
    if (e__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s\n", cudaGetErrorString(e__));      \
      std::exit(2);                                                          \
    }                                                                        \
  } while (0)

// Time a launch closure over `iters` runs (ms), after a warmup.
template <class F>
double time_ms(F&& launch, int iters) {
  cudaEvent_t a, b;
  BK_CK(cudaEventCreate(&a));
  BK_CK(cudaEventCreate(&b));
  launch();  // warmup
  BK_CK(cudaDeviceSynchronize());
  BK_CK(cudaEventRecord(a));
  for (int i = 0; i < iters; ++i) launch();
  BK_CK(cudaEventRecord(b));
  BK_CK(cudaEventSynchronize(b));
  float ms = 0.f;
  BK_CK(cudaEventElapsedTime(&ms, a, b));
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return ms / iters;
}
}  // namespace

int main(int argc, char** argv) {
  std::vector<int> sizes = {512, 1024, 2048};
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--sizes") && i + 1 < argc) {
      sizes.clear();
      char* s = argv[++i];
      for (char* tok = std::strtok(s, ","); tok; tok = std::strtok(nullptr, ","))
        sizes.push_back(std::atoi(tok));
    }

  cudaDeviceProp prop{};
  BK_CK(cudaGetDeviceProperties(&prop, 0));
  // T4 (sm_75) FP32 peak ~8.1 TFLOP/s, memory bandwidth ~320 GB/s.
  std::printf("GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  std::printf("%6s %12s %12s %12s   %s\n", "N", "naive", "tiled", "cuBLAS",
              "tiled/cuBLAS");

  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  for (int n : sizes) {
    const size_t sz = (size_t)n * n;
    std::vector<float> hA(sz), hB(sz);
    for (size_t i = 0; i < sz; ++i) { hA[i] = dist(rng); hB[i] = dist(rng); }
    float *A, *B, *C, *Cref;
    BK_CK(cudaMalloc(&A, sz * sizeof(float)));
    BK_CK(cudaMalloc(&B, sz * sizeof(float)));
    BK_CK(cudaMalloc(&C, sz * sizeof(float)));
    BK_CK(cudaMalloc(&Cref, sz * sizeof(float)));
    BK_CK(cudaMemcpy(A, hA.data(), sz * sizeof(float), cudaMemcpyHostToDevice));
    BK_CK(cudaMemcpy(B, hB.data(), sz * sizeof(float), cudaMemcpyHostToDevice));

    // Correctness: tiled vs cuBLAS (once, at each size).
    mtrt::cuda::matmul(A, B, Cref, n, n, n);
    mtrt::cuda::gemm_tiled(A, B, C, n, n, n);
    BK_CK(cudaDeviceSynchronize());
    std::vector<float> hc(sz), hr(sz);
    BK_CK(cudaMemcpy(hc.data(), C, sz * sizeof(float), cudaMemcpyDeviceToHost));
    BK_CK(cudaMemcpy(hr.data(), Cref, sz * sizeof(float), cudaMemcpyDeviceToHost));
    double max_abs = 0.0;
    for (size_t i = 0; i < sz; ++i) max_abs = std::fmax(max_abs, std::fabs((double)hc[i] - hr[i]));

    const double flop = 2.0 * n * n * n;
    auto gflops = [&](double ms) { return flop / (ms * 1e6); };
    const int it = n >= 2048 ? 10 : 30;
    const double gn = gflops(time_ms([&] { mtrt::cuda::gemm_naive(A, B, C, n, n, n); }, it / 2 + 1));
    const double gt = gflops(time_ms([&] { mtrt::cuda::gemm_tiled(A, B, C, n, n, n); }, it));
    const double gc = gflops(time_ms([&] { mtrt::cuda::matmul(A, B, C, n, n, n); }, it));

    std::printf("%6d %10.1f %10.1f %10.1f   %10.1f%%   (tiled vs cuBLAS max_abs=%.2e)\n",
                n, gn, gt, gc, 100.0 * gt / gc, max_abs);

    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(Cref);
  }
  mtrt::cuda::shutdown();
  std::printf("[note] T4 FP32 peak ~8100 GFLOP/s, bandwidth ~320 GB/s.\n");
  return 0;
}
