// FP16 tensor-core GEMM roofline (H): our hand-written WMMA kernel vs cuBLAS FP16
// (tensor-op path), in GFLOP/s against the T4's ~65 TFLOP/s FP16 tensor-core peak.
// The FP16 analogue of bench_gemm_cuda (FP32 CUDA cores vs cuBLAS). Also reports
// the FP32-cuBLAS GFLOP/s so the tensor-core speedup over FP32 is visible, and the
// accuracy cost of FP16. Square, row-major, sizes multiples of 16.
//
//   ./build/backends/cuda/bench_gemm_fp16 [--sizes 512,1024,2048]

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_fp16.h>
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

__global__ void f2h(const float* in, half* out, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2half(in[i]);
}

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
      for (char* t = std::strtok(argv[++i], ","); t; t = std::strtok(nullptr, ","))
        sizes.push_back(std::atoi(t));
    }

  cudaDeviceProp prop{};
  BK_CK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  std::printf("%6s %12s %12s %12s   %s\n", "N", "wmma-f16", "cuBLAS-f16",
              "cuBLAS-f32", "wmma/cublas16");

  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  for (int n : sizes) {
    const size_t sz = (size_t)n * n;
    std::vector<float> hA(sz), hB(sz);
    for (size_t i = 0; i < sz; ++i) { hA[i] = dist(rng); hB[i] = dist(rng); }

    float *A, *B, *Cf32, *Cwmma, *Ccub;
    half *Ah, *Bh;
    BK_CK(cudaMalloc(&A, sz * sizeof(float)));
    BK_CK(cudaMalloc(&B, sz * sizeof(float)));
    BK_CK(cudaMalloc(&Cf32, sz * sizeof(float)));
    BK_CK(cudaMalloc(&Cwmma, sz * sizeof(float)));
    BK_CK(cudaMalloc(&Ccub, sz * sizeof(float)));
    BK_CK(cudaMalloc(&Ah, sz * sizeof(half)));
    BK_CK(cudaMalloc(&Bh, sz * sizeof(half)));
    BK_CK(cudaMemcpy(A, hA.data(), sz * sizeof(float), cudaMemcpyHostToDevice));
    BK_CK(cudaMemcpy(B, hB.data(), sz * sizeof(float), cudaMemcpyHostToDevice));
    const int thr = 256;
    f2h<<<(sz + thr - 1) / thr, thr>>>(A, Ah, sz);
    f2h<<<(sz + thr - 1) / thr, thr>>>(B, Bh, sz);
    BK_CK(cudaGetLastError());

    // Accuracy: WMMA FP16 vs cuBLAS FP16 (both tensor cores) and vs FP32 cuBLAS.
    mtrt::cuda::gemm_wmma_fp16(Ah, Bh, Cwmma, n, n, n);
    mtrt::cuda::matmul_fp16(Ah, Bh, Ccub, n, n, n);
    mtrt::cuda::matmul(A, B, Cf32, n, n, n);
    BK_CK(cudaDeviceSynchronize());
    std::vector<float> hw(sz), hc(sz), hf(sz);
    BK_CK(cudaMemcpy(hw.data(), Cwmma, sz * sizeof(float), cudaMemcpyDeviceToHost));
    BK_CK(cudaMemcpy(hc.data(), Ccub, sz * sizeof(float), cudaMemcpyDeviceToHost));
    BK_CK(cudaMemcpy(hf.data(), Cf32, sz * sizeof(float), cudaMemcpyDeviceToHost));
    double e_vs_cub = 0.0, e_vs_f32 = 0.0;
    for (size_t i = 0; i < sz; ++i) {
      e_vs_cub = std::fmax(e_vs_cub, std::fabs((double)hw[i] - hc[i]));
      e_vs_f32 = std::fmax(e_vs_f32, std::fabs((double)hw[i] - hf[i]));
    }

    const double flop = 2.0 * n * n * n;
    auto gflops = [&](double ms) { return flop / (ms * 1e6); };
    const int it = n >= 2048 ? 20 : 50;
    const double gw = gflops(time_ms([&] { mtrt::cuda::gemm_wmma_fp16(Ah, Bh, Cwmma, n, n, n); }, it));
    const double gc = gflops(time_ms([&] { mtrt::cuda::matmul_fp16(Ah, Bh, Ccub, n, n, n); }, it));
    const double gf = gflops(time_ms([&] { mtrt::cuda::matmul(A, B, Cf32, n, n, n); }, it));

    std::printf("%6d %10.1f %10.1f %10.1f   %9.1f%%   (err vs cuBLAS-f16=%.2e, vs f32=%.2e)\n",
                n, gw, gc, gf, 100.0 * gw / gc, e_vs_cub, e_vs_f32);

    cudaFree(A); cudaFree(B); cudaFree(Cf32); cudaFree(Cwmma); cudaFree(Ccub);
    cudaFree(Ah); cudaFree(Bh);
  }
  mtrt::cuda::shutdown();
  std::printf("[note] T4 FP16 tensor-core peak ~65 TFLOP/s; FP32 ~8.1 TFLOP/s. FP16 error vs FP32 is expected (reduced mantissa).\n");
  return 0;
}
