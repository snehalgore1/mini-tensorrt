// M0 smoke test for the CUDA backend: prove the toolchain end to end on the
// target GPU (e.g. a Colab T4). It queries the device, uploads a vector, launches
// a trivial kernel, downloads the result, and verifies it -- exercising nvcc, the
// cudart link, kernel launch, and host<->device copies before any real kernel is
// written (M1+). Build with -DMTRT_CUDA=ON; run: ./build/backends/cuda/cuda_smoke
//
// This file only ever compiles when MTRT_CUDA=ON, so the macOS/CPU build never
// sees it.

#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

namespace {
__global__ void saxpy2(const float* x, float* y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = 2.0f * x[i] + 1.0f;
}

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    const cudaError_t err__ = (call);                                        \
    if (err__ != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                   cudaGetErrorString(err__), __FILE__, __LINE__);           \
      return 1;                                                              \
    }                                                                        \
  } while (0)
}  // namespace

int main() {
  int dev_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&dev_count));
  if (dev_count == 0) {
    std::fprintf(stderr, "[CUDA] no device found\n");
    return 1;
  }
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("[CUDA] device 0: %s (sm_%d%d, %.1f GB)\n", prop.name, prop.major,
              prop.minor, prop.totalGlobalMem / 1e9);

  const int n = 1 << 20;
  std::vector<float> h(n);
  for (int i = 0; i < n; ++i) h[i] = static_cast<float>(i);

  float *dx = nullptr, *dy = nullptr;
  CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dy, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dx, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  const int threads = 256;
  saxpy2<<<(n + threads - 1) / threads, threads>>>(dx, dy, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> out(n);
  CUDA_CHECK(cudaMemcpy(out.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dy));

  // Verify y = 2x + 1.
  int bad = 0;
  for (int i = 0; i < n; ++i)
    if (out[i] != 2.0f * h[i] + 1.0f) ++bad;
  if (bad) {
    std::fprintf(stderr, "[CUDA] smoke FAILED: %d mismatched elements\n", bad);
    return 1;
  }
  std::printf("[CUDA] smoke OK: roundtrip + kernel launch verified on %s\n",
              prop.name);
  return 0;
}
