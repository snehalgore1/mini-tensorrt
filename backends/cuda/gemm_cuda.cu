#include "backends/cuda/kernels.h"

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// Hand-written FP32 GEMM kernels for the GPU roofline study (bench_gemm_cuda),
// the GPU analogue of the CPU NEON ladder. Row-major C[M,N] = A[M,K] @ B[K,N].

namespace mtrt::cuda {
namespace {

#define GK_CK(call)                                                          \
  do {                                                                       \
    const cudaError_t e__ = (call);                                          \
    if (e__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                   cudaGetErrorString(e__), __FILE__, __LINE__);             \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

// 0. Naive: one thread per C element, streams A row and B column from global mem.
__global__ void gemm_naive_k(const float* A, const float* B, float* C, int M, int N, int K) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
  C[row * N + col] = acc;
}

// 1. Tiled: cooperatively stage TILE x TILE blocks of A and B into shared memory,
// so each global element is reused TILE times (the classic locality win).
constexpr int TILE = 32;
__global__ void gemm_tiled_k(const float* A, const float* B, float* C, int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  const int ty = threadIdx.y, tx = threadIdx.x;
  const int row = blockIdx.y * TILE + ty;
  const int col = blockIdx.x * TILE + tx;
  float acc = 0.f;
  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
    const int aCol = t * TILE + tx;
    const int bRow = t * TILE + ty;
    As[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.f;
    Bs[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.f;
    __syncthreads();
    for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

}  // namespace

void gemm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
  const dim3 block(16, 16);
  const dim3 grid((N + 15) / 16, (M + 15) / 16);
  gemm_naive_k<<<grid, block>>>(A, B, C, M, N, K);
  GK_CK(cudaGetLastError());
}

void gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
  const dim3 block(TILE, TILE);
  const dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  gemm_tiled_k<<<grid, block>>>(A, B, C, M, N, K);
  GK_CK(cudaGetLastError());
}

}  // namespace mtrt::cuda
