#include "backends/cuda/kernels.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace mtrt::cuda {
namespace {

#define CUDA_CK(call)                                                        \
  do {                                                                       \
    const cudaError_t e__ = (call);                                          \
    if (e__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                   cudaGetErrorString(e__), __FILE__, __LINE__);             \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

constexpr int kThreads = 256;
int blocks(int n) { return (n + kThreads - 1) / kThreads; }

__global__ void add_elem(const float* a, const float* b, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

__global__ void add_bias(const float* a, const float* b, float* out, int M, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < M * N) out[i] = a[i] + b[i % N];  // b is [N], broadcast per column
}

__global__ void scale_k(const float* x, float* out, int n, float s) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = x[i] * s;
}

__global__ void gelu_tanh_k(const float* x, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float v = x[i];
    const float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    out[i] = 0.5f * v * (1.0f + tanhf(inner));
  }
}

cublasHandle_t g_handle = nullptr;
cublasHandle_t handle() {
  if (!g_handle) {
    if (cublasCreate(&g_handle) != CUBLAS_STATUS_SUCCESS) {
      std::fprintf(stderr, "cublasCreate failed\n");
      std::abort();
    }
  }
  return g_handle;
}

}  // namespace

void add(const float* a, const float* b, float* out, int M, int N,
         bool bias_broadcast) {
  const int n = M * N;
  if (bias_broadcast)
    add_bias<<<blocks(n), kThreads>>>(a, b, out, M, N);
  else
    add_elem<<<blocks(n), kThreads>>>(a, b, out, n);
  CUDA_CK(cudaGetLastError());
}

void scale(const float* x, float* out, int n, float s) {
  scale_k<<<blocks(n), kThreads>>>(x, out, n, s);
  CUDA_CK(cudaGetLastError());
}

void gelu_tanh(const float* x, float* out, int n) {
  gelu_tanh_k<<<blocks(n), kThreads>>>(x, out, n);
  CUDA_CK(cudaGetLastError());
}

// Row-major C[M,N] = A[M,K] @ B[K,N]. cuBLAS is column-major, so we compute the
// column-major product of the same buffers with M<->N and A<->B swapped: cuBLAS
// then sees B as [N,K]^T and A as [K,M]^T and writes C row-major. (Standard trick.)
void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
  const float alpha = 1.0f, beta = 0.0f;
  const cublasStatus_t st = cublasSgemm(
      handle(), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
  if (st != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cublasSgemm failed: %d\n", (int)st);
    std::abort();
  }
}

void shutdown() {
  if (g_handle) { cublasDestroy(g_handle); g_handle = nullptr; }
}

}  // namespace mtrt::cuda
