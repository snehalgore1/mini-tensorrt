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

// ---- M2: LayerNorm, CausalSoftmax, Transpose, Reshape, BatchedMatMul, Gather --

namespace {
// One thread per row (correctness-first; D is small, e.g. 768).
__global__ void layernorm_k(const float* x, const float* g, const float* b,
                            float* y, int rows, int D, float eps) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const float* row = x + (size_t)r * D;
  float* orow = y + (size_t)r * D;
  float mean = 0.f;
  for (int j = 0; j < D; ++j) mean += row[j];
  mean /= D;
  float var = 0.f;
  for (int j = 0; j < D; ++j) { const float c = row[j] - mean; var += c * c; }
  var /= D;
  const float inv = rsqrtf(var + eps);
  for (int j = 0; j < D; ++j) orow[j] = (row[j] - mean) * inv * g[j] + b[j];
}

__global__ void causal_softmax_k(const float* x, float* y, int rows, int Sq, int Sk) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const int q = r % Sq;                 // query index within its [Sq,Sk] block
  const float* row = x + (size_t)r * Sk;
  float* orow = y + (size_t)r * Sk;
  const int valid = q + 1;              // keys 0..q visible
  float m = row[0];
  for (int j = 1; j < valid; ++j) m = row[j] > m ? row[j] : m;
  float sum = 0.f;
  for (int j = 0; j < valid; ++j) { const float e = expf(row[j] - m); orow[j] = e; sum += e; }
  const float inv = 1.f / sum;
  for (int j = 0; j < valid; ++j) orow[j] *= inv;
  for (int j = valid; j < Sk; ++j) orow[j] = 0.f;
}

struct TMeta { int rank; int in_strides[8]; int out_shape[8]; int perm[8]; };

__global__ void transpose_k(const float* x, float* y, TMeta t, int numel) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) return;
  int rem = idx, in_off = 0;
  for (int k = t.rank - 1; k >= 0; --k) {
    const int c = rem % t.out_shape[k];
    rem /= t.out_shape[k];
    in_off += c * t.in_strides[t.perm[k]];
  }
  y[idx] = x[in_off];
}

__global__ void gather_k(const float* table, const int* ids, float* out, int T, int D) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= T * D) return;
  const int t = i / D, dcol = i % D;
  out[i] = table[(size_t)ids[t] * D + dcol];
}
}  // namespace

void layernorm(const float* x, const float* g, const float* b, float* y,
               int rows, int D, float eps) {
  layernorm_k<<<blocks(rows), kThreads>>>(x, g, b, y, rows, D, eps);
  CUDA_CK(cudaGetLastError());
}

void causal_softmax(const float* x, float* y, int rows, int Sq, int Sk) {
  causal_softmax_k<<<blocks(rows), kThreads>>>(x, y, rows, Sq, Sk);
  CUDA_CK(cudaGetLastError());
}

void transpose(const float* x, float* y, const int* in_shape, const int* perm,
               int rank, int numel) {
  TMeta t{};
  t.rank = rank;
  // Contiguous strides of the input, and the output shape (in_shape[perm[k]]).
  int in_str[8];
  in_str[rank - 1] = 1;
  for (int k = rank - 2; k >= 0; --k) in_str[k] = in_str[k + 1] * in_shape[k + 1];
  for (int k = 0; k < rank; ++k) {
    t.in_strides[k] = in_str[k];
    t.perm[k] = perm[k];
    t.out_shape[k] = in_shape[perm[k]];
  }
  transpose_k<<<blocks(numel), kThreads>>>(x, y, t, numel);
  CUDA_CK(cudaGetLastError());
}

void reshape(const float* x, float* y, int numel) {
  CUDA_CK(cudaMemcpy(y, x, (size_t)numel * sizeof(float), cudaMemcpyDeviceToDevice));
}

void batched_matmul(const float* A, const float* B, float* C, int batch, int M,
                    int N, int K) {
  const float alpha = 1.0f, beta = 0.0f;
  const cublasStatus_t st = cublasSgemmStridedBatched(
      handle(), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
      B, N, (long long)K * N, A, K, (long long)M * K, &beta,
      C, N, (long long)M * N, batch);
  if (st != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cublasSgemmStridedBatched failed: %d\n", (int)st);
    std::abort();
  }
}

void gather(const float* table, const int* ids, float* out, int T, int D) {
  gather_k<<<blocks(T * D), kThreads>>>(table, ids, out, T, D);
  CUDA_CK(cudaGetLastError());
}

void shutdown() {
  if (g_handle) { cublasDestroy(g_handle); g_handle = nullptr; }
}

}  // namespace mtrt::cuda
