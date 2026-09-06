#pragma once

// CUDA kernel launchers for the GPT-2 operator set. Every pointer here is a
// DEVICE pointer; all tensors are row-major, contiguous, FP32. These mirror the
// CPU kernels in src/ops/ and are validated against the same PyTorch goldens
// (backends/cuda/cuda_test.cu). Host-callable (compiled by nvcc).

namespace mtrt::cuda {

// out[M,N] = a[M,N] + b. If bias_broadcast, b is [N] (or [1,N]) added per column;
// otherwise b is [M,N] elementwise.
void add(const float* a, const float* b, float* out, int M, int N,
         bool bias_broadcast);

// out[n] = x[n] * s.
void scale(const float* x, float* out, int n, float s);

// out[n] = gelu_tanh(x[n])  (GPT-2 "gelu_new").
void gelu_tanh(const float* x, float* out, int n);

// C[M,N] = A[M,K] @ B[K,N] via cuBLAS (SGEMM). Row-major inputs/outputs.
void matmul(const float* A, const float* B, float* C, int M, int N, int K);

// Free the shared cuBLAS handle (optional; process teardown).
void shutdown();

}  // namespace mtrt::cuda
