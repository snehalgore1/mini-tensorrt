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

// out[n] = gelu(x[n])  (exact, erf-based).
void gelu(const float* x, float* out, int n);

// Plain softmax over the last dim: x is [rows, n].
void softmax(const float* x, float* out, int rows, int n);

// C[M,N] = A[M,K] @ B[K,N] via cuBLAS (SGEMM). Row-major inputs/outputs.
void matmul(const float* A, const float* B, float* C, int M, int N, int K);

// Hand-written FP32 GEMM (C[M,N] = A[M,K] @ B[K,N], row-major) for the roofline
// study vs cuBLAS (bench_gemm_cuda): a naive one-thread-per-output kernel and a
// shared-memory tiled kernel.
void gemm_naive(const float* A, const float* B, float* C, int M, int N, int K);
void gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K);

// LayerNorm over the last dim: out = (x-mean)/sqrt(var+eps)*gamma + beta, biased
// variance (÷D), matching PyTorch. x is [rows, D]; gamma/beta are [D].
void layernorm(const float* x, const float* gamma, const float* beta, float* out,
               int rows, int D, float eps);

// Causal (masked) softmax over the last dim of a [.., Sq, Sk] score tensor:
// for query row q, key j>q is masked. rows = numel/Sk, Sq==Sk.
void causal_softmax(const float* x, float* out, int rows, int Sq, int Sk);

// n-d transpose: out[k] = in[perm[k]]. in_shape/perm are host arrays of length
// rank (<=8). numel = product(in_shape).
void transpose(const float* x, float* out, const int* in_shape, const int* perm,
               int rank, int numel);

// Contiguous reshape: same bytes, new shape (a device-to-device copy).
void reshape(const float* x, float* out, int numel);

// Batched matmul: A[B,M,K] @ B[B,K,N] -> C[B,M,N] via cuBLAS strided-batched.
void batched_matmul(const float* A, const float* Bmat, float* C, int batch, int M,
                    int N, int K);

// Embedding gather: out[t,:] = table[ids[t],:]. table [V,D] (f32), ids [T] (i32).
void gather(const float* table, const int* ids, float* out, int T, int D);

// Free the shared cuBLAS handle (optional; process teardown).
void shutdown();

}  // namespace mtrt::cuda
