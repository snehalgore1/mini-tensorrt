#pragma once

// GPT-OSS MoE layer on the GPU: the routing / data-movement kernels shared by every 1B variant
// (docs/moe/GPU_DISPATCH_DESIGN.md, approved at Checkpoint 3), plus a row-major cuBLAS SGEMM.
//
// Rules (GPU_DISPATCH_DESIGN.md §5):
// - every launcher takes an explicit stream; nothing here touches the legacy default stream
// - no launcher allocates, copies to/from the host, or synchronizes
// All pointers are DEVICE pointers, row-major, contiguous. Index tensors are int32.

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace mtrt::cuda::moe {

// t[T,H] = x * 1/sqrt(mean(x^2) + eps) * scale. One block per row, fp32 tree reduction.
void rmsnorm(const float* x, const float* scale, float* t, int T, int H, float eps,
             cudaStream_t s);

// logits[T,E] += b[E] (router bias), in place.
void add_row_bias(float* logits, const float* b, int T, int E, cudaStream_t s);

// Per token: the k largest logits, sorted descending, ties to the LOWER expert index, then a
// softmax over those k. One warp per token. Requires E <= 128, k <= 8, finite logits.
void topk_softmax(const float* logits, float* w, int* idx, int T, int E, int k,
                  cudaStream_t s);

// Stable grouping of the Tk = T*k assignments (a = token*k + slot) by expert, "rank by
// counting": one block per expert, no atomics, no workspace, one launch.
//   offsets[e] = #{a : idx[a] < e},  offsets[E] = Tk
//   src[j] = a for the j-th row (ascending a within an expert),  inv[src[j]] = j
// Output is identical to the CPU MoEDispatch kernel (a stable sort is unique).
void dispatch(const int* idx, int* src, int* inv, int* offsets, int Tk, int E,
              cudaStream_t s);

// t_sorted[j,:] = t[src[j] / k, :]. One block per sorted row.
void gather_rows(const float* t, const int* src, float* t_sorted, int Tk, int H, int k,
                 cudaStream_t s);

// C[j,:] += b[e,:] with e = idx[src[j]] (the expert that owns sorted row j).
void add_expert_bias(float* C, const float* b, const int* idx, const int* src, int Tk, int N,
                     cudaStream_t s);

// a[r,j] = g*sigmoid(alpha*g)*(l+1), g = min(h[r,2j], limit), l = clamp(h[r,2j+1], +-limit).
void swiglu_interleaved(const float* h, float* a, int R, int I, float alpha, float limit,
                        cudaStream_t s);

// m[t,:] = sum_{s=0..k-1} w[t,s] * y_sorted[inv[t*k+s], :], fixed slot order, one writer.
void combine(const float* y_sorted, const int* inv, const float* w, float* m, int T, int H,
             int k, cudaStream_t s);

// out[i] = a[i] + b[i].
void add(const float* a, const float* b, float* out, int n, cudaStream_t s);

// Row-major C[M,N] = A[M,K] @ B[K,N] with cuBLAS. The handle must already be bound to the
// intended stream (cublasSetStream) and given a fixed workspace (cublasSetWorkspace) so that
// cuBLAS never allocates in a timed path. Aborts on error.
void sgemm_rowmajor(cublasHandle_t h, const float* A, const float* B, float* C, int M, int N,
                    int K);

}  // namespace mtrt::cuda::moe
