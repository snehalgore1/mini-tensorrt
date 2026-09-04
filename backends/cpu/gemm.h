#pragma once

// FP32 GEMM optimization ladder: C[M,N] = A[M,K] @ B[K,N], row-major. Each step
// is a separate, independently-benchmarkable variant, all retained as the
// ablation baseline (DESIGN D9, D11). All handle general (non-multiple) sizes.

namespace mtrt {
class ThreadPool;
}

namespace mtrt::cpu {

void gemm_naive(const float* A, const float* B, float* C, int M, int N, int K);
void gemm_reorder(const float* A, const float* B, float* C, int M, int N, int K);
void gemm_register(const float* A, const float* B, float* C, int M, int N, int K);
void gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K);
void gemm_packed(const float* A, const float* B, float* C, int M, int N, int K);
void gemm_neon(const float* A, const float* B, float* C, int M, int N, int K);
void gemm_threaded(const float* A, const float* B, float* C, int M, int N, int K,
                   ThreadPool& pool);

}  // namespace mtrt::cpu
