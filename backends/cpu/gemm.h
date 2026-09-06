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
// NEON GEMM with runtime cache-block sizes (M/N/K blocking), for the autotuning
// sweep in benchmarks/bench_autotune.cpp. gemm_neon uses the tuned defaults.
void gemm_neon_blocked(const float* A, const float* B, float* C, int M, int N, int K,
                       int bMC, int bNC, int bKC);
void gemm_threaded(const float* A, const float* B, float* C, int M, int N, int K,
                   ThreadPool& pool);

// Convenience dispatcher for the model executor's MatMul kernels: picks
// gemm_neon vs gemm_threaded by problem size, reusing a shared internal thread
// pool (built once). The env var MTRT_MATMUL=naive|neon|threaded forces a single
// variant, which keeps the naive triple loop selectable as the ablation baseline
// (DESIGN D9, D11).
void gemm_auto(const float* A, const float* B, float* C, int M, int N, int K);

}  // namespace mtrt::cpu
