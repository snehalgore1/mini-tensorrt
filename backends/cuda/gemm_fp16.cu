#include "backends/cuda/kernels.h"

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>

// FP16 tensor-core GEMM (H): C[M,N] = A[M,K] @ B[K,N] with half inputs and FP32
// accumulate, using the WMMA API (16x16x16 fragments) that targets the T4's
// tensor cores (sm_70+). Plus a cuBLAS FP16 reference (cublasGemmEx with the
// tensor-op path). This is the FP16 analogue of the FP32 roofline in
// gemm_cuda.cu / bench_gemm_cuda.cu -- the real GPU inference datapath, where
// tensor cores give the large step over FP32 CUDA cores.
//
// Requires M, N, K to be multiples of 16 (the WMMA tile). Row-major throughout.

namespace mtrt::cuda {
namespace {

#define FP16_CK(call)                                                        \
  do {                                                                       \
    const cudaError_t e__ = (call);                                          \
    if (e__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                   cudaGetErrorString(e__), __FILE__, __LINE__);             \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

constexpr int WM = 16, WN = 16, WK = 16;

// One warp computes one 16x16 output tile, accumulating over K in 16-wide steps.
// A is row-major (lda=K), B is row-major (ldb=N); C is FP32 row-major (ldc=N).
// The WMMA API (nvcuda::wmma, <mma.h>) is only defined for __CUDA_ARCH__ >= 700,
// so it must be used inside the device pass only -- referencing it at file scope
// (or in the host pass) fails to find the namespace. Guard the body accordingly;
// the host pass sees an empty stub, the sm_75 device pass sees the tensor-core code.
__global__ void wmma_gemm_k(const half* A, const half* B, float* C,
                            int M, int N, int K) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  using namespace nvcuda;
  const int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  const int warpN = blockIdx.y * blockDim.y + threadIdx.y;
  const int aRow = warpM * WM;
  const int bCol = warpN * WN;
  if (aRow >= M || bCol >= N) return;

  wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (int k = 0; k < K; k += WK) {
    wmma::load_matrix_sync(a_frag, A + (long)aRow * K + k, K);
    wmma::load_matrix_sync(b_frag, B + (long)k * N + bCol, N);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  wmma::store_matrix_sync(C + (long)aRow * N + bCol, c_frag, N, wmma::mem_row_major);
#else
  (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
#endif
}

cublasHandle_t g_fp16_handle = nullptr;
cublasHandle_t fp16_handle() {
  if (!g_fp16_handle) {
    if (cublasCreate(&g_fp16_handle) != CUBLAS_STATUS_SUCCESS) {
      std::fprintf(stderr, "cublasCreate (fp16) failed\n");
      std::abort();
    }
  }
  return g_fp16_handle;
}

}  // namespace

void gemm_wmma_fp16(const half* A, const half* B, float* C, int M, int N, int K) {
  if (M % WM || N % WN || K % WK) {
    std::fprintf(stderr, "gemm_wmma_fp16 requires M,N,K multiples of 16 (got %d,%d,%d)\n",
                 M, N, K);
    std::abort();
  }
  // blockDim (128,4): 4 warps in x, 4 in y -> each block covers 64x64 of C.
  const dim3 block(128, 4);
  const dim3 grid((M + (WM * (block.x / 32)) - 1) / (WM * (block.x / 32)),
                  (N + (WN * block.y) - 1) / (WN * block.y));
  wmma_gemm_k<<<grid, block>>>(A, B, C, M, N, K);
  FP16_CK(cudaGetLastError());
}

// cuBLAS FP16 reference: half inputs, FP32 accumulate, tensor-op path. Row-major
// C[M,N]=A@B via the same operand-swap trick as the FP32 matmul() (compute Cᵀ).
void matmul_fp16(const half* A, const half* B, float* C, int M, int N, int K) {
  const float alpha = 1.0f, beta = 0.0f;
  const cublasStatus_t st = cublasGemmEx(
      fp16_handle(), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
      B, CUDA_R_16F, N, A, CUDA_R_16F, K, &beta,
      C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (st != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cublasGemmEx (fp16) failed: %d\n", (int)st);
    std::abort();
  }
}

}  // namespace mtrt::cuda
