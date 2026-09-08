#include "backends/cuda/kernels.h"

#include <cstdio>
#include <cstdlib>
#include <math.h>

#include <cuda_runtime.h>

// CUDA FlashAttention-style fused causal attention with online (streaming)
// softmax, the GPU analogue of the CPU kernel in src/ops/kernels_gpt.cpp:
//   out[H,S,d] = softmax(scale * Q Kᵀ + causal_mask) @ V
// computed WITHOUT materializing the [H,S,S] score matrix. It mirrors the CPU
// kernel's math exactly (same running max/denominator/output rescale), so it is
// validated against the same PyTorch golden (op_flashattn_*) in cuda_test.cu.
//
// Parallelization: one thread per (head, query). Each thread streams keys 0..i,
// keeping the running softmax state in registers (O(d), not O(S²)) -- that O(d)
// scratch is the whole point of flash attention. This is correctness-first (it
// maps 1:1 onto the CPU kernel); shared-memory key/query tiling across a thread
// block is the next optimization.

namespace mtrt::cuda {
namespace {

#define FA_CK(call)                                                          \
  do {                                                                       \
    const cudaError_t e__ = (call);                                          \
    if (e__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                   cudaGetErrorString(e__), __FILE__, __LINE__);             \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

// Max supported head dim; GPT-2-small uses d=64. The running weighted output is
// held in per-thread registers/local memory of this size.
constexpr int kMaxHeadDim = 128;

__global__ void flash_attention_k(const float* Q, const float* K, const float* V,
                                  float* O, int H, int S, int d, float scale) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)H * S) return;
  const int h = (int)(idx / S);
  const int i = (int)(idx % S);

  const float* qh = Q + (long)h * S * d;
  const float* kh = K + (long)h * S * d;
  const float* vh = V + (long)h * S * d;
  const float* qi = qh + (long)i * d;

  float acc[kMaxHeadDim];
  for (int t = 0; t < d; ++t) acc[t] = 0.f;
  float m = -INFINITY, l = 0.f;

  for (int j = 0; j <= i; ++j) {  // causal: keys 0..i
    const float* kj = kh + (long)j * d;
    float s = 0.f;
    for (int t = 0; t < d; ++t) s += qi[t] * kj[t];
    s *= scale;
    const float m_new = s > m ? s : m;
    const float corr = expf(m - m_new);  // m=-inf on first j -> corr=0
    const float p = expf(s - m_new);
    l = l * corr + p;
    const float* vj = vh + (long)j * d;
    for (int t = 0; t < d; ++t) acc[t] = acc[t] * corr + p * vj[t];
    m = m_new;
  }

  const float inv = 1.f / l;
  float* oi = O + (long)h * S * d + (long)i * d;
  for (int t = 0; t < d; ++t) oi[t] = acc[t] * inv;
}

}  // namespace

void flash_attention(const float* Q, const float* K, const float* V, float* O,
                     int H, int S, int d, float scale) {
  if (d > kMaxHeadDim) {
    std::fprintf(stderr, "flash_attention: head dim %d exceeds max %d\n", d,
                 kMaxHeadDim);
    std::abort();
  }
  const int threads = 128;
  const int blocks = (H * S + threads - 1) / threads;
  flash_attention_k<<<blocks, threads>>>(Q, K, V, O, H, S, d, scale);
  FA_CK(cudaGetLastError());
}

}  // namespace mtrt::cuda
