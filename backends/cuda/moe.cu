#include "backends/cuda/moe.h"

#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cublas_v2.h>
#include <cuda_runtime.h>

// MoE routing / data-movement kernels for the GPU (docs/moe/GPU_DISPATCH_DESIGN.md). Each one
// mirrors a CPU kernel in src/ops/kernels_moe.cpp; backends/cuda/moe_cuda_test.cu checks the
// index outputs EXACTLY against the CPU kernels and the float outputs against the goldens.

namespace mtrt::cuda::moe {
namespace {

#define MOE_CK(call)                                                         \
  do {                                                                       \
    const cudaError_t e__ = (call);                                          \
    if (e__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                   cudaGetErrorString(e__), __FILE__, __LINE__);             \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

constexpr int kThreads = 256;               // all block-per-row kernels; multiple of 32
constexpr int kWarps = kThreads / 32;
constexpr unsigned kFull = 0xffffffffu;
constexpr int kMaxLanesPerExpertSlot = 4;   // top-k supports E <= 32 * 4 = 128
constexpr int kMaxK = 8;

int grid_for(long long n) { return static_cast<int>((n + kThreads - 1) / kThreads); }

// Deterministic block-wide sum (fixed reduction tree). `red` is __shared__[kWarps].
template <class Tv>
__device__ Tv block_sum(Tv v, Tv* red) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(kFull, v, o);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) red[warp] = v;
  __syncthreads();
  Tv r = 0;
  if (warp == 0) {
    r = lane < kWarps ? red[lane] : Tv(0);
    for (int o = 16; o > 0; o >>= 1) r += __shfl_xor_sync(kFull, r, o);
    if (lane == 0) red[0] = r;
  }
  __syncthreads();
  r = red[0];
  __syncthreads();  // red may be reused by the caller right after
  return r;
}

// ---- RMSNorm: one block per row ------------------------------------------------------------
__global__ void rmsnorm_k(const float* x, const float* scale, float* t, int H, float eps) {
  __shared__ float red[kWarps];
  const float* row = x + static_cast<size_t>(blockIdx.x) * H;
  float* out = t + static_cast<size_t>(blockIdx.x) * H;
  float ss = 0.f;
  for (int j = threadIdx.x; j < H; j += blockDim.x) ss += row[j] * row[j];
  ss = block_sum(ss, red);
  const float inv = 1.0f / sqrtf(ss / static_cast<float>(H) + eps);
  for (int j = threadIdx.x; j < H; j += blockDim.x) out[j] = (row[j] * inv) * scale[j];
}

__global__ void add_row_bias_k(float* logits, const float* b, int n, int E) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) logits[i] += b[i % E];
}

// ---- Top-k + softmax: one warp per token ---------------------------------------------------
// Key order: larger value first; on equal values the LOWER expert index first (the CPU rule).
__device__ bool better(float va, int ia, float vb, int ib) {
  return va > vb || (va == vb && ia < ib);
}

__global__ void topk_softmax_k(const float* logits, float* w, int* idx, int T, int E, int k) {
  const int lane = threadIdx.x & 31;
  const int token = blockIdx.x * kWarps + (threadIdx.x >> 5);
  if (token >= T) return;  // warp-uniform: the whole warp leaves together
  const float* row = logits + static_cast<size_t>(token) * E;

  float v[kMaxLanesPerExpertSlot];
  bool live[kMaxLanesPerExpertSlot];
#pragma unroll
  for (int q = 0; q < kMaxLanesPerExpertSlot; ++q) {
    const int e = lane + 32 * q;
    live[q] = e < E;
    v[q] = live[q] ? row[e] : -INFINITY;
  }
  float win_v[kMaxK];
  int win_i[kMaxK];
  for (int r = 0; r < k; ++r) {
    float bv = -INFINITY;
    int bi = INT_MAX;  // "no candidate": loses every comparison against a live one
#pragma unroll
    for (int q = 0; q < kMaxLanesPerExpertSlot; ++q) {
      const int e = lane + 32 * q;
      if (live[q] && better(v[q], e, bv, bi)) { bv = v[q]; bi = e; }
    }
    for (int o = 16; o > 0; o >>= 1) {  // butterfly: every lane ends with the warp winner
      const float ov = __shfl_xor_sync(kFull, bv, o);
      const int oi = __shfl_xor_sync(kFull, bi, o);
      if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
    }
#pragma unroll
    for (int q = 0; q < kMaxLanesPerExpertSlot; ++q)
      if (lane + 32 * q == bi) live[q] = false;  // the owner drops the winner
    win_v[r] = bv;
    win_i[r] = bi;
  }
  if (lane == 0) {
    float ex[kMaxK];
    float sum = 0.f;
    for (int r = 0; r < k; ++r) {
      ex[r] = expf(win_v[r] - win_v[0]);
      sum += ex[r];
    }
    for (int r = 0; r < k; ++r) {
      w[static_cast<size_t>(token) * k + r] = ex[r] / sum;
      idx[static_cast<size_t>(token) * k + r] = win_i[r];
    }
  }
}

// ---- Dispatch: rank by counting, one block per expert ---------------------------------------
__global__ void dispatch_k(const int* idx, int* src, int* inv, int* offsets, int Tk, int E) {
  __shared__ int red[kWarps];
  __shared__ int warp_cnt[kWarps];
  const int e = blockIdx.x;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;

  // Pass 1: start of this expert's rows = number of assignments with a smaller expert id.
  int smaller = 0;
  for (int a = threadIdx.x; a < Tk; a += blockDim.x) smaller += idx[a] < e ? 1 : 0;
  const int start = block_sum(smaller, red);
  if (threadIdx.x == 0) {
    offsets[e] = start;
    if (e == E - 1) offsets[E] = Tk;
  }

  // Pass 2: stable compaction. Chunks of blockDim assignments in ascending a; inside a chunk,
  // rank = (matches in earlier warps) + (matches in earlier lanes of my warp).
  int cursor = 0;
  for (int base = 0; base < Tk; base += blockDim.x) {  // uniform trip count for all threads
    const int a = base + threadIdx.x;
    const bool pred = a < Tk && idx[a] == e;
    const unsigned ballot = __ballot_sync(kFull, pred);
    if (lane == 0) warp_cnt[warp] = __popc(ballot);
    __syncthreads();
    int before = 0, total = 0;
    for (int i = 0; i < kWarps; ++i) {
      const int c = warp_cnt[i];
      if (i < warp) before += c;
      total += c;
    }
    if (pred) {
      const int j = start + cursor + before + __popc(ballot & ((1u << lane) - 1u));
      src[j] = a;
      inv[a] = j;
    }
    cursor += total;
    __syncthreads();  // warp_cnt is rewritten by the next chunk
  }
}

// ---- Row gather, expert bias, SwiGLU, combine, add -----------------------------------------
__global__ void gather_rows_k(const float* t, const int* src, float* ts, int H, int k) {
  const int j = blockIdx.x;
  const float* in = t + static_cast<size_t>(src[j] / k) * H;
  float* out = ts + static_cast<size_t>(j) * H;
  if ((H & 3) == 0) {  // rows are 16-byte aligned when H % 4 == 0 (cudaMalloc is 256-aligned)
    const float4* in4 = reinterpret_cast<const float4*>(in);
    float4* out4 = reinterpret_cast<float4*>(out);
    for (int c = threadIdx.x; c < H / 4; c += blockDim.x) out4[c] = in4[c];
  } else {
    for (int c = threadIdx.x; c < H; c += blockDim.x) out[c] = in[c];
  }
}

__global__ void add_expert_bias_k(float* C, const float* b, const int* idx, const int* src,
                                  int N) {
  const int j = blockIdx.x;
  const int e = idx[src[j]];
  float* row = C + static_cast<size_t>(j) * N;
  const float* bias = b + static_cast<size_t>(e) * N;
  for (int c = threadIdx.x; c < N; c += blockDim.x) row[c] += bias[c];
}

__global__ void swiglu_k(const float* h, float* a, long long n, int I, float alpha,
                         float limit) {
  const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const long long r = i / I, j = i % I;
  const float* row = h + r * 2LL * I;
  const float g = fminf(row[2 * j], limit);
  const float l = fminf(fmaxf(row[2 * j + 1], -limit), limit);
  const float sig = 1.0f / (1.0f + expf(-alpha * g));
  a[i] = g * sig * (l + 1.0f);
}

__global__ void combine_k(const float* y, const int* inv, const float* w, float* m, int H,
                          int k) {
  const int t = blockIdx.x;
  for (int c = threadIdx.x; c < H; c += blockDim.x) {
    float acc = 0.f;
    for (int s = 0; s < k; ++s) {  // fixed slot order, same as the CPU kernel
      const int j = inv[t * k + s];
      const float v = w[t * k + s] * y[static_cast<size_t>(j) * H + c];
      acc = s == 0 ? v : acc + v;
    }
    m[static_cast<size_t>(t) * H + c] = acc;
  }
}

__global__ void add_k(const float* a, const float* b, float* out, long long n) {
  const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

}  // namespace

void rmsnorm(const float* x, const float* scale, float* t, int T, int H, float eps,
             cudaStream_t s) {
  if (T == 0) return;
  rmsnorm_k<<<T, kThreads, 0, s>>>(x, scale, t, H, eps);
  MOE_CK(cudaGetLastError());
}

void add_row_bias(float* logits, const float* b, int T, int E, cudaStream_t s) {
  const int n = T * E;
  if (n == 0) return;
  add_row_bias_k<<<grid_for(n), kThreads, 0, s>>>(logits, b, n, E);
  MOE_CK(cudaGetLastError());
}

void topk_softmax(const float* logits, float* w, int* idx, int T, int E, int k,
                  cudaStream_t s) {
  if (E > 32 * kMaxLanesPerExpertSlot || k < 1 || k > kMaxK || k > E) {
    std::fprintf(stderr, "moe::topk_softmax: need E <= 128 and 1 <= k <= min(8, E)\n");
    std::abort();
  }
  if (T == 0) return;
  topk_softmax_k<<<(T + kWarps - 1) / kWarps, kThreads, 0, s>>>(logits, w, idx, T, E, k);
  MOE_CK(cudaGetLastError());
}

void dispatch(const int* idx, int* src, int* inv, int* offsets, int Tk, int E,
              cudaStream_t s) {
  dispatch_k<<<E, kThreads, 0, s>>>(idx, src, inv, offsets, Tk, E);
  MOE_CK(cudaGetLastError());
}

void gather_rows(const float* t, const int* src, float* t_sorted, int Tk, int H, int k,
                 cudaStream_t s) {
  if (Tk == 0) return;
  gather_rows_k<<<Tk, kThreads, 0, s>>>(t, src, t_sorted, H, k);
  MOE_CK(cudaGetLastError());
}

void add_expert_bias(float* C, const float* b, const int* idx, const int* src, int Tk, int N,
                     cudaStream_t s) {
  if (Tk == 0) return;
  add_expert_bias_k<<<Tk, kThreads, 0, s>>>(C, b, idx, src, N);
  MOE_CK(cudaGetLastError());
}

void swiglu_interleaved(const float* h, float* a, int R, int I, float alpha, float limit,
                        cudaStream_t s) {
  const long long n = static_cast<long long>(R) * I;
  if (n == 0) return;
  swiglu_k<<<grid_for(n), kThreads, 0, s>>>(h, a, n, I, alpha, limit);
  MOE_CK(cudaGetLastError());
}

void combine(const float* y_sorted, const int* inv, const float* w, float* m, int T, int H,
             int k, cudaStream_t s) {
  if (T == 0) return;
  combine_k<<<T, kThreads, 0, s>>>(y_sorted, inv, w, m, H, k);
  MOE_CK(cudaGetLastError());
}

void add(const float* a, const float* b, float* out, int n, cudaStream_t s) {
  if (n == 0) return;
  add_k<<<grid_for(n), kThreads, 0, s>>>(a, b, out, n);
  MOE_CK(cudaGetLastError());
}

// cuBLAS is column-major: computing the column-major product of the same buffers with
// M<->N and A<->B swapped writes C row-major (same trick as kernels.cu matmul()).
void sgemm_rowmajor(cublasHandle_t h, const float* A, const float* B, float* C, int M, int N,
                    int K) {
  const float alpha = 1.0f, beta = 0.0f;
  const cublasStatus_t st =
      cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
  if (st != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "moe::sgemm_rowmajor: cublasSgemm failed (%d)\n", static_cast<int>(st));
    std::abort();
  }
}

}  // namespace mtrt::cuda::moe
