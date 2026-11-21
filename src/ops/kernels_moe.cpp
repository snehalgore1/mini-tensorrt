#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "backends/cpu/gemm.h"
#include "kernels.h"
#include "mtrt/support/assert.h"
#include "mtrt/tensor.h"

// GPT-OSS Mixture-of-Experts block, CPU FP32 (Milestone 1A). Semantics are verified against
// the official MLPBlock in docs/moe/SEMANTICS.md; the op split and buffer layout are
// docs/moe/CPU_DESIGN.md; the decisions are logged in docs/ENGINEERING_DECISIONS.md.
//
// Every kernel works on STATIC shapes. Only the VALUES of idx / src / inv / offsets depend on
// the routing, and no kernel allocates (core invariants 3 and 4).

namespace mtrt {
namespace {

constexpr int64_t kMaxTopK = 8;  // MoETopK keeps its candidates in a stack array

int64_t attr_i(const OpContext& ctx, const char* name) {
  return std::get<int64_t>(ctx.node.attrs.at(name));
}
float attr_f(const OpContext& ctx, const char* name) {
  return static_cast<float>(std::get<double>(ctx.node.attrs.at(name)));
}

// ---------------------------------------------------------------------------------------
// RMSNorm (model.py:43-47):  t = x * rsqrt(mean(x^2) + eps) * scale
// No mean subtraction and no bias (unlike LayerNorm). Inputs x[T,H], scale[H]. Attr eps.
// The sum of squares is accumulated in double: over H = 2880 terms an fp32 running sum
// carries ~sqrt(H)*eps32 relative error into every output; double removes it for ~free
// (H multiply-adds per token).
void rmsnorm_f32(const OpContext& ctx) {
  MTRT_ASSERT(ctx.inputs.size() == 2 && ctx.outputs.size() == 1, "RMSNorm: x, scale -> t");
  const Tensor& x = *ctx.inputs[0];
  const Tensor& scale = *ctx.inputs[1];
  Tensor& y = *ctx.outputs[0];
  MTRT_ASSERT(x.is_contiguous() && y.is_contiguous(), "RMSNorm needs contiguous");
  const int64_t H = x.shape().back();
  const int64_t rows = H == 0 ? 0 : x.numel() / H;
  MTRT_ASSERT(scale.numel() == H && y.numel() == x.numel(), "RMSNorm shape");
  const float eps = attr_f(ctx, "eps");

  const float* px = x.data<float>();
  const float* ps = scale.data<float>();
  float* py = y.data<float>();
  for (int64_t r = 0; r < rows; ++r) {
    const float* row = px + r * H;
    float* out = py + r * H;
    double ss = 0.0;
    for (int64_t j = 0; j < H; ++j) ss += static_cast<double>(row[j]) * row[j];
    const float inv = 1.0f / std::sqrt(static_cast<float>(ss / H) + eps);
    for (int64_t j = 0; j < H; ++j) out[j] = (row[j] * inv) * ps[j];  // official order
  }
}

// ---------------------------------------------------------------------------------------
// MoETopK (model.py:315-317): per token, the k largest logits sorted descending, then a
// softmax over ONLY those k values (so the k weights sum to 1).
// Inputs logits[T,E]. Outputs (f32 first -- the registry keys on outputs[0]):
//   w[T,k] f32, idx[T,k] i32. Attr k.
//
// One pass over the E logits with an insertion-sorted buffer of k. A new logit is
// inserted only where it is STRICTLY greater than an existing entry; experts are visited
// in ascending index, so on equal logits the lower index stays ahead. That is the tie rule
// (SEMANTICS.md) -- torch.topk leaves tie order unspecified. O(E*k) per token, no sort.
void moe_topk_f32(const OpContext& ctx) {
  MTRT_ASSERT(ctx.inputs.size() == 1 && ctx.outputs.size() == 2, "MoETopK: logits -> w, idx");
  const Tensor& logits = *ctx.inputs[0];
  Tensor& w = *ctx.outputs[0];
  Tensor& idx = *ctx.outputs[1];
  MTRT_ASSERT(w.dtype() == DType::kF32 && idx.dtype() == DType::kI32, "MoETopK dtypes");
  const int64_t T = logits.shape()[0], E = logits.shape()[1];
  const int64_t k = attr_i(ctx, "k");
  MTRT_ASSERT(k >= 1 && k <= kMaxTopK && k <= E, "MoETopK: need 1 <= k <= min(E, 8)");
  MTRT_ASSERT(w.numel() == T * k && idx.numel() == T * k, "MoETopK output shape");

  const float* pl = logits.data<float>();
  float* pw = w.data<float>();
  int32_t* pi = idx.data<int32_t>();
  for (int64_t t = 0; t < T; ++t) {
    const float* row = pl + t * E;
    float val[kMaxTopK];
    int32_t id[kMaxTopK];
    int64_t n = 0;  // filled entries, val[0..n) descending
    for (int64_t e = 0; e < E; ++e) {
      const float v = row[e];
      if (n == k && !(v > val[k - 1])) continue;  // not strictly better than the k-th
      int64_t p = (n < k) ? n : k - 1;            // slot to fill (drops the k-th if full)
      while (p > 0 && v > val[p - 1]) {           // strict: equal values keep their order
        val[p] = val[p - 1];
        id[p] = id[p - 1];
        --p;
      }
      val[p] = v;
      id[p] = static_cast<int32_t>(e);
      if (n < k) ++n;
    }
    // Softmax over the k selected logits; subtract the max (val[0]) first, as torch does.
    float sum = 0.0f;
    float ex[kMaxTopK];
    for (int64_t j = 0; j < k; ++j) {
      ex[j] = std::exp(val[j] - val[0]);
      sum += ex[j];
    }
    for (int64_t j = 0; j < k; ++j) {
      pw[t * k + j] = ex[j] / sum;
      pi[t * k + j] = id[j];
    }
  }
}

// ---------------------------------------------------------------------------------------
// MoEDispatch: group the T*k (token, slot) assignments by expert.
// Inputs t[T,H] f32, idx[T,k] i32. Outputs (f32 first):
//   t_sorted[T*k,H] f32   row j = t[src[j] / k]
//   src[T*k] i32          assignment id a = token*k + slot stored at sorted row j
//   inv[T*k] i32          sorted row holding assignment a (inv[src[j]] == j)
//   offsets[E+1] i32      expert e owns rows [offsets[e], offsets[e+1]); offsets[E] == T*k
// Attr num_experts.
//
// Stable counting sort with NO scratch memory (D-MoE-5; checked exhaustively by
// python/verify_counting_sort.py). `offsets` itself is the write cursor:
//   1. histogram into offsets[e+1]
//   2. prefix sum            -> offsets[e] = start of e
//   3. scatter, a ascending  -> stable; afterwards offsets[e] = END of e
//   4. shift right by one (downward, so each read happens before its slot is overwritten)
// O(T*k + E), no comparisons, and the row gather is fused into the scatter.
void moe_dispatch_f32(const OpContext& ctx) {
  MTRT_ASSERT(ctx.inputs.size() == 2 && ctx.outputs.size() == 4,
              "MoEDispatch: t, idx -> t_sorted, src, inv, offsets");
  const Tensor& t = *ctx.inputs[0];
  const Tensor& idx = *ctx.inputs[1];
  Tensor& ts = *ctx.outputs[0];
  Tensor& src_t = *ctx.outputs[1];
  Tensor& inv_t = *ctx.outputs[2];
  Tensor& off_t = *ctx.outputs[3];
  const int64_t T = t.shape()[0], H = t.shape()[1];
  const int64_t k = idx.shape()[1];
  const int64_t E = attr_i(ctx, "num_experts");
  const int64_t Tk = T * k;
  MTRT_ASSERT(idx.dtype() == DType::kI32 && idx.shape()[0] == T, "MoEDispatch idx shape");
  MTRT_ASSERT(src_t.dtype() == DType::kI32 && inv_t.dtype() == DType::kI32 &&
                  off_t.dtype() == DType::kI32,
              "MoEDispatch index outputs must be i32");
  MTRT_ASSERT(ts.numel() == Tk * H && src_t.numel() == Tk && inv_t.numel() == Tk &&
                  off_t.numel() == E + 1,
              "MoEDispatch output shapes");

  const float* pt = t.data<float>();
  const int32_t* pidx = idx.data<int32_t>();
  float* pts = ts.data<float>();
  int32_t* src = src_t.data<int32_t>();
  int32_t* inv = inv_t.data<int32_t>();
  int32_t* off = off_t.data<int32_t>();

  std::fill(off, off + E + 1, 0);
  for (int64_t a = 0; a < Tk; ++a) {  // 1. histogram
    MTRT_ASSERT(pidx[a] >= 0 && pidx[a] < E, "MoEDispatch: expert id out of range");
    off[pidx[a] + 1]++;
  }
  for (int64_t e = 0; e < E; ++e) off[e + 1] += off[e];  // 2. prefix sum
  for (int64_t a = 0; a < Tk; ++a) {                      // 3. stable scatter + gather
    const int32_t j = off[pidx[a]]++;
    src[j] = static_cast<int32_t>(a);
    inv[a] = j;
    std::memcpy(pts + static_cast<int64_t>(j) * H, pt + (a / k) * H,
                static_cast<size_t>(H) * sizeof(float));
  }
  for (int64_t e = E; e > 0; --e) off[e] = off[e - 1];  // 4. shift (must go downward)
  off[0] = 0;
  MTRT_ASSERT(off[E] == Tk, "MoEDispatch: offsets[E] != T*k");
}

// ---------------------------------------------------------------------------------------
// MoEGroupedMatMul: one GEMM per expert over its contiguous segment of sorted rows.
//   for each e with m_e = offsets[e+1]-offsets[e] > 0:
//       C[off_e : off_e+m_e] = A[off_e : off_e+m_e] @ W[e] + b[e]
// Inputs A[Tk,K] f32, offsets[E+1] i32, W[E,K,N] f32, b[E,N] f32. Output C[Tk,N].
// W is the runtime (transposed) layout [in, out] (D-MoE-3), so cpu::gemm_auto -- the tuned
// NEON/threaded GEMM, which wants row-major B[K,N] -- runs unmodified on each segment.
// The segments partition [0, Tk), so every row of C is written; an empty expert is skipped
// and its weights are never read. Experts run serially; parallelism is inside gemm_auto
// (D-MoE-9 records the small-m weakness).
void moe_grouped_matmul_f32(const OpContext& ctx) {
  MTRT_ASSERT(ctx.inputs.size() == 4 && ctx.outputs.size() == 1,
              "MoEGroupedMatMul: A, offsets, W, b -> C");
  const Tensor& A = *ctx.inputs[0];
  const Tensor& off_t = *ctx.inputs[1];
  const Tensor& W = *ctx.inputs[2];
  const Tensor& b = *ctx.inputs[3];
  Tensor& C = *ctx.outputs[0];
  MTRT_ASSERT(A.is_contiguous() && W.is_contiguous() && C.is_contiguous(),
              "MoEGroupedMatMul needs contiguous");
  [[maybe_unused]] const int64_t Tk = A.shape()[0];  // used by Debug asserts
  const int64_t K = A.shape()[1];
  const int64_t E = W.shape()[0], N = W.shape()[2];
  MTRT_ASSERT(W.shape()[1] == K && b.numel() == E * N && off_t.numel() == E + 1,
              "MoEGroupedMatMul input shapes");
  MTRT_ASSERT(off_t.dtype() == DType::kI32, "MoEGroupedMatMul offsets must be i32");
  MTRT_ASSERT(C.shape()[0] == Tk && C.shape()[1] == N, "MoEGroupedMatMul output shape");

  const int32_t* off = off_t.data<int32_t>();
  MTRT_ASSERT(off[0] == 0 && off[E] == Tk, "MoEGroupedMatMul: offsets must cover [0, Tk)");
  const float* pa = A.data<float>();
  const float* pw = W.data<float>();
  const float* pb = b.data<float>();
  float* pc = C.data<float>();
  for (int64_t e = 0; e < E; ++e) {
    const int64_t lo = off[e], m = off[e + 1] - off[e];
    MTRT_ASSERT(m >= 0, "MoEGroupedMatMul: offsets not monotone");
    if (m == 0) continue;
    float* c = pc + lo * N;
    cpu::gemm_auto(pa + lo * K, pw + e * K * N, c, static_cast<int>(m), static_cast<int>(N),
                   static_cast<int>(K));
    const float* bias = pb + e * N;
    for (int64_t r = 0; r < m; ++r)
      for (int64_t j = 0; j < N; ++j) c[r * N + j] += bias[j];
  }
}

// ---------------------------------------------------------------------------------------
// SwiGLUInterleaved (model.py:249-256). INTERLEAVED halves: even columns are the gate,
// odd columns the linear part (NOT first-half / second-half).
//   g = min(h[2j], limit)              upper clamp only
//   l = clamp(h[2j+1], -limit, limit)
//   a[j] = g * sigmoid(alpha * g) * (l + 1)
// Input h[R,2I]. Output a[R,I]. Attrs alpha, limit.
void swiglu_interleaved_f32(const OpContext& ctx) {
  MTRT_ASSERT(ctx.inputs.size() == 1 && ctx.outputs.size() == 1, "SwiGLUInterleaved: h -> a");
  const Tensor& h = *ctx.inputs[0];
  Tensor& a = *ctx.outputs[0];
  const int64_t R = h.shape()[0], I2 = h.shape()[1], I = I2 / 2;
  MTRT_ASSERT(I2 % 2 == 0 && a.shape()[0] == R && a.shape()[1] == I, "SwiGLU shapes");
  const float alpha = attr_f(ctx, "alpha");
  const float limit = attr_f(ctx, "limit");
  const float* ph = h.data<float>();
  float* pa = a.data<float>();
  for (int64_t r = 0; r < R; ++r) {
    const float* row = ph + r * I2;
    float* out = pa + r * I;
    for (int64_t j = 0; j < I; ++j) {
      const float g = std::min(row[2 * j], limit);
      const float l = std::min(std::max(row[2 * j + 1], -limit), limit);
      const float sig = 1.0f / (1.0f + std::exp(-alpha * g));
      out[j] = g * sig * (l + 1.0f);
    }
  }
}

// ---------------------------------------------------------------------------------------
// MoECombine: m[t] = sum_{s=0..k-1} w[t,s] * y_sorted[inv[t*k+s]]
// Inputs y_sorted[Tk,H] f32, inv[Tk] i32, w[T,k] f32. Output m[T,H].
// A GATHER through inv: each output row has exactly one writer, reads its k rows, and sums
// them in a fixed slot order. No atomics, no scatter-add, bitwise-reproducible run to run.
// The residual (+x) is the following Add node (fusing it is a later, measured change).
void moe_combine_f32(const OpContext& ctx) {
  MTRT_ASSERT(ctx.inputs.size() == 3 && ctx.outputs.size() == 1, "MoECombine: y, inv, w -> m");
  const Tensor& y = *ctx.inputs[0];
  const Tensor& inv_t = *ctx.inputs[1];
  const Tensor& w = *ctx.inputs[2];
  Tensor& m = *ctx.outputs[0];
  const int64_t T = w.shape()[0], k = w.shape()[1], H = y.shape()[1];
  MTRT_ASSERT(y.shape()[0] == T * k && inv_t.numel() == T * k, "MoECombine input shapes");
  MTRT_ASSERT(inv_t.dtype() == DType::kI32, "MoECombine inv must be i32");
  MTRT_ASSERT(m.shape()[0] == T && m.shape()[1] == H, "MoECombine output shape");
  const float* py = y.data<float>();
  const int32_t* inv = inv_t.data<int32_t>();
  const float* pw = w.data<float>();
  float* pm = m.data<float>();
  for (int64_t t = 0; t < T; ++t) {
    float* out = pm + t * H;
    for (int64_t s = 0; s < k; ++s) {
      const int64_t j = inv[t * k + s];
      MTRT_ASSERT(j >= 0 && j < T * k, "MoECombine: inv out of range");
      const float ws = pw[t * k + s];
      const float* row = py + j * H;
      if (s == 0) {
        for (int64_t c = 0; c < H; ++c) out[c] = ws * row[c];
      } else {
        for (int64_t c = 0; c < H; ++c) out[c] += ws * row[c];
      }
    }
  }
}

}  // namespace

void register_moe_kernels(KernelRegistry& registry) {
  registry.register_kernel("RMSNorm", DType::kF32, &rmsnorm_f32);
  registry.register_kernel("MoETopK", DType::kF32, &moe_topk_f32);
  registry.register_kernel("MoEDispatch", DType::kF32, &moe_dispatch_f32);
  registry.register_kernel("MoEGroupedMatMul", DType::kF32, &moe_grouped_matmul_f32);
  registry.register_kernel("SwiGLUInterleaved", DType::kF32, &swiglu_interleaved_f32);
  registry.register_kernel("MoECombine", DType::kF32, &moe_combine_f32);
}

}  // namespace mtrt
