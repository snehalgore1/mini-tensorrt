#include <cmath>
#include <cstring>
#include <vector>

#include "backends/cpu/gemm.h"
#include "kernels.h"
#include "mtrt/support/assert.h"
#include "mtrt/tensor.h"

// Kernels needed for a multi-head transformer block, beyond the reused MatMul /
// Softmax / Gelu / Add. FP32, contiguous.

namespace mtrt {
namespace {

// LayerNorm over the last dim: y = (x-mean)/sqrt(var+eps) * gamma + beta.
// Biased variance (divide by D), matching PyTorch F.layer_norm. Inputs: x,
// gamma[D], beta[D]. Attr: eps (double).
void layernorm_f32(const OpContext& ctx) {
  MTRT_ASSERT(ctx.inputs.size() == 3, "LayerNorm expects x, gamma, beta");
  MTRT_ASSERT(ctx.outputs.size() == 1, "LayerNorm expects 1 output");
  const Tensor& x = *ctx.inputs[0];
  const Tensor& gamma = *ctx.inputs[1];
  const Tensor& beta = *ctx.inputs[2];
  Tensor& y = *ctx.outputs[0];
  MTRT_ASSERT(x.is_contiguous() && y.is_contiguous(), "LayerNorm needs contiguous");

  const auto eps = static_cast<float>(std::get<double>(ctx.node.attrs.at("eps")));
  const int64_t d = x.shape().back();
  const int64_t rows = d == 0 ? 0 : x.numel() / d;
  MTRT_ASSERT(gamma.numel() == d && beta.numel() == d, "LayerNorm gamma/beta size");

  const float* px = x.data<float>();
  const float* pg = gamma.data<float>();
  const float* pb = beta.data<float>();
  float* py = y.data<float>();

  for (int64_t r = 0; r < rows; ++r) {
    const float* row = px + r * d;
    float* orow = py + r * d;
    float mean = 0.0f;
    for (int64_t j = 0; j < d; ++j) mean += row[j];
    mean /= static_cast<float>(d);
    float var = 0.0f;
    for (int64_t j = 0; j < d; ++j) {
      const float c = row[j] - mean;
      var += c * c;
    }
    var /= static_cast<float>(d);
    const float inv = 1.0f / std::sqrt(var + eps);
    for (int64_t j = 0; j < d; ++j)
      orow[j] = (row[j] - mean) * inv * pg[j] + pb[j];
  }
}

// y = x * scale. Attr: scale (double).
void scale_f32(const OpContext& ctx) {
  MTRT_ASSERT(ctx.inputs.size() == 1, "Scale expects 1 input");
  MTRT_ASSERT(ctx.outputs.size() == 1, "Scale expects 1 output");
  const Tensor& x = *ctx.inputs[0];
  Tensor& y = *ctx.outputs[0];
  MTRT_ASSERT(x.is_contiguous() && y.is_contiguous(), "Scale needs contiguous");
  const auto s = static_cast<float>(std::get<double>(ctx.node.attrs.at("scale")));
  const float* px = x.data<float>();
  float* py = y.data<float>();
  const int64_t n = y.numel();
  for (int64_t i = 0; i < n; ++i) py[i] = px[i] * s;
}

// Contiguous reshape: same bytes, new shape (validated). A copy, so the output
// gets its own arena slot; keeps the executor's buffer model simple.
void reshape_f32(const OpContext& ctx) {
  MTRT_ASSERT(ctx.inputs.size() == 1, "Reshape expects 1 input");
  MTRT_ASSERT(ctx.outputs.size() == 1, "Reshape expects 1 output");
  const Tensor& x = *ctx.inputs[0];
  Tensor& y = *ctx.outputs[0];
  MTRT_ASSERT(x.is_contiguous() && y.is_contiguous(), "Reshape needs contiguous");
  MTRT_ASSERT(x.numel() == y.numel(), "Reshape element count mismatch");
  std::memcpy(y.data<float>(), x.data<float>(),
              static_cast<size_t>(x.numel()) * sizeof(float));
}

// General n-d transpose. Attr: perm (vector<int64>). out.shape[k] = in.shape[perm[k]].
void transpose_f32(const OpContext& ctx) {
  MTRT_ASSERT(ctx.inputs.size() == 1, "Transpose expects 1 input");
  MTRT_ASSERT(ctx.outputs.size() == 1, "Transpose expects 1 output");
  const Tensor& x = *ctx.inputs[0];
  Tensor& y = *ctx.outputs[0];
  MTRT_ASSERT(x.is_contiguous() && y.is_contiguous(), "Transpose needs contiguous");
  const auto& perm = std::get<std::vector<int64_t>>(ctx.node.attrs.at("perm"));
  const int64_t rank = x.rank();
  MTRT_ASSERT(static_cast<int64_t>(perm.size()) == rank, "perm rank mismatch");

  // Fixed-rank stack scratch (rank is tiny -- <=4 for our models). Keeps the op
  // allocation-free in the executor hot path (invariant 4); it previously heap-
  // allocated two vectors (strides + coords) on every Transpose node.
  constexpr int kMaxRank = 8;
  MTRT_ASSERT(rank <= kMaxRank, "Transpose rank exceeds supported max (8)");
  int64_t in_strides[kMaxRank];
  int64_t s = 1;
  for (int64_t k = rank - 1; k >= 0; --k) { in_strides[k] = s; s *= x.shape()[k]; }
  const std::vector<int64_t>& out_shape = y.shape();
  const float* px = x.data<float>();
  float* py = y.data<float>();
  const int64_t n = y.numel();

  int64_t co[kMaxRank];
  for (int64_t idx = 0; idx < n; ++idx) {
    // decode idx -> out coords co (row-major over out_shape)
    int64_t rem = idx;
    for (int64_t k = rank - 1; k >= 0; --k) {
      co[static_cast<size_t>(k)] = rem % out_shape[static_cast<size_t>(k)];
      rem /= out_shape[static_cast<size_t>(k)];
    }
    // in coords: in[perm[k]] = co[k]  ->  in_off = sum co[k]*in_strides[perm[k]]
    int64_t in_off = 0;
    for (int64_t k = 0; k < rank; ++k)
      in_off += co[static_cast<size_t>(k)] *
                in_strides[static_cast<size_t>(perm[static_cast<size_t>(k)])];
    py[idx] = px[in_off];
  }
}

// Batched matmul: A[B,M,K] @ B[B,K,N] -> C[B,M,N], 2D matmul per batch slice.
void batched_matmul_f32(const OpContext& ctx) {
  MTRT_ASSERT(ctx.inputs.size() == 2, "BatchedMatMul expects 2 inputs");
  MTRT_ASSERT(ctx.outputs.size() == 1, "BatchedMatMul expects 1 output");
  const Tensor& A = *ctx.inputs[0];
  const Tensor& B = *ctx.inputs[1];
  Tensor& C = *ctx.outputs[0];
  MTRT_ASSERT(A.rank() == 3 && B.rank() == 3 && C.rank() == 3,
              "BatchedMatMul operands must be rank 3");
  MTRT_ASSERT(A.is_contiguous() && B.is_contiguous() && C.is_contiguous(),
              "BatchedMatMul needs contiguous");
  const int64_t bt = A.shape()[0];
  const int64_t M = A.shape()[1];
  const int64_t K = A.shape()[2];
  const int64_t N = B.shape()[2];
  MTRT_ASSERT(B.shape()[0] == bt && B.shape()[1] == K, "batched inner dims");
  MTRT_ASSERT(C.shape()[0] == bt && C.shape()[1] == M && C.shape()[2] == N,
              "batched output shape");

  const float* a = A.data<float>();
  const float* b = B.data<float>();
  float* c = C.data<float>();
  // Each batch slice is a plain [M,K]@[K,N] row-major GEMM (both attention uses --
  // QKᵀ and attn·V -- pre-transpose their operand, so A@B here is standard). Route
  // it through the tuned Week-5 ladder instead of the triple loop; the profiler
  // flagged this as the single hottest op once MatMul was tuned (docs/RESULTS.md).
  // gemm_auto honors MTRT_MATMUL=naive, so the ablation baseline stays selectable.
  for (int64_t bi = 0; bi < bt; ++bi) {
    cpu::gemm_auto(a + bi * M * K, b + bi * K * N, c + bi * M * N,
                   static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
  }
}

}  // namespace

void register_transformer_kernels(KernelRegistry& registry) {
  registry.register_kernel("LayerNorm", DType::kF32, &layernorm_f32);
  registry.register_kernel("Scale", DType::kF32, &scale_f32);
  registry.register_kernel("Reshape", DType::kF32, &reshape_f32);
  registry.register_kernel("Transpose", DType::kF32, &transpose_f32);
  registry.register_kernel("BatchedMatMul", DType::kF32, &batched_matmul_f32);
}

}  // namespace mtrt
