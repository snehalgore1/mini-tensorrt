#include <cmath>
#include <cstdint>
#include <cstring>

#include "kernels.h"
#include "mtrt/support/assert.h"
#include "mtrt/tensor.h"

// Kernels specific to GPT-style models: embedding gather (int32 indices) and the
// tanh-approximation GELU that GPT-2 uses ("gelu_new").

namespace mtrt {
namespace {

// Embedding gather (ONNX Gather semantics): out[t, :] = table[ids[t], :].
// inputs: table [V, D] (f32), ids [T] (i32). output: [T, D] (f32).
void gather_f32(const OpContext& ctx) {
  MTRT_ASSERT(ctx.inputs.size() == 2, "Gather expects table and indices");
  MTRT_ASSERT(ctx.outputs.size() == 1, "Gather expects 1 output");
  const Tensor& table = *ctx.inputs[0];
  const Tensor& ids = *ctx.inputs[1];
  Tensor& out = *ctx.outputs[0];
  MTRT_ASSERT(table.dtype() == DType::kF32, "Gather table must be f32");
  MTRT_ASSERT(ids.dtype() == DType::kI32, "Gather indices must be i32");
  MTRT_ASSERT(table.is_contiguous() && out.is_contiguous(), "Gather needs contiguous");

  [[maybe_unused]] const int64_t V = table.shape()[0];  // used in the bounds assert
  const int64_t D = table.shape()[1];
  const int64_t T = ids.numel();
  const float* tb = table.data<float>();
  const int32_t* idx = ids.data<int32_t>();
  float* o = out.data<float>();
  for (int64_t t = 0; t < T; ++t) {
    const int32_t id = idx[t];
    MTRT_ASSERT(id >= 0 && id < V, "Gather index out of range");
    std::memcpy(o + t * D, tb + static_cast<int64_t>(id) * D,
                static_cast<size_t>(D) * sizeof(float));
  }
}

// tanh-approximation GELU (GPT-2 "gelu_new"):
//   0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x^3) ))
void gelu_tanh_f32(const OpContext& ctx) {
  MTRT_ASSERT(ctx.inputs.size() == 1, "GeluTanh expects 1 input");
  MTRT_ASSERT(ctx.outputs.size() == 1, "GeluTanh expects 1 output");
  const Tensor& in = *ctx.inputs[0];
  Tensor& out = *ctx.outputs[0];
  MTRT_ASSERT(in.shape() == out.shape(), "GeluTanh output shape mismatch");
  MTRT_ASSERT(in.is_contiguous() && out.is_contiguous(), "GeluTanh needs contiguous");

  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  const float* pi = in.data<float>();
  float* po = out.data<float>();
  const int64_t n = out.numel();
  for (int64_t i = 0; i < n; ++i) {
    const float x = pi[i];
    const float inner = kSqrt2OverPi * (x + 0.044715f * x * x * x);
    po[i] = 0.5f * x * (1.0f + std::tanh(inner));
  }
}

// Causal (masked) softmax over the last dim of a [.., Sq, Sk] score tensor, as in
// GPT-2 self-attention: for query row q, key j > q is masked out (attends only to
// the past + self). Sq == Sk (square). Numerically stable (subtract row max over
// the valid range). Replaces a separate additive mask + Softmax, which our Add
// broadcast rules ([H,Sq,Sk] + [Sq,Sk]) don't cover.
void causal_softmax_f32(const OpContext& ctx) {
  MTRT_ASSERT(ctx.inputs.size() == 1, "CausalSoftmax expects 1 input");
  MTRT_ASSERT(ctx.outputs.size() == 1, "CausalSoftmax expects 1 output");
  const Tensor& x = *ctx.inputs[0];
  Tensor& y = *ctx.outputs[0];
  MTRT_ASSERT(x.is_contiguous() && y.is_contiguous(), "CausalSoftmax needs contiguous");
  MTRT_ASSERT(x.rank() >= 2, "CausalSoftmax needs rank >= 2");

  const int64_t Sk = x.shape().back();
  const int64_t Sq = x.shape()[static_cast<size_t>(x.rank() - 2)];
  MTRT_ASSERT(Sq == Sk, "CausalSoftmax expects a square score matrix");
  const int64_t rows = Sk == 0 ? 0 : x.numel() / Sk;

  const float* px = x.data<float>();
  float* py = y.data<float>();
  for (int64_t r = 0; r < rows; ++r) {
    const int64_t q = r % Sq;          // query index within its [Sq, Sk] block
    const float* row = px + r * Sk;
    float* orow = py + r * Sk;
    const int64_t valid = q + 1;       // keys 0..q are visible

    float m = row[0];
    for (int64_t j = 1; j < valid; ++j) m = row[j] > m ? row[j] : m;
    float sum = 0.0f;
    for (int64_t j = 0; j < valid; ++j) {
      const float e = std::exp(row[j] - m);
      orow[j] = e;
      sum += e;
    }
    const float inv = 1.0f / sum;
    for (int64_t j = 0; j < valid; ++j) orow[j] *= inv;
    for (int64_t j = valid; j < Sk; ++j) orow[j] = 0.0f;  // masked future
  }
}

}  // namespace

void register_gpt_kernels(KernelRegistry& registry) {
  registry.register_kernel("Gather", DType::kF32, &gather_f32);
  registry.register_kernel("GeluTanh", DType::kF32, &gelu_tanh_f32);
  registry.register_kernel("CausalSoftmax", DType::kF32, &causal_softmax_f32);
}

}  // namespace mtrt
