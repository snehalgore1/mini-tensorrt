#include <cmath>

#include "kernels.h"
#include "mtrt/support/assert.h"
#include "mtrt/tensor.h"

namespace mtrt {
namespace {

// Exact, erf-based GELU: 0.5 * x * (1 + erf(x / sqrt(2))). Matches PyTorch's
// default nn.GELU() (approximate='none'). FP32, contiguous.
void gelu_f32(const OpContext& ctx) {
  MTRT_ASSERT(ctx.inputs.size() == 1, "Gelu expects 1 input");
  MTRT_ASSERT(ctx.outputs.size() == 1, "Gelu expects 1 output");
  const Tensor& in = *ctx.inputs[0];
  Tensor& out = *ctx.outputs[0];
  MTRT_ASSERT(in.shape() == out.shape(), "Gelu output shape mismatch");
  MTRT_ASSERT(in.is_contiguous() && out.is_contiguous(),
              "Gelu requires contiguous tensors");

  constexpr float kInvSqrt2 = 0.7071067811865476f;
  const float* pi = in.data<float>();
  float* po = out.data<float>();
  const int64_t n = out.numel();
  for (int64_t i = 0; i < n; ++i) {
    po[i] = 0.5f * pi[i] * (1.0f + std::erf(pi[i] * kInvSqrt2));
  }
}

// Softmax over the last axis. Numerically stable (subtract row max). FP32,
// contiguous. Treats the tensor as rows of length D = last dim.
void softmax_f32(const OpContext& ctx) {
  MTRT_ASSERT(ctx.inputs.size() == 1, "Softmax expects 1 input");
  MTRT_ASSERT(ctx.outputs.size() == 1, "Softmax expects 1 output");
  const Tensor& in = *ctx.inputs[0];
  Tensor& out = *ctx.outputs[0];
  MTRT_ASSERT(in.shape() == out.shape(), "Softmax output shape mismatch");
  MTRT_ASSERT(in.is_contiguous() && out.is_contiguous(),
              "Softmax requires contiguous tensors");
  MTRT_ASSERT(in.rank() >= 1, "Softmax needs at least rank 1");

  const int64_t d = in.shape().back();
  const int64_t rows = d == 0 ? 0 : in.numel() / d;
  const float* pi = in.data<float>();
  float* po = out.data<float>();

  for (int64_t r = 0; r < rows; ++r) {
    const float* row = pi + r * d;
    float* orow = po + r * d;
    float mx = row[0];
    for (int64_t j = 1; j < d; ++j) mx = std::max(mx, row[j]);
    float sum = 0.0f;
    for (int64_t j = 0; j < d; ++j) {
      const float e = std::exp(row[j] - mx);
      orow[j] = e;
      sum += e;
    }
    const float inv = 1.0f / sum;
    for (int64_t j = 0; j < d; ++j) orow[j] *= inv;
  }
}

}  // namespace

void register_activation_kernels(KernelRegistry& registry) {
  registry.register_kernel("Gelu", DType::kF32, &gelu_f32);
  registry.register_kernel("Softmax", DType::kF32, &softmax_f32);
}

}  // namespace mtrt
