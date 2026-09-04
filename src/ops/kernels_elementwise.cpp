#include <algorithm>

#include "mtrt/support/assert.h"
#include "mtrt/tensor.h"
#include "kernels.h"

namespace mtrt {
namespace {

// out = a + b, contiguous, FP32. Two shapes supported: identical (elementwise,
// e.g. residual add) and last-dim bias broadcast (b is [D] or [1,D] added to
// every row of a's last dim, e.g. Linear bias over a sequence). Broadening
// beyond this stays out of scope until a model needs it.
void add_f32(const OpContext& ctx) {
  MTRT_ASSERT(ctx.inputs.size() == 2, "Add expects 2 inputs");
  MTRT_ASSERT(ctx.outputs.size() == 1, "Add expects 1 output");
  const Tensor& a = *ctx.inputs[0];
  const Tensor& b = *ctx.inputs[1];
  Tensor& out = *ctx.outputs[0];
  MTRT_ASSERT(a.shape() == out.shape(), "Add output shape must match first input");
  MTRT_ASSERT(a.is_contiguous() && b.is_contiguous() && out.is_contiguous(),
              "Add requires contiguous tensors");

  const float* pa = a.data<float>();
  const float* pb = b.data<float>();
  float* po = out.data<float>();
  const int64_t n = out.numel();

  if (a.shape() == b.shape()) {
    for (int64_t i = 0; i < n; ++i) po[i] = pa[i] + pb[i];
  } else {
    const int64_t d = a.shape().back();
    MTRT_ASSERT(b.numel() == d, "Add broadcast requires b to match the last dim");
    for (int64_t i = 0; i < n; ++i) po[i] = pa[i] + pb[i % d];
  }
}

// out = max(0, in), elementwise. Contiguous, FP32.
void relu_f32(const OpContext& ctx) {
  MTRT_ASSERT(ctx.inputs.size() == 1, "Relu expects 1 input");
  MTRT_ASSERT(ctx.outputs.size() == 1, "Relu expects 1 output");
  const Tensor& in = *ctx.inputs[0];
  Tensor& out = *ctx.outputs[0];
  MTRT_ASSERT(in.shape() == out.shape(), "Relu output shape mismatch");
  MTRT_ASSERT(in.is_contiguous() && out.is_contiguous(),
              "Relu requires contiguous tensors");

  const float* pi = in.data<float>();
  float* po = out.data<float>();
  const int64_t n = out.numel();
  for (int64_t i = 0; i < n; ++i) po[i] = std::max(0.0f, pi[i]);
}

}  // namespace

void register_elementwise_kernels(KernelRegistry& registry) {
  registry.register_kernel("Add", DType::kF32, &add_f32);
  registry.register_kernel("Relu", DType::kF32, &relu_f32);
}

}  // namespace mtrt
