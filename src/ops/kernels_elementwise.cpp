#include <algorithm>

#include "mtrt/registry.h"
#include "mtrt/support/assert.h"
#include "mtrt/tensor.h"

namespace mtrt {
namespace {

// out = a + b, elementwise. Week 1: identical shapes, contiguous, FP32.
// No broadcasting until a model needs it (scope tripwire).
void add_f32(const OpContext& ctx) {
  MTRT_ASSERT(ctx.inputs.size() == 2, "Add expects 2 inputs");
  MTRT_ASSERT(ctx.outputs.size() == 1, "Add expects 1 output");
  const Tensor& a = *ctx.inputs[0];
  const Tensor& b = *ctx.inputs[1];
  Tensor& out = *ctx.outputs[0];
  MTRT_ASSERT(a.shape() == b.shape(), "Add requires identical input shapes");
  MTRT_ASSERT(a.shape() == out.shape(), "Add output shape mismatch");
  MTRT_ASSERT(a.is_contiguous() && b.is_contiguous() && out.is_contiguous(),
              "Add requires contiguous tensors");

  const float* pa = a.data<float>();
  const float* pb = b.data<float>();
  float* po = out.data<float>();
  const int64_t n = out.numel();
  for (int64_t i = 0; i < n; ++i) po[i] = pa[i] + pb[i];
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

void register_builtin_kernels(KernelRegistry& registry) {
  registry.register_kernel("Add", DType::kF32, &add_f32);
  registry.register_kernel("Relu", DType::kF32, &relu_f32);
}

}  // namespace mtrt
