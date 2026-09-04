#include <algorithm>
#include <cmath>

#include "kernels.h"
#include "mtrt/optimizer/fusion.h"
#include "mtrt/support/assert.h"
#include "mtrt/tensor.h"

namespace mtrt {
namespace {

// gelu(A[M,K] @ B[K,N] + bias) in one pass. bias is [1,N] or [N] (added per
// column). No t0/t1 intermediates -- that is the memory win; keeping the output
// tile hot across matmul+bias+activation is the locality win. FP32, contiguous.
void fused_matmul_bias_gelu_f32(const OpContext& ctx) {
  MTRT_ASSERT(ctx.inputs.size() == 3, "FusedMatMulBiasGelu expects 3 inputs");
  MTRT_ASSERT(ctx.outputs.size() == 1, "FusedMatMulBiasGelu expects 1 output");
  const Tensor& A = *ctx.inputs[0];
  const Tensor& B = *ctx.inputs[1];
  const Tensor& bias = *ctx.inputs[2];
  Tensor& C = *ctx.outputs[0];
  MTRT_ASSERT(A.rank() == 2 && B.rank() == 2 && C.rank() == 2,
              "FusedMatMulBiasGelu operands must be rank 2");
  MTRT_ASSERT(A.is_contiguous() && B.is_contiguous() && C.is_contiguous() &&
                  bias.is_contiguous(),
              "FusedMatMulBiasGelu requires contiguous tensors");

  const int64_t M = A.shape()[0];
  const int64_t K = A.shape()[1];
  const int64_t N = B.shape()[1];
  MTRT_ASSERT(B.shape()[0] == K, "inner dimensions disagree");
  MTRT_ASSERT(C.shape()[0] == M && C.shape()[1] == N, "output shape wrong");
  MTRT_ASSERT(bias.numel() == N, "bias length must equal N");

  constexpr float kInvSqrt2 = 0.7071067811865476f;
  const float* a = A.data<float>();
  const float* b = B.data<float>();
  const float* bs = bias.data<float>();
  float* c = C.data<float>();

  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float acc = bs[j];
      for (int64_t k = 0; k < K; ++k) acc += a[i * K + k] * b[k * N + j];
      c[i * N + j] = 0.5f * acc * (1.0f + std::erf(acc * kInvSqrt2));
    }
  }
}

}  // namespace

void register_fused_kernels(KernelRegistry& registry) {
  registry.register_kernel(kFusedMatMulBiasGelu, DType::kF32,
                           &fused_matmul_bias_gelu_f32);
}

}  // namespace mtrt
