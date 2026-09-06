#include <algorithm>
#include <cmath>

#include "backends/cpu/gemm.h"
#include "kernels.h"
#include "mtrt/optimizer/fusion.h"
#include "mtrt/support/assert.h"
#include "mtrt/tensor.h"

namespace mtrt {
namespace {

// act(A[M,K] @ B[K,N] + bias) in one node. bias is [1,N] or [N] (added per
// column). Fusing away the t0/t1 intermediates is the memory win; computing the
// matmul through the tuned GEMM ladder (gemm_auto) then applying a cheap
// bias+activation epilogue keeps this on par with the standalone kernel while
// still eliminating the intermediates. `tanh_approx` selects GPT-2's gelu_new.
// FP32, contiguous.
void fused_matmul_bias_act_f32(const OpContext& ctx, bool tanh_approx) {
  MTRT_ASSERT(ctx.inputs.size() == 3, "FusedMatMulBias* expects 3 inputs");
  MTRT_ASSERT(ctx.outputs.size() == 1, "FusedMatMulBias* expects 1 output");
  const Tensor& A = *ctx.inputs[0];
  const Tensor& B = *ctx.inputs[1];
  const Tensor& bias = *ctx.inputs[2];
  Tensor& C = *ctx.outputs[0];
  MTRT_ASSERT(A.rank() == 2 && B.rank() == 2 && C.rank() == 2,
              "FusedMatMulBias* operands must be rank 2");
  MTRT_ASSERT(A.is_contiguous() && B.is_contiguous() && C.is_contiguous() &&
                  bias.is_contiguous(),
              "FusedMatMulBias* requires contiguous tensors");

  const int64_t M = A.shape()[0];
  const int64_t K = A.shape()[1];
  const int64_t N = B.shape()[1];
  MTRT_ASSERT(B.shape()[0] == K, "inner dimensions disagree");
  MTRT_ASSERT(C.shape()[0] == M && C.shape()[1] == N, "output shape wrong");
  MTRT_ASSERT(bias.numel() == N, "bias length must equal N");

  const float* bs = bias.data<float>();
  float* c = C.data<float>();

  // C = A @ B via the tuned GEMM, then bias + activation over the output tile.
  cpu::gemm_auto(A.data<float>(), B.data<float>(), c, static_cast<int>(M),
                 static_cast<int>(N), static_cast<int>(K));

  constexpr float kInvSqrt2 = 0.7071067811865476f;      // erf GELU
  constexpr float kSqrt2OverPi = 0.7978845608028654f;   // tanh GELU
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      const float v = c[i * N + j] + bs[j];
      if (tanh_approx) {
        const float inner = kSqrt2OverPi * (v + 0.044715f * v * v * v);
        c[i * N + j] = 0.5f * v * (1.0f + std::tanh(inner));
      } else {
        c[i * N + j] = 0.5f * v * (1.0f + std::erf(v * kInvSqrt2));
      }
    }
  }
}

}  // namespace

void register_fused_kernels(KernelRegistry& registry) {
  registry.register_kernel(
      kFusedMatMulBiasGelu, DType::kF32,
      [](const OpContext& ctx) { fused_matmul_bias_act_f32(ctx, false); });
  registry.register_kernel(
      kFusedMatMulBiasGeluTanh, DType::kF32,
      [](const OpContext& ctx) { fused_matmul_bias_act_f32(ctx, true); });
}

}  // namespace mtrt
