#include "backends/cpu/gemm.h"
#include "kernels.h"
#include "mtrt/support/assert.h"
#include "mtrt/tensor.h"

namespace mtrt {
namespace {

// C[M,N] = A[M,K] @ B[K,N]. Dispatches to the Week-5 optimized GEMM ladder
// (packed NEON microkernel, multithreaded above a size threshold) via gemm_auto;
// MTRT_MATMUL=naive restores the triple-loop ablation baseline (DESIGN D9, D11).
// This is what lets the tuned GEMM actually accelerate a real model, not just
// the standalone bench_gemm. FP32, contiguous, rank 2.
void matmul_f32(const OpContext& ctx) {
  MTRT_ASSERT(ctx.inputs.size() == 2, "MatMul expects 2 inputs");
  MTRT_ASSERT(ctx.outputs.size() == 1, "MatMul expects 1 output");
  const Tensor& A = *ctx.inputs[0];
  const Tensor& B = *ctx.inputs[1];
  Tensor& C = *ctx.outputs[0];
  MTRT_ASSERT(A.rank() == 2 && B.rank() == 2 && C.rank() == 2,
              "MatMul operands must be rank 2");
  MTRT_ASSERT(A.is_contiguous() && B.is_contiguous() && C.is_contiguous(),
              "MatMul requires contiguous tensors");

  const int64_t M = A.shape()[0];
  const int64_t K = A.shape()[1];
  const int64_t N = B.shape()[1];
  MTRT_ASSERT(B.shape()[0] == K, "MatMul inner dimensions disagree");
  MTRT_ASSERT(C.shape()[0] == M && C.shape()[1] == N, "MatMul output shape wrong");

  cpu::gemm_auto(A.data<float>(), B.data<float>(), C.data<float>(),
                 static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
}

// Weight-only INT8 matmul (#5): C[M,N] = A[M,K] @ (W_q8[K,N] dequantized), where W
// is symmetric **per-channel** (per output column j) INT8-quantized with a scale
// vector scale[N]. Per-channel is essential for transformers: a single per-tensor
// scale is dominated by weight outliers and wrecks accuracy. The int8 weight is
// dequantized in the inner loop, f32 accumulate: C[i,j] = scale[j] * sum_k A[i,k]*W[k,j].
// This is the LLM size win (weights 4x smaller); a speed win needs int8 SIMD (NEON
// SDOT) + activation quantization -- future work. Inputs: A (f32), W (i8), scale (f32[N]).
void matmul_q8_f32(const OpContext& ctx) {
  MTRT_ASSERT(ctx.inputs.size() == 3, "MatMulQ expects A, W_int8, scale");
  MTRT_ASSERT(ctx.outputs.size() == 1, "MatMulQ expects 1 output");
  const Tensor& A = *ctx.inputs[0];
  const Tensor& W = *ctx.inputs[1];
  const Tensor& scale = *ctx.inputs[2];
  Tensor& C = *ctx.outputs[0];
  MTRT_ASSERT(A.dtype() == DType::kF32 && W.dtype() == DType::kI8,
              "MatMulQ: A must be f32, W must be i8");
  MTRT_ASSERT(A.rank() == 2 && W.rank() == 2 && C.rank() == 2, "MatMulQ ranks");
  MTRT_ASSERT(A.is_contiguous() && W.is_contiguous() && C.is_contiguous(),
              "MatMulQ requires contiguous tensors");

  const int64_t M = A.shape()[0];
  const int64_t K = A.shape()[1];
  const int64_t N = W.shape()[1];
  MTRT_ASSERT(W.shape()[0] == K, "MatMulQ inner dimensions disagree");
  MTRT_ASSERT(C.shape()[0] == M && C.shape()[1] == N, "MatMulQ output shape wrong");
  MTRT_ASSERT(scale.numel() == N, "MatMulQ scale must be per-column [N]");

  const float* a = A.data<float>();
  const int8_t* w = W.data<int8_t>();
  const float* s = scale.data<float>();
  float* c = C.data<float>();
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k)
        acc += a[i * K + k] * static_cast<float>(w[k * N + j]);
      c[i * N + j] = acc * s[j];
    }
  }
}

}  // namespace

void register_matmul_kernels(KernelRegistry& registry) {
  registry.register_kernel("MatMul", DType::kF32, &matmul_f32);
  registry.register_kernel("MatMulQ", DType::kF32, &matmul_q8_f32);
}

}  // namespace mtrt
