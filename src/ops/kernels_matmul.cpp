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

}  // namespace

void register_matmul_kernels(KernelRegistry& registry) {
  registry.register_kernel("MatMul", DType::kF32, &matmul_f32);
}

}  // namespace mtrt
