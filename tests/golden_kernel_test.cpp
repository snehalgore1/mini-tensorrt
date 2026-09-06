#include <gtest/gtest.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "mtrt/node.h"
#include "mtrt/op_context.h"
#include "mtrt/registry.h"
#include "mtrt/tensor.h"
#include "tests/support/golden.h"

using namespace mtrt;
using namespace mtrt::testing;

namespace {

void run_kernel_attrs(const std::string& op,
                      const std::vector<const Tensor*>& ins, Tensor& out,
                      std::unordered_map<std::string, Attribute> attrs) {
  KernelRegistry reg;
  register_builtin_kernels(reg);
  KernelFn fn = reg.lookup(op, DType::kF32);
  ASSERT_NE(fn, nullptr) << "no kernel for " << op;
  Node node{op, {}, {}, std::move(attrs)};
  std::vector<Tensor*> outs{&out};
  OpContext ctx{ins, outs, node};
  fn(ctx);
}

void run_kernel(const std::string& op, const std::vector<const Tensor*>& ins,
                Tensor& out) {
  run_kernel_attrs(op, ins, out, {});
}

}  // namespace

TEST(Golden, Add) {
  Tensor a = tensor_from_npy(load_golden("op_add_a"));
  Tensor b = tensor_from_npy(load_golden("op_add_b"));
  NpyArray exp = load_golden("op_add_out");
  Tensor out = Tensor::owning(DType::kF32, a.shape());
  run_kernel("Add", {&a, &b}, out);
  ExpectGolden(out, exp);
}

TEST(Golden, Relu) {
  Tensor in = tensor_from_npy(load_golden("op_relu_in"));
  NpyArray exp = load_golden("op_relu_out");
  Tensor out = Tensor::owning(DType::kF32, in.shape());
  run_kernel("Relu", {&in}, out);
  ExpectGolden(out, exp);
}

TEST(Golden, MatMul) {
  Tensor a = tensor_from_npy(load_golden("op_matmul_a"));
  Tensor b = tensor_from_npy(load_golden("op_matmul_b"));
  NpyArray exp = load_golden("op_matmul_out");
  Tensor out = Tensor::owning(DType::kF32, {a.shape()[0], b.shape()[1]});
  run_kernel("MatMul", {&a, &b}, out);
  ExpectGolden(out, exp);
}

TEST(Golden, Gelu) {
  Tensor in = tensor_from_npy(load_golden("op_gelu_in"));
  NpyArray exp = load_golden("op_gelu_out");
  Tensor out = Tensor::owning(DType::kF32, in.shape());
  run_kernel("Gelu", {&in}, out);
  ExpectGolden(out, exp);
}

TEST(Golden, Softmax) {
  Tensor in = tensor_from_npy(load_golden("op_softmax_in"));
  NpyArray exp = load_golden("op_softmax_out");
  Tensor out = Tensor::owning(DType::kF32, in.shape());
  run_kernel("Softmax", {&in}, out);
  ExpectGolden(out, exp, kDefaultRtol, 1e-5f);  // Softmax: looser atol per CLAUDE.md
}

TEST(Golden, LayerNorm) {
  Tensor x = tensor_from_npy(load_golden("op_layernorm_in"));
  Tensor gamma = tensor_from_npy(load_golden("op_layernorm_gamma"));
  Tensor beta = tensor_from_npy(load_golden("op_layernorm_beta"));
  Tensor out = Tensor::owning(DType::kF32, x.shape());
  run_kernel_attrs("LayerNorm", {&x, &gamma, &beta}, out,
                   {{"eps", static_cast<double>(1e-5)}});
  ExpectGolden(out, load_golden("op_layernorm_out"), kDefaultRtol, 1e-5f);
}

TEST(Golden, Scale) {
  Tensor x = tensor_from_npy(load_golden("op_scale_in"));
  Tensor out = Tensor::owning(DType::kF32, x.shape());
  run_kernel_attrs("Scale", {&x}, out, {{"scale", 0.25}});
  ExpectGolden(out, load_golden("op_scale_out"));
}

TEST(Golden, Reshape) {
  Tensor x = tensor_from_npy(load_golden("op_reshape_in"));
  Tensor out = Tensor::owning(DType::kF32, {3, 4});
  run_kernel("Reshape", {&x}, out);
  ExpectGolden(out, load_golden("op_reshape_out"));
}

TEST(Golden, Transpose) {
  Tensor x = tensor_from_npy(load_golden("op_transpose_in"));  // [2,3,4]
  Tensor out = Tensor::owning(DType::kF32, {3, 2, 4});
  run_kernel_attrs("Transpose", {&x}, out,
                   {{"perm", std::vector<int64_t>{1, 0, 2}}});
  ExpectGolden(out, load_golden("op_transpose_out"));
}

TEST(Golden, BatchedMatMul) {
  Tensor a = tensor_from_npy(load_golden("op_bmm_a"));  // [2,3,4]
  Tensor b = tensor_from_npy(load_golden("op_bmm_b"));  // [2,4,5]
  Tensor out = Tensor::owning(DType::kF32, {2, 3, 5});
  run_kernel("BatchedMatMul", {&a, &b}, out);
  ExpectGolden(out, load_golden("op_bmm_out"));
}

TEST(Golden, GeluTanh) {
  Tensor in = tensor_from_npy(load_golden("op_gelutanh_in"));
  Tensor out = Tensor::owning(DType::kF32, in.shape());
  run_kernel("GeluTanh", {&in}, out);
  ExpectGolden(out, load_golden("op_gelutanh_out"), kDefaultRtol, 1e-5f);
}

TEST(Golden, Gather) {
  Tensor table = tensor_from_npy(load_golden("op_gather_table"));  // [10,4]
  // Indices kept in sync with GATHER_IDS in gen_goldens.py.
  const int32_t ids_v[] = {3, 1, 4, 1, 5};
  Tensor ids = Tensor::owning(DType::kI32, {5});
  for (int i = 0; i < 5; ++i) ids.data<int32_t>()[i] = ids_v[i];
  Tensor out = Tensor::owning(DType::kF32, {5, table.shape()[1]});
  run_kernel("Gather", {&table, &ids}, out);
  ExpectGolden(out, load_golden("op_gather_out"));
}

TEST(Golden, CausalSoftmax) {
  Tensor in = tensor_from_npy(load_golden("op_causalsoftmax_in"));  // [2,4,4]
  Tensor out = Tensor::owning(DType::kF32, in.shape());
  run_kernel("CausalSoftmax", {&in}, out);
  ExpectGolden(out, load_golden("op_causalsoftmax_out"));
}

TEST(Golden, FlashAttention) {
  Tensor Q = tensor_from_npy(load_golden("op_flashattn_q"));  // [2,5,3]
  Tensor K = tensor_from_npy(load_golden("op_flashattn_k"));
  Tensor V = tensor_from_npy(load_golden("op_flashattn_v"));
  Tensor out = Tensor::owning(DType::kF32, Q.shape());
  const double scale = 1.0 / std::sqrt(3.0);  // d = 3
  run_kernel_attrs("FlashAttention", {&Q, &K, &V}, out, {{"scale", scale}});
  ExpectGolden(out, load_golden("op_flashattn_out"), kDefaultRtol, 1e-5f);
}

TEST(Golden, MatMulQ) {
  // A @ B, with B symmetric per-channel (per column) INT8-quantized -- the
  // weight-only INT8 path used for real GPT-2.
  Tensor A = tensor_from_npy(load_golden("op_matmul_a"));  // [2,3]
  NpyArray B = load_golden("op_matmul_b");                 // [3,5]
  const int K = static_cast<int>(B.shape[0]), N = static_cast<int>(B.shape[1]);
  Tensor W = Tensor::owning(DType::kI8, B.shape);
  Tensor s = Tensor::owning(DType::kF32, {static_cast<int64_t>(N)});
  for (int j = 0; j < N; ++j) {
    float amax = 0.0f;
    for (int k = 0; k < K; ++k) amax = std::max(amax, std::abs(B.data[k * N + j]));
    const float sc = amax / 127.0f;
    s.data<float>()[j] = sc;
    for (int k = 0; k < K; ++k)
      W.data<int8_t>()[k * N + j] = static_cast<int8_t>(std::lround(B.data[k * N + j] / sc));
  }
  Tensor out = Tensor::owning(DType::kF32, {2, 5});
  run_kernel("MatMulQ", {&A, &W, &s}, out);
  // vs full-precision A@B, within a quantization-sized tolerance.
  ExpectGolden(out, load_golden("op_matmul_out"), 0.05f, 0.05f);
}
