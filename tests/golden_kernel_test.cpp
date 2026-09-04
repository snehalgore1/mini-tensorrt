#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "mtrt/node.h"
#include "mtrt/op_context.h"
#include "mtrt/registry.h"
#include "mtrt/tensor.h"
#include "tests/support/golden.h"

using namespace mtrt;
using namespace mtrt::testing;

namespace {

void run_kernel(const std::string& op, const std::vector<const Tensor*>& ins,
                Tensor& out) {
  KernelRegistry reg;
  register_builtin_kernels(reg);
  KernelFn fn = reg.lookup(op, DType::kF32);
  ASSERT_NE(fn, nullptr) << "no kernel for " << op;
  Node node{op, {}, {}, {}};
  std::vector<Tensor*> outs{&out};
  OpContext ctx{ins, outs, node};
  fn(ctx);
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
