#include <gtest/gtest.h>

#include "mtrt/node.h"
#include "mtrt/op_context.h"
#include "mtrt/registry.h"
#include "mtrt/tensor.h"

using namespace mtrt;

namespace {
KernelRegistry make_registry() {
  KernelRegistry reg;
  register_builtin_kernels(reg);
  return reg;
}
}  // namespace

TEST(Kernel, AddF32) {
  KernelRegistry reg = make_registry();
  KernelFn add = reg.lookup("Add", DType::kF32);
  ASSERT_NE(add, nullptr);

  Tensor a = Tensor::owning(DType::kF32, {4});
  Tensor b = Tensor::owning(DType::kF32, {4});
  Tensor out = Tensor::owning(DType::kF32, {4});
  const float av[] = {1, 2, 3, 4};
  const float bv[] = {10, 20, 30, 40};
  for (int i = 0; i < 4; ++i) {
    a.data<float>()[i] = av[i];
    b.data<float>()[i] = bv[i];
  }

  Node node{"Add", {}, {}, {}};
  std::vector<const Tensor*> ins{&a, &b};
  std::vector<Tensor*> outs{&out};
  OpContext ctx{ins, outs, node};
  add(ctx);

  const float expected[] = {11, 22, 33, 44};
  for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(out.data<float>()[i], expected[i]);
}

TEST(Kernel, ReluF32) {
  KernelRegistry reg = make_registry();
  KernelFn relu = reg.lookup("Relu", DType::kF32);
  ASSERT_NE(relu, nullptr);

  Tensor in = Tensor::owning(DType::kF32, {5});
  Tensor out = Tensor::owning(DType::kF32, {5});
  const float iv[] = {-2, -0.5, 0, 0.5, 3};
  for (int i = 0; i < 5; ++i) in.data<float>()[i] = iv[i];

  Node node{"Relu", {}, {}, {}};
  std::vector<const Tensor*> ins{&in};
  std::vector<Tensor*> outs{&out};
  OpContext ctx{ins, outs, node};
  relu(ctx);

  const float expected[] = {0, 0, 0, 0.5, 3};
  for (int i = 0; i < 5; ++i) EXPECT_FLOAT_EQ(out.data<float>()[i], expected[i]);
}
