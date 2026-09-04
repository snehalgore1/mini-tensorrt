#include "mtrt/executor.h"

#include <gtest/gtest.h>

#include "mtrt/graph.h"
#include "mtrt/registry.h"
#include "mtrt/tensor.h"

using namespace mtrt;

namespace {
TensorInfo info(std::string name, TensorKind kind) {
  TensorInfo t;
  t.name = std::move(name);
  t.dtype = DType::kF32;
  t.shape = {4};
  t.kind = kind;
  return t;
}
}  // namespace

// Week 1 DoD: a hand-built Add -> Relu graph executes and returns correct values.
TEST(Executor, AddThenReluEndToEnd) {
  Graph g;
  TensorId a = g.add_tensor(info("a", TensorKind::kInput));
  TensorId b = g.add_tensor(info("b", TensorKind::kInput));
  TensorId sum = g.add_tensor(info("sum", TensorKind::kIntermediate));
  TensorId out = g.add_tensor(info("out", TensorKind::kOutput));

  g.add_node(Node{"Add", {a, b}, {sum}, {}});
  g.add_node(Node{"Relu", {sum}, {out}, {}});
  g.mark_input(a);
  g.mark_input(b);
  g.mark_output(out);

  KernelRegistry reg;
  register_builtin_kernels(reg);
  Executor exec(g, reg);

  Tensor ta = Tensor::owning(DType::kF32, {4});
  Tensor tb = Tensor::owning(DType::kF32, {4});
  // a + b = {-3, 1, 0, 6}; relu -> {0, 1, 0, 6}
  const float av[] = {-1, 2, -5, 4};
  const float bv[] = {-2, -1, 5, 2};
  for (int i = 0; i < 4; ++i) {
    ta.data<float>()[i] = av[i];
    tb.data<float>()[i] = bv[i];
  }

  std::vector<Tensor> outputs = exec.run({{a, ta}, {b, tb}});
  ASSERT_EQ(outputs.size(), 1u);
  const float expected[] = {0, 1, 0, 6};
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(outputs[0].data<float>()[i], expected[i]);
  }
}

TEST(Executor, UnregisteredOpThrowsAtSetup) {
  Graph g;
  TensorId in = g.add_tensor(info("in", TensorKind::kInput));
  TensorId out = g.add_tensor(info("out", TensorKind::kOutput));
  g.add_node(Node{"Gelu", {in}, {out}, {}});
  g.mark_input(in);
  g.mark_output(out);

  KernelRegistry reg;
  register_builtin_kernels(reg);
  EXPECT_THROW(Executor(g, reg), GraphError);
}
