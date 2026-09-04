#include "mtrt/optimizer/const_fold.h"

#include <unordered_map>

#include <gtest/gtest.h>

#include "mtrt/executor.h"
#include "mtrt/frontend/json_loader.h"
#include "mtrt/graph.h"
#include "mtrt/registry.h"
#include "tests/support/golden.h"

using namespace mtrt;
using namespace mtrt::frontend;
using namespace mtrt::testing;

namespace {
TensorInfo info(std::string name, TensorKind kind) {
  TensorInfo t;
  t.name = std::move(name);
  t.dtype = DType::kF32;
  t.shape = {4};
  t.kind = kind;
  return t;
}

Tensor vec4(float a, float b, float c, float d) {
  Tensor t = Tensor::owning(DType::kF32, {4});
  t.data<float>()[0] = a;
  t.data<float>()[1] = b;
  t.data<float>()[2] = c;
  t.data<float>()[3] = d;
  return t;
}
}  // namespace

// Graph with a constant subexpression: Add(W0, W1) -> c (both weights, foldable),
// then Add(c, x) -> out (x is runtime, not foldable).
TEST(ConstFold, FoldsConstantSubexpression) {
  Graph g;
  TensorId w0 = g.add_tensor(info("w0", TensorKind::kWeight));
  TensorId w1 = g.add_tensor(info("w1", TensorKind::kWeight));
  TensorId x = g.add_tensor(info("x", TensorKind::kInput));
  TensorId c = g.add_tensor(info("c", TensorKind::kIntermediate));
  TensorId out = g.add_tensor(info("out", TensorKind::kOutput));
  g.add_node(Node{"Add", {w0, w1}, {c}, {}});
  g.add_node(Node{"Add", {c, x}, {out}, {}});
  g.mark_input(x);
  g.mark_output(out);

  KernelRegistry reg;
  register_builtin_kernels(reg);
  std::unordered_map<TensorId, Tensor> constants;
  constants.emplace(w0, vec4(1, 2, 3, 4));
  constants.emplace(w1, vec4(10, 20, 30, 40));

  FoldResult r = fold_constants(g, constants, reg);

  // The first Add is folded away; only Add(c, x) remains.
  EXPECT_EQ(r.graph.num_nodes(), 1);
  // c is now a constant with the precomputed value w0 + w1.
  ASSERT_TRUE(r.constants.count(c));
  const float* cv = r.constants.at(c).data<float>();
  EXPECT_FLOAT_EQ(cv[0], 11);
  EXPECT_FLOAT_EQ(cv[3], 44);
  EXPECT_EQ(r.graph.tensor(c).kind, TensorKind::kWeight);
}

// Folded graph must produce the same output as the original.
TEST(ConstFold, PreservesNumerics) {
  Graph g;
  TensorId w0 = g.add_tensor(info("w0", TensorKind::kWeight));
  TensorId w1 = g.add_tensor(info("w1", TensorKind::kWeight));
  TensorId x = g.add_tensor(info("x", TensorKind::kInput));
  TensorId c = g.add_tensor(info("c", TensorKind::kIntermediate));
  TensorId out = g.add_tensor(info("out", TensorKind::kOutput));
  g.add_node(Node{"Add", {w0, w1}, {c}, {}});
  g.add_node(Node{"Add", {c, x}, {out}, {}});
  g.mark_input(x);
  g.mark_output(out);

  KernelRegistry reg;
  register_builtin_kernels(reg);
  std::unordered_map<TensorId, Tensor> constants;
  constants.emplace(w0, vec4(1, 2, 3, 4));
  constants.emplace(w1, vec4(10, 20, 30, 40));
  Tensor xv = vec4(100, 200, 300, 400);

  Executor before(g, reg);
  auto binds_before = constants;
  binds_before.emplace(x, xv);
  std::vector<Tensor> out_before = before.run(binds_before);

  FoldResult r = fold_constants(g, constants, reg);
  Executor after(r.graph, reg);
  auto binds_after = r.constants;
  binds_after.emplace(x, xv);
  std::vector<Tensor> out_after = after.run(binds_after);

  ASSERT_EQ(out_before[0].numel(), out_after[0].numel());
  for (int64_t i = 0; i < out_before[0].numel(); ++i)
    EXPECT_FLOAT_EQ(out_before[0].data<float>()[i], out_after[0].data<float>()[i]);
}

// The MLP has no constant subexpression (every op touches x or an intermediate),
// so folding is a correct no-op.
TEST(ConstFold, NoOpOnMlp) {
  LoadedModel m = load_json_model(std::string(MTRT_MODELS_DIR) + "/mlp.json");
  KernelRegistry reg;
  register_builtin_kernels(reg);
  FoldResult r = fold_constants(m.graph, m.weights, reg);
  EXPECT_EQ(r.graph.num_nodes(), m.graph.num_nodes());
}
