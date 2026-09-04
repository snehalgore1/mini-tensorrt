#include <string>
#include <unordered_map>

#include <gtest/gtest.h>

#include "mtrt/executor.h"
#include "mtrt/frontend/json_loader.h"
#include "mtrt/optimizer/fusion.h"
#include "mtrt/registry.h"
#include "mtrt/tensor.h"
#include "tests/support/golden.h"

using namespace mtrt;
using namespace mtrt::frontend;
using namespace mtrt::testing;

namespace {
std::string tf_json() { return std::string(MTRT_MODELS_DIR) + "/transformer.json"; }

std::unordered_map<std::string, Tensor> inputs_by_name(const LoadedModel& m) {
  std::unordered_map<std::string, Tensor> by_name;
  for (const auto& [id, w] : m.weights) by_name.emplace(m.graph.tensor(id).name, w);
  by_name.emplace(m.graph.tensor(m.graph.graph_inputs()[0]).name,
                  tensor_from_npy(load_golden("tf_input")));
  return by_name;
}

std::unordered_map<TensorId, Tensor> bindings_for(
    const Graph& g, const std::unordered_map<std::string, Tensor>& by_name) {
  std::unordered_map<TensorId, Tensor> b;
  for (TensorId id = 0; id < g.num_tensors(); ++id) {
    auto it = by_name.find(g.tensor(id).name);
    if (it != by_name.end()) b.emplace(id, it->second);
  }
  return b;
}

int count_op(const Graph& g, const std::string& op) {
  int c = 0;
  for (const Node& n : g.nodes())
    if (n.op_type == op) c++;
  return c;
}
}  // namespace

// The whole multi-head transformer block matches PyTorch within tolerance.
TEST(Transformer, BlockMatchesPyTorch) {
  LoadedModel m = load_json_model(tf_json());
  KernelRegistry reg;
  register_builtin_kernels(reg);
  Executor exec(m.graph, reg);

  auto by_name = inputs_by_name(m);
  std::vector<Tensor> out = exec.run(bindings_for(m.graph, by_name));
  ASSERT_EQ(out.size(), 1u);
  ExpectGolden(out[0], load_golden("tf_output"), kDefaultRtol, 1e-5f);
}

// Fusion still fires on the block's MLP sub-block, and preserves numerics.
TEST(Transformer, FusionFiresAndPreservesNumerics) {
  LoadedModel m = load_json_model(tf_json());
  Graph fused = fuse_matmul_bias_gelu(m.graph);
  EXPECT_EQ(count_op(fused, kFusedMatMulBiasGelu), 1);  // the h2->W1->b1->Gelu chain

  KernelRegistry reg;
  register_builtin_kernels(reg);
  auto by_name = inputs_by_name(m);

  Executor unfused(m.graph, reg);
  Executor fused_exec(fused, reg);
  std::vector<Tensor> a = unfused.run(bindings_for(m.graph, by_name));
  std::vector<Tensor> b = fused_exec.run(bindings_for(fused, by_name));

  ASSERT_EQ(a[0].numel(), b[0].numel());
  for (int64_t i = 0; i < a[0].numel(); ++i)
    EXPECT_NEAR(a[0].data<float>()[i], b[0].data<float>()[i], 1e-6f);
  ExpectGolden(b[0], load_golden("tf_output"), kDefaultRtol, 1e-5f);
}
