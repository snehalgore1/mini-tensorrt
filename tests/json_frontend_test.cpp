#include "mtrt/frontend/json_loader.h"

#include <string>

#include <gtest/gtest.h>

#include "mtrt/graph.h"

using namespace mtrt;
using namespace mtrt::frontend;

namespace {
std::string mlp_json() { return std::string(MTRT_MODELS_DIR) + "/mlp.json"; }
}  // namespace

TEST(JsonFrontend, LoadsMlpTopology) {
  LoadedModel m = load_json_model(mlp_json());
  EXPECT_EQ(m.graph.num_tensors(), 11);  // x, W0,b0, t0,t1,t2, W1,b1, t3,t4, out
  EXPECT_EQ(m.graph.num_nodes(), 6);
  ASSERT_EQ(m.graph.graph_inputs().size(), 1u);
  ASSERT_EQ(m.graph.graph_outputs().size(), 1u);
  EXPECT_EQ(m.weights.size(), 4u);  // W0, b0, W1, b1
}

TEST(JsonFrontend, ProducesExecutableOrder) {
  LoadedModel m = load_json_model(mlp_json());
  std::vector<NodeId> order;
  ASSERT_NO_THROW(order = m.graph.topo_order());
  ASSERT_EQ(order.size(), 6u);
  EXPECT_EQ(m.graph.node(order.front()).op_type, "MatMul");
  EXPECT_EQ(m.graph.node(order.back()).op_type, "Softmax");
}

TEST(JsonFrontend, WeightTensorsHaveData) {
  LoadedModel m = load_json_model(mlp_json());
  for (const auto& [id, w] : m.weights) {
    EXPECT_TRUE(w.defined());
    EXPECT_EQ(w.dtype(), DType::kF32);
    EXPECT_GT(w.numel(), 0);
  }
}
