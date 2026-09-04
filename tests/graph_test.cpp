#include "mtrt/graph.h"

#include <algorithm>

#include <gtest/gtest.h>

using namespace mtrt;

namespace {
TensorInfo make_info(std::string name, TensorKind kind) {
  TensorInfo info;
  info.name = std::move(name);
  info.dtype = DType::kF32;
  info.shape = {4};
  info.kind = kind;
  return info;
}

// Position of a node in an ordering.
int pos(const std::vector<NodeId>& order, NodeId id) {
  return static_cast<int>(
      std::find(order.begin(), order.end(), id) - order.begin());
}
}  // namespace

TEST(Graph, TopoOrderRespectsDependencies) {
  // in --(Relu)--> mid --(Add with in)--> out
  Graph g;
  TensorId in = g.add_tensor(make_info("in", TensorKind::kInput));
  TensorId mid = g.add_tensor(make_info("mid", TensorKind::kIntermediate));
  TensorId out = g.add_tensor(make_info("out", TensorKind::kOutput));

  NodeId relu = g.add_node(Node{"Relu", {in}, {mid}, {}});
  NodeId add = g.add_node(Node{"Add", {mid, in}, {out}, {}});
  g.mark_input(in);
  g.mark_output(out);

  std::vector<NodeId> order = g.topo_order();
  ASSERT_EQ(order.size(), 2u);
  EXPECT_LT(pos(order, relu), pos(order, add));
}

TEST(Graph, CycleThrows) {
  // a -> b (node0), b -> a (node1): a cycle.
  Graph g;
  TensorId a = g.add_tensor(make_info("a", TensorKind::kIntermediate));
  TensorId b = g.add_tensor(make_info("b", TensorKind::kIntermediate));
  g.add_node(Node{"Relu", {a}, {b}, {}});
  g.add_node(Node{"Relu", {b}, {a}, {}});
  EXPECT_THROW(g.topo_order(), GraphError);
}

TEST(Graph, ConsumedButNeverProducedThrows) {
  // Intermediate tensor read with no producer -> error.
  Graph g;
  TensorId x = g.add_tensor(make_info("x", TensorKind::kIntermediate));
  TensorId y = g.add_tensor(make_info("y", TensorKind::kOutput));
  g.add_node(Node{"Relu", {x}, {y}, {}});
  EXPECT_THROW(g.topo_order(), GraphError);
}

TEST(Graph, MultipleProducersThrows) {
  Graph g;
  TensorId in = g.add_tensor(make_info("in", TensorKind::kInput));
  TensorId out = g.add_tensor(make_info("out", TensorKind::kOutput));
  g.add_node(Node{"Relu", {in}, {out}, {}});
  g.add_node(Node{"Relu", {in}, {out}, {}});  // second producer of out
  EXPECT_THROW(g.topo_order(), GraphError);
}
