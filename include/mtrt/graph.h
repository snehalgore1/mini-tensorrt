#pragma once

#include <stdexcept>
#include <vector>

#include "mtrt/node.h"

// core invariant 2: the IR is frontend-agnostic. src/graph/ must never include
// anything from src/frontend/.

namespace mtrt {

// Thrown at load/setup time for malformed graphs (cycles, dangling tensors).
// Setup and parsing may throw; the execution hot path must not.
struct GraphError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

class Graph {
 public:
  // Register a tensor and return its id.
  TensorId add_tensor(TensorInfo info);

  // Register a node and return its id.
  NodeId add_node(Node node);

  void mark_input(TensorId id) { graph_inputs_.push_back(id); }
  void mark_output(TensorId id) { graph_outputs_.push_back(id); }

  const std::vector<TensorInfo>& tensors() const { return tensors_; }
  const std::vector<Node>& nodes() const { return nodes_; }
  const TensorInfo& tensor(TensorId id) const;
  const Node& node(NodeId id) const;

  const std::vector<TensorId>& graph_inputs() const { return graph_inputs_; }
  const std::vector<TensorId>& graph_outputs() const { return graph_outputs_; }

  int64_t num_tensors() const { return static_cast<int64_t>(tensors_.size()); }
  int64_t num_nodes() const { return static_cast<int64_t>(nodes_.size()); }

  // Topological execution order via Kahn's algorithm. Throws GraphError on a
  // cycle or a tensor consumed but never produced (and not Input/Weight).
  std::vector<NodeId> topo_order() const;

 private:
  std::vector<TensorInfo> tensors_;
  std::vector<Node> nodes_;
  std::vector<TensorId> graph_inputs_;
  std::vector<TensorId> graph_outputs_;
};

}  // namespace mtrt
