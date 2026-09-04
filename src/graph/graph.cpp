#include "mtrt/graph.h"

#include <queue>
#include <string>

namespace mtrt {

TensorId Graph::add_tensor(TensorInfo info) {
  tensors_.push_back(std::move(info));
  return static_cast<TensorId>(tensors_.size() - 1);
}

NodeId Graph::add_node(Node node) {
  nodes_.push_back(std::move(node));
  return static_cast<NodeId>(nodes_.size() - 1);
}

const TensorInfo& Graph::tensor(TensorId id) const {
  if (id < 0 || id >= num_tensors()) {
    throw GraphError("tensor id out of range: " + std::to_string(id));
  }
  return tensors_[static_cast<size_t>(id)];
}

const Node& Graph::node(NodeId id) const {
  if (id < 0 || id >= num_nodes()) {
    throw GraphError("node id out of range: " + std::to_string(id));
  }
  return nodes_[static_cast<size_t>(id)];
}

std::vector<NodeId> Graph::topo_order() const {
  const size_t n = nodes_.size();

  // producer[t] = node that writes tensor t, or -1 if none (Input/Weight).
  std::vector<NodeId> producer(tensors_.size(), -1);
  for (size_t ni = 0; ni < n; ++ni) {
    for (TensorId out : nodes_[ni].outputs) {
      if (out < 0 || out >= num_tensors()) {
        throw GraphError("node writes out-of-range tensor id");
      }
      if (producer[static_cast<size_t>(out)] != -1) {
        throw GraphError("tensor has more than one producer (not SSA-like)");
      }
      producer[static_cast<size_t>(out)] = static_cast<NodeId>(ni);
    }
  }

  // in-degree = number of input tensors produced by another node.
  std::vector<int> indeg(n, 0);
  std::vector<std::vector<NodeId>> consumers(n);  // producer node -> dependents
  for (size_t ni = 0; ni < n; ++ni) {
    for (TensorId in : nodes_[ni].inputs) {
      if (in < 0 || in >= num_tensors()) {
        throw GraphError("node reads out-of-range tensor id");
      }
      NodeId prod = producer[static_cast<size_t>(in)];
      if (prod == -1) {
        // Consumed but not produced: legal only for Input/Weight tensors.
        TensorKind kind = tensors_[static_cast<size_t>(in)].kind;
        if (kind != TensorKind::kInput && kind != TensorKind::kWeight) {
          throw GraphError("tensor consumed but never produced: " +
                           tensors_[static_cast<size_t>(in)].name);
        }
      } else {
        consumers[static_cast<size_t>(prod)].push_back(static_cast<NodeId>(ni));
        indeg[ni]++;
      }
    }
  }

  std::queue<NodeId> ready;
  for (size_t ni = 0; ni < n; ++ni) {
    if (indeg[ni] == 0) ready.push(static_cast<NodeId>(ni));
  }

  std::vector<NodeId> order;
  order.reserve(n);
  while (!ready.empty()) {
    NodeId ni = ready.front();
    ready.pop();
    order.push_back(ni);
    for (NodeId dep : consumers[static_cast<size_t>(ni)]) {
      if (--indeg[static_cast<size_t>(dep)] == 0) ready.push(dep);
    }
  }

  if (order.size() != n) {
    throw GraphError("graph contains a cycle");
  }
  return order;
}

}  // namespace mtrt
