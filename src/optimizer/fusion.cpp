#include "mtrt/optimizer/fusion.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mtrt {

namespace {

// How many distinct nodes consume a given tensor.
std::vector<int> consumer_counts(const Graph& g) {
  std::vector<int> counts(static_cast<size_t>(g.num_tensors()), 0);
  for (const Node& n : g.nodes()) {
    for (TensorId in : n.inputs) counts[static_cast<size_t>(in)]++;
  }
  return counts;
}

}  // namespace

Graph fuse_matmul_bias_gelu(const Graph& graph) {
  const auto& nodes = graph.nodes();
  const std::vector<int> consumers = consumer_counts(graph);

  std::unordered_set<TensorId> graph_outputs(graph.graph_outputs().begin(),
                                             graph.graph_outputs().end());

  auto is_fusable_intermediate = [&](TensorId t) {
    return graph.tensor(t).kind == TensorKind::kIntermediate &&
           consumers[static_cast<size_t>(t)] == 1 && !graph_outputs.count(t);
  };

  // For each fusion, remember the anchor MatMul node and the two nodes to drop
  // (Add, activation), the bias input, the final output tensor, and which fused
  // op to emit (depends on the activation matched).
  struct Fusion {
    TensorId x, w, bias, out;
    const char* fused_op;
  };
  std::unordered_map<NodeId, Fusion> fusion_at_anchor;  // anchor node -> fusion
  std::unordered_set<NodeId> dropped_nodes;             // add + gelu nodes
  std::unordered_set<TensorId> dropped_tensors;         // t0 (mm out), t1 (add out)

  for (size_t ni = 0; ni < nodes.size(); ++ni) {
    const Node& mm = nodes[ni];
    if (mm.op_type != "MatMul" || mm.outputs.size() != 1) continue;
    const TensorId t0 = mm.outputs[0];
    if (!is_fusable_intermediate(t0)) continue;

    // Find the single consumer of t0: it must be an Add.
    NodeId add_ni = -1;
    for (size_t k = 0; k < nodes.size(); ++k) {
      for (TensorId in : nodes[k].inputs) {
        if (in == t0) {
          add_ni = static_cast<NodeId>(k);
          break;
        }
      }
      if (add_ni != -1) break;
    }
    if (add_ni < 0) continue;
    const Node& add = nodes[static_cast<size_t>(add_ni)];
    if (add.op_type != "Add" || add.inputs.size() != 2 || add.outputs.size() != 1)
      continue;
    const TensorId bias = add.inputs[0] == t0 ? add.inputs[1] : add.inputs[0];
    if (bias == t0) continue;  // degenerate
    const TensorId t1 = add.outputs[0];
    if (!is_fusable_intermediate(t1)) continue;

    // The single consumer of t1 must be a Gelu.
    NodeId gel_ni = -1;
    for (size_t k = 0; k < nodes.size(); ++k) {
      for (TensorId in : nodes[k].inputs) {
        if (in == t1) {
          gel_ni = static_cast<NodeId>(k);
          break;
        }
      }
      if (gel_ni != -1) break;
    }
    if (gel_ni < 0) continue;
    const Node& gel = nodes[static_cast<size_t>(gel_ni)];
    if (gel.inputs.size() != 1 || gel.outputs.size() != 1) continue;
    const char* fused_op = nullptr;
    if (gel.op_type == "Gelu") fused_op = kFusedMatMulBiasGelu;
    else if (gel.op_type == "GeluTanh") fused_op = kFusedMatMulBiasGeluTanh;
    else continue;

    fusion_at_anchor[static_cast<NodeId>(ni)] =
        Fusion{mm.inputs[0], mm.inputs[1], bias, gel.outputs[0], fused_op};
    dropped_nodes.insert(add_ni);
    dropped_nodes.insert(gel_ni);
    dropped_tensors.insert(t0);
    dropped_tensors.insert(t1);
  }

  // Rebuild a compact graph: keep every tensor except the dropped ones, remap
  // ids, then emit nodes (fused node in the anchor's place; drop add/gelu).
  Graph out;
  std::unordered_map<TensorId, TensorId> remap;
  for (TensorId id = 0; id < graph.num_tensors(); ++id) {
    if (dropped_tensors.count(id)) continue;
    remap[id] = out.add_tensor(graph.tensor(id));
  }
  auto rid = [&](TensorId old) { return remap.at(old); };

  for (size_t ni = 0; ni < nodes.size(); ++ni) {
    auto fit = fusion_at_anchor.find(static_cast<NodeId>(ni));
    if (fit != fusion_at_anchor.end()) {
      const Fusion& f = fit->second;
      Node fused;
      fused.op_type = f.fused_op;
      fused.inputs = {rid(f.x), rid(f.w), rid(f.bias)};
      fused.outputs = {rid(f.out)};
      out.add_node(std::move(fused));
      continue;
    }
    if (dropped_nodes.count(static_cast<NodeId>(ni))) continue;

    Node n = nodes[ni];  // copy op_type/attrs
    for (TensorId& in : n.inputs) in = rid(in);
    for (TensorId& o : n.outputs) o = rid(o);
    out.add_node(std::move(n));
  }

  for (TensorId in : graph.graph_inputs()) out.mark_input(rid(in));
  for (TensorId o : graph.graph_outputs()) out.mark_output(rid(o));
  return out;
}

}  // namespace mtrt
