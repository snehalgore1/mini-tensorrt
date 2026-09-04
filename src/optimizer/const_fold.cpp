#include "mtrt/optimizer/const_fold.h"

#include <unordered_set>
#include <vector>

#include "mtrt/op_context.h"

namespace mtrt {

FoldResult fold_constants(const Graph& graph,
                          const std::unordered_map<TensorId, Tensor>& constants,
                          const KernelRegistry& registry) {
  FoldResult result;
  result.constants = constants;

  // is_const[id]: tensor is a compile-time constant (weight or folded).
  std::vector<char> is_const(static_cast<size_t>(graph.num_tensors()), 0);
  for (TensorId id = 0; id < graph.num_tensors(); ++id) {
    if (graph.tensor(id).kind == TensorKind::kWeight ||
        result.constants.count(id)) {
      is_const[static_cast<size_t>(id)] = 1;
    }
  }

  // Working copy of tensor kinds; folded outputs become kWeight.
  std::vector<TensorInfo> tensors(graph.tensors().begin(), graph.tensors().end());
  std::unordered_set<NodeId> folded;

  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t ni = 0; ni < graph.nodes().size(); ++ni) {
      if (folded.count(static_cast<NodeId>(ni))) continue;
      const Node& n = graph.nodes()[ni];

      bool all_const = !n.inputs.empty();
      for (TensorId in : n.inputs) {
        if (!is_const[static_cast<size_t>(in)]) {
          all_const = false;
          break;
        }
      }
      if (!all_const || n.outputs.empty()) continue;

      const DType dtype = graph.tensor(n.outputs[0]).dtype;
      KernelFn fn = registry.lookup(n.op_type, dtype);
      if (fn == nullptr) continue;  // no kernel -> can't fold, leave as runtime

      // Evaluate the node now.
      std::vector<const Tensor*> ins;
      for (TensorId in : n.inputs) ins.push_back(&result.constants.at(in));
      std::vector<Tensor> out_storage;
      out_storage.reserve(n.outputs.size());
      for (TensorId out : n.outputs) {
        const TensorInfo& info = graph.tensor(out);
        out_storage.push_back(Tensor::owning(info.dtype, info.shape));
      }
      std::vector<Tensor*> outs;
      for (Tensor& t : out_storage) outs.push_back(&t);

      OpContext ctx{ins, outs, n};
      fn(ctx);

      for (size_t k = 0; k < n.outputs.size(); ++k) {
        const TensorId out = n.outputs[k];
        result.constants.emplace(out, std::move(out_storage[k]));
        is_const[static_cast<size_t>(out)] = 1;
        tensors[static_cast<size_t>(out)].kind = TensorKind::kWeight;
      }
      folded.insert(static_cast<NodeId>(ni));
      changed = true;
    }
  }

  // Rebuild the graph with the same tensor ids: updated kinds, folded nodes
  // dropped. Folded outputs now have no producer but are kWeight, which
  // topo_order() permits.
  Graph& out = result.graph;
  for (const TensorInfo& info : tensors) out.add_tensor(info);
  for (size_t ni = 0; ni < graph.nodes().size(); ++ni) {
    if (folded.count(static_cast<NodeId>(ni))) continue;
    out.add_node(graph.nodes()[ni]);
  }
  for (TensorId in : graph.graph_inputs()) out.mark_input(in);
  for (TensorId o : graph.graph_outputs()) out.mark_output(o);
  return result;
}

}  // namespace mtrt
