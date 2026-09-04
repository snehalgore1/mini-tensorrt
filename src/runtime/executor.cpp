#include "mtrt/executor.h"

#include <stdexcept>
#include <string>

namespace mtrt {

namespace {
// Minimal Week 1 shape check. Full shape inference arrives with the frontend in
// Week 2; here shapes are already present in TensorInfo, so we only validate the
// two operators we support.
void validate_node_shapes(const Graph& g, const Node& n) {
  auto shape_of = [&](TensorId id) { return g.tensor(id).shape; };
  if (n.op_type == "Add") {
    if (n.inputs.size() != 2 || n.outputs.size() != 1) {
      throw GraphError("Add expects 2 inputs and 1 output");
    }
    if (shape_of(n.inputs[0]) != shape_of(n.inputs[1]) ||
        shape_of(n.inputs[0]) != shape_of(n.outputs[0])) {
      throw GraphError("Add requires identical input/output shapes");
    }
  } else if (n.op_type == "Relu") {
    if (n.inputs.size() != 1 || n.outputs.size() != 1) {
      throw GraphError("Relu expects 1 input and 1 output");
    }
    if (shape_of(n.inputs[0]) != shape_of(n.outputs[0])) {
      throw GraphError("Relu must preserve shape");
    }
  }
}
}  // namespace

Executor::Executor(const Graph& graph, const KernelRegistry& registry)
    : graph_(graph) {
  const std::vector<NodeId> order = graph_.topo_order();

  // Allocate one owning tensor per Intermediate/Output id (naive baseline).
  // Input/Weight slots stay empty until bound in run(). Sized once and never
  // resized, so plan pointers into pool_ stay valid.
  pool_.resize(static_cast<size_t>(graph_.num_tensors()));
  for (TensorId id = 0; id < graph_.num_tensors(); ++id) {
    const TensorInfo& info = graph_.tensor(id);
    if (info.kind == TensorKind::kIntermediate ||
        info.kind == TensorKind::kOutput) {
      pool_[static_cast<size_t>(id)] = Tensor::owning(info.dtype, info.shape);
    }
  }

  // Resolve each node to a kernel once and record it in the plan.
  plan_.reserve(order.size());
  for (NodeId nid : order) {
    const Node& n = graph_.node(nid);
    validate_node_shapes(graph_, n);

    if (n.outputs.empty()) {
      throw GraphError("node has no outputs: " + n.op_type);
    }
    const DType dtype = graph_.tensor(n.outputs[0]).dtype;
    KernelFn fn = registry.lookup(n.op_type, dtype);
    if (fn == nullptr) {
      throw GraphError("no kernel registered for op '" + n.op_type + "' dtype " +
                       dtype_name(dtype));
    }

    PlanStep step;
    step.fn = fn;
    step.node = &n;
    for (TensorId in : n.inputs) {
      step.inputs.push_back(&pool_[static_cast<size_t>(in)]);
    }
    for (TensorId out : n.outputs) {
      step.outputs.push_back(&pool_[static_cast<size_t>(out)]);
    }
    plan_.push_back(std::move(step));
  }
}

std::vector<Tensor> Executor::run(
    const std::unordered_map<TensorId, Tensor>& bindings) {
  // Bind inputs/weights into the pool (shallow -- shares caller storage).
  for (const auto& [id, tensor] : bindings) {
    if (id < 0 || id >= graph_.num_tensors()) {
      throw std::runtime_error("binding for unknown tensor id");
    }
    pool_[static_cast<size_t>(id)] = tensor;
  }

  // Hot loop: direct calls through cached function pointers. No allocation, no
  // map lookups, no virtual dispatch.
  for (PlanStep& s : plan_) {
    OpContext ctx{s.inputs, s.outputs, *s.node};
    s.fn(ctx);
  }

  std::vector<Tensor> outputs;
  outputs.reserve(graph_.graph_outputs().size());
  for (TensorId id : graph_.graph_outputs()) {
    outputs.push_back(pool_[static_cast<size_t>(id)]);
  }
  return outputs;
}

}  // namespace mtrt
