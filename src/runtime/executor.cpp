#include "mtrt/executor.h"

#include <chrono>
#include <stdexcept>
#include <string>

#include "mtrt/memory/memory_planner.h"

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
    if (shape_of(n.inputs[0]) != shape_of(n.outputs[0])) {
      throw GraphError("Add output shape must match the first input");
    }
    // Second input is either identical-shape or a last-dim bias broadcast.
    const auto& a = shape_of(n.inputs[0]);
    const auto& b = shape_of(n.inputs[1]);
    int64_t b_numel = 1;
    for (int64_t d : b) b_numel *= d;
    if (b != a && !(!a.empty() && b_numel == a.back())) {
      throw GraphError("Add second input must match shape or broadcast last dim");
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

namespace {
int64_t tensor_bytes(const TensorInfo& info) {
  int64_t n = 1;
  for (int64_t d : info.shape) n *= d;
  return n * static_cast<int64_t>(dtype_size(info.dtype));
}
}  // namespace

Executor::Executor(const Graph& graph, const KernelRegistry& registry,
                   AllocatorKind allocator)
    : graph_(graph) {
  const std::vector<NodeId> order = graph_.topo_order();

  // pool_ is sized once and never resized after this, so the raw Tensor* stored
  // in the plan stay valid (run() only assigns into existing slots).
  pool_.resize(static_cast<size_t>(graph_.num_tensors()));

  // Outputs get their own owning buffers in both modes (they must outlive the
  // arena). Sum of intermediate bytes is the naive peak / the reuse baseline.
  int64_t intermediate_sum = 0;
  int64_t intermediate_count = 0;
  for (TensorId id = 0; id < graph_.num_tensors(); ++id) {
    const TensorInfo& info = graph_.tensor(id);
    if (info.kind == TensorKind::kOutput) {
      pool_[static_cast<size_t>(id)] = Tensor::owning(info.dtype, info.shape);
    } else if (info.kind == TensorKind::kIntermediate) {
      intermediate_sum += tensor_bytes(info);
      intermediate_count++;
    }
  }

  if (allocator == AllocatorKind::kNaive) {
    // One owning buffer per intermediate -- no reuse (DESIGN D11 baseline).
    for (TensorId id = 0; id < graph_.num_tensors(); ++id) {
      const TensorInfo& info = graph_.tensor(id);
      if (info.kind == TensorKind::kIntermediate) {
        pool_[static_cast<size_t>(id)] = Tensor::owning(info.dtype, info.shape);
      }
    }
    stats_ = MemoryStats{intermediate_sum, intermediate_count, 0};
  } else {
    // Planned: one arena, intermediates are views at their assigned offsets.
    const MemoryPlan mplan = plan_memory(graph_);
    arena_ = Arena(mplan.total_bytes);
    for (const auto& [id, offset] : mplan.offset) {
      const TensorInfo& info = graph_.tensor(id);
      pool_[static_cast<size_t>(id)] =
          arena_.view_at(offset, info.dtype, info.shape);
    }
    stats_ = MemoryStats{mplan.total_bytes, mplan.total_bytes > 0 ? 1 : 0,
                         intermediate_sum - mplan.total_bytes};
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
  // map lookups, no virtual dispatch. The profiler branch is chosen once, not
  // per node, so the untimed path stays clean.
  if (profiler_ == nullptr) {
    for (PlanStep& s : plan_) {
      OpContext ctx{s.inputs, s.outputs, *s.node};
      s.fn(ctx);
    }
  } else {
    const auto run_start = std::chrono::steady_clock::now();
    for (PlanStep& s : plan_) {
      OpContext ctx{s.inputs, s.outputs, *s.node};
      const auto t0 = std::chrono::steady_clock::now();
      s.fn(ctx);
      const auto t1 = std::chrono::steady_clock::now();
      const auto ts =
          std::chrono::duration_cast<std::chrono::nanoseconds>(t0 - run_start)
              .count();
      const auto dur =
          std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
      profiler_->record(s.node->op_type, ts, dur);
    }
  }

  std::vector<Tensor> outputs;
  outputs.reserve(graph_.graph_outputs().size());
  for (TensorId id : graph_.graph_outputs()) {
    outputs.push_back(pool_[static_cast<size_t>(id)]);
  }
  return outputs;
}

}  // namespace mtrt
