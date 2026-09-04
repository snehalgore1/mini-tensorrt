#pragma once

#include <unordered_map>
#include <vector>

#include "mtrt/graph.h"
#include "mtrt/memory/arena.h"
#include "mtrt/op_context.h"
#include "mtrt/registry.h"
#include "mtrt/tensor.h"

namespace mtrt {

// Intermediate-memory strategy. The naive per-tensor allocator is retained as
// the ablation baseline (DESIGN D11); the planner is the real path (D4).
enum class AllocatorKind {
  kNaive,
  kPlanned,
};

// Intermediate-memory instrumentation. Inputs/weights/outputs are excluded --
// they are handled identically under both allocators, so this isolates the
// planner's effect.
struct MemoryStats {
  int64_t peak_bytes = 0;        // naive: sum of intermediate bytes; planned: arena size
  int64_t allocation_count = 0;  // naive: one per intermediate; planned: 1 arena
  int64_t bytes_reused = 0;      // naive_sum - planned_peak (0 for naive)
};

// Naive executor. All allocation happens in the constructor (setup). run() never
// allocates and never throws (core invariant 4). This week's allocation strategy
// -- one owning tensor per id -- is the naive baseline retained behind a flag in
// Week 3 (DESIGN D11).
//
// The graph and registry must outlive the Executor: the execution plan holds
// pointers back into the graph's nodes.
class Executor {
 public:
  Executor(const Graph& graph, const KernelRegistry& registry,
           AllocatorKind allocator = AllocatorKind::kPlanned);

  // Bind graph inputs/weights (by tensor id), run, and return the graph outputs
  // in graph_outputs() order. Bindings share storage with the caller (no copy).
  std::vector<Tensor> run(const std::unordered_map<TensorId, Tensor>& bindings);

  MemoryStats memory_stats() const { return stats_; }

 private:
  struct PlanStep {
    KernelFn fn;                        // resolved once at setup
    std::vector<const Tensor*> inputs;  // pointers into pool_
    std::vector<Tensor*> outputs;       // pointers into pool_
    const Node* node;
  };

  const Graph& graph_;
  std::vector<Tensor> pool_;   // runtime tensor storage, indexed by TensorId
  std::vector<PlanStep> plan_;
  Arena arena_;                // backs planned intermediates (empty if naive)
  MemoryStats stats_;
};

}  // namespace mtrt
