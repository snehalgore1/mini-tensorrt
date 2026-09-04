#pragma once

#include <unordered_map>
#include <vector>

#include "mtrt/graph.h"
#include "mtrt/op_context.h"
#include "mtrt/registry.h"
#include "mtrt/tensor.h"

namespace mtrt {

// Naive executor. All allocation happens in the constructor (setup). run() never
// allocates and never throws (core invariant 4). This week's allocation strategy
// -- one owning tensor per id -- is the naive baseline retained behind a flag in
// Week 3 (DESIGN D11).
//
// The graph and registry must outlive the Executor: the execution plan holds
// pointers back into the graph's nodes.
class Executor {
 public:
  Executor(const Graph& graph, const KernelRegistry& registry);

  // Bind graph inputs/weights (by tensor id), run, and return the graph outputs
  // in graph_outputs() order. Bindings share storage with the caller (no copy).
  std::vector<Tensor> run(const std::unordered_map<TensorId, Tensor>& bindings);

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
};

}  // namespace mtrt
