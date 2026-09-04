#pragma once

#include <unordered_map>

#include "mtrt/graph.h"
#include "mtrt/registry.h"
#include "mtrt/tensor.h"

// Constant folding (optimizer pass). Any node whose inputs are all constants
// (weights, or tensors produced by already-folded nodes) is evaluated once at
// load time and replaced by its result, which becomes a new constant. Unlike
// fusion, this needs the constant tensor DATA and the kernel registry, so it can
// actually run the op -- the same approach ONNX/TF constant folding uses.

namespace mtrt {

struct FoldResult {
  Graph graph;  // folded nodes removed; folded outputs are now kWeight
  // Original weights plus every newly folded constant, keyed by tensor id (ids
  // are preserved by this pass). Pass these as run() bindings.
  std::unordered_map<TensorId, Tensor> constants;
};

// `constants` maps each kWeight tensor id to its data. Returns the rewritten
// graph and the augmented constant set. Iterates to a fixpoint so a folded
// output can enable folding its consumers.
FoldResult fold_constants(const Graph& graph,
                          const std::unordered_map<TensorId, Tensor>& constants,
                          const KernelRegistry& registry);

}  // namespace mtrt
