#pragma once

#include <string>
#include <unordered_map>

#include "mtrt/graph.h"
#include "mtrt/tensor.h"

// JSON frontend (DESIGN D3). Produces frontend-agnostic IR: nothing here leaks
// the JSON library into the public surface -- callers see only Graph and Tensor.

namespace mtrt::frontend {

struct LoadedModel {
  Graph graph;
  // Owning tensors holding weight data read from the companion .bin, keyed by
  // the weight's TensorId in `graph`. Pass these as run() bindings alongside the
  // graph inputs.
  std::unordered_map<TensorId, Tensor> weights;
};

// Load a topology JSON file and its companion weights .bin (the .bin path is
// taken from the JSON's "weights_file" field, resolved relative to the JSON
// file's directory). Throws GraphError on malformed input.
LoadedModel load_json_model(const std::string& json_path);

}  // namespace mtrt::frontend
