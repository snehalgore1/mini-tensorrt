#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "mtrt/dtype.h"

namespace mtrt {

using TensorId = int32_t;
using NodeId = int32_t;

constexpr TensorId kInvalidTensor = -1;

// Node attribute value. Extend the variant as operators require it.
using Attribute = std::variant<int64_t, double, std::vector<int64_t>>;

// Role of a tensor in the graph. Drives which tensors the executor allocates
// versus binds from caller-provided data.
enum class TensorKind {
  kInput,
  kOutput,
  kWeight,
  kIntermediate,
};

// IR-level tensor metadata -- NOT storage. Storage is bound by the runtime.
struct TensorInfo {
  std::string name;
  DType dtype = DType::kF32;
  std::vector<int64_t> shape;
  TensorKind kind = TensorKind::kIntermediate;
};

// A graph node. Tensors are referenced by integer id so passes can rewrite the
// IR by editing vectors.
struct Node {
  std::string op_type;
  std::vector<TensorId> inputs;
  std::vector<TensorId> outputs;
  std::unordered_map<std::string, Attribute> attrs;
};

}  // namespace mtrt
