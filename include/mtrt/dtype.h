#pragma once

#include <cstddef>

namespace mtrt {

// FP32 is the compute dtype (DESIGN D2). kI32 exists only for integer index
// inputs (token IDs into an embedding Gather) -- no arithmetic kernels operate
// on it. The registry is keyed on (op_type, dtype) so this stays additive.
enum class DType {
  kF32,
  kI32,
  kI8,  // symmetric per-tensor INT8 weights (with a companion f32 scale). #5.
};

inline std::size_t dtype_size(DType dt) {
  switch (dt) {
    case DType::kF32:
      return 4;
    case DType::kI32:
      return 4;
    case DType::kI8:
      return 1;
  }
  return 0;
}

inline const char* dtype_name(DType dt) {
  switch (dt) {
    case DType::kF32:
      return "f32";
    case DType::kI32:
      return "i32";
    case DType::kI8:
      return "i8";
  }
  return "?";
}

}  // namespace mtrt
