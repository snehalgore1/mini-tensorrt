#pragma once

#include <cstddef>

namespace mtrt {

// FP32 only until Week 7 (DESIGN D2). The enum exists so the op registry can be
// keyed on (op_type, dtype) from day one, making INT8 additive rather than a
// rewrite.
enum class DType {
  kF32,
};

inline std::size_t dtype_size(DType dt) {
  switch (dt) {
    case DType::kF32:
      return 4;
  }
  return 0;
}

inline const char* dtype_name(DType dt) {
  switch (dt) {
    case DType::kF32:
      return "f32";
  }
  return "?";
}

}  // namespace mtrt
