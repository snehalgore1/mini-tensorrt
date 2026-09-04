#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <unordered_map>

#include "mtrt/dtype.h"
#include "mtrt/op_context.h"

namespace mtrt {

// Maps (op_type, dtype) -> kernel function pointer. Lookups happen once at
// setup, never in the hot loop. Explicit registration (register_builtin_kernels)
// is used instead of self-registering static initializers, which the linker can
// silently strip from a static library (DESIGN D5).
class KernelRegistry {
 public:
  void register_kernel(std::string op_type, DType dtype, KernelFn fn);

  // Returns nullptr if no kernel is registered for (op_type, dtype).
  KernelFn lookup(const std::string& op_type, DType dtype) const;

 private:
  struct Key {
    std::string op_type;
    DType dtype;
    bool operator==(const Key& o) const {
      return dtype == o.dtype && op_type == o.op_type;
    }
  };
  struct KeyHash {
    std::size_t operator()(const Key& k) const {
      return std::hash<std::string>{}(k.op_type) ^
             (std::hash<int>{}(static_cast<int>(k.dtype)) << 1);
    }
  };

  std::unordered_map<Key, KernelFn, KeyHash> table_;
};

// Registers the built-in kernels (Add, Relu) into the given registry.
void register_builtin_kernels(KernelRegistry& registry);

}  // namespace mtrt
