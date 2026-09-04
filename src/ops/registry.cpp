#include "mtrt/registry.h"

namespace mtrt {

void KernelRegistry::register_kernel(std::string op_type, DType dtype,
                                     KernelFn fn) {
  table_[Key{std::move(op_type), dtype}] = fn;
}

KernelFn KernelRegistry::lookup(const std::string& op_type, DType dtype) const {
  auto it = table_.find(Key{op_type, dtype});
  return it == table_.end() ? nullptr : it->second;
}

}  // namespace mtrt
