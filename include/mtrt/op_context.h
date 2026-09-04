#pragma once

#include <vector>

#include "mtrt/node.h"
#include "mtrt/tensor.h"

namespace mtrt {

// Everything a kernel needs to run one node. Holds non-owning pointers into the
// executor's tensor pool plus the IR node (for attributes). Kernels are plain
// free functions taking this by const reference -- no virtual dispatch.
struct OpContext {
  const std::vector<const Tensor*>& inputs;
  const std::vector<Tensor*>& outputs;
  const Node& node;
};

// A kernel is a plain function pointer, NOT a virtual method (DESIGN D5).
using KernelFn = void (*)(const OpContext&);

}  // namespace mtrt
