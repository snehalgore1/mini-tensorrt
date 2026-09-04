#pragma once

// Internal: per-group kernel registration. Kernels stay as free functions with
// internal linkage inside their .cpp files; each file exposes one registration
// entry point, and register_builtin_kernels() (registry.cpp) calls them all.
// This keeps registration explicit rather than relying on static initializers
// the linker can strip from a static library (DESIGN D5).

#include "mtrt/registry.h"

namespace mtrt {

void register_elementwise_kernels(KernelRegistry& registry);
void register_matmul_kernels(KernelRegistry& registry);
void register_activation_kernels(KernelRegistry& registry);
void register_fused_kernels(KernelRegistry& registry);
void register_transformer_kernels(KernelRegistry& registry);

}  // namespace mtrt
