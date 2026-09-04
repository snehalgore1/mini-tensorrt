#pragma once

#include "mtrt/graph.h"

// IR rewrite passes (fusion). A pass takes IR and returns rewritten IR; numerics
// must be preserved within tolerance (core invariant 6), which the on/off
// equivalence tests enforce.

namespace mtrt {

// Op type of the fused node produced below.
inline constexpr const char* kFusedMatMulBiasGelu = "FusedMatMulBiasGelu";

// Fuse every MatMul -> Add(bias) -> Gelu chain into a single node computing
// gelu(A @ B + bias). Only fuses when the MatMul and Add outputs are
// intermediates with a single consumer and are not graph outputs, so no other
// node depends on the eliminated tensors. Returns a new, compacted graph; the
// input graph is unchanged.
Graph fuse_matmul_bias_gelu(const Graph& graph);

}  // namespace mtrt
