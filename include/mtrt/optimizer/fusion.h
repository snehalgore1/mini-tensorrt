#pragma once

#include "mtrt/graph.h"

// IR rewrite passes (fusion). A pass takes IR and returns rewritten IR; numerics
// must be preserved within tolerance (core invariant 6), which the on/off
// equivalence tests enforce.

namespace mtrt {

// Op types of the fused nodes produced below (erf GELU and GPT-2's tanh-approx
// "gelu_new" respectively).
inline constexpr const char* kFusedMatMulBiasGelu = "FusedMatMulBiasGelu";
inline constexpr const char* kFusedMatMulBiasGeluTanh = "FusedMatMulBiasGeluTanh";

// Fuse every MatMul -> Add(bias) -> Gelu/GeluTanh chain into a single node
// computing act(A @ B + bias), where act matches the activation found (Gelu ->
// kFusedMatMulBiasGelu, GeluTanh -> kFusedMatMulBiasGeluTanh). Only fuses when
// the MatMul and Add outputs are intermediates with a single consumer and are
// not graph outputs, so no other node depends on the eliminated tensors. Returns
// a new, compacted graph; the input graph is unchanged.
Graph fuse_matmul_bias_gelu(const Graph& graph);

}  // namespace mtrt
