#pragma once

#include <string>

#include "mtrt/frontend/json_loader.h"  // LoadedModel

// ONNX frontend (DESIGN D3, Week 6). Parses a real `.onnx` file into the same
// frontend-agnostic IR the JSON loader produces -- proving the IR is truly
// frontend-agnostic by adding a second real frontend rather than asserting it.
// Nothing ONNX (protobuf, onnx.pb.h) leaks into this public header; callers see
// only Graph and Tensor via LoadedModel.
//
// Strict subset (Week 6 DoD): every operator must map to a registered kernel and
// every tensor must have a static shape, otherwise load throws GraphError with a
// clear message -- it never silently misbehaves. Shapes come from the file's
// value_info (run `onnx.shape_inference.infer_shapes` at export, as
// python/export_onnx.py does); dynamic/symbolic dims are rejected (DESIGN D1).
//
// Only compiled when -DMTRT_ONNX=ON; guard call sites with MTRT_HAVE_ONNX.

namespace mtrt::frontend {

LoadedModel load_onnx_model(const std::string& onnx_path);

}  // namespace mtrt::frontend
