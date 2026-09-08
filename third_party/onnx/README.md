# Vendored ONNX protobuf schema

`onnx.proto` is vendored verbatim from the `onnx` Python package (onnx 1.22.0,
IR_VERSION 13), pinned so the C++ ONNX frontend builds reproducibly without a
network fetch. It is `proto2`, `package onnx`, and self-contained (no imports).

At build time (only when `-DMTRT_ONNX=ON`) CMake runs `protoc --cpp_out` on this
file to generate `onnx.pb.{h,cc}`, which the loader in `src/frontend/onnx/` uses
to parse `.onnx` files into MiniTensorRT's frontend-agnostic IR. Using the schema
+ protobuf (rather than a hand-written parser) follows CLAUDE.md's non-goal.
