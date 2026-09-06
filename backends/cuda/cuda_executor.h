#pragma once

#include <vector>

#include "mtrt/frontend/json_loader.h"  // LoadedModel
#include "mtrt/graph.h"

// Runs a whole model graph on the GPU. Mirrors the CPU Executor
// (src/runtime/executor.cpp) but keeps every tensor in device memory: weights are
// uploaded once at construction, one device buffer is allocated per tensor
// (naive allocator -- a device arena is the M5 optimization), and each node is
// dispatched to the matching CUDA kernel in backends/cuda/kernels.h. Reuses the
// frontend-agnostic Graph IR unchanged (core invariant 2).

namespace mtrt::cuda {

class CudaExecutor {
 public:
  explicit CudaExecutor(const LoadedModel& m);
  ~CudaExecutor();
  CudaExecutor(const CudaExecutor&) = delete;
  CudaExecutor& operator=(const CudaExecutor&) = delete;

  // Bind the single graph input from raw host bytes (matching its dtype: f32 or
  // i32), run the graph on the GPU, and return the single graph output as host
  // floats.
  std::vector<float> run(const void* input_host);

 private:
  const LoadedModel& m_;
  std::vector<void*> d_;        // device buffer per tensor id
  std::vector<int64_t> bytes_;  // byte size per tensor id
};

}  // namespace mtrt::cuda
