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

// Device intermediate-memory stats, analogous to the CPU MemoryStats: naive =
// one cudaMalloc per intermediate; planned = all intermediates packed into one
// device arena via the greedy planner, reusing space between disjoint lifetimes.
struct DeviceMemStats {
  int64_t naive_peak_bytes = 0;   // sum of intermediate bytes
  int64_t naive_alloc_count = 0;  // one per intermediate
  int64_t arena_bytes = 0;        // single device arena size (planned peak)
  int64_t bytes_reused = 0;       // naive_peak - arena_bytes
};

class CudaExecutor {
 public:
  explicit CudaExecutor(const frontend::LoadedModel& m);
  ~CudaExecutor();
  CudaExecutor(const CudaExecutor&) = delete;
  CudaExecutor& operator=(const CudaExecutor&) = delete;

  // Bind the single graph input from raw host bytes (matching its dtype: f32 or
  // i32), run the graph on the GPU, and return the single graph output as host
  // floats.
  std::vector<float> run(const void* input_host);

  DeviceMemStats memory_stats() const { return stats_; }

 private:
  const frontend::LoadedModel& m_;
  std::vector<void*> d_;         // device pointer per tensor id (may view arena_)
  std::vector<int64_t> bytes_;   // byte size per tensor id
  std::vector<bool> owned_;      // true if d_[id] is its own cudaMalloc (not a view)
  void* arena_ = nullptr;        // single device arena backing the intermediates
  DeviceMemStats stats_;
};

}  // namespace mtrt::cuda
