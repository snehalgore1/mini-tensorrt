#include "backends/cuda/cuda_executor.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <cuda_runtime.h>

#include "backends/cuda/kernels.h"
#include "mtrt/memory/memory_planner.h"
#include "mtrt/tensor.h"

namespace mtrt::cuda {
namespace {

#define CE_CK(call)                                                          \
  do {                                                                       \
    const cudaError_t e__ = (call);                                          \
    if (e__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                   cudaGetErrorString(e__), __FILE__, __LINE__);             \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

int64_t numel(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}
}  // namespace

CudaExecutor::CudaExecutor(const frontend::LoadedModel& m) : m_(m) {
  const int64_t nt = m_.graph.num_tensors();
  d_.assign(static_cast<size_t>(nt), nullptr);
  bytes_.assign(static_cast<size_t>(nt), 0);
  owned_.assign(static_cast<size_t>(nt), false);

  // Plan all intermediates into one device arena (greedy planner, reused from the
  // CPU path). Non-intermediates (weights, input, output) keep their own buffers,
  // exactly as the CPU executor excludes them from planning.
  const MemoryPlan plan = plan_memory(m_.graph);
  CE_CK(cudaMalloc(&arena_, static_cast<size_t>(plan.total_bytes > 0 ? plan.total_bytes : 1)));

  int64_t naive_sum = 0, naive_count = 0;
  for (TensorId id = 0; id < nt; ++id) {
    const TensorInfo& t = m_.graph.tensor(id);
    const int64_t nb = numel(t.shape) * static_cast<int64_t>(dtype_size(t.dtype));
    bytes_[static_cast<size_t>(id)] = nb;

    if (t.kind == TensorKind::kIntermediate) {
      d_[static_cast<size_t>(id)] =
          static_cast<char*>(arena_) + plan.offset.at(id);  // view into the arena
      naive_sum += nb;
      ++naive_count;
      continue;
    }
    // Own buffer for weights / input / output.
    CE_CK(cudaMalloc(&d_[static_cast<size_t>(id)], static_cast<size_t>(nb)));
    owned_[static_cast<size_t>(id)] = true;
    auto it = m_.weights.find(id);
    if (it != m_.weights.end()) {
      const Tensor& w = it->second;
      const void* src = t.dtype == DType::kI32
                            ? static_cast<const void*>(w.data<int32_t>())
                            : static_cast<const void*>(w.data<float>());
      CE_CK(cudaMemcpy(d_[static_cast<size_t>(id)], src, static_cast<size_t>(nb),
                       cudaMemcpyHostToDevice));
    }
  }
  stats_ = DeviceMemStats{naive_sum, naive_count, plan.total_bytes,
                          naive_sum - plan.total_bytes};
}

CudaExecutor::~CudaExecutor() {
  for (size_t i = 0; i < d_.size(); ++i)
    if (owned_[i] && d_[i]) cudaFree(d_[i]);  // intermediates are views into arena_
  if (arena_) cudaFree(arena_);
}

std::vector<float> CudaExecutor::run(const void* input_host) {
  const TensorId in_id = m_.graph.graph_inputs()[0];
  CE_CK(cudaMemcpy(d_[static_cast<size_t>(in_id)], input_host,
                   static_cast<size_t>(bytes_[static_cast<size_t>(in_id)]),
                   cudaMemcpyHostToDevice));

  auto shape = [&](TensorId id) -> const std::vector<int64_t>& {
    return m_.graph.tensor(id).shape;
  };
  auto F = [&](TensorId id) { return static_cast<float*>(d_[static_cast<size_t>(id)]); };
  auto I = [&](TensorId id) { return static_cast<int*>(d_[static_cast<size_t>(id)]); };
  auto dbl = [&](const Node& n, const char* k) {
    return static_cast<float>(std::get<double>(n.attrs.at(k)));
  };

  for (NodeId nid : m_.graph.topo_order()) {
    const Node& n = m_.graph.node(nid);
    const std::string& op = n.op_type;
    const auto& out = n.outputs;
    const auto& in = n.inputs;

    if (op == "Gather") {
      gather(F(in[0]), I(in[1]), F(out[0]),
             static_cast<int>(numel(shape(in[1]))), static_cast<int>(shape(in[0])[1]));
    } else if (op == "Add") {
      const auto& a = shape(in[0]);
      const int N = static_cast<int>(a.back());
      const int M = static_cast<int>(numel(a) / a.back());
      const int64_t bn = numel(shape(in[1]));
      const bool bias = (bn == a.back() && bn != numel(a));
      add(F(in[0]), F(in[1]), F(out[0]), M, N, bias);
    } else if (op == "MatMul") {
      matmul(F(in[0]), F(in[1]), F(out[0]), static_cast<int>(shape(in[0])[0]),
             static_cast<int>(shape(in[1])[1]), static_cast<int>(shape(in[0])[1]));
    } else if (op == "Reshape") {
      reshape(F(in[0]), F(out[0]), static_cast<int>(numel(shape(in[0]))));
    } else if (op == "Transpose") {
      const auto& perm64 = std::get<std::vector<int64_t>>(n.attrs.at("perm"));
      const auto& s = shape(in[0]);
      int ishape[8], perm[8];
      const int rank = static_cast<int>(s.size());
      for (int k = 0; k < rank; ++k) { ishape[k] = static_cast<int>(s[k]); perm[k] = static_cast<int>(perm64[k]); }
      transpose(F(in[0]), F(out[0]), ishape, perm, rank, static_cast<int>(numel(s)));
    } else if (op == "Scale") {
      scale(F(in[0]), F(out[0]), static_cast<int>(numel(shape(in[0]))), dbl(n, "scale"));
    } else if (op == "Softmax") {
      const auto& s = shape(in[0]);
      const int nlast = static_cast<int>(s.back());
      softmax(F(in[0]), F(out[0]), static_cast<int>(numel(s) / s.back()), nlast);
    } else if (op == "CausalSoftmax") {
      const auto& s = shape(in[0]);
      const int Sk = static_cast<int>(s.back());
      const int Sq = static_cast<int>(s[s.size() - 2]);
      causal_softmax(F(in[0]), F(out[0]), static_cast<int>(numel(s) / s.back()), Sq, Sk);
    } else if (op == "BatchedMatMul") {
      const auto& a = shape(in[0]);  // [B,M,K]
      const auto& b = shape(in[1]);  // [B,K,N]
      batched_matmul(F(in[0]), F(in[1]), F(out[0]), static_cast<int>(a[0]),
                     static_cast<int>(a[1]), static_cast<int>(b[2]), static_cast<int>(a[2]));
    } else if (op == "Gelu") {
      gelu(F(in[0]), F(out[0]), static_cast<int>(numel(shape(in[0]))));
    } else if (op == "GeluTanh") {
      gelu_tanh(F(in[0]), F(out[0]), static_cast<int>(numel(shape(in[0]))));
    } else if (op == "LayerNorm") {
      const auto& s = shape(in[0]);
      const int D = static_cast<int>(s.back());
      layernorm(F(in[0]), F(in[1]), F(in[2]), F(out[0]),
                static_cast<int>(numel(s) / s.back()), D, dbl(n, "eps"));
    } else {
      throw std::runtime_error("CudaExecutor: unsupported op '" + op + "'");
    }
  }
  CE_CK(cudaDeviceSynchronize());

  const TensorId oid = m_.graph.graph_outputs()[0];
  std::vector<float> host(static_cast<size_t>(numel(shape(oid))));
  CE_CK(cudaMemcpy(host.data(), d_[static_cast<size_t>(oid)],
                   static_cast<size_t>(bytes_[static_cast<size_t>(oid)]),
                   cudaMemcpyDeviceToHost));
  return host;
}

}  // namespace mtrt::cuda
