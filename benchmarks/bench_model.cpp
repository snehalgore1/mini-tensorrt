// Whole-model latency benchmark: unfused vs fused MLP.
//
//   ./build/benchmarks/bench_model [--model path] [--iters N] [--trace path]
//
// Reports p50/p95/mean wall time per run() and planned peak intermediate bytes
// for both graphs, and optionally writes a Chrome trace of one profiled run.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "mtrt/executor.h"
#include "mtrt/frontend/json_loader.h"
#include "mtrt/optimizer/fusion.h"
#include "mtrt/profiler/profiler.h"
#include "mtrt/registry.h"

using namespace mtrt;
using namespace mtrt::frontend;

namespace {

std::unordered_map<TensorId, Tensor> bindings_for(
    const Graph& g, const std::unordered_map<std::string, Tensor>& by_name) {
  std::unordered_map<TensorId, Tensor> b;
  for (TensorId id = 0; id < g.num_tensors(); ++id) {
    auto it = by_name.find(g.tensor(id).name);
    if (it != by_name.end()) b.emplace(id, it->second);
  }
  return b;
}

double percentile(std::vector<double>& v, double p) {
  std::sort(v.begin(), v.end());
  const size_t idx = static_cast<size_t>(p * (v.size() - 1));
  return v[idx];
}

struct LatResult {
  double p50_ns, p95_ns, mean_ns;
  int64_t peak_bytes;
};

LatResult bench(const Graph& g,
                const std::unordered_map<std::string, Tensor>& by_name,
                const KernelRegistry& reg, int iters) {
  Executor exec(g, reg);
  auto bindings = bindings_for(g, by_name);

  for (int i = 0; i < 50; ++i) exec.run(bindings);  // warm up

  std::vector<double> samples;
  samples.reserve(iters);
  double sum = 0.0;
  for (int i = 0; i < iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    volatile auto out = exec.run(bindings);
    (void)out;
    const auto t1 = std::chrono::steady_clock::now();
    const double ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    samples.push_back(ns);
    sum += ns;
  }
  return LatResult{percentile(samples, 0.50), percentile(samples, 0.95),
                   sum / iters, exec.memory_stats().peak_bytes};
}

}  // namespace

int main(int argc, char** argv) {
  std::string model = std::string(MTRT_MODELS_DIR) + "/mlp.json";
  std::string trace_path;
  int iters = 2000;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
    else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--trace") && i + 1 < argc) trace_path = argv[++i];
  }

  LoadedModel m = load_json_model(model);
  KernelRegistry reg;
  register_builtin_kernels(reg);

  // Synthesize an input; weights come from the model file.
  std::unordered_map<std::string, Tensor> by_name;
  for (const auto& [id, w] : m.weights) by_name.emplace(m.graph.tensor(id).name, w);
  const TensorId in_id = m.graph.graph_inputs()[0];
  Tensor x = Tensor::owning(DType::kF32, m.graph.tensor(in_id).shape);
  for (int64_t i = 0; i < x.numel(); ++i) x.data<float>()[i] = 0.1f;
  by_name.emplace(m.graph.tensor(in_id).name, x);

  Graph fused = fuse_matmul_bias_gelu(m.graph);

  LatResult unf = bench(m.graph, by_name, reg, iters);
  LatResult fus = bench(fused, by_name, reg, iters);

  std::cout << "model=" << model << " iters=" << iters << "\n";
  std::cout << "                  p50(ns)   p95(ns)  mean(ns)  peak_bytes\n";
  std::cout << "unfused         " << unf.p50_ns << "     " << unf.p95_ns
            << "    " << unf.mean_ns << "      " << unf.peak_bytes << "\n";
  std::cout << "fused           " << fus.p50_ns << "     " << fus.p95_ns
            << "    " << fus.mean_ns << "      " << fus.peak_bytes << "\n";

  if (!trace_path.empty()) {
    Executor exec(m.graph, reg);
    Profiler prof;
    exec.set_profiler(&prof);
    exec.run(bindings_for(m.graph, by_name));
    prof.write_chrome_trace(trace_path);
    std::cout << "wrote trace: " << trace_path << "\n";
  }
  return 0;
}
