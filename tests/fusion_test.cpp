#include "mtrt/optimizer/fusion.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <unordered_map>

#include <gtest/gtest.h>

#include "mtrt/executor.h"
#include "mtrt/frontend/json_loader.h"
#include "mtrt/profiler/profiler.h"
#include "mtrt/registry.h"
#include "tests/support/golden.h"

using namespace mtrt;
using namespace mtrt::frontend;
using namespace mtrt::testing;

namespace {
std::string mlp_json() { return std::string(MTRT_MODELS_DIR) + "/mlp.json"; }

// Bindings keyed by tensor id, resolved from a name->Tensor map. Robust across
// graph rewrites that renumber tensor ids (fusion), since names are preserved.
std::unordered_map<TensorId, Tensor> bindings_for(
    const Graph& g, const std::unordered_map<std::string, Tensor>& by_name) {
  std::unordered_map<TensorId, Tensor> b;
  for (TensorId id = 0; id < g.num_tensors(); ++id) {
    auto it = by_name.find(g.tensor(id).name);
    if (it != by_name.end()) b.emplace(id, it->second);
  }
  return b;
}

// weights + input, keyed by name.
std::unordered_map<std::string, Tensor> inputs_by_name(const LoadedModel& m) {
  std::unordered_map<std::string, Tensor> by_name;
  for (const auto& [id, w] : m.weights) by_name.emplace(m.graph.tensor(id).name, w);
  by_name.emplace(m.graph.tensor(m.graph.graph_inputs()[0]).name,
                  tensor_from_npy(load_golden("mlp_input")));
  return by_name;
}

int count_op(const Graph& g, const std::string& op) {
  int c = 0;
  for (const Node& n : g.nodes())
    if (n.op_type == op) c++;
  return c;
}
}  // namespace

TEST(Fusion, RewritesChain) {
  LoadedModel m = load_json_model(mlp_json());
  Graph fused = fuse_matmul_bias_gelu(m.graph);

  // One MatMul->Add->Gelu chain collapses: 6 nodes -> 4, 11 tensors -> 9.
  EXPECT_EQ(m.graph.num_nodes(), 6);
  EXPECT_EQ(fused.num_nodes(), 4);
  EXPECT_EQ(fused.num_tensors(), 9);
  EXPECT_EQ(count_op(fused, kFusedMatMulBiasGelu), 1);
  EXPECT_EQ(count_op(fused, "Gelu"), 0);
  // The second Linear is followed by Softmax (not Gelu), so it does NOT fuse.
  EXPECT_EQ(count_op(fused, "MatMul"), 1);
  EXPECT_NO_THROW(fused.topo_order());
}

// Core invariant 6: the pass preserves numerics within tolerance.
TEST(Fusion, PreservesNumerics) {
  LoadedModel m = load_json_model(mlp_json());
  Graph fused = fuse_matmul_bias_gelu(m.graph);
  auto by_name = inputs_by_name(m);

  KernelRegistry reg;
  register_builtin_kernels(reg);

  Executor unfused_exec(m.graph, reg);
  Executor fused_exec(fused, reg);
  std::vector<Tensor> a = unfused_exec.run(bindings_for(m.graph, by_name));
  std::vector<Tensor> b = fused_exec.run(bindings_for(fused, by_name));

  ASSERT_EQ(a.size(), 1u);
  ASSERT_EQ(b.size(), 1u);
  ASSERT_EQ(a[0].numel(), b[0].numel());
  for (int64_t i = 0; i < a[0].numel(); ++i)
    EXPECT_NEAR(a[0].data<float>()[i], b[0].data<float>()[i], 1e-6f);

  ExpectGolden(b[0], load_golden("mlp_output"), kDefaultRtol, 1e-5f);
}

TEST(Fusion, ReducesPeakMemory) {
  LoadedModel m = load_json_model(mlp_json());
  Graph fused = fuse_matmul_bias_gelu(m.graph);
  KernelRegistry reg;
  register_builtin_kernels(reg);

  Executor unfused_exec(m.graph, reg);
  Executor fused_exec(fused, reg);
  const int64_t unf = unfused_exec.memory_stats().peak_bytes;
  const int64_t fus = fused_exec.memory_stats().peak_bytes;
  std::cout << "[RESULTS] planned peak bytes: unfused=" << unf
            << " fused=" << fus << std::endl;
  EXPECT_LT(fus, unf);  // fusion removes the t0/t1 intermediates
}

TEST(Profiler, RecordsPerNodeAndWritesTrace) {
  LoadedModel m = load_json_model(mlp_json());
  auto by_name = inputs_by_name(m);
  KernelRegistry reg;
  register_builtin_kernels(reg);

  Executor exec(m.graph, reg);
  Profiler prof;
  exec.set_profiler(&prof);
  exec.run(bindings_for(m.graph, by_name));

  EXPECT_EQ(prof.events().size(), static_cast<size_t>(m.graph.num_nodes()));
  EXPECT_EQ(prof.events().front().name, "MatMul");

  const std::string path = ::testing::TempDir() + "/mtrt_trace.json";
  ASSERT_NO_THROW(prof.write_chrome_trace(path));
  std::ifstream f(path);
  std::string contents((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
  EXPECT_NE(contents.find("traceEvents"), std::string::npos);
  EXPECT_NE(contents.find("MatMul"), std::string::npos);
}
