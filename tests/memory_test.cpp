#include "mtrt/memory/memory_planner.h"

#include <string>
#include <unordered_map>

#include <gtest/gtest.h>

#include "mtrt/executor.h"
#include "mtrt/frontend/json_loader.h"
#include "mtrt/registry.h"
#include "tests/support/golden.h"

using namespace mtrt;
using namespace mtrt::frontend;
using namespace mtrt::testing;

namespace {
std::string mlp_json() { return std::string(MTRT_MODELS_DIR) + "/mlp.json"; }
std::string gpt2_json() { return std::string(MTRT_MODELS_DIR) + "/gpt2_block.json"; }

int64_t interval_bytes(const MemoryPlan& p, TensorId id) {
  for (const auto& iv : p.intervals)
    if (iv.id == id) return iv.nbytes;
  return 0;
}
}  // namespace

// The core planner invariant: any two tensors whose lifetimes conflict must not
// share overlapping byte ranges in the arena.
TEST(MemoryPlanner, NoConflictingByteOverlap) {
  LoadedModel m = load_json_model(mlp_json());
  MemoryPlan plan = plan_memory(m.graph);

  for (size_t i = 0; i < plan.intervals.size(); ++i) {
    for (size_t j = i + 1; j < plan.intervals.size(); ++j) {
      const LiveInterval& a = plan.intervals[i];
      const LiveInterval& b = plan.intervals[j];
      if (!intervals_overlap(a, b)) continue;  // may share space -> skip
      const int64_t a0 = plan.offset.at(a.id);
      const int64_t a1 = a0 + interval_bytes(plan, a.id);
      const int64_t b0 = plan.offset.at(b.id);
      const int64_t b1 = b0 + interval_bytes(plan, b.id);
      const bool byte_overlap = a0 < b1 && b0 < a1;
      EXPECT_FALSE(byte_overlap)
          << "tensors " << a.id << " and " << b.id
          << " have conflicting lifetimes but overlapping byte ranges";
    }
  }
}

TEST(MemoryPlanner, PlannedNotLargerThanNaive) {
  LoadedModel m = load_json_model(mlp_json());
  MemoryPlan plan = plan_memory(m.graph);

  int64_t naive_sum = 0;
  for (const auto& iv : plan.intervals) naive_sum += iv.nbytes;
  EXPECT_LE(plan.total_bytes, naive_sum);
  EXPECT_GT(plan.total_bytes, 0);
}

// Outputs must be identical under both allocators (and both must match PyTorch).
TEST(MemoryPlanner, NaiveAndPlannedAgree) {
  LoadedModel m = load_json_model(mlp_json());
  KernelRegistry reg;
  register_builtin_kernels(reg);

  Tensor x = tensor_from_npy(load_golden("mlp_input"));
  auto make_bindings = [&]() {
    std::unordered_map<TensorId, Tensor> b = m.weights;
    b.emplace(m.graph.graph_inputs()[0], x);
    return b;
  };

  Executor naive(m.graph, reg, AllocatorKind::kNaive);
  Executor planned(m.graph, reg, AllocatorKind::kPlanned);
  std::vector<Tensor> out_naive = naive.run(make_bindings());
  std::vector<Tensor> out_planned = planned.run(make_bindings());

  ASSERT_EQ(out_naive.size(), 1u);
  ASSERT_EQ(out_planned.size(), 1u);
  ASSERT_EQ(out_naive[0].numel(), out_planned[0].numel());
  const float* a = out_naive[0].data<float>();
  const float* b = out_planned[0].data<float>();
  for (int64_t i = 0; i < out_naive[0].numel(); ++i) {
    EXPECT_FLOAT_EQ(a[i], b[i]) << "allocators disagree at " << i;
  }

  NpyArray expected = load_golden("mlp_output");
  ExpectGolden(out_planned[0], expected, kDefaultRtol, 1e-5f);
}

// Emit the measured naive-vs-planned stats for docs/RESULTS.md.
TEST(MemoryPlanner, ReportStats) {
  LoadedModel m = load_json_model(mlp_json());
  KernelRegistry reg;
  register_builtin_kernels(reg);

  Executor naive(m.graph, reg, AllocatorKind::kNaive);
  Executor planned(m.graph, reg, AllocatorKind::kPlanned);
  MemoryStats ns = naive.memory_stats();
  MemoryStats ps = planned.memory_stats();

  std::cout << "[RESULTS] MLP naive:   peak_bytes=" << ns.peak_bytes
            << " alloc_count=" << ns.allocation_count << std::endl;
  std::cout << "[RESULTS] MLP planned: peak_bytes=" << ps.peak_bytes
            << " alloc_count=" << ps.allocation_count
            << " bytes_reused=" << ps.bytes_reused << std::endl;

  EXPECT_EQ(ps.allocation_count, 1);
  EXPECT_LT(ps.peak_bytes, ns.peak_bytes);
  EXPECT_EQ(ps.bytes_reused, ns.peak_bytes - ps.peak_bytes);
}

// Same ablation on the GPT-2-small block: the numbers that actually matter for
// RESULTS.md (megabyte-scale, unlike the toy MLP's bytes). Skipped unless the
// large, gitignored model has been exported (python/export_models.py).
TEST(MemoryPlanner, ReportStatsGpt2) {
  if (!file_exists(gpt2_json())) {
    GTEST_SKIP() << "gpt2_block.json absent; run python/export_models.py";
  }
  LoadedModel m = load_json_model(gpt2_json());
  KernelRegistry reg;
  register_builtin_kernels(reg);

  Executor naive(m.graph, reg, AllocatorKind::kNaive);
  Executor planned(m.graph, reg, AllocatorKind::kPlanned);
  MemoryStats ns = naive.memory_stats();
  MemoryStats ps = planned.memory_stats();

  const double mb = 1.0 / (1024.0 * 1024.0);
  std::cout << "[RESULTS] GPT2 naive:   peak_bytes=" << ns.peak_bytes << " ("
            << ns.peak_bytes * mb << " MB) alloc_count=" << ns.allocation_count
            << std::endl;
  std::cout << "[RESULTS] GPT2 planned: peak_bytes=" << ps.peak_bytes << " ("
            << ps.peak_bytes * mb << " MB) alloc_count=" << ps.allocation_count
            << " bytes_reused=" << ps.bytes_reused << " ("
            << ps.bytes_reused * mb << " MB, "
            << (100.0 * ps.bytes_reused / ns.peak_bytes) << "% of naive peak)"
            << std::endl;

  EXPECT_LT(ps.peak_bytes, ns.peak_bytes);
  EXPECT_EQ(ps.bytes_reused, ns.peak_bytes - ps.peak_bytes);
}
