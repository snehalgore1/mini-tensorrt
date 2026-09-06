// Real GPT-2 (124M) end-to-end: our runtime's logits must match HuggingFace.
// This is the correctness core of "#2 — make it a real workload". Skipped unless
// the large, gitignored model + golden have been exported:
//   python python/export_gpt2_hf.py --seq-len 16 --verify
//
// The golden (models/goldens/gpt2_124m_logits.npy) is HF's logits for the canonical
// id sequence arange(10, 10+S); the ids here are kept in sync with export_gpt2_hf.py.

#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "mtrt/executor.h"
#include "mtrt/frontend/json_loader.h"
#include "mtrt/registry.h"
#include "tests/support/golden.h"

using namespace mtrt;
using namespace mtrt::frontend;
using namespace mtrt::testing;

namespace {
std::string gpt2_json() { return std::string(MTRT_MODELS_DIR) + "/gpt2_124m.json"; }
std::string gpt2_flash_json() { return std::string(MTRT_MODELS_DIR) + "/gpt2_124m_flash.json"; }

// argmax over one row of length n.
int64_t argmax(const float* row, int64_t n) {
  int64_t best = 0;
  for (int64_t j = 1; j < n; ++j)
    if (row[j] > row[best]) best = j;
  return best;
}
}  // namespace

TEST(Gpt2Real, LogitsMatchHuggingFace) {
  if (!file_exists(gpt2_json()) || !golden_exists("gpt2_124m_logits")) {
    GTEST_SKIP() << "gpt2_124m fixtures absent; run "
                    "python python/export_gpt2_hf.py --seq-len 16 --verify";
  }
  LoadedModel m = load_json_model(gpt2_json());
  KernelRegistry reg;
  register_builtin_kernels(reg);

  const TensorId in_id = m.graph.graph_inputs()[0];
  const int64_t S = m.graph.tensor(in_id).shape[0];

  // Canonical ids = arange(10, 10+S), matching export_gpt2_hf.py's golden.
  Tensor ids = Tensor::owning(DType::kI32, {S});
  for (int64_t i = 0; i < S; ++i) ids.data<int32_t>()[i] = static_cast<int32_t>(10 + i);

  std::unordered_map<TensorId, Tensor> bindings = m.weights;
  bindings.emplace(in_id, ids);

  Executor exec(m.graph, reg);
  std::vector<Tensor> outs = exec.run(bindings);
  ASSERT_EQ(outs.size(), 1u);
  const Tensor& logits = outs[0];              // [S, V]
  const int64_t V = logits.shape()[1];
  ASSERT_EQ(logits.shape()[0], S);

  NpyArray hf = load_golden("gpt2_124m_logits");  // [S, V]
  ASSERT_EQ(hf.numel(), logits.numel());

  // Strong signal: argmax (predicted next token) must match HF at every position.
  int mism = 0;
  double max_abs = 0.0;
  for (int64_t i = 0; i < S; ++i) {
    const float* ours = logits.data<float>() + i * V;
    const float* ref = hf.data.data() + i * V;
    if (argmax(ours, V) != argmax(ref, V)) ++mism;
    for (int64_t j = 0; j < V; ++j) {
      const double d = std::abs(static_cast<double>(ours[j]) - ref[j]);
      if (d > max_abs) max_abs = d;
    }
  }
  std::cout << "[RESULTS] GPT-2 124M vs HuggingFace: positions=" << S
            << " argmax_mismatches=" << mism << " max_abs_logit_err=" << max_abs
            << std::endl;

  EXPECT_EQ(mism, 0) << "next-token argmax disagrees with HuggingFace";
  // Loosened tol: 12 layers of FP32 with reassociating (threaded NEON) GEMM.
  EXPECT_LT(max_abs, 5e-2) << "logit magnitude drift too large";
}

// FlashAttention (fused, online-softmax) model vs the standard attention model:
// identical logits, and the [H,S,S] attention scratch is gone -> lower peak memory.
TEST(Gpt2Real, FlashAttentionMatchesAndSavesMemory) {
  if (!file_exists(gpt2_json()) || !file_exists(gpt2_flash_json())) {
    GTEST_SKIP() << "run python python/export_gpt2_hf.py --seq-len 64 --flash";
  }
  LoadedModel base = load_json_model(gpt2_json());
  LoadedModel flash = load_json_model(gpt2_flash_json());
  KernelRegistry reg;
  register_builtin_kernels(reg);

  const TensorId in_id = base.graph.graph_inputs()[0];
  const int64_t S = base.graph.tensor(in_id).shape[0];
  const int64_t V = base.graph.tensor(base.graph.graph_outputs()[0]).shape[1];
  Tensor ids = Tensor::owning(DType::kI32, {S});
  for (int64_t i = 0; i < S; ++i) ids.data<int32_t>()[i] = static_cast<int32_t>(10 + i);

  auto run = [&](LoadedModel& m) {
    Executor exec(m.graph, reg);
    std::unordered_map<TensorId, Tensor> b = m.weights;
    b.emplace(m.graph.graph_inputs()[0], ids);
    return std::make_pair(exec.run(b), exec.memory_stats().peak_bytes);
  };
  auto [ob, mb] = run(base);
  auto [of, mf] = run(flash);

  int mism = 0;
  double max_abs = 0.0;
  for (int64_t i = 0; i < S; ++i)
    for (int64_t j = 0; j < V; ++j) {
      const double d = std::abs(static_cast<double>(of[0].data<float>()[i * V + j]) -
                                ob[0].data<float>()[i * V + j]);
      if (d > max_abs) max_abs = d;
    }
  for (int64_t i = 0; i < S; ++i)
    if (argmax(of[0].data<float>() + i * V, V) != argmax(ob[0].data<float>() + i * V, V))
      ++mism;

  const double mbf = 1.0 / (1024.0 * 1024.0);
  std::cout << "[RESULTS] FlashAttention vs standard (S=" << S << "): argmax_mism=" << mism
            << " max_abs_logit_err=" << max_abs << "; planned peak "
            << mb * mbf << " MB -> " << mf * mbf << " MB ("
            << (100.0 * (mb - mf) / mb) << "% less; the O(S^2) win grows with S)"
            << std::endl;

  EXPECT_EQ(mism, 0);
  EXPECT_LT(max_abs, 1e-2);   // online softmax is numerically equivalent
  EXPECT_LE(mf, mb);          // never worse; strictly less once S^2 scratch dominates
}
