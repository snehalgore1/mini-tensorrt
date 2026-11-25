// Golden + invariant tests for the GPT-OSS MoE CPU operators (Milestone 1A).
//
// Fixtures come from python/gen_goldens.py --only-moe, i.e. from moe_ref.py, which
// check_official_moe.py proves equal to the official GPT-OSS MLPBlock. Each operator is
// fed the GOLDEN inputs of its stage, so a failure points at exactly one kernel.
//
// Tolerances: float stages use the combined rule |got-ref| <= 1e-6 + 1e-5*|ref| (CLAUDE.md
// standard), with max abs / max rel reported separately (tests/support/golden.h).
// Index stages (idx, src, inv, offsets) must match EXACTLY, and so must t_sorted (a pure
// row copy, no arithmetic).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "mtrt/executor.h"
#include "mtrt/frontend/json_loader.h"
#include "mtrt/node.h"
#include "mtrt/op_context.h"
#include "mtrt/registry.h"
#include "mtrt/tensor.h"
#include "tests/support/golden.h"
#include "tests/support/moe_graph.h"
#include "tests/support/moe_tolerance.h"

using namespace mtrt;
using namespace mtrt::testing;

namespace {

const char* kCases[] = {"small", "four", "clamp", "e32", "sparse"};

// Runs one registered kernel. The registry is keyed on outputs[0]'s dtype, exactly as the
// executor does it.
void run_op(const std::string& op, const std::vector<const Tensor*>& ins,
            const std::vector<Tensor*>& outs,
            std::unordered_map<std::string, Attribute> attrs = {}) {
  KernelRegistry reg;
  register_builtin_kernels(reg);
  KernelFn fn = reg.lookup(op, outs[0]->dtype());
  ASSERT_NE(fn, nullptr) << "no kernel registered for " << op;
  Node node{op, {}, {}, std::move(attrs)};
  OpContext ctx{ins, outs, node};
  fn(ctx);
}

Tensor F(const std::string& c, const std::string& n) {
  return tensor_from_npy(load_golden("moe_" + c + "_" + n));
}
Tensor I32(const std::string& c, const std::string& n) {
  return tensor_from_npy_i32(load_golden_i32("moe_" + c + "_" + n));
}
NpyArray GF(const std::string& c, const std::string& n) {
  return load_golden("moe_" + c + "_" + n);
}

::testing::AssertionResult EqualI32(const Tensor& got, const Tensor& ref) {
  if (got.numel() != ref.numel()) return ::testing::AssertionFailure() << "numel differs";
  for (int64_t i = 0; i < got.numel(); ++i)
    if (got.data<int32_t>()[i] != ref.data<int32_t>()[i])
      return ::testing::AssertionFailure() << "index " << i << ": got " << got.data<int32_t>()[i]
                                           << " expected " << ref.data<int32_t>()[i];
  return ::testing::AssertionSuccess();
}

// The dispatch invariants from CPU_DESIGN.md D-MoE-4/5, checked on any routing.
::testing::AssertionResult DispatchInvariants(const int32_t* idx, int64_t T, int64_t k,
                                              int64_t E, const int32_t* src,
                                              const int32_t* inv, const int32_t* off) {
  const int64_t Tk = T * k;
  if (off[0] != 0 || off[E] != Tk)
    return ::testing::AssertionFailure() << "offsets[0]/offsets[E] wrong: " << off[0] << ", "
                                         << off[E];
  std::vector<int64_t> count(static_cast<size_t>(E), 0);
  for (int64_t a = 0; a < Tk; ++a) count[static_cast<size_t>(idx[a])]++;
  for (int64_t e = 0; e < E; ++e) {
    if (off[e + 1] - off[e] != count[static_cast<size_t>(e)])
      return ::testing::AssertionFailure() << "segment " << e << " size != routing count";
    for (int64_t j = off[e]; j < off[e + 1]; ++j) {
      if (idx[src[j]] != e) return ::testing::AssertionFailure() << "row " << j << " in wrong expert";
      if (j > off[e] && src[j - 1] >= src[j])
        return ::testing::AssertionFailure() << "not stable inside expert " << e;
    }
  }
  std::vector<int> seen(static_cast<size_t>(Tk), 0);
  for (int64_t j = 0; j < Tk; ++j) {
    if (src[j] < 0 || src[j] >= Tk || seen[static_cast<size_t>(src[j])]++)
      return ::testing::AssertionFailure() << "src is not a permutation (j=" << j << ")";
    if (inv[src[j]] != j) return ::testing::AssertionFailure() << "inv[src[j]] != j at " << j;
  }
  // Every token appears exactly k times (nothing lost or duplicated).
  std::vector<int> per_tok(static_cast<size_t>(T), 0);
  for (int64_t j = 0; j < Tk; ++j) per_tok[static_cast<size_t>(src[j] / k)]++;
  for (int64_t t = 0; t < T; ++t)
    if (per_tok[static_cast<size_t>(t)] != k)
      return ::testing::AssertionFailure() << "token " << t << " has " << per_tok[t] << " rows";
  return ::testing::AssertionSuccess();
}

// Whole-layer check: gate on the term magnitude, and also report the plain |ref| stats.
::testing::AssertionResult LayerClose(const std::string& what, const std::string& c,
                                      const Tensor& got) {
  NpyArray ref = GF(c, "out");
  const std::vector<float> mag = layer_term_magnitude(c);
  const CloseStats plain = close_stats(got.data<float>(), ref.data.data(), got.numel(),
                                       kDefaultRtol, kDefaultAtol);
  const CloseStats s = close_stats(got.data<float>(), ref.data.data(), got.numel(),
                                   kDefaultRtol, kDefaultAtol, mag.data());
  std::printf("  [%s %s] max_abs=%.3e max_rel=%.3e (|ref|>=0.1, near_zero=%lld) "
              "tol_used(term-mag)=%.3f tol_used(|ref|)=%.3f\n",
              c.c_str(), what.c_str(), s.max_abs, s.max_rel, (long long)s.near_zero, s.worst,
              plain.worst);
  if (s.worst <= 1.0) return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << what << " " << c << ": used " << s.worst
                                       << "x of the term-magnitude tolerance at element "
                                       << s.worst_index << " (max_abs=" << s.max_abs << ")";
}

class MoECase : public ::testing::TestWithParam<const char*> {};

}  // namespace

// ---------------------------------------------------------------- per-operator goldens

TEST_P(MoECase, RMSNorm) {
  const std::string c = GetParam();
  Tensor x = F(c, "x"), scale = F(c, "norm_scale");
  Tensor t = Tensor::owning(DType::kF32, x.shape());
  run_op("RMSNorm", {&x, &scale}, {&t}, {{"eps", 1e-5}});
  MTRT_EXPECT_CLOSE("t", t, GF(c, "t"));
}

TEST_P(MoECase, TopK) {
  const std::string c = GetParam();
  Tensor logits = F(c, "logits");
  Tensor idx_ref = I32(c, "idx");
  const int64_t T = idx_ref.shape()[0], k = idx_ref.shape()[1];
  Tensor w = Tensor::owning(DType::kF32, {T, k});
  Tensor idx = Tensor::owning(DType::kI32, {T, k});
  run_op("MoETopK", {&logits}, {&w, &idx}, {{"k", k}});
  EXPECT_TRUE(EqualI32(idx, idx_ref));
  MTRT_EXPECT_CLOSE("w", w, GF(c, "w"));
  for (int64_t t = 0; t < T; ++t) {  // softmax over the k selected: sums to 1
    double s = 0;
    for (int64_t j = 0; j < k; ++j) s += w.data<float>()[t * k + j];
    EXPECT_NEAR(s, 1.0, 1e-6) << "token " << t;
  }
}

TEST_P(MoECase, Dispatch) {
  const std::string c = GetParam();
  Tensor t = F(c, "t"), idx = I32(c, "idx");
  Tensor src_ref = I32(c, "src"), inv_ref = I32(c, "inv"), off_ref = I32(c, "offsets");
  const int64_t T = idx.shape()[0], k = idx.shape()[1], H = t.shape()[1];
  const int64_t E = off_ref.numel() - 1;
  Tensor ts = Tensor::owning(DType::kF32, {T * k, H});
  Tensor src = Tensor::owning(DType::kI32, {T * k});
  Tensor inv = Tensor::owning(DType::kI32, {T * k});
  Tensor off = Tensor::owning(DType::kI32, {E + 1});
  run_op("MoEDispatch", {&t, &idx}, {&ts, &src, &inv, &off}, {{"num_experts", E}});
  EXPECT_TRUE(EqualI32(src, src_ref));
  EXPECT_TRUE(EqualI32(inv, inv_ref));
  EXPECT_TRUE(EqualI32(off, off_ref));
  NpyArray ts_ref = GF(c, "t_sorted");  // pure row copy: must be bit-exact
  EXPECT_EQ(0, std::memcmp(ts.data<float>(), ts_ref.data.data(),
                           static_cast<size_t>(ts.numel()) * sizeof(float)));
  EXPECT_TRUE(DispatchInvariants(idx.data<int32_t>(), T, k, E, src.data<int32_t>(),
                                 inv.data<int32_t>(), off.data<int32_t>()));
}

TEST_P(MoECase, GroupedMatMul1) {
  const std::string c = GetParam();
  Tensor ts = F(c, "t_sorted"), off = I32(c, "offsets"), w1t = F(c, "w1t"), b1 = F(c, "b1");
  Tensor h = Tensor::owning(DType::kF32, {ts.shape()[0], w1t.shape()[2]});
  run_op("MoEGroupedMatMul", {&ts, &off, &w1t, &b1}, {&h});
  MTRT_EXPECT_CLOSE("h", h, GF(c, "h"));
}

TEST_P(MoECase, SwiGLUInterleaved) {
  const std::string c = GetParam();
  Tensor h = F(c, "h");
  Tensor a = Tensor::owning(DType::kF32, {h.shape()[0], h.shape()[1] / 2});
  run_op("SwiGLUInterleaved", {&h}, {&a}, {{"alpha", 1.702}, {"limit", 7.0}});
  MTRT_EXPECT_CLOSE("a", a, GF(c, "a"));
}

TEST_P(MoECase, GroupedMatMul2) {
  const std::string c = GetParam();
  Tensor a = F(c, "a"), off = I32(c, "offsets"), w2t = F(c, "w2t"), b2 = F(c, "b2");
  Tensor y = Tensor::owning(DType::kF32, {a.shape()[0], w2t.shape()[2]});
  run_op("MoEGroupedMatMul", {&a, &off, &w2t, &b2}, {&y});
  MTRT_EXPECT_CLOSE("y_sorted", y, GF(c, "y_sorted"));
}

TEST_P(MoECase, Combine) {
  const std::string c = GetParam();
  Tensor y = F(c, "y_sorted"), inv = I32(c, "inv"), w = F(c, "w");
  Tensor moe = Tensor::owning(DType::kF32, {w.shape()[0], y.shape()[1]});
  run_op("MoECombine", {&y, &inv, &w}, {&moe});
  MTRT_EXPECT_CLOSE("moe", moe, GF(c, "moe"));
}

// ---------------------------------------------------------------- whole layer via Executor

TEST_P(MoECase, LayerThroughPlannedExecutor) {
  const std::string c = GetParam();
  Tensor x = F(c, "x"), w1t = F(c, "w1t"), w2t = F(c, "w2t");
  const MoEDims d{x.shape()[0], x.shape()[1], w2t.shape()[1], w1t.shape()[0],
                  I32(c, "idx").shape()[1]};
  MoEGraph m = build_moe_graph(d);
  KernelRegistry reg;
  register_builtin_kernels(reg);
  for (AllocatorKind kind : {AllocatorKind::kPlanned, AllocatorKind::kNaive}) {
    Executor exec(m.graph, reg, kind);
    std::vector<Tensor> out = exec.run({{m.x, x}, {m.norm_scale, F(c, "norm_scale")},
                                        {m.gate_wT, F(c, "gate_wT")}, {m.gate_b, F(c, "gate_b")},
                                        {m.w1t, w1t}, {m.b1, F(c, "b1")},
                                        {m.w2t, w2t}, {m.b2, F(c, "b2")}});
    EXPECT_TRUE(LayerClose(kind == AllocatorKind::kPlanned ? "planned" : "naive-alloc", c,
                           out[0]));
  }
}

INSTANTIATE_TEST_SUITE_P(Goldens, MoECase, ::testing::ValuesIn(kCases),
                         [](const auto& info) { return std::string(info.param); });

// ---------------------------------------------------------------- non-golden properties

// One executor, consecutive run()s whose routings differ (dense -> 4 empty experts -> dense):
// static shapes, data-dependent offsets. Catches state leaking between runs (e.g. a stale
// offsets/cursor, or rows of an empty expert left over from the previous run).
TEST(MoE, RepeatedRunsWithDifferentRouting) {
  Tensor x = F("small", "x");
  const MoEDims d{6, 8, 8, 8, 4};
  MoEGraph m = build_moe_graph(d);
  KernelRegistry reg;
  register_builtin_kernels(reg);
  Executor exec(m.graph, reg);
  auto bind = [&](const std::string& c) {
    return std::unordered_map<TensorId, Tensor>{
        {m.x, F(c, "x")},         {m.norm_scale, F(c, "norm_scale")},
        {m.gate_wT, F(c, "gate_wT")}, {m.gate_b, F(c, "gate_b")},
        {m.w1t, F(c, "w1t")},     {m.b1, F(c, "b1")},
        {m.w2t, F(c, "w2t")},     {m.b2, F(c, "b2")}};
  };
  for (const char* c : {"small", "four", "small", "four"}) {
    std::vector<Tensor> out = exec.run(bind(c));
    EXPECT_TRUE(LayerClose("repeated", c, out[0]));
  }
}

// Tie rule: equal logits go to the LOWER expert index (SEMANTICS.md). torch.topk leaves this
// unspecified, so it is a hand-built case rather than a golden.
TEST(MoE, TopKTiesPreferLowerIndex) {
  Tensor logits = Tensor::owning(DType::kF32, {2, 6});
  const float v[12] = {1, 3, 3, 0.5f, 3, 2,      // three-way tie at 3 -> 1,2,4 then 5
                       0, 0, 0, 0, 0, 0};        // all equal -> 0,1,2,3
  std::copy(v, v + 12, logits.data<float>());
  Tensor w = Tensor::owning(DType::kF32, {2, 4});
  Tensor idx = Tensor::owning(DType::kI32, {2, 4});
  run_op("MoETopK", {&logits}, {&w, &idx}, {{"k", int64_t{4}}});
  const int32_t want[8] = {1, 2, 4, 5, 0, 1, 2, 3};
  for (int i = 0; i < 8; ++i) EXPECT_EQ(idx.data<int32_t>()[i], want[i]) << i;
  for (int j = 0; j < 4; ++j) EXPECT_FLOAT_EQ(w.data<float>()[4 + j], 0.25f);
}

// Dispatch invariants on many random routings, including all tokens on the same k experts
// and decode-like T=1. Uses only the property checks (no golden needed).
TEST(MoE, DispatchInvariantsRandomRoutings) {
  std::mt19937 rng(7);
  for (int trial = 0; trial < 500; ++trial) {
    const int64_t E = (trial % 2) ? 32 : 8, k = 4, T = 1 + (trial * 7) % 40, H = 3;
    const bool same = trial % 5 == 0;
    std::vector<int32_t> experts(static_cast<size_t>(E));
    std::iota(experts.begin(), experts.end(), 0);
    std::shuffle(experts.begin(), experts.end(), rng);
    Tensor idx = Tensor::owning(DType::kI32, {T, k});
    for (int64_t t = 0; t < T; ++t) {
      if (!same) std::shuffle(experts.begin(), experts.end(), rng);
      for (int64_t s = 0; s < k; ++s) idx.data<int32_t>()[t * k + s] = experts[s];
    }
    Tensor tok = Tensor::owning(DType::kF32, {T, H});
    for (int64_t i = 0; i < T * H; ++i) tok.data<float>()[i] = static_cast<float>(i);
    Tensor ts = Tensor::owning(DType::kF32, {T * k, H});
    Tensor src = Tensor::owning(DType::kI32, {T * k}), inv = Tensor::owning(DType::kI32, {T * k});
    Tensor off = Tensor::owning(DType::kI32, {E + 1});
    // Poison the outputs: the kernel must fully define them on every run.
    std::fill(off.data<int32_t>(), off.data<int32_t>() + E + 1, -7);
    run_op("MoEDispatch", {&tok, &idx}, {&ts, &src, &inv, &off}, {{"num_experts", E}});
    ASSERT_TRUE(DispatchInvariants(idx.data<int32_t>(), T, k, E, src.data<int32_t>(),
                                   inv.data<int32_t>(), off.data<int32_t>()))
        << "trial " << trial;
    for (int64_t j = 0; j < T * k; ++j)  // row j is a copy of token src[j]/k
      ASSERT_EQ(ts.data<float>()[j * H], static_cast<float>((src.data<int32_t>()[j] / k) * H));
  }
}

// The exported JSON graph (python/export_models.py --only-moe) runs end to end and matches
// the "small" golden: this is the frontend -> IR -> planned executor integration path.
TEST(MoE, JsonModelMatchesGolden) {
  const std::string path = std::string(MTRT_MODELS_DIR) + "/moe_layer.json";
  frontend::LoadedModel model = frontend::load_json_model(path);
  KernelRegistry reg;
  register_builtin_kernels(reg);
  Executor exec(model.graph, reg);
  std::unordered_map<TensorId, Tensor> b = model.weights;
  b[model.graph.graph_inputs()[0]] = F("small", "x");
  std::vector<Tensor> out = exec.run(b);
  EXPECT_TRUE(LayerClose("json", "small", out[0]));
}
