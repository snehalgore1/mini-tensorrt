#pragma once

// Builds the one-layer GPT-OSS MoE graph (docs/moe/CPU_DESIGN.md §2) programmatically, for
// the MoE tests and bench_moe. python/export_models.py emits the identical topology as
// models/moe_layer.json; the JSON test checks that path separately.
//
//   x -> RMSNorm -> t -> MatMul(gate_wT) -> Add(gate_b) -> logits -> MoETopK -> w, idx
//   t, idx -> MoEDispatch -> t_sorted, src, inv, offsets
//   -> MoEGroupedMatMul(w1t, b1) -> h -> SwiGLUInterleaved -> a
//   -> MoEGroupedMatMul(w2t, b2) -> y_sorted -> MoECombine(inv, w) -> moe -> Add(x) -> out

#include <cstdint>
#include <string>
#include <vector>

#include "mtrt/graph.h"

namespace mtrt::testing {

struct MoEDims {
  int64_t T, H, I, E, k;
};

struct MoEGraph {
  Graph graph;
  TensorId x, norm_scale, gate_wT, gate_b, w1t, b1, w2t, b2;
  TensorId out;
};

inline MoEGraph build_moe_graph(const MoEDims& d, double eps = 1e-5, double alpha = 1.702,
                                double limit = 7.0) {
  MoEGraph m;
  Graph& g = m.graph;
  auto T = [&](const std::string& name, DType dt, std::vector<int64_t> shape,
               TensorKind kind) { return g.add_tensor(TensorInfo{name, dt, shape, kind}); };
  const int64_t Tk = d.T * d.k;
  const auto kF = DType::kF32, kI = DType::kI32;
  const auto In = TensorKind::kInput, W = TensorKind::kWeight,
             Mid = TensorKind::kIntermediate, Out = TensorKind::kOutput;

  m.x = T("x", kF, {d.T, d.H}, In);
  m.norm_scale = T("norm_scale", kF, {d.H}, W);
  m.gate_wT = T("gate_wT", kF, {d.H, d.E}, W);
  m.gate_b = T("gate_b", kF, {d.E}, W);
  m.w1t = T("w1t", kF, {d.E, d.H, 2 * d.I}, W);
  m.b1 = T("b1", kF, {d.E, 2 * d.I}, W);
  m.w2t = T("w2t", kF, {d.E, d.I, d.H}, W);
  m.b2 = T("b2", kF, {d.E, d.H}, W);

  const TensorId t = T("t", kF, {d.T, d.H}, Mid);
  const TensorId z = T("z", kF, {d.T, d.E}, Mid);
  const TensorId logits = T("logits", kF, {d.T, d.E}, Mid);
  const TensorId w = T("w", kF, {d.T, d.k}, Mid);
  const TensorId idx = T("idx", kI, {d.T, d.k}, Mid);
  const TensorId t_sorted = T("t_sorted", kF, {Tk, d.H}, Mid);
  const TensorId src = T("src", kI, {Tk}, Mid);
  const TensorId inv = T("inv", kI, {Tk}, Mid);
  const TensorId offsets = T("offsets", kI, {d.E + 1}, Mid);
  const TensorId h = T("h", kF, {Tk, 2 * d.I}, Mid);
  const TensorId a = T("a", kF, {Tk, d.I}, Mid);
  const TensorId y_sorted = T("y_sorted", kF, {Tk, d.H}, Mid);
  const TensorId moe = T("moe", kF, {d.T, d.H}, Mid);
  m.out = T("out", kF, {d.T, d.H}, Out);

  g.add_node(Node{"RMSNorm", {m.x, m.norm_scale}, {t}, {{"eps", eps}}});
  g.add_node(Node{"MatMul", {t, m.gate_wT}, {z}, {}});
  g.add_node(Node{"Add", {z, m.gate_b}, {logits}, {}});
  g.add_node(Node{"MoETopK", {logits}, {w, idx}, {{"k", d.k}}});
  g.add_node(Node{"MoEDispatch", {t, idx}, {t_sorted, src, inv, offsets},
                  {{"num_experts", d.E}}});
  g.add_node(Node{"MoEGroupedMatMul", {t_sorted, offsets, m.w1t, m.b1}, {h}, {}});
  g.add_node(Node{"SwiGLUInterleaved", {h}, {a}, {{"alpha", alpha}, {"limit", limit}}});
  g.add_node(Node{"MoEGroupedMatMul", {a, offsets, m.w2t, m.b2}, {y_sorted}, {}});
  g.add_node(Node{"MoECombine", {y_sorted, inv, w}, {moe}, {}});
  g.add_node(Node{"Add", {m.x, moe}, {m.out}, {}});
  g.mark_input(m.x);
  g.mark_output(m.out);
  return m;
}

}  // namespace mtrt::testing
