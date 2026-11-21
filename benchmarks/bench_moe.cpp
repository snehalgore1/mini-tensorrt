// GPT-OSS MoE layer on CPU at real gpt-oss-20b MoE dims (H = I = 2880, E = 32, k = 4), FP32.
//
//   ./build/benchmarks/bench_moe [--T 1,4,16,64,256] [--routing uniform|same4|both]
//                                [--iters N] [--warmup N] [--csv path]
//
// For each token count T it reports:
//   1. Whole layer through the planned Executor (RMSNorm .. residual): p50 / p95 wall time,
//      plus a per-op breakdown from the profiler (median over the timed runs).
//   2. The ABLATION -- only the expert section, same inputs, same GEMM (cpu::gemm_auto):
//        grouped : MoEDispatch + MoEGroupedMatMul + SwiGLUInterleaved + MoEGroupedMatMul
//                  + MoECombine   (one GEMM per expert over its contiguous rows)
//        naive   : for each (token, slot): h = t @ W1t[e] + b1 (an M=1 GEMM), SwiGLU,
//                  y = a @ W2t[e] + b2 (M=1), out[t] += w * y     (D-MoE-8)
//      The single variable is "rows of one expert batched into one GEMM" vs "one row at a
//      time". Error between the two is reported as max abs / max rel / normwise rel.
//   3. Memory planning: the planned arena size vs the sum of all intermediates.
//   4. A DERIVED number: expert-weight bytes actually touched (U distinct experts x 99.5 MB)
//      divided by the measured grouped expert-section time. It is labelled derived because
//      the bytes come from a formula, not a hardware counter.
//
// Weights are random (fan-in scaled), not the real checkpoint: speed does not depend on the
// values, and correctness is covered by tests/moe_test.cpp. Always build Release.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "backends/cpu/gemm.h"
#include "mtrt/executor.h"
#include "mtrt/profiler/profiler.h"
#include "mtrt/registry.h"
#include "tests/support/moe_graph.h"

using namespace mtrt;
using mtrt::testing::MoEDims;

namespace {

constexpr int64_t H = 2880, I = 2880, E = 32, K_TOP = 4;
constexpr float kAlpha = 1.702f, kLimit = 7.0f;

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

double pct(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  return v[static_cast<size_t>(p * static_cast<double>(v.size() - 1) + 0.5)];
}

// Deterministic fast fill: xorshift -> uniform(-1,1) * scale (scale = sqrt(3/fan_in) gives
// unit-variance-per-fan-in, like a fan-in-scaled normal init).
void fill(Tensor& t, uint64_t seed, float scale) {
  uint64_t s = seed * 0x9E3779B97F4A7C15ull + 1;
  float* p = t.data<float>();
  for (int64_t i = 0; i < t.numel(); ++i) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    p[i] = scale * (static_cast<float>(s >> 40) * (2.0f / 16777216.0f) - 1.0f);
  }
}

struct Weights {
  Tensor norm_scale, gate_wT, gate_b_uniform, gate_b_same4, w1t, b1, w2t, b2;
};

Weights make_weights() {
  Weights w;
  w.norm_scale = Tensor::owning(DType::kF32, {H});
  std::fill(w.norm_scale.data<float>(), w.norm_scale.data<float>() + H, 1.0f);
  w.gate_wT = Tensor::owning(DType::kF32, {H, E});
  fill(w.gate_wT, 1, std::sqrt(3.0f / H));
  w.gate_b_uniform = Tensor::owning(DType::kF32, {E});
  std::fill(w.gate_b_uniform.data<float>(), w.gate_b_uniform.data<float>() + E, 0.0f);
  w.gate_b_same4 = w.gate_b_uniform.clone();
  for (int e : {1, 3, 4, 6}) w.gate_b_same4.data<float>()[e] = 100.0f;  // all tokens -> 4
  w.w1t = Tensor::owning(DType::kF32, {E, H, 2 * I});
  fill(w.w1t, 2, std::sqrt(3.0f / H));
  w.b1 = Tensor::owning(DType::kF32, {E, 2 * I});
  fill(w.b1, 3, 0.1f);
  w.w2t = Tensor::owning(DType::kF32, {E, I, H});
  fill(w.w2t, 4, std::sqrt(3.0f / I));
  w.b2 = Tensor::owning(DType::kF32, {E, H});
  fill(w.b2, 5, 0.1f);
  return w;
}

struct Op {
  KernelFn fn;
  Node node;
};

// Calls one registered kernel directly (for the expert-section ablation).
void call(const Op& op, std::vector<const Tensor*> ins, std::vector<Tensor*> outs) {
  OpContext ctx{ins, outs, op.node};
  op.fn(ctx);
}

// Naive expert section: one (token, slot) at a time with M = 1 GEMMs. `h` and `a` are
// preallocated single-row buffers, so the loop itself allocates nothing.
void naive_experts(const Tensor& t, const Tensor& idx, const Tensor& w, const Weights& W,
                   Tensor& out, std::vector<float>& h, std::vector<float>& a,
                   std::vector<float>& y) {
  const int64_t T = t.shape()[0];
  const float* pt = t.data<float>();
  const int32_t* pi = idx.data<int32_t>();
  const float* pw = w.data<float>();
  float* po = out.data<float>();
  std::fill(po, po + T * H, 0.0f);
  for (int64_t tok = 0; tok < T; ++tok) {
    for (int64_t s = 0; s < K_TOP; ++s) {
      const int64_t e = pi[tok * K_TOP + s];
      cpu::gemm_auto(pt + tok * H, W.w1t.data<float>() + e * H * 2 * I, h.data(), 1,
                     static_cast<int>(2 * I), static_cast<int>(H));
      const float* b1 = W.b1.data<float>() + e * 2 * I;
      for (int64_t j = 0; j < I; ++j) {
        const float g = std::min(h[2 * j] + b1[2 * j], kLimit);
        const float l = std::min(std::max(h[2 * j + 1] + b1[2 * j + 1], -kLimit), kLimit);
        a[j] = g * (1.0f / (1.0f + std::exp(-kAlpha * g))) * (l + 1.0f);
      }
      cpu::gemm_auto(a.data(), W.w2t.data<float>() + e * I * H, y.data(), 1,
                     static_cast<int>(H), static_cast<int>(I));
      const float* b2 = W.b2.data<float>() + e * H;
      const float ws = pw[tok * K_TOP + s];
      float* o = po + tok * H;
      for (int64_t c = 0; c < H; ++c) o[c] += ws * (y[c] + b2[c]);
    }
  }
}

std::vector<int64_t> parse_list(const std::string& s) {
  std::vector<int64_t> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) v.push_back(std::stoll(tok));
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int64_t> Ts = {1, 4, 16, 64, 256};
  std::vector<std::string> routings = {"uniform", "same4"};
  int iters_override = 0, warmup = 2;
  std::string csv_path;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&] { return std::string(argv[++i]); };
    if (a == "--T") Ts = parse_list(next());
    else if (a == "--routing") {
      const std::string r = next();
      routings = r == "both" ? std::vector<std::string>{"uniform", "same4"}
                             : std::vector<std::string>{r};
    } else if (a == "--iters") iters_override = std::stoi(next());
    else if (a == "--warmup") warmup = std::stoi(next());
    else if (a == "--csv") csv_path = next();
  }

  std::printf("bench_moe: H=%lld I=%lld E=%lld k=%lld FP32, hardware threads=%u, "
              "MTRT_MATMUL=%s\n",
              (long long)H, (long long)I, (long long)E, (long long)K_TOP,
              std::thread::hardware_concurrency(),
              std::getenv("MTRT_MATMUL") ? std::getenv("MTRT_MATMUL") : "auto");
  const auto tw = Clock::now();
  const Weights W = make_weights();
  const double expert_bytes = 4.0 * (2.0 * I * H + 2.0 * I + 1.0 * I * H + H);  // per expert
  std::printf("weights built in %.1f s (%.2f GB of expert weights, %.1f MB per expert)\n\n",
              ms_since(tw) / 1e3, E * expert_bytes / 1e9, expert_bytes / 1e6);

  KernelRegistry reg;
  register_builtin_kernels(reg);
  FILE* csv = csv_path.empty() ? nullptr : std::fopen(csv_path.c_str(), "w");
  if (csv)
    std::fprintf(csv, "routing,T,experts_touched,layer_p50_ms,layer_p95_ms,grouped_p50_ms,"
                      "naive_p50_ms,speedup,derived_weight_GBps,max_abs,max_rel,normwise_rel,"
                      "arena_MB,intermediates_sum_MB\n");

  for (const std::string& routing : routings) {
    std::printf("== routing: %s ==\n", routing == "uniform"
                                           ? "uniform (random router weights)"
                                           : "same4 (every token -> experts {1,3,4,6})");
    std::printf("%5s %4s | %9s %9s | %10s %10s %7s | %9s | %9s %9s %9s | %8s %8s\n", "T", "U",
                "layer p50", "p95 ms", "grouped", "naive ms", "speedup", "GB/s(der)",
                "max_abs", "max_rel", "norm_rel", "arena MB", "sum MB");
    for (const int64_t T : Ts) {
      const int iters = iters_override > 0 ? iters_override : (T <= 16 ? 20 : T <= 64 ? 10 : 5);
      const MoEDims d{T, H, I, E, K_TOP};
      mtrt::testing::MoEGraph mg = mtrt::testing::build_moe_graph(d);
      Executor exec(mg.graph, reg);
      Tensor x = Tensor::owning(DType::kF32, {T, H});
      fill(x, 100 + static_cast<uint64_t>(T), 1.7f);
      std::unordered_map<TensorId, Tensor> b{
          {mg.x, x}, {mg.norm_scale, W.norm_scale}, {mg.gate_wT, W.gate_wT},
          {mg.gate_b, routing == "uniform" ? W.gate_b_uniform : W.gate_b_same4},
          {mg.w1t, W.w1t}, {mg.b1, W.b1}, {mg.w2t, W.w2t}, {mg.b2, W.b2}};

      // 1. whole layer, planned executor
      for (int i = 0; i < warmup; ++i) exec.run(b);
      std::vector<double> lat;
      for (int i = 0; i < iters; ++i) {
        const auto t0 = Clock::now();
        exec.run(b);
        lat.push_back(ms_since(t0));
      }
      Profiler prof;
      exec.set_profiler(&prof);
      std::map<std::string, std::vector<double>> per_op;
      for (int i = 0; i < std::min(iters, 5); ++i) {
        prof.clear();
        exec.run(b);
        std::map<std::string, double> sum;
        for (const auto& ev : prof.events()) sum[ev.name] += ev.dur_ns / 1e6;
        for (const auto& [k, v] : sum) per_op[k].push_back(v);
      }
      exec.set_profiler(nullptr);

      // 2. expert-section ablation on identical inputs (t, idx, w from the real kernels)
      auto F = [](std::vector<int64_t> s) { return Tensor::owning(DType::kF32, std::move(s)); };
      auto Ii = [](std::vector<int64_t> s) { return Tensor::owning(DType::kI32, std::move(s)); };
      const int64_t Tk = T * K_TOP;
      Tensor t = F({T, H}), z = F({T, E}), logits = F({T, E}), w = F({T, K_TOP}),
             idx = Ii({T, K_TOP}), ts = F({Tk, H}), src = Ii({Tk}), inv = Ii({Tk}),
             off = Ii({E + 1}), h = F({Tk, 2 * I}), a = F({Tk, I}), y = F({Tk, H}),
             moe_g = F({T, H}), moe_n = F({T, H});
      auto op = [&](const char* name, DType dt, std::unordered_map<std::string, Attribute> at) {
        return Op{reg.lookup(name, dt), Node{name, {}, {}, std::move(at)}};
      };
      const Op rms = op("RMSNorm", DType::kF32, {{"eps", 1e-5}});
      const Op mm = op("MatMul", DType::kF32, {});
      const Op add = op("Add", DType::kF32, {});
      const Op topk = op("MoETopK", DType::kF32, {{"k", K_TOP}});
      const Op disp = op("MoEDispatch", DType::kF32, {{"num_experts", E}});
      const Op gmm = op("MoEGroupedMatMul", DType::kF32, {});
      const Op swi = op("SwiGLUInterleaved", DType::kF32,
                        {{"alpha", double{kAlpha}}, {"limit", double{kLimit}}});
      const Op comb = op("MoECombine", DType::kF32, {});
      const Tensor& gb = routing == "uniform" ? W.gate_b_uniform : W.gate_b_same4;
      call(rms, {&x, &W.norm_scale}, {&t});
      call(mm, {&t, &W.gate_wT}, {&z});
      call(add, {&z, &gb}, {&logits});
      call(topk, {&logits}, {&w, &idx});

      auto grouped = [&] {
        call(disp, {&t, &idx}, {&ts, &src, &inv, &off});
        call(gmm, {&ts, &off, &W.w1t, &W.b1}, {&h});
        call(swi, {&h}, {&a});
        call(gmm, {&a, &off, &W.w2t, &W.b2}, {&y});
        call(comb, {&y, &inv, &w}, {&moe_g});
      };
      std::vector<float> hrow(2 * I), arow(I), yrow(H);
      auto naive = [&] { naive_experts(t, idx, w, W, moe_n, hrow, arow, yrow); };
      std::vector<double> tg, tn;
      for (int i = 0; i < warmup; ++i) { grouped(); naive(); }
      for (int i = 0; i < iters; ++i) {  // interleaved, so drift hits both equally
        auto t0 = Clock::now();
        grouped();
        tg.push_back(ms_since(t0));
        t0 = Clock::now();
        naive();
        tn.push_back(ms_since(t0));
      }

      int64_t U = 0;
      for (int64_t e = 0; e < E; ++e) U += off.data<int32_t>()[e + 1] > off.data<int32_t>()[e];
      double max_abs = 0, max_rel = 0, ref_max = 0;
      for (int64_t i = 0; i < T * H; ++i) {
        const double r = moe_n.data<float>()[i];
        const double dd = std::abs(static_cast<double>(moe_g.data<float>()[i]) - r);
        max_abs = std::max(max_abs, dd);
        ref_max = std::max(ref_max, std::abs(r));
        if (std::abs(r) >= 0.1) max_rel = std::max(max_rel, dd / std::abs(r));  // atol/rtol
      }
      const double g50 = pct(tg, 0.5), n50 = pct(tn, 0.5);
      const double gbps = U * expert_bytes / (g50 / 1e3) / 1e9;
      const MemoryStats ms = exec.memory_stats();
      const double arena_mb = ms.peak_bytes / 1e6;
      const double sum_mb = (ms.peak_bytes + ms.bytes_reused) / 1e6;

      std::printf("%5lld %4lld | %9.3f %9.3f | %10.3f %10.3f %6.2fx | %9.1f | %9.2e %9.2e %9.2e"
                  " | %8.2f %8.2f\n",
                  (long long)T, (long long)U, pct(lat, 0.5), pct(lat, 0.95), g50, n50, n50 / g50,
                  gbps, max_abs, max_rel, max_abs / ref_max, arena_mb, sum_mb);
      std::printf("        per-op median ms:");
      for (const auto& [k, v] : per_op) std::printf(" %s=%.3f", k.c_str(), pct(v, 0.5));
      std::printf("\n");
      if (csv)
        std::fprintf(csv, "%s,%lld,%lld,%.4f,%.4f,%.4f,%.4f,%.3f,%.2f,%.3e,%.3e,%.3e,%.3f,%.3f\n",
                     routing.c_str(), (long long)T, (long long)U, pct(lat, 0.5), pct(lat, 0.95),
                     g50, n50, n50 / g50, gbps, max_abs, max_rel, max_abs / ref_max, arena_mb,
                     sum_mb);
    }
    std::printf("\n");
  }
  if (csv) std::fclose(csv);
  return 0;
}
