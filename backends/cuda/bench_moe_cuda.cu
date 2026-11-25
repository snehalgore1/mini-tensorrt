// Milestone 1B benchmark: one GPT-OSS MoE layer on the GPU at real gpt-oss-20b MoE dims
// (H = I = 2880, E = 32, k = 4), FP32, GPU 0 only.
//
//   ./build/backends/cuda/bench_moe_cuda [--T 1,4,16,64,256,1024,4096]
//       [--routing uniform,zipf,same4] [--iters 200] [--warmup 20] [--stages]
//       [--check-max-T 64] [--no-scrub] [--sha <git sha>] [--csv path]
//
// Variants so far: V0 and V0-oracle (moe_layer.h). For every (routing, T):
//   - routing is FORCED from one synthetic generator, so both variants see identical expert
//     segments (the router and top-k still run and are timed)
//   - timing: CUDA events around one enqueue() on the layer stream; the host waits for the
//     stop event OUTSIDE the measured interval; iterations interleave V0 / V0-oracle so clock
//     drift hits both; optional 64 MB L2 scrub before each iteration (outside the interval)
//   - p50 / p95 / mean / CV over --iters, after --warmup
//   - host time spent inside enqueue() (for V0 this includes the blocking sync), as weak
//     supporting evidence only
//   - correctness on every row: V0 == V0-oracle bitwise; for T <= --check-max-T also GPU vs the
//     CPU kernels on the same forced routing (max abs / max rel / normwise rel)
//   - --stages: extra instrumented runs with an event after each stage (H5 breakdown)
// Weights are random (fan-in scaled): speed does not depend on their values; correctness is
// covered by moe_cuda_test. All setup (allocation, upload, routing upload) is outside timing.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "backends/cuda/moe_layer.h"
#include "mtrt/node.h"
#include "mtrt/op_context.h"
#include "mtrt/registry.h"
#include "mtrt/tensor.h"

using namespace mtrt;
using cuda::MoEVariant;

namespace {

#define BK(call)                                                             \
  do {                                                                       \
    const cudaError_t e__ = (call);                                          \
    if (e__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e__), \
                   __FILE__, __LINE__);                                      \
      std::exit(2);                                                          \
    }                                                                        \
  } while (0)

constexpr int H = 2880, I = 2880, E = 32, K_TOP = 4;

void fill(std::vector<float>& v, uint64_t seed, float scale) {
  uint64_t s = seed * 0x9E3779B97F4A7C15ull + 1;
  for (float& x : v) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    x = scale * (static_cast<float>(s >> 40) * (2.0f / 16777216.0f) - 1.0f);
  }
}

double pct(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  return v[static_cast<size_t>(p * static_cast<double>(v.size() - 1) + 0.5)];
}
double mean(const std::vector<double>& v) {
  return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}
double cv(const std::vector<double>& v) {
  const double m = mean(v);
  double ss = 0;
  for (double x : v) ss += (x - m) * (x - m);
  return std::sqrt(ss / static_cast<double>(v.size())) / m;
}

// Synthetic routing: idx[T,k] (k distinct experts per token) and w[T,k] (positive, sum 1).
void make_routing(const std::string& kind, int T, uint64_t seed, std::vector<int>& idx,
                  std::vector<float>& w) {
  std::mt19937_64 rng(seed);
  idx.assign(static_cast<size_t>(T) * K_TOP, 0);
  w.assign(static_cast<size_t>(T) * K_TOP, 0.f);
  std::vector<double> pop(E, 1.0);
  if (kind == "zipf") {  // popularity ~ 1/rank^1.2, ranks randomly assigned to experts
    std::vector<int> rank(E);
    std::iota(rank.begin(), rank.end(), 1);
    std::shuffle(rank.begin(), rank.end(), rng);
    for (int e = 0; e < E; ++e) pop[e] = 1.0 / std::pow(static_cast<double>(rank[e]), 1.2);
  }
  std::uniform_real_distribution<double> U(0.0, 1.0);
  for (int t = 0; t < T; ++t) {
    if (kind == "same4") {
      const int fixed[K_TOP] = {1, 3, 4, 6};
      for (int j = 0; j < K_TOP; ++j) idx[t * K_TOP + j] = fixed[j];
    } else {  // k draws without replacement, proportional to pop
      std::vector<double> p = pop;
      for (int j = 0; j < K_TOP; ++j) {
        double total = std::accumulate(p.begin(), p.end(), 0.0), r = U(rng) * total;
        int e = 0;
        while (e < E - 1 && (r -= p[e]) > 0) ++e;
        while (p[e] == 0.0) e = (e + 1) % E;  // guard against rounding at the boundary
        idx[t * K_TOP + j] = e;
        p[e] = 0.0;
      }
    }
    float sum = 0.f, v[K_TOP];
    for (int j = 0; j < K_TOP; ++j) sum += (v[j] = static_cast<float>(0.1 + U(rng)));
    for (int j = 0; j < K_TOP; ++j) w[t * K_TOP + j] = v[j] / sum;
  }
}

// CPU reference on the same forced routing, through the registered CPU kernels.
std::vector<float> cpu_reference(int T, const std::vector<float>& x, const std::vector<int>& idx,
                                 const std::vector<float>& wv,
                                 const std::unordered_map<std::string, std::vector<float>*>& W) {
  static KernelRegistry reg = [] {
    KernelRegistry r;
    register_builtin_kernels(r);
    return r;
  }();
  auto F = [](std::vector<int64_t> s) { return Tensor::owning(DType::kF32, std::move(s)); };
  auto Ii = [](std::vector<int64_t> s) { return Tensor::owning(DType::kI32, std::move(s)); };
  auto from = [&](const std::vector<float>& v, std::vector<int64_t> s) {
    Tensor t = F(std::move(s));
    std::copy(v.begin(), v.end(), t.data<float>());
    return t;
  };
  auto op = [&](const char* name, std::vector<const Tensor*> in, std::vector<Tensor*> out,
                std::unordered_map<std::string, Attribute> at) {
    Node n{name, {}, {}, std::move(at)};
    OpContext ctx{in, out, n};
    reg.lookup(name, out[0]->dtype())(ctx);
  };
  const int64_t Tk = static_cast<int64_t>(T) * K_TOP;
  Tensor tx = from(x, {T, H}), ns = from(*W.at("norm_scale"), {H});
  Tensor w1 = from(*W.at("w1t"), {E, H, 2 * I}), b1 = from(*W.at("b1"), {E, 2 * I});
  Tensor w2 = from(*W.at("w2t"), {E, I, H}), b2 = from(*W.at("b2"), {E, H});
  Tensor ti = Ii({T, K_TOP});
  std::copy(idx.begin(), idx.end(), ti.data<int32_t>());
  Tensor tw = from(wv, {T, K_TOP});
  Tensor t = F({T, H}), ts = F({Tk, H}), src = Ii({Tk}), inv = Ii({Tk}), off = Ii({E + 1});
  Tensor h = F({Tk, 2 * I}), a = F({Tk, I}), y = F({Tk, H}), m = F({T, H}), out = F({T, H});
  op("RMSNorm", {&tx, &ns}, {&t}, {{"eps", 1e-5}});
  op("MoEDispatch", {&t, &ti}, {&ts, &src, &inv, &off}, {{"num_experts", int64_t{E}}});
  op("MoEGroupedMatMul", {&ts, &off, &w1, &b1}, {&h}, {});
  op("SwiGLUInterleaved", {&h}, {&a}, {{"alpha", 1.702}, {"limit", 7.0}});
  op("MoEGroupedMatMul", {&a, &off, &w2, &b2}, {&y}, {});
  op("MoECombine", {&y, &inv, &tw}, {&m}, {});
  op("Add", {&tx, &m}, {&out}, {});
  return std::vector<float>(out.data<float>(), out.data<float>() + out.numel());
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) v.push_back(tok);
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> Ts = {1, 4, 16, 64, 256, 1024, 4096};
  std::vector<std::string> routings = {"uniform", "zipf", "same4"};
  int iters = 200, warmup = 20, check_max_T = 64;
  bool stages = false, scrub = true;
  std::string sha = "unknown", csv_path;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&] { return std::string(argv[++i]); };
    if (a == "--T") { Ts.clear(); for (auto& s : split(next())) Ts.push_back(std::stoi(s)); }
    else if (a == "--routing") routings = split(next());
    else if (a == "--iters") iters = std::stoi(next());
    else if (a == "--warmup") warmup = std::stoi(next());
    else if (a == "--check-max-T") check_max_T = std::stoi(next());
    else if (a == "--stages") stages = true;
    else if (a == "--no-scrub") scrub = false;
    else if (a == "--sha") sha = next();
    else if (a == "--csv") csv_path = next();
  }

  // ---- environment header -------------------------------------------------------------------
  int dev = 0, drv = 0, rt = 0, sm_clk = 0, mem_clk = 0, bus = 0, sms = 0;
  BK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  BK(cudaGetDeviceProperties(&prop, dev));
  BK(cudaDriverGetVersion(&drv));
  BK(cudaRuntimeGetVersion(&rt));
  BK(cudaDeviceGetAttribute(&sm_clk, cudaDevAttrClockRate, dev));          // kHz, max
  BK(cudaDeviceGetAttribute(&mem_clk, cudaDevAttrMemoryClockRate, dev));   // kHz, max
  BK(cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, dev));  // bits
  BK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  const char* vis = std::getenv("CUDA_VISIBLE_DEVICES");
  char header[1024];
  std::snprintf(header, sizeof(header),
                "# device=%s sm_%d%d SMs=%d max_sm_clock_MHz=%d mem_clock_MHz=%d bus_bits=%d "
                "spec_bw_GBps=%.1f driver=%d runtime=%d CUDA_VISIBLE_DEVICES=%s sha=%s\n"
                "# dims H=%d I=%d E=%d k=%d fp32 iters=%d warmup=%d l2_scrub=%d\n",
                prop.name, prop.major, prop.minor, sms, sm_clk / 1000, mem_clk / 1000, bus,
                2.0 * mem_clk * 1e3 * (bus / 8.0) / 1e9, drv, rt, vis ? vis : "(unset)",
                sha.c_str(), H, I, E, K_TOP, iters, warmup, scrub ? 1 : 0);
  std::printf("%s", header);
  FILE* csv = csv_path.empty() ? nullptr : std::fopen(csv_path.c_str(), "w");
  if (csv) {
    std::fputs(header, csv);
    std::fputs("routing,T,experts_touched,variant,p50_ms,p95_ms,mean_ms,cv,host_enqueue_p50_ms,"
               "v0_eq_oracle,max_abs,max_rel,normwise_rel,layer_device_MB,used_device_MB\n", csv);
  }

  // ---- weights (host, runtime layout), generated once ----------------------------------------
  std::vector<float> norm_scale(H, 1.0f), gate_wT((size_t)H * E), gate_b(E, 0.f),
      w1t((size_t)E * H * 2 * I), b1((size_t)E * 2 * I), w2t((size_t)E * I * H), b2((size_t)E * H);
  fill(gate_wT, 1, std::sqrt(3.0f / H));
  fill(w1t, 2, std::sqrt(3.0f / H));
  fill(b1, 3, 0.1f);
  fill(w2t, 4, std::sqrt(3.0f / I));
  fill(b2, 5, 0.1f);
  const cuda::MoEWeightsHost wh{norm_scale.data(), gate_wT.data(), gate_b.data(), w1t.data(),
                                b1.data(), w2t.data(), b2.data()};
  const std::unordered_map<std::string, std::vector<float>*> Wmap{
      {"norm_scale", &norm_scale}, {"w1t", &w1t}, {"b1", &b1}, {"w2t", &w2t}, {"b2", &b2}};

  void* scrub_buf = nullptr;
  const size_t kScrub = 64u << 20;  // > T4's 4 MB L2
  BK(cudaMalloc(&scrub_buf, kScrub));
  const MoEVariant variants[2] = {MoEVariant::kV0, MoEVariant::kV0Oracle};

  for (const int T : Ts) {
    size_t free0 = 0, total = 0;
    BK(cudaMemGetInfo(&free0, &total));
    cuda::MoELayerCuda layer({T, H, I, E, K_TOP}, wh);
    size_t free1 = 0;
    BK(cudaMemGetInfo(&free1, &total));
    const double layer_mb = layer.device_bytes() / 1e6;
    const double used_mb = (total - free1) / 1e6;
    cudaEvent_t ev0, ev1, st[cuda::kNumStages];
    BK(cudaEventCreate(&ev0));
    BK(cudaEventCreate(&ev1));
    for (auto& e : st) BK(cudaEventCreate(&e));

    std::vector<float> x((size_t)T * H);
    fill(x, 1000 + T, 1.7f);
    layer.set_input(x.data());

    for (const std::string& routing : routings) {
      std::vector<int> idx;
      std::vector<float> wv;
      make_routing(routing, T, 77 + T, idx, wv);
      layer.set_forced_routing(idx.data(), wv.data());
      layer.prepare_oracle();
      std::vector<int> cnt(E, 0);
      for (int e : idx) cnt[e]++;
      const int U = (int)std::count_if(cnt.begin(), cnt.end(), [](int c) { return c > 0; });

      // correctness for this row
      layer.enqueue(MoEVariant::kV0);
      BK(cudaStreamSynchronize(layer.stream()));
      const auto out_v0 = layer.download_f(layer.buffers().out, (int64_t)T * H);
      layer.enqueue(MoEVariant::kV0Oracle);
      BK(cudaStreamSynchronize(layer.stream()));
      const bool eq = layer.download_f(layer.buffers().out, (int64_t)T * H) == out_v0;
      double max_abs = -1, max_rel = -1, norm_rel = -1;
      if (T <= check_max_T) {
        const auto ref = cpu_reference(T, x, idx, wv, Wmap);
        double rmax = 0;
        max_abs = 0;
        max_rel = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
          const double d = std::fabs((double)out_v0[i] - ref[i]);
          max_abs = std::max(max_abs, d);
          rmax = std::max(rmax, (double)std::fabs(ref[i]));
          if (std::fabs(ref[i]) >= 0.1) max_rel = std::max(max_rel, d / std::fabs(ref[i]));
        }
        norm_rel = max_abs / rmax;
      }

      // timing, interleaved
      std::vector<double> ms[2], host[2];
      for (int it = 0; it < warmup + iters; ++it) {
        for (int vi = 0; vi < 2; ++vi) {
          if (scrub) BK(cudaMemsetAsync(scrub_buf, it & 0xff, kScrub, layer.stream()));
          BK(cudaEventRecord(ev0, layer.stream()));
          const auto h0 = std::chrono::steady_clock::now();
          layer.enqueue(variants[vi]);
          const auto h1 = std::chrono::steady_clock::now();
          BK(cudaEventRecord(ev1, layer.stream()));
          BK(cudaEventSynchronize(ev1));  // outside the measured [ev0, ev1] interval
          float t_ms = 0.f;
          BK(cudaEventElapsedTime(&t_ms, ev0, ev1));
          if (it >= warmup) {
            ms[vi].push_back(t_ms);
            host[vi].push_back(std::chrono::duration<double, std::milli>(h1 - h0).count());
          }
        }
      }
      for (int vi = 0; vi < 2; ++vi) {
        std::printf("%-8s T=%5d U=%2d %-9s p50=%9.3f ms p95=%9.3f mean=%9.3f cv=%.3f "
                    "host_enq_p50=%8.3f ms  v0==oracle:%s  max_abs=%.2e max_rel=%.2e norm_rel=%.2e"
                    "  layer=%.0f MB used=%.0f MB\n",
                    routing.c_str(), T, U, cuda::variant_name(variants[vi]), pct(ms[vi], 0.5),
                    pct(ms[vi], 0.95), mean(ms[vi]), cv(ms[vi]), pct(host[vi], 0.5),
                    eq ? "yes" : "NO", max_abs, max_rel, norm_rel, layer_mb, used_mb);
        if (csv)
          std::fprintf(csv, "%s,%d,%d,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%.3e,%.3e,%.3e,%.1f,%.1f\n",
                       routing.c_str(), T, U, cuda::variant_name(variants[vi]), pct(ms[vi], 0.5),
                       pct(ms[vi], 0.95), mean(ms[vi]), cv(ms[vi]), pct(host[vi], 0.5), eq ? 1 : 0,
                       max_abs, max_rel, norm_rel, layer_mb, used_mb);
      }
      std::printf("%-8s T=%5d  V0 - V0-oracle (p50) = %.3f ms\n", routing.c_str(), T,
                  pct(ms[0], 0.5) - pct(ms[1], 0.5));

      if (stages) {  // instrumented runs: median duration of each stage
        for (int vi = 0; vi < 2; ++vi) {
          std::vector<std::vector<double>> per(cuda::kNumStages);
          for (int it = 0; it < 30; ++it) {
            if (scrub) BK(cudaMemsetAsync(scrub_buf, 0, kScrub, layer.stream()));
            BK(cudaEventRecord(ev0, layer.stream()));
            layer.enqueue(variants[vi], st);
            BK(cudaEventSynchronize(st[cuda::kNumStages - 1]));
            float prev = 0.f;
            for (int sidx = 0; sidx < cuda::kNumStages; ++sidx) {
              float t_ms = 0.f;
              BK(cudaEventElapsedTime(&t_ms, ev0, st[sidx]));
              if (it >= 5) per[sidx].push_back(t_ms - prev);
              prev = t_ms;
            }
          }
          std::printf("  stages %-9s:", cuda::variant_name(variants[vi]));
          for (int sidx = 0; sidx < cuda::kNumStages; ++sidx)
            std::printf(" %s=%.3f", cuda::stage_name(sidx), pct(per[sidx], 0.5));
          std::printf(" (ms, median)\n");
        }
      }
    }
    BK(cudaEventDestroy(ev0));
    BK(cudaEventDestroy(ev1));
    for (auto& e : st) BK(cudaEventDestroy(e));
    (void)free0;
  }
  BK(cudaFree(scrub_buf));
  if (csv) std::fclose(csv);
  return 0;
}
