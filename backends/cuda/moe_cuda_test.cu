// GPU MoE tests (Milestone 1B, before any variant is timed; GPU_DISPATCH_DESIGN.md §8).
//
//   1. Per-kernel goldens: each GPU kernel gets its stage's GOLDEN inputs, for all 5 MoE cases.
//      Index outputs (idx, src, inv, offsets) and t_sorted (a pure copy) must be EXACT;
//      float outputs use the combined tolerance with max abs / max rel reported (D-MoE-10).
//   2. Dispatch vs the CPU MoEDispatch kernel, EXACT, on random routings (uniform, same-4,
//      T = 1 .. 4096, E = 8 / 32 / 128, Tk not a multiple of the block size), with the device
//      outputs poisoned (0xFF) before every launch so an unwritten slot cannot hide.
//   3. Top-k vs the CPU MoETopK kernel on tie-heavy integer logits: idx EXACT.
//   4. Whole layer, V0 and V0-oracle: out vs golden (term-magnitude gate), V0 == V0-oracle
//      bitwise, 10 repeated runs bitwise identical, and forced routing == model routing.
//
// Run: ./build/backends/cuda/moe_cuda_test   (exit 0 = all pass). Also under compute-sanitizer.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "backends/cuda/moe.h"
#include "backends/cuda/moe_layer.h"
#include "mtrt/node.h"
#include "mtrt/op_context.h"
#include "mtrt/registry.h"
#include "mtrt/tensor.h"
#include "tests/support/close_stats.h"
#include "tests/support/moe_tolerance.h"
#include "tests/support/npy.h"

using namespace mtrt;
using namespace mtrt::testing;
namespace gm = mtrt::cuda::moe;

namespace {

int g_fail = 0, g_pass = 0;
constexpr float kRtol = 1e-5f, kAtol = 1e-6f;

#define CK(call)                                                             \
  do {                                                                       \
    const cudaError_t e__ = (call);                                          \
    if (e__ != cudaSuccess) {                                                \
      std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e__),       \
                  __FILE__, __LINE__);                                       \
      std::exit(2);                                                          \
    }                                                                        \
  } while (0)

void report(const std::string& name, bool ok, const std::string& detail = "") {
  std::printf("[MOE-CUDA] %-46s %s  %s\n", name.c_str(), ok ? "PASS" : "FAIL", detail.c_str());
  (ok ? g_pass : g_fail)++;
}

template <class Tv>
Tv* up(const std::vector<Tv>& h) {
  Tv* d = nullptr;
  CK(cudaMalloc(&d, std::max<size_t>(h.size(), 1) * sizeof(Tv)));
  CK(cudaMemcpy(d, h.data(), h.size() * sizeof(Tv), cudaMemcpyHostToDevice));
  return d;
}
template <class Tv>
Tv* dnew(size_t n) {
  Tv* d = nullptr;
  CK(cudaMalloc(&d, std::max<size_t>(n, 1) * sizeof(Tv)));
  CK(cudaMemset(d, 0xFF, std::max<size_t>(n, 1) * sizeof(Tv)));  // poisoned
  return d;
}
template <class Tv>
std::vector<Tv> down(const Tv* d, size_t n) {
  std::vector<Tv> h(n);
  CK(cudaMemcpy(h.data(), d, n * sizeof(Tv), cudaMemcpyDeviceToHost));
  return h;
}

std::vector<float> GF(const std::string& c, const std::string& n) {
  return load_golden("moe_" + c + "_" + n).data;
}
std::vector<int> GI(const std::string& c, const std::string& n) {
  return load_golden_i32("moe_" + c + "_" + n).idata;
}

std::string stats_str(const CloseStats& s) {
  char buf[160];
  std::snprintf(buf, sizeof(buf), "max_abs=%.3e max_rel=%.3e near0=%lld tol_used=%.3f",
                s.max_abs, s.max_rel, (long long)s.near_zero, s.worst);
  return buf;
}

void expect_close(const std::string& name, const std::vector<float>& got,
                  const std::vector<float>& ref, const float* mag = nullptr) {
  if (got.size() != ref.size()) return report(name, false, "size mismatch");
  const CloseStats s =
      close_stats(got.data(), ref.data(), (int64_t)got.size(), kRtol, kAtol, mag);
  report(name, s.worst <= 1.0, stats_str(s));
}

template <class Tv>
void expect_exact(const std::string& name, const std::vector<Tv>& got,
                  const std::vector<Tv>& ref) {
  bool ok = got.size() == ref.size() &&
            std::memcmp(got.data(), ref.data(), got.size() * sizeof(Tv)) == 0;
  long first = -1;
  for (size_t i = 0; ok == false && i < std::min(got.size(), ref.size()); ++i)
    if (std::memcmp(&got[i], &ref[i], sizeof(Tv)) != 0) { first = (long)i; break; }
  report(name, ok, ok ? "exact" : "first mismatch at " + std::to_string(first));
}

// ---- CPU reference kernels (the registered MoE ops in src/ops/kernels_moe.cpp) -------------
KernelRegistry& cpu_reg() {
  static KernelRegistry r = [] {
    KernelRegistry k;
    register_builtin_kernels(k);
    return k;
  }();
  return r;
}
void cpu_op(const char* op, std::vector<const Tensor*> in, std::vector<Tensor*> out,
            std::unordered_map<std::string, Attribute> attrs) {
  KernelFn fn = cpu_reg().lookup(op, out[0]->dtype());
  if (!fn) { std::printf("no CPU kernel %s\n", op); std::exit(2); }
  Node node{op, {}, {}, std::move(attrs)};
  OpContext ctx{in, out, node};
  fn(ctx);
}

// ---- 1. per-kernel goldens -----------------------------------------------------------------
void per_kernel_goldens(const std::string& c, cudaStream_t s, cublasHandle_t cb) {
  const NpyArray x = load_golden("moe_" + c + "_x");
  const NpyArray w1t = load_golden("moe_" + c + "_w1t"), w2t = load_golden("moe_" + c + "_w2t");
  const NpyArray idxg = load_golden_i32("moe_" + c + "_idx");
  const int T = (int)x.shape[0], H = (int)x.shape[1], E = (int)w1t.shape[0];
  const int I = (int)w2t.shape[1], k = (int)idxg.shape[1], Tk = T * k;
  const std::string p = c + ": ";

  // RMSNorm(golden x)
  float* dx = up(x.data); float* dscale = up(GF(c, "norm_scale"));
  float* dt = dnew<float>((size_t)T * H);
  gm::rmsnorm(dx, dscale, dt, T, H, 1e-5f, s);
  // router(golden t)
  float* dtg = up(GF(c, "t")); float* dgw = up(GF(c, "gate_wT")); float* dgb = up(GF(c, "gate_b"));
  float* dlog = dnew<float>((size_t)T * E);
  gm::sgemm_rowmajor(cb, dtg, dgw, dlog, T, E, H);
  gm::add_row_bias(dlog, dgb, T, E, s);
  // top-k(golden logits)
  float* dlg = up(GF(c, "logits"));
  float* dw = dnew<float>(Tk); int* didx = dnew<int>(Tk);
  gm::topk_softmax(dlg, dw, didx, T, E, k, s);
  // dispatch(golden idx)
  int* didxg = up(idxg.idata);
  int* dsrc = dnew<int>(Tk); int* dinv = dnew<int>(Tk); int* doff = dnew<int>(E + 1);
  gm::dispatch(didxg, dsrc, dinv, doff, Tk, E, s);
  // gather(golden t, golden src)
  int* dsrcg = up(GI(c, "src"));
  float* dts = dnew<float>((size_t)Tk * H);
  gm::gather_rows(dtg, dsrcg, dts, Tk, H, k, s);
  // expert GEMM 1 (+bias) on golden t_sorted with golden offsets, the V0 host loop
  const std::vector<int> offg = GI(c, "offsets");
  float* dtsg = up(GF(c, "t_sorted")); float* dw1 = up(w1t.data); float* db1 = up(GF(c, "b1"));
  float* dh = dnew<float>((size_t)Tk * 2 * I);
  for (int e = 0; e < E; ++e)
    if (offg[e + 1] > offg[e])
      gm::sgemm_rowmajor(cb, dtsg + (size_t)offg[e] * H, dw1 + (size_t)e * H * 2 * I,
                         dh + (size_t)offg[e] * 2 * I, offg[e + 1] - offg[e], 2 * I, H);
  gm::add_expert_bias(dh, db1, didxg, dsrcg, Tk, 2 * I, s);
  // SwiGLU(golden h)
  float* dhg = up(GF(c, "h")); float* da = dnew<float>((size_t)Tk * I);
  gm::swiglu_interleaved(dhg, da, Tk, I, 1.702f, 7.0f, s);
  // expert GEMM 2 (+bias) on golden a
  float* dag = up(GF(c, "a")); float* dw2 = up(w2t.data); float* db2 = up(GF(c, "b2"));
  float* dy = dnew<float>((size_t)Tk * H);
  for (int e = 0; e < E; ++e)
    if (offg[e + 1] > offg[e])
      gm::sgemm_rowmajor(cb, dag + (size_t)offg[e] * I, dw2 + (size_t)e * I * H,
                         dy + (size_t)offg[e] * H, offg[e + 1] - offg[e], H, I);
  gm::add_expert_bias(dy, db2, didxg, dsrcg, Tk, H, s);
  // combine(golden y_sorted, inv, w) and residual
  float* dyg = up(GF(c, "y_sorted")); int* dinvg = up(GI(c, "inv")); float* dwg = up(GF(c, "w"));
  float* dm = dnew<float>((size_t)T * H);
  gm::combine(dyg, dinvg, dwg, dm, T, H, k, s);
  float* dmg = up(GF(c, "moe")); float* dout = dnew<float>((size_t)T * H);
  gm::add(dx, dmg, dout, T * H, s);
  CK(cudaStreamSynchronize(s));

  expect_close(p + "rmsnorm", down(dt, (size_t)T * H), GF(c, "t"));
  expect_close(p + "router", down(dlog, (size_t)T * E), GF(c, "logits"));
  expect_exact(p + "topk idx", down(didx, Tk), GI(c, "idx"));
  expect_close(p + "topk w", down(dw, Tk), GF(c, "w"));
  expect_exact(p + "dispatch src", down(dsrc, Tk), GI(c, "src"));
  expect_exact(p + "dispatch inv", down(dinv, Tk), GI(c, "inv"));
  expect_exact(p + "dispatch offsets", down(doff, E + 1), offg);
  expect_exact(p + "gather t_sorted (bitwise)", down(dts, (size_t)Tk * H), GF(c, "t_sorted"));
  expect_close(p + "gemm1+bias h", down(dh, (size_t)Tk * 2 * I), GF(c, "h"));
  expect_close(p + "swiglu a", down(da, (size_t)Tk * I), GF(c, "a"));
  expect_close(p + "gemm2+bias y_sorted", down(dy, (size_t)Tk * H), GF(c, "y_sorted"));
  expect_close(p + "combine moe", down(dm, (size_t)T * H), GF(c, "moe"));
  expect_close(p + "residual out", down(dout, (size_t)T * H), GF(c, "out"));

  for (void* q : std::vector<void*>{dx, dscale, dt, dtg, dgw, dgb, dlog, dlg, dw, didx, didxg,
                                    dsrc, dinv, doff, dsrcg, dts, dtsg, dw1, db1, dh, dhg, da,
                                    dag, dw2, db2, dy, dyg, dinvg, dwg, dm, dmg, dout})
    CK(cudaFree(q));
}

// ---- 2. dispatch vs CPU, exact, random routings ---------------------------------------------
void dispatch_vs_cpu(cudaStream_t s) {
  std::mt19937 rng(11);
  int cases = 0, bad = 0;
  for (int E : {8, 32, 128}) {
    for (int T : {1, 2, 3, 63, 64, 65, 1000, 4096}) {
      for (int same = 0; same < 2; ++same) {
        const int k = 4, Tk = T * k;
        std::vector<int> experts(E);
        std::iota(experts.begin(), experts.end(), 0);
        std::shuffle(experts.begin(), experts.end(), rng);
        std::vector<int> idx(Tk);
        for (int t = 0; t < T; ++t) {
          if (!same) std::shuffle(experts.begin(), experts.end(), rng);
          for (int j = 0; j < k; ++j) idx[t * k + j] = experts[j];
        }
        // CPU reference
        Tensor ti = Tensor::owning(DType::kI32, {T, k});
        std::copy(idx.begin(), idx.end(), ti.data<int32_t>());
        Tensor tok = Tensor::owning(DType::kF32, {T, 1});
        Tensor ts = Tensor::owning(DType::kF32, {Tk, 1});
        Tensor cs = Tensor::owning(DType::kI32, {Tk}), cinv = Tensor::owning(DType::kI32, {Tk});
        Tensor coff = Tensor::owning(DType::kI32, {E + 1});
        cpu_op("MoEDispatch", {&tok, &ti}, {&ts, &cs, &cinv, &coff},
               {{"num_experts", int64_t{E}}});
        // GPU (outputs poisoned by dnew)
        int* di = up(idx);
        int* ds = dnew<int>(Tk); int* dv = dnew<int>(Tk); int* dof = dnew<int>(E + 1);
        gm::dispatch(di, ds, dv, dof, Tk, E, s);
        CK(cudaStreamSynchronize(s));
        const auto gs = down(ds, Tk), gv = down(dv, Tk), go = down(dof, E + 1);
        const bool ok = std::equal(gs.begin(), gs.end(), cs.data<int32_t>()) &&
                        std::equal(gv.begin(), gv.end(), cinv.data<int32_t>()) &&
                        std::equal(go.begin(), go.end(), coff.data<int32_t>());
        if (!ok) {
          ++bad;
          std::printf("  dispatch mismatch: E=%d T=%d same=%d\n", E, T, same);
        }
        ++cases;
        CK(cudaFree(di)); CK(cudaFree(ds)); CK(cudaFree(dv)); CK(cudaFree(dof));
      }
    }
  }
  report("dispatch == CPU kernel (random routings)", bad == 0,
         std::to_string(cases) + " routings, E in {8,32,128}, T up to 4096, poisoned outputs");
}

// ---- 3. top-k vs CPU on ties -----------------------------------------------------------------
void topk_vs_cpu(cudaStream_t s) {
  std::mt19937 rng(5);
  int bad = 0, rows = 0;
  for (int E : {8, 32, 33, 128}) {
    const int T = 512, k = 4;
    std::vector<float> lg((size_t)T * E);
    for (float& v : lg) v = (float)((int)(rng() % 7) - 3);  // integers -> many ties
    Tensor tl = Tensor::owning(DType::kF32, {T, E});
    std::copy(lg.begin(), lg.end(), tl.data<float>());
    Tensor cw = Tensor::owning(DType::kF32, {T, k}), ci = Tensor::owning(DType::kI32, {T, k});
    cpu_op("MoETopK", {&tl}, {&cw, &ci}, {{"k", int64_t{k}}});
    float* dl = up(lg); float* dw = dnew<float>((size_t)T * k); int* di = dnew<int>((size_t)T * k);
    gm::topk_softmax(dl, dw, di, T, E, k, s);
    CK(cudaStreamSynchronize(s));
    const auto gi = down(di, (size_t)T * k);
    for (int t = 0; t < T; ++t) {
      ++rows;
      if (!std::equal(gi.begin() + t * k, gi.begin() + (t + 1) * k, ci.data<int32_t>() + t * k))
        ++bad;
    }
    CK(cudaFree(dl)); CK(cudaFree(dw)); CK(cudaFree(di));
  }
  report("topk idx == CPU kernel (tie-heavy)", bad == 0,
         std::to_string(rows) + " rows, E in {8,32,33,128}, " + std::to_string(bad) + " bad");
}

// ---- 4. whole layer: V0, V0-oracle, determinism, forced routing ------------------------------
void layer_case(const std::string& c) {
  const NpyArray x = load_golden("moe_" + c + "_x");
  const NpyArray w1t = load_golden("moe_" + c + "_w1t"), w2t = load_golden("moe_" + c + "_w2t");
  const int T = (int)x.shape[0], H = (int)x.shape[1], E = (int)w1t.shape[0];
  const int I = (int)w2t.shape[1], k = (int)load_golden_i32("moe_" + c + "_idx").shape[1];
  const auto ns = GF(c, "norm_scale"), gw = GF(c, "gate_wT"), gb = GF(c, "gate_b"),
             b1 = GF(c, "b1"), b2 = GF(c, "b2");
  cuda::MoEWeightsHost wh{ns.data(), gw.data(), gb.data(), w1t.data.data(), b1.data(),
                          w2t.data.data(), b2.data()};
  cuda::MoELayerCuda layer({T, H, I, E, k}, wh);
  layer.set_input(x.data.data());
  const size_t n = (size_t)T * H;
  const std::vector<float> mag = layer_term_magnitude(c);
  const std::vector<float> ref = GF(c, "out");

  layer.enqueue(cuda::MoEVariant::kV0);
  CK(cudaStreamSynchronize(layer.stream()));
  const auto out_v0 = layer.download_f(layer.buffers().out, n);
  expect_close(c + ": layer V0 out (term-magnitude gate)", out_v0, ref, mag.data());
  expect_exact(c + ": layer V0 routing idx", layer.download_i(layer.buffers().idx_model, T * k),
               GI(c, "idx"));

  layer.prepare_oracle();
  layer.enqueue(cuda::MoEVariant::kV0Oracle);
  CK(cudaStreamSynchronize(layer.stream()));
  expect_exact(c + ": V0-oracle == V0 (bitwise)", layer.download_f(layer.buffers().out, n),
               out_v0);

  bool same = true;
  for (int r = 0; r < 10 && same; ++r) {
    layer.enqueue(cuda::MoEVariant::kV0);
    CK(cudaStreamSynchronize(layer.stream()));
    same = layer.download_f(layer.buffers().out, n) == out_v0;
  }
  report(c + ": 10 repeated runs bitwise identical", same);

  // Forced routing = the golden routing must reproduce the same output.
  const auto gi = GI(c, "idx");
  const auto gwt = GF(c, "w");
  layer.set_forced_routing(gi.data(), gwt.data());
  layer.enqueue(cuda::MoEVariant::kV0);
  CK(cudaStreamSynchronize(layer.stream()));
  expect_close(c + ": forced routing (golden idx,w) out", layer.download_f(layer.buffers().out, n),
               ref, mag.data());
}

}  // namespace

int main() {
  int dev = 0;
  CK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, dev));
  std::printf("[MOE-CUDA] device %d: %s (sm_%d%d)\n", dev, prop.name, prop.major, prop.minor);

  cudaStream_t s;
  CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  cublasHandle_t cb;
  if (cublasCreate(&cb) != CUBLAS_STATUS_SUCCESS || cublasSetStream(cb, s) != CUBLAS_STATUS_SUCCESS) {
    std::printf("cuBLAS init failed\n");
    return 2;
  }
  for (const char* c : {"small", "four", "clamp", "e32", "sparse"}) per_kernel_goldens(c, s, cb);
  dispatch_vs_cpu(s);
  topk_vs_cpu(s);
  for (const char* c : {"small", "four", "clamp", "e32", "sparse"}) layer_case(c);
  cublasDestroy(cb);
  CK(cudaStreamDestroy(s));

  std::printf("[MOE-CUDA] %d passed, %d failed -> %s\n", g_pass, g_fail,
              g_fail ? "FAILED" : "all passed");
  return g_fail ? 1 : 0;
}
