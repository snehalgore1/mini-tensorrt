#include "backends/cuda/moe_layer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "backends/cuda/moe.h"

namespace mtrt::cuda {
namespace {

#define ML_CK(call)                                                          \
  do {                                                                       \
    const cudaError_t e__ = (call);                                          \
    if (e__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                   cudaGetErrorString(e__), __FILE__, __LINE__);             \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

#define ML_CB(call)                                                          \
  do {                                                                       \
    const cublasStatus_t s__ = (call);                                       \
    if (s__ != CUBLAS_STATUS_SUCCESS) {                                      \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", static_cast<int>(s__), \
                   __FILE__, __LINE__);                                      \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

// Fixed cuBLAS workspace, so cuBLAS never allocates inside a timed enqueue().
constexpr size_t kCublasWorkspaceBytes = 32u << 20;

}  // namespace

const char* variant_name(MoEVariant v) {
  switch (v) {
    case MoEVariant::kV0: return "V0";
    case MoEVariant::kV0Oracle: return "V0-oracle";
  }
  return "?";
}

const char* stage_name(int s) {
  static const char* names[kNumStages] = {
      "norm", "router", "topk", "dispatch", "gather", "routing_visible", "gemm1",
      "bias1_swiglu", "gemm2", "bias2", "combine", "residual"};
  return s >= 0 && s < kNumStages ? names[s] : "?";
}

void* MoELayerCuda::dalloc(size_t bytes) {
  void* p = nullptr;
  ML_CK(cudaMalloc(&p, bytes > 0 ? bytes : 4));
  ML_CK(cudaMemset(p, 0, bytes > 0 ? bytes : 4));
  allocs_.push_back(p);
  device_bytes_ += static_cast<int64_t>(bytes);
  return p;
}

MoELayerCuda::MoELayerCuda(const MoEDimsCuda& d, const MoEWeightsHost& w, float eps,
                           float alpha, float limit)
    : d_(d), eps_(eps), alpha_(alpha), limit_(limit) {
  const size_t T = d.T, H = d.H, I = d.I, E = d.E, k = d.k, Tk = T * k;
  // A non-blocking stream never synchronizes implicitly with the legacy default stream.
  ML_CK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  ML_CB(cublasCreate(&cublas_));
  ML_CB(cublasSetStream(cublas_, stream_));
  ML_CB(cublasSetMathMode(cublas_, CUBLAS_DEFAULT_MATH));  // plain FP32 SGEMM
  cublas_ws_ = dalloc(kCublasWorkspaceBytes);
  ML_CB(cublasSetWorkspace(cublas_, cublas_ws_, kCublasWorkspaceBytes));

  auto up = [&](const float* host, size_t n) {
    float* p = static_cast<float*>(dalloc(n * sizeof(float)));
    ML_CK(cudaMemcpy(p, host, n * sizeof(float), cudaMemcpyHostToDevice));
    return p;
  };
  norm_scale_ = up(w.norm_scale, H);
  gate_wT_ = up(w.gate_wT, H * E);
  gate_b_ = up(w.gate_b, E);
  w1t_ = up(w.w1t, E * H * 2 * I);
  b1_ = up(w.b1, E * 2 * I);
  w2t_ = up(w.w2t, E * I * H);
  b2_ = up(w.b2, E * H);

  auto F = [&](size_t n) { return static_cast<float*>(dalloc(n * sizeof(float))); };
  auto Ii = [&](size_t n) { return static_cast<int*>(dalloc(n * sizeof(int))); };
  b_.x = F(T * H);
  b_.t = F(T * H);
  b_.logits = F(T * E);
  b_.w_model = F(T * k);
  b_.idx_model = Ii(T * k);
  b_.w_forced = F(T * k);
  b_.idx_forced = Ii(T * k);
  b_.src = Ii(Tk);
  b_.inv = Ii(Tk);
  b_.offsets = Ii(E + 1);
  b_.t_sorted = F(Tk * H);
  b_.h = F(Tk * 2 * I);
  b_.a = F(Tk * I);
  b_.y_sorted = F(Tk * H);
  b_.moe = F(T * H);
  b_.out = F(T * H);

  // void** form: the C API overload exists in every CUDA version (the templated one's default
  // flags argument is not something to rely on).
  ML_CK(cudaMallocHost(reinterpret_cast<void**>(&offsets_pinned_), (E + 1) * sizeof(int)));
  oracle_offsets_.assign(E + 1, 0);
  ML_CK(cudaDeviceSynchronize());  // setup complete before any timed work
}

MoELayerCuda::~MoELayerCuda() {
  cudaStreamSynchronize(stream_);
  for (void* p : allocs_) cudaFree(p);
  if (offsets_pinned_) cudaFreeHost(offsets_pinned_);
  if (cublas_) cublasDestroy(cublas_);
  if (stream_) cudaStreamDestroy(stream_);
}

void MoELayerCuda::set_input(const float* x_host) {
  ML_CK(cudaMemcpy(b_.x, x_host, sizeof(float) * d_.T * d_.H, cudaMemcpyHostToDevice));
  oracle_ready_ = false;  // in model-routing mode the routing may have changed
}

void MoELayerCuda::set_forced_routing(const int* idx_host, const float* w_host) {
  const size_t n = static_cast<size_t>(d_.T) * d_.k;
  for (size_t i = 0; i < n; ++i) {
    if (idx_host[i] < 0 || idx_host[i] >= d_.E) {
      std::fprintf(stderr, "set_forced_routing: expert id out of range\n");
      std::abort();
    }
  }
  ML_CK(cudaMemcpy(b_.idx_forced, idx_host, n * sizeof(int), cudaMemcpyHostToDevice));
  ML_CK(cudaMemcpy(b_.w_forced, w_host, n * sizeof(float), cudaMemcpyHostToDevice));
  forced_ = true;
  oracle_ready_ = false;
}

void MoELayerCuda::clear_forced_routing() {
  forced_ = false;
  oracle_ready_ = false;
}

void MoELayerCuda::prepare_oracle() {
  enqueue(MoEVariant::kV0);  // V0 produces the host copy of offsets as a side effect
  ML_CK(cudaStreamSynchronize(stream_));
  std::memcpy(oracle_offsets_.data(), offsets_pinned_, (d_.E + 1) * sizeof(int));
  oracle_ready_ = true;
}

// One cuBLAS SGEMM per non-empty expert over its contiguous rows. host_off must hold the
// CURRENT routing's offsets (V0: just copied; V0-oracle: from prepare_oracle()).
void MoELayerCuda::expert_gemms(const int* host_off, const float* A, const float* W, float* C,
                                int K, int N) {
  for (int e = 0; e < d_.E; ++e) {
    const int lo = host_off[e], m = host_off[e + 1] - host_off[e];
    if (m == 0) continue;
    moe::sgemm_rowmajor(cublas_, A + static_cast<size_t>(lo) * K,
                        W + static_cast<size_t>(e) * K * N, C + static_cast<size_t>(lo) * N, m,
                        N, K);
  }
}

// SYNC AUDIT (docs/ENGINEERING_DECISIONS.md D-MoE-17). Calls in this function:
//   V0:        kernel launches + cuBLAS (async on stream_), cudaEventRecord (optional),
//              ONE cudaMemcpyAsync D2H (offsets, pinned) + ONE cudaStreamSynchronize.
//   V0-oracle: kernel launches + cuBLAS + cudaEventRecord only. No copy, no sync.
// No cudaMalloc/cudaFree, no synchronous cudaMemcpy, no legacy default stream (stream_ is
// non-blocking; cuBLAS is bound to it and has a fixed workspace).
void MoELayerCuda::enqueue(MoEVariant v, cudaEvent_t* ev) {
  const cudaStream_t s = stream_;
  const int T = d_.T, H = d_.H, I = d_.I, E = d_.E, k = d_.k, Tk = T * k;
  auto rec = [&](int stage) {
    if (ev) ML_CK(cudaEventRecord(ev[stage], s));
  };

  moe::rmsnorm(b_.x, norm_scale_, b_.t, T, H, eps_, s);
  rec(kStNorm);
  moe::sgemm_rowmajor(cublas_, b_.t, gate_wT_, b_.logits, T, E, H);
  moe::add_row_bias(b_.logits, gate_b_, T, E, s);
  rec(kStRouter);
  moe::topk_softmax(b_.logits, b_.w_model, b_.idx_model, T, E, k, s);
  rec(kStTopK);
  moe::dispatch(idx_active(), b_.src, b_.inv, b_.offsets, Tk, E, s);
  rec(kStDispatch);
  moe::gather_rows(b_.t, b_.src, b_.t_sorted, Tk, H, k, s);
  rec(kStGather);

  const int* off = nullptr;
  if (v == MoEVariant::kV0) {
    ML_CK(cudaMemcpyAsync(offsets_pinned_, b_.offsets, sizeof(int) * (E + 1),
                          cudaMemcpyDeviceToHost, s));
    ML_CK(cudaStreamSynchronize(s));  // the host needs the segment sizes to issue the GEMMs
    off = offsets_pinned_;
  } else {
    if (!oracle_ready_) {
      std::fprintf(stderr, "V0-oracle: call prepare_oracle() after any input/routing change\n");
      std::abort();
    }
    off = oracle_offsets_.data();
  }
  rec(kStRoutingVisible);

  expert_gemms(off, b_.t_sorted, w1t_, b_.h, H, 2 * I);
  rec(kStGemm1);
  moe::add_expert_bias(b_.h, b1_, idx_active(), b_.src, Tk, 2 * I, s);
  moe::swiglu_interleaved(b_.h, b_.a, Tk, I, alpha_, limit_, s);
  rec(kStBias1Swiglu);
  expert_gemms(off, b_.a, w2t_, b_.y_sorted, I, H);
  rec(kStGemm2);
  moe::add_expert_bias(b_.y_sorted, b2_, idx_active(), b_.src, Tk, H, s);
  rec(kStBias2);
  moe::combine(b_.y_sorted, b_.inv, w_active(), b_.moe, T, H, k, s);
  rec(kStCombine);
  moe::add(b_.x, b_.moe, b_.out, T * H, s);
  rec(kStResidual);
}

std::vector<float> MoELayerCuda::download_f(const float* dptr, int64_t n) const {
  std::vector<float> h(static_cast<size_t>(n));
  ML_CK(cudaMemcpy(h.data(), dptr, sizeof(float) * n, cudaMemcpyDeviceToHost));
  return h;
}

std::vector<int> MoELayerCuda::download_i(const int* dptr, int64_t n) const {
  std::vector<int> h(static_cast<size_t>(n));
  ML_CK(cudaMemcpy(h.data(), dptr, sizeof(int) * n, cudaMemcpyDeviceToHost));
  return h;
}

}  // namespace mtrt::cuda
