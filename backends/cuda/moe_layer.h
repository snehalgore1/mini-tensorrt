#pragma once

// MoELayerCuda: one GPT-OSS MoE layer on the GPU for the Milestone 1B study
// (docs/moe/GPU_DISPATCH_DESIGN.md §7, approved at Checkpoint 3).
//
// Owns every device buffer and ONE copy of the weights, shared by all variants. Everything is
// allocated and uploaded in the constructor; enqueue() never allocates, never copies weights,
// and (except V0's one intended sync) never synchronizes.
//
// Variants implemented so far (Checkpoint 4, one variant per approval):
//   V0         device router + top-k + dispatch + gather, then D2H copy of offsets[E+1] into
//              pinned host memory + cudaStreamSynchronize, then a HOST loop issuing one cuBLAS
//              SGEMM per non-empty expert (x2 projections).
//   V0-oracle  identical kernels and identical cuBLAS calls in identical order, but the host
//              loop reads offsets that prepare_oracle() copied OUTSIDE the timed region, so the
//              D2H copy and the stream sync are absent. V0 - V0-oracle = the cost of making the
//              routing host-visible (D2H + sync + the GPU idling while the host waits). It is a
//              measurement device, not a deployable path: it is only valid while the routing is
//              unchanged since prepare_oracle().
//
// Routing source: by default dispatch consumes the router's own top-k output. For the study,
// set_forced_routing() uploads synthetic (idx, w) once; the router and top-k STILL run every
// time (their cost stays in the measurement), but dispatch / bias / combine read the forced
// buffers, so every variant sees identical expert segments.

#include <cstdint>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace mtrt::cuda {

struct MoEDimsCuda {
  int T, H, I, E, k;
};

// Host weights in the runtime layout (docs/ENGINEERING_DECISIONS.md D-MoE-3).
struct MoEWeightsHost {
  const float* norm_scale;  // [H]
  const float* gate_wT;     // [H, E]
  const float* gate_b;      // [E]
  const float* w1t;         // [E, H, 2I]
  const float* b1;          // [E, 2I]
  const float* w2t;         // [E, I, H]
  const float* b2;          // [E, H]
};

enum class MoEVariant { kV0, kV0Oracle };
const char* variant_name(MoEVariant v);

// Stage boundaries for the instrumented (H5) runs. Events are recorded on the layer stream
// AFTER each stage; the caller records its own start event before enqueue().
enum MoEStage {
  kStNorm, kStRouter, kStTopK, kStDispatch, kStGather, kStRoutingVisible, kStGemm1,
  kStBias1Swiglu, kStGemm2, kStBias2, kStCombine, kStResidual, kNumStages
};
const char* stage_name(int s);

class MoELayerCuda {
 public:
  MoELayerCuda(const MoEDimsCuda& d, const MoEWeightsHost& w, float eps = 1e-5f,
               float alpha = 1.702f, float limit = 7.0f);
  ~MoELayerCuda();
  MoELayerCuda(const MoELayerCuda&) = delete;
  MoELayerCuda& operator=(const MoELayerCuda&) = delete;

  // Setup-time (synchronous) uploads. Not for timed regions.
  void set_input(const float* x_host);
  void set_forced_routing(const int* idx_host, const float* w_host);  // [T,k] each
  void clear_forced_routing();

  // V0-oracle only: runs the layer once and copies offsets to the host (synchronously), so the
  // timed V0-oracle run can skip the D2H + sync. Call after any input/routing change.
  void prepare_oracle();

  // The timed path: enqueue one layer on stream(). If `stage_events` is non-null it must hold
  // kNumStages events; each is recorded after its stage.
  void enqueue(MoEVariant v, cudaEvent_t* stage_events = nullptr);

  cudaStream_t stream() const { return stream_; }
  const MoEDimsCuda& dims() const { return d_; }

  // Synchronous downloads for tests (after cudaStreamSynchronize(stream())).
  std::vector<float> download_f(const float* dptr, int64_t n) const;
  std::vector<int> download_i(const int* dptr, int64_t n) const;

  // Device buffers, exposed read-only for tests and the benchmark's correctness check.
  struct Buffers {
    float *x, *t, *logits, *w_model, *t_sorted, *h, *a, *y_sorted, *moe, *out;
    int *idx_model, *src, *inv, *offsets;
    float* w_forced;
    int* idx_forced;
  };
  const Buffers& buffers() const { return b_; }
  const float* w_active() const { return forced_ ? b_.w_forced : b_.w_model; }
  const int* idx_active() const { return forced_ ? b_.idx_forced : b_.idx_model; }

  // Bytes of device memory this object allocated (weights + activations + workspace).
  int64_t device_bytes() const { return device_bytes_; }
  const std::vector<int>& oracle_offsets() const { return oracle_offsets_; }

 private:
  void* dalloc(size_t bytes);
  void expert_gemms(const int* host_off, const float* A, const float* W, float* C, int K, int N);

  MoEDimsCuda d_;
  float eps_, alpha_, limit_;
  cudaStream_t stream_ = nullptr;
  cublasHandle_t cublas_ = nullptr;
  void* cublas_ws_ = nullptr;
  std::vector<void*> allocs_;
  int64_t device_bytes_ = 0;

  // weights (one copy, shared by every variant)
  float *norm_scale_, *gate_wT_, *gate_b_, *w1t_, *b1_, *w2t_, *b2_;
  Buffers b_{};
  bool forced_ = false;

  int* offsets_pinned_ = nullptr;     // V0: D2H target (pinned, allocated once)
  std::vector<int> oracle_offsets_;   // V0-oracle: filled by prepare_oracle()
  bool oracle_ready_ = false;
};

}  // namespace mtrt::cuda
