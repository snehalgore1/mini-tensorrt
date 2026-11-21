// Enforces core invariant 4: "The executor never allocates during run()." All
// intermediate memory comes from the arena and per-op/GEMM scratch is reused, so
// a warmed-up run() must not heap-allocate per node -- the only per-run
// allocations are the returned std::vector<Tensor> and its output tensors.
//
// Its own executable (not linked into mtrt_tests) so the global operator new
// override stays isolated. Counts allocations during a warm run and asserts the
// count is small and does NOT scale with the node count (the property that would
// break if any kernel or the GEMM/thread-pool allocated per call again).

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
#include <unordered_map>

#include "mtrt/executor.h"
#include "mtrt/frontend/json_loader.h"
#include "mtrt/registry.h"
#include "mtrt/tensor.h"
#include "backends/cpu/gemm.h"
#include "tests/support/moe_graph.h"

namespace {
std::atomic<long> g_allocs{0};
bool g_count = false;
}  // namespace

void* operator new(std::size_t n) {
  if (g_count) g_allocs.fetch_add(1, std::memory_order_relaxed);
  void* p = std::malloc(n ? n : 1);
  if (!p) throw std::bad_alloc();
  return p;
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

using namespace mtrt;
using namespace mtrt::frontend;

namespace {

// MoE adds a hazard the dense graphs do not have: the per-expert GEMM sizes CHANGE from
// run to run with the routing, while the graph's shapes stay fixed. Any scratch that grows
// with problem size, or that is created lazily per worker thread, can allocate in a later
// run that sees a bigger expert segment than the warm-up did.
//
// Worst case, made deterministic: warm up with "decode-like" routing (every expert segment
// is exactly 1 row), then measure with every token sent to the same 4 experts (segments of
// T rows). Routing is steered through the router weights, which are run() bindings.
int check_moe() {
  using mtrt::testing::build_moe_graph;
  const mtrt::testing::MoEDims d{/*T=*/8, /*H=*/64, /*I=*/64, /*E=*/32, /*k=*/4};
  auto mg = build_moe_graph(d);
  KernelRegistry reg;
  register_builtin_kernels(reg);
  Executor exec(mg.graph, reg);

  auto filled = [](std::vector<int64_t> shape, float v) {
    Tensor t = Tensor::owning(DType::kF32, std::move(shape));
    for (int64_t i = 0; i < t.numel(); ++i) t.data<float>()[i] = v;
    return t;
  };
  // x[t] = one-hot at dim t, so after RMSNorm token t only "sees" router row t.
  Tensor x = filled({d.T, d.H}, 0.0f);
  for (int64_t t = 0; t < d.T; ++t) x.data<float>()[t * d.H + t] = 1.0f;
  // Router: token t prefers experts 4t..4t+3  ->  32 distinct experts, 1 row each.
  Tensor gate_wT = filled({d.H, d.E}, 0.0f);
  for (int64_t t = 0; t < d.T; ++t)
    for (int64_t s = 0; s < d.k; ++s) gate_wT.data<float>()[t * d.E + t * d.k + s] = 1.0f + 0.1f * s;
  Tensor gate_b_spread = filled({d.E}, 0.0f);
  Tensor gate_b_same4 = filled({d.E}, 0.0f);  // +100 on experts 0..3: all tokens pick them
  for (int64_t e = 0; e < 4; ++e) gate_b_same4.data<float>()[e] = 100.0f;

  std::unordered_map<TensorId, Tensor> b{
      {mg.x, x}, {mg.norm_scale, filled({d.H}, 1.0f)}, {mg.gate_wT, gate_wT},
      {mg.gate_b, gate_b_spread}, {mg.w1t, filled({d.E, d.H, 2 * d.I}, 0.01f)},
      {mg.b1, filled({d.E, 2 * d.I}, 0.0f)}, {mg.w2t, filled({d.E, d.I, d.H}, 0.01f)},
      {mg.b2, filled({d.E, d.H}, 0.0f)}};

  exec.run(b);
  exec.run(b);  // warm-up: decode-like routing, every expert segment has 1 row

  b[mg.gate_b] = gate_b_same4;  // building this map entry is outside the counted region
  g_count = true;
  exec.run(b);  // measured: 4 experts x 8 rows, 28 experts empty
  g_count = false;

  const long allocs = g_allocs.exchange(0);
  const int64_t outputs = static_cast<int64_t>(mg.graph.graph_outputs().size());
  const long bound = 3 * outputs + 2;
  std::printf("[ALLOC] MoE warm run() after routing change (1-row -> 8-row segments): "
              "%ld heap allocs (nodes=%lld, bound=%ld)\n",
              allocs, (long long)mg.graph.num_nodes(), bound);
  if (allocs > bound) {
    std::printf("FAIL: MoE run() allocated after a routing change -- invariant 4 broken\n");
    return 1;
  }
  std::printf("PASS: MoE compute path is allocation-free across routing changes\n");
  return 0;
}

// GEMM scratch first-touch, isolated. The tuned GEMM keeps its packing panels in
// `static thread_local` grow-only vectors sized by the cache-block constants (independent of
// M, N, K), so each THREAD allocates them once, on its first GEMM. In a threaded GEMM, the
// workers claim row ranges dynamically, so which workers have touched their scratch after a
// warm-up depends on scheduling and on the M values seen so far. A small first GEMM (M = 1:
// one range, one worker) followed by a large one makes the gap visible.
int check_gemm_first_touch() {
  std::vector<float> A(256 * 64, 0.5f), B(64 * 64, 0.25f), C(256 * 64);
  mtrt::cpu::gemm_auto(A.data(), B.data(), C.data(), /*M=*/1, 64, 64);  // first call ever
  g_count = true;
  for (int rep = 0; rep < 4; ++rep)
    mtrt::cpu::gemm_auto(A.data(), B.data(), C.data(), /*M=*/256, 64, 64);
  g_count = false;
  const long allocs = g_allocs.exchange(0);
  std::printf("[ALLOC] threaded GEMM M=256 after a single M=1 warm-up: %ld heap allocs\n",
              allocs);
  if (allocs != 0) {
    std::printf("FAIL: GEMM worker scratch allocated after the first call\n");
    return 1;
  }
  std::printf("PASS: GEMM scratch for every worker exists after the first call\n");
  return 0;
}

int check_transformer() {

  const std::string path = std::string(MTRT_MODELS_DIR) + "/transformer.json";
  LoadedModel m = load_json_model(path);
  KernelRegistry reg;
  register_builtin_kernels(reg);
  Executor exec(m.graph, reg);

  std::unordered_map<TensorId, Tensor> b = m.weights;
  const TensorId in = m.graph.graph_inputs()[0];
  Tensor x = Tensor::owning(m.graph.tensor(in).dtype, m.graph.tensor(in).shape);
  for (int64_t i = 0; i < x.numel(); ++i) x.data<float>()[i] = 0.1f;
  b[in] = x;

  exec.run(b);
  exec.run(b);  // warm up: grow thread-local scratch, spin up the pool

  g_count = true;
  exec.run(b);
  g_count = false;

  const long allocs = g_allocs.exchange(0);
  const int64_t nodes = m.graph.num_nodes();
  const int64_t outputs = static_cast<int64_t>(m.graph.graph_outputs().size());
  // The returned vector<Tensor> plus each output tensor's shape/strides are the
  // only allowed allocations; the compute path (kernels + GEMM + pool) must add
  // none. Bound is O(outputs), and must be far below the node count.
  const long bound = 3 * outputs + 2;
  std::printf("[ALLOC] warm run(): %ld heap allocs (nodes=%lld, outputs=%lld, bound=%ld)\n",
              allocs, (long long)nodes, (long long)outputs, bound);
  if (allocs > bound) {
    std::printf("FAIL: run() allocates more than the return value -- invariant 4 broken\n");
    return 1;
  }
  if (allocs >= nodes) {
    std::printf("FAIL: allocations scale with node count -- per-op allocation regressed\n");
    return 1;
  }
  std::printf("PASS: executor compute path is allocation-free (invariant 4)\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  // Force the threaded GEMM path -- the most allocation-prone (packing buffers +
  // thread-pool dispatch). Read once by gemm_auto, so set before any run.
  setenv("MTRT_MATMUL", "threaded", 1);
  // Each check that needs a cold process (no GEMM call yet) gets its own ctest entry.
  if (argc > 1 && std::string(argv[1]) == "gemm-first-touch") return check_gemm_first_touch();
  // MoE runs FIRST, in a fresh process, so no earlier graph has already warmed the GEMM's
  // per-thread scratch for it.
  const int moe = check_moe();
  g_allocs = 0;
  return check_transformer() | moe;
}
