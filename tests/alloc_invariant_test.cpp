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

int main() {
  // Force the threaded GEMM path -- the most allocation-prone (packing buffers +
  // thread-pool dispatch). Read once by gemm_auto, so set before any run.
  setenv("MTRT_MATMUL", "threaded", 1);

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

  const long allocs = g_allocs.load();
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
