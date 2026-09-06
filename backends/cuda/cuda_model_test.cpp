// M3: run a whole model graph on the GPU (CudaExecutor) and check it matches the
// CPU executor (the reference oracle, itself validated against PyTorch/HuggingFace).
//
//   ./build/backends/cuda/cuda_model_test
//
// Always runs the committed tiny transformer block (no Python needed). If real
// GPT-2 has been exported (python/export_gpt2_hf.py), it also runs that on the GPU
// and checks next-token argmax + logits vs the CPU path. Exit 0 = all pass.

#include <cmath>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#include "backends/cuda/cuda_executor.h"
#include "mtrt/executor.h"
#include "mtrt/frontend/json_loader.h"
#include "mtrt/registry.h"
#include "tests/support/npy.h"

using namespace mtrt;
using namespace mtrt::frontend;
using namespace mtrt::testing;

namespace {
int g_fail = 0;

int64_t argmax(const float* row, int64_t n) {
  int64_t b = 0;
  for (int64_t j = 1; j < n; ++j)
    if (row[j] > row[b]) b = j;
  return b;
}

// Compare GPU vs CPU output; report max abs error and (optionally) argmax match.
void report(const std::string& name, const std::vector<float>& gpu,
            const Tensor& cpu, int64_t rows, int64_t cols, bool check_argmax) {
  double max_abs = 0.0;
  int argmax_mism = 0;
  const float* c = cpu.data<float>();
  for (int64_t i = 0; i < static_cast<int64_t>(gpu.size()); ++i) {
    const double d = std::fabs(static_cast<double>(gpu[i]) - c[i]);
    if (d > max_abs) max_abs = d;
  }
  if (check_argmax)
    for (int64_t r = 0; r < rows; ++r)
      if (argmax(gpu.data() + r * cols, cols) != argmax(c + r * cols, cols)) ++argmax_mism;

  const bool ok = max_abs < 1e-2 && argmax_mism == 0;
  std::printf("[CUDA-MODEL] %-14s %s  (max_abs_err=%.3e%s)\n", name.c_str(),
              ok ? "PASS" : "FAIL", max_abs,
              check_argmax ? (", argmax_mism=" + std::to_string(argmax_mism)).c_str() : "");
  if (!ok) ++g_fail;
}

std::vector<Tensor> run_cpu(const LoadedModel& m, TensorId in_id, const Tensor& input) {
  KernelRegistry reg;
  register_builtin_kernels(reg);
  Executor exec(m.graph, reg);
  std::unordered_map<TensorId, Tensor> b = m.weights;
  b.emplace(in_id, input);
  return exec.run(b);
}
}  // namespace

int main() {
  const std::string dir = std::string(MTRT_MODELS_DIR) + "/";

  // 1) Committed tiny transformer block (float input) -- always available.
  {
    LoadedModel m = load_json_model(dir + "transformer.json");
    const TensorId in_id = m.graph.graph_inputs()[0];
    Tensor x = tensor_from_npy(load_golden("tf_input"));
    std::vector<Tensor> cpu = run_cpu(m, in_id, x);
    cuda::CudaExecutor ce(m);
    std::vector<float> gpu = ce.run(x.data<float>());
    report("transformer", gpu, cpu[0], cpu[0].shape()[0],
           cpu[0].shape().back(), /*check_argmax=*/false);
  }

  // 2) Real GPT-2 124M (int32 ids), only if it has been exported.
  if (file_exists(dir + "gpt2_124m.json")) {
    LoadedModel m = load_json_model(dir + "gpt2_124m.json");
    const TensorId in_id = m.graph.graph_inputs()[0];
    const int64_t S = m.graph.tensor(in_id).shape[0];
    Tensor ids = Tensor::owning(DType::kI32, {S});
    for (int64_t i = 0; i < S; ++i) ids.data<int32_t>()[i] = static_cast<int32_t>(10 + i);
    std::vector<Tensor> cpu = run_cpu(m, in_id, ids);
    cuda::CudaExecutor ce(m);
    std::vector<float> gpu = ce.run(ids.data<int32_t>());
    report("gpt2_124m", gpu, cpu[0], cpu[0].shape()[0], cpu[0].shape().back(),
           /*check_argmax=*/true);
  } else {
    std::printf("[CUDA-MODEL] gpt2_124m      SKIP  (export it: "
                "python python/export_gpt2_hf.py --seq-len 64)\n");
  }

  std::printf(g_fail ? "[CUDA-MODEL] %d FAILED\n" : "[CUDA-MODEL] all passed\n", g_fail);
  return g_fail ? 1 : 0;
}
