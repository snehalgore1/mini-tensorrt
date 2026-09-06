// M1 GPU kernel tests: run each CUDA kernel on the device and compare against the
// same committed PyTorch goldens the CPU kernels use (models/goldens/*.npy). This
// is the golden-test discipline carried onto the GPU. Build with -DMTRT_CUDA=ON;
// run: ./build/backends/cuda/cuda_test  (exit 0 = all pass).

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "backends/cuda/kernels.h"
#include "tests/support/npy.h"

using mtrt::testing::load_golden;
using mtrt::testing::NpyArray;

namespace {
int g_fail = 0;

#define CK(call)                                                             \
  do {                                                                       \
    const cudaError_t e__ = (call);                                          \
    if (e__ != cudaSuccess) {                                                \
      std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e__),       \
                  __FILE__, __LINE__);                                       \
      std::exit(2);                                                          \
    }                                                                        \
  } while (0)

float* upload(const std::vector<float>& h) {
  float* d = nullptr;
  CK(cudaMalloc(&d, h.size() * sizeof(float)));
  CK(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
  return d;
}
std::vector<float> download(const float* d, int64_t n) {
  std::vector<float> h((size_t)n);
  CK(cudaMemcpy(h.data(), d, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
  return h;
}

void check(const std::string& name, const std::vector<float>& got,
           const std::vector<float>& exp, float atol = 1e-4f, float rtol = 1e-5f) {
  double max_abs = 0.0;
  int bad = -1;
  for (size_t i = 0; i < exp.size(); ++i) {
    const double d = std::fabs((double)got[i] - exp[i]);
    if (d > max_abs) max_abs = d;
    if (d > atol + (double)rtol * std::fabs(exp[i]) && bad < 0) bad = (int)i;
  }
  const bool ok = bad < 0 && got.size() == exp.size();
  std::printf("[CUDA-TEST] %-12s %s  (max_abs_err=%.3e)\n", name.c_str(),
              ok ? "PASS" : "FAIL", max_abs);
  if (!ok) ++g_fail;
}
}  // namespace

int main() {
  // Add (identical shape): op_add_a + op_add_b.
  {
    NpyArray a = load_golden("op_add_a"), b = load_golden("op_add_b"),
             o = load_golden("op_add_out");
    const int M = (int)a.shape[0], N = (int)a.shape[1];
    float *da = upload(a.data), *db = upload(b.data), *dc;
    CK(cudaMalloc(&dc, a.data.size() * sizeof(float)));
    mtrt::cuda::add(da, db, dc, M, N, /*bias_broadcast=*/false);
    check("Add", download(dc, a.numel()), o.data);
    cudaFree(da); cudaFree(db); cudaFree(dc);
  }
  // Add (bias broadcast): [2,3] + [3] per column, vs a CPU-computed expected.
  {
    std::vector<float> a = {1, 2, 3, 4, 5, 6}, b = {10, 20, 30};
    std::vector<float> exp = {11, 22, 33, 14, 25, 36};
    float *da = upload(a), *db = upload(b), *dc;
    CK(cudaMalloc(&dc, a.size() * sizeof(float)));
    mtrt::cuda::add(da, db, dc, 2, 3, /*bias_broadcast=*/true);
    check("Add(bias)", download(dc, 6), exp);
    cudaFree(da); cudaFree(db); cudaFree(dc);
  }
  // Scale by 0.25.
  {
    NpyArray in = load_golden("op_scale_in"), o = load_golden("op_scale_out");
    float* dx = upload(in.data); float* dy;
    CK(cudaMalloc(&dy, in.data.size() * sizeof(float)));
    mtrt::cuda::scale(dx, dy, (int)in.numel(), 0.25f);
    check("Scale", download(dy, in.numel()), o.data);
    cudaFree(dx); cudaFree(dy);
  }
  // GeluTanh (gelu_new).
  {
    NpyArray in = load_golden("op_gelutanh_in"), o = load_golden("op_gelutanh_out");
    float* dx = upload(in.data); float* dy;
    CK(cudaMalloc(&dy, in.data.size() * sizeof(float)));
    mtrt::cuda::gelu_tanh(dx, dy, (int)in.numel());
    check("GeluTanh", download(dy, in.numel()), o.data);
    cudaFree(dx); cudaFree(dy);
  }
  // MatMul via cuBLAS: A[2,3] @ B[3,5] = C[2,5].
  {
    NpyArray a = load_golden("op_matmul_a"), b = load_golden("op_matmul_b"),
             o = load_golden("op_matmul_out");
    const int M = (int)a.shape[0], K = (int)a.shape[1], N = (int)b.shape[1];
    float *da = upload(a.data), *db = upload(b.data), *dc;
    CK(cudaMalloc(&dc, (size_t)M * N * sizeof(float)));
    mtrt::cuda::matmul(da, db, dc, M, N, K);
    CK(cudaDeviceSynchronize());
    check("MatMul", download(dc, (int64_t)M * N), o.data);
    cudaFree(da); cudaFree(db); cudaFree(dc);
  }

  mtrt::cuda::shutdown();
  std::printf(g_fail ? "[CUDA-TEST] %d FAILED\n" : "[CUDA-TEST] all passed\n", g_fail);
  return g_fail ? 1 : 0;
}
