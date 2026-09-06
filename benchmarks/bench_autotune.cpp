// GEMM cache-block autotuning sweep: measure GFLOP/s of the NEON kernel across a
// grid of M/N/K block sizes, to find the optimum and explain it with the cache
// hierarchy (a depth artifact -- design-space exploration over a known technique).
//
//   ./build/benchmarks/bench_autotune [--sizes 1024,2048]
//
// Only meaningful on arm64 (NEON); off arm64 the kernel falls back to the scalar
// packed path and block sizes have little effect.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "backends/cpu/gemm.h"

namespace {
double gflops(int n, int iters, int mc, int nc, int kc, const float* A,
              const float* B, float* C) {
  using namespace mtrt::cpu;
  gemm_neon_blocked(A, B, C, n, n, n, mc, nc, kc);  // warmup
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) gemm_neon_blocked(A, B, C, n, n, n, mc, nc, kc);
  const auto t1 = std::chrono::steady_clock::now();
  const double s = std::chrono::duration<double>(t1 - t0).count() / iters;
  return 2.0 * n * n * n / s / 1e9;
}
}  // namespace

int main(int argc, char** argv) {
  std::vector<int> sizes = {1024, 2048};
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--sizes") && i + 1 < argc) {
      sizes.clear();
      for (char* t = std::strtok(argv[++i], ","); t; t = std::strtok(nullptr, ","))
        sizes.push_back(std::atoi(t));
    }

  const int MCs[] = {64, 128, 256};
  const int NCs[] = {64, 128, 256, 512};
  const int KCs[] = {128, 256, 512};

  std::mt19937 rng(0);
  std::uniform_real_distribution<float> d(-1.f, 1.f);

  for (int n : sizes) {
    std::vector<float> A((size_t)n * n), B((size_t)n * n), C((size_t)n * n);
    for (auto& x : A) x = d(rng);
    for (auto& x : B) x = d(rng);
    const int iters = n >= 2048 ? 3 : 8;

    double best = 0; int bmc = 0, bnc = 0, bkc = 0;
    std::printf("== N=%d ==   MC   NC   KC   GFLOP/s\n", n);
    for (int mc : MCs)
      for (int nc : NCs)
        for (int kc : KCs) {
          const double g = gflops(n, iters, mc, nc, kc, A.data(), B.data(), C.data());
          std::printf("           %4d %4d %4d   %7.1f\n", mc, nc, kc, g);
          if (g > best) { best = g; bmc = mc; bnc = nc; bkc = kc; }
        }
    std::printf(">> N=%d best: MC=%d NC=%d KC=%d -> %.1f GFLOP/s "
                "(default is MC=128 NC=128 KC=256)\n\n", n, bmc, bnc, bkc, best);
  }
  return 0;
}
