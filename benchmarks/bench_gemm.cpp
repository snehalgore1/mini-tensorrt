// GEMM optimization-ladder benchmark: GFLOP/s per variant across sizes, plus
// Apple Accelerate (cblas_sgemm) and this machine's roofline ceilings (peak NEON
// FMA throughput + memory bandwidth). Emits a JSON results file for plotting.
//
//   ./build/benchmarks/bench_gemm [--sizes 256,512,1024,2048] [--reps 5]
//                                 [--threads N] [--out gemm_results.json]

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "backends/cpu/gemm.h"
#include "backends/cpu/thread_pool.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

using namespace mtrt;
using Clock = std::chrono::steady_clock;

namespace {

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

std::vector<float> random_matrix(int rows, int cols, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  std::vector<float> m(static_cast<size_t>(rows) * cols);
  for (float& v : m) v = d(rng);
  return m;
}

// Median GFLOP/s over adaptive reps (target ~0.4s of work, >=1 rep).
double bench_gflops(const std::function<void()>& run, double flops) {
  run();  // warm up
  std::vector<double> samples;
  int reps = 5;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = Clock::now();
    run();
    const double s = seconds_since(t0);
    samples.push_back(flops / s / 1e9);
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

// Peak single-core NEON FP32 FMA throughput (GFLOP/s).
double measure_peak_fma() {
#if defined(__ARM_NEON)
  constexpr int NV = 20;
  float32x4_t acc[NV];
  for (int i = 0; i < NV; ++i) acc[i] = vdupq_n_f32(1.0f);
  float32x4_t a = vdupq_n_f32(1.0000001f);
  float32x4_t b = vdupq_n_f32(0.9999999f);
  const long iters = 300'000'000;
  const auto t0 = Clock::now();
  for (long t = 0; t < iters; ++t)
    for (int i = 0; i < NV; ++i) acc[i] = vfmaq_f32(acc[i], a, b);
  const double s = seconds_since(t0);
  float sink = 0.0f;
  for (int i = 0; i < NV; ++i) sink += vaddvq_f32(acc[i]);
  volatile float keep = sink;
  (void)keep;
  const double flops = static_cast<double>(iters) * NV * 4 * 2;
  return flops / s / 1e9;
#else
  return 0.0;
#endif
}

// Peak single-thread memory bandwidth via a STREAM triad (GB/s), counting 2
// reads + 1 write.
double measure_peak_bw() {
  const size_t n = 1u << 24;  // 16M floats = 64 MB per array
  std::vector<float> a(n), b(n), c(n);
  for (size_t i = 0; i < n; ++i) { b[i] = 1.0f; c[i] = 2.0f; }
  const float s = 3.0f;
  for (size_t i = 0; i < n; ++i) a[i] = b[i] + s * c[i];  // warm up
  const int reps = 20;
  const auto t0 = Clock::now();
  for (int r = 0; r < reps; ++r)
    for (size_t i = 0; i < n; ++i) a[i] = b[i] + s * c[i];
  const double sec = seconds_since(t0);
  volatile float keep = a[0] + a[n - 1];
  (void)keep;
  const double bytes = 3.0 * n * sizeof(float) * reps;
  return bytes / sec / 1e9;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> sizes = {256, 512, 1024, 2048};
  int threads = 0;
  std::string out = "gemm_results.json";
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--sizes") && i + 1 < argc) {
      sizes.clear();
      std::string s = argv[++i];
      size_t p = 0, q;
      while ((q = s.find(',', p)) != std::string::npos) {
        sizes.push_back(std::stoi(s.substr(p, q - p)));
        p = q + 1;
      }
      sizes.push_back(std::stoi(s.substr(p)));
    } else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) {
      threads = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) {
      out = argv[++i];
    }
  }

  const double peak_fma = measure_peak_fma();
  const double peak_bw = measure_peak_bw();
  ThreadPool pool(threads);

  struct Variant { const char* name; std::function<void(const float*, const float*, float*, int, int, int)> fn; };
  std::vector<Variant> variants = {
      {"naive", cpu::gemm_naive},
      {"reorder", cpu::gemm_reorder},
      {"register", cpu::gemm_register},
      {"tiled", cpu::gemm_tiled},
      {"packed", cpu::gemm_packed},
      {"neon", cpu::gemm_neon},
      {"threaded", [&](const float* A, const float* B, float* C, int M, int N, int K) {
         cpu::gemm_threaded(A, B, C, M, N, K, pool);
       }},
  };

  // results[variant][size] = GFLOP/s (NaN if skipped).
  std::vector<std::vector<double>> results(variants.size(),
                                           std::vector<double>(sizes.size(), 0.0 / 0.0));
  std::vector<double> accel(sizes.size(), 0.0 / 0.0);

  printf("peak NEON FMA (1 core): %.1f GFLOP/s   peak BW (triad): %.1f GB/s\n",
         peak_fma, peak_bw);
  printf("%-10s", "size");
  for (auto& v : variants) printf("%10s", v.name);
  printf("%12s\n", "Accelerate");

  for (size_t si = 0; si < sizes.size(); ++si) {
    const int n = sizes[si];
    const double flops = 2.0 * n * n * n;
    const std::vector<float> A = random_matrix(n, n, 1);
    const std::vector<float> B = random_matrix(n, n, 2);
    std::vector<float> C(static_cast<size_t>(n) * n);

    printf("%-10d", n);
    for (size_t vi = 0; vi < variants.size(); ++vi) {
      // Naive/reorder get impractically slow at large n; skip to keep the run short.
      if ((vi == 0 && n > 1024) || (vi == 1 && n > 2048)) {
        printf("%10s", "-");
        continue;
      }
      const auto& fn = variants[vi].fn;
      const double g = bench_gflops([&] { fn(A.data(), B.data(), C.data(), n, n, n); }, flops);
      results[vi][si] = g;
      printf("%10.1f", g);
    }
    const double ag = bench_gflops(
        [&] {
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f,
                      A.data(), n, B.data(), n, 0.0f, C.data(), n);
        },
        flops);
    accel[si] = ag;
    printf("%12.1f\n", ag);
  }

  // Write JSON results for the plotter.
  std::ofstream f(out);
  f << "{\n  \"peak_fma_gflops\": " << peak_fma
    << ",\n  \"peak_bw_gbps\": " << peak_bw << ",\n  \"sizes\": [";
  for (size_t i = 0; i < sizes.size(); ++i) f << (i ? ", " : "") << sizes[i];
  f << "],\n  \"variants\": {\n";
  for (size_t vi = 0; vi < variants.size(); ++vi) {
    f << "    \"" << variants[vi].name << "\": [";
    for (size_t si = 0; si < sizes.size(); ++si) {
      const double g = results[vi][si];
      f << (si ? ", " : "") << (g == g ? std::to_string(g) : "null");
    }
    f << "]" << (vi + 1 < variants.size() ? "," : "") << "\n";
  }
  f << "  },\n  \"accelerate\": [";
  for (size_t si = 0; si < sizes.size(); ++si)
    f << (si ? ", " : "") << (accel[si] == accel[si] ? std::to_string(accel[si]) : "null");
  f << "]\n}\n";
  printf("wrote %s\n", out.c_str());
  return 0;
}
