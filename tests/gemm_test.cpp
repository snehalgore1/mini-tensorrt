#include "backends/cpu/gemm.h"

#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "backends/cpu/thread_pool.h"
#include "tests/support/golden.h"

using namespace mtrt;
using namespace mtrt::testing;

namespace {

std::vector<float> random_matrix(int rows, int cols, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> m(static_cast<size_t>(rows) * cols);
  for (float& v : m) v = dist(rng);
  return m;
}

// Double-precision reference so tolerance reflects the variants' fp32 error, not
// the reference's.
std::vector<float> ref_gemm(const std::vector<float>& A,
                            const std::vector<float>& B, int M, int N, int K) {
  std::vector<float> C(static_cast<size_t>(M) * N);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k)
        acc += static_cast<double>(A[i * K + k]) * B[k * N + j];
      C[i * N + j] = static_cast<float>(acc);
    }
  return C;
}

using GemmFn = void (*)(const float*, const float*, float*, int, int, int);

void check_variant(GemmFn fn, int M, int N, int K, unsigned seed) {
  const std::vector<float> A = random_matrix(M, K, seed);
  const std::vector<float> B = random_matrix(K, N, seed + 1);
  const std::vector<float> expected = ref_gemm(A, B, M, N, K);
  std::vector<float> C(static_cast<size_t>(M) * N, -1.0f);
  fn(A.data(), B.data(), C.data(), M, N, K);
  EXPECT_TRUE(AllClose(C.data(), expected.data(), static_cast<int64_t>(C.size()),
                       1e-3f, 1e-4f));
}

}  // namespace

// gemm_naive against the PyTorch op_matmul golden (2x3 @ 3x5).
TEST(Gemm, NaiveMatchesGolden) {
  NpyArray a = load_golden("op_matmul_a");
  NpyArray b = load_golden("op_matmul_b");
  NpyArray exp = load_golden("op_matmul_out");
  const int M = 2, N = 5, K = 3;
  std::vector<float> C(static_cast<size_t>(M) * N);
  cpu::gemm_naive(a.data.data(), b.data.data(), C.data(), M, N, K);
  EXPECT_TRUE(AllClose(C.data(), exp.data.data(), M * N, kDefaultRtol, kDefaultAtol));
}

// Every variant matches the reference on square, larger, and non-square (edge)
// shapes.
TEST(Gemm, VariantsMatchReference) {
  struct Case { int M, N, K; };
  const Case cases[] = {{64, 64, 64}, {128, 96, 160}, {37, 53, 41}};
  const GemmFn fns[] = {cpu::gemm_naive,  cpu::gemm_reorder, cpu::gemm_register,
                        cpu::gemm_tiled, cpu::gemm_packed,  cpu::gemm_neon};
  unsigned seed = 7;
  for (const Case& c : cases)
    for (GemmFn fn : fns) check_variant(fn, c.M, c.N, c.K, seed++);
}

TEST(Gemm, ThreadedMatchesReference) {
  ThreadPool pool(4);
  struct Case { int M, N, K; };
  const Case cases[] = {{128, 128, 128}, {130, 71, 97}};
  unsigned seed = 100;
  for (const Case& c : cases) {
    const std::vector<float> A = random_matrix(c.M, c.K, seed);
    const std::vector<float> B = random_matrix(c.K, c.N, seed + 1);
    const std::vector<float> expected = ref_gemm(A, B, c.M, c.N, c.K);
    std::vector<float> C(static_cast<size_t>(c.M) * c.N, -1.0f);
    cpu::gemm_threaded(A.data(), B.data(), C.data(), c.M, c.N, c.K, pool);
    EXPECT_TRUE(AllClose(C.data(), expected.data(),
                         static_cast<int64_t>(C.size()), 1e-3f, 1e-4f));
    seed += 2;
  }
}
