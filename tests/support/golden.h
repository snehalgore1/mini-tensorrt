#pragma once

// Golden-comparison helpers shared by the kernel and model tests. Tolerance
// follows CLAUDE.md: rtol 1e-5 / atol 1e-6 for elementwise and GEMM; Softmax may
// need atol 1e-5.

#include <cmath>
#include <string>

#include <gtest/gtest.h>

#include "mtrt/tensor.h"
#include "tests/support/npy.h"

namespace mtrt::testing {

constexpr float kDefaultRtol = 1e-5f;
constexpr float kDefaultAtol = 1e-6f;

struct ErrorStats {
  double max_abs = 0.0;
  double max_rel = 0.0;
};

// Max absolute and max relative error between got and expected, elementwise.
inline ErrorStats compute_error(const float* got, const float* exp, int64_t n) {
  ErrorStats s;
  for (int64_t i = 0; i < n; ++i) {
    const double diff = std::abs(static_cast<double>(got[i]) - exp[i]);
    if (diff > s.max_abs) s.max_abs = diff;
    if (exp[i] != 0.0f) {
      const double rel = diff / std::abs(exp[i]);
      if (rel > s.max_rel) s.max_rel = rel;
    }
  }
  return s;
}

// numpy-style allclose: |got - exp| <= atol + rtol*|exp|, elementwise.
inline ::testing::AssertionResult AllClose(const float* got, const float* exp,
                                           int64_t n, float rtol, float atol) {
  double max_abs = 0.0, max_rel = 0.0;
  int64_t bad = -1;
  for (int64_t i = 0; i < n; ++i) {
    const double diff = std::abs(static_cast<double>(got[i]) - exp[i]);
    const double tol = atol + static_cast<double>(rtol) * std::abs(exp[i]);
    if (diff > max_abs) max_abs = diff;
    const double rel = exp[i] != 0.0f ? diff / std::abs(exp[i]) : 0.0;
    if (rel > max_rel) max_rel = rel;
    if (diff > tol && bad < 0) bad = i;
  }
  if (bad < 0) return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure()
         << "mismatch at index " << bad << ": got " << got[bad] << " expected "
         << exp[bad] << " (max_abs_err=" << max_abs << ", max_rel_err=" << max_rel
         << ", rtol=" << rtol << ", atol=" << atol << ")";
}

inline void ExpectGolden(const Tensor& got, const NpyArray& expected,
                         float rtol = kDefaultRtol, float atol = kDefaultAtol) {
  ASSERT_EQ(got.numel(), expected.numel());
  EXPECT_TRUE(AllClose(got.data<float>(), expected.data.data(), got.numel(),
                       rtol, atol));
}

}  // namespace mtrt::testing
