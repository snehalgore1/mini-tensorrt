#pragma once

// Golden-comparison helpers shared by the kernel and model tests. Tolerance
// follows CLAUDE.md: rtol 1e-5 / atol 1e-6 for elementwise and GEMM; Softmax may
// need atol 1e-5.

#include <algorithm>
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

// Combined-tolerance comparison with SEPARATELY reported statistics (used by the MoE
// tests). Pass condition per element, as numpy.allclose:
//     |got - ref| <= atol + rtol * mag,   mag = |ref|  (or a caller-supplied scale)
// Reported:
//   max_abs  max |got - ref|
//   max_rel  max |got - ref| / |ref| ONLY over elements with |ref| >= atol / rtol. Below
//            that floor the atol term dominates the tolerance and relative error is
//            meaningless (it diverges as ref -> 0); those elements are counted in near_zero.
//   worst    max |got - ref| / (atol + rtol * mag): the fraction of the tolerance used.
struct CloseStats {
  double max_abs = 0.0;
  double max_rel = 0.0;
  int64_t near_zero = 0;
  double worst = 0.0;
  int64_t worst_index = -1;
};

inline CloseStats close_stats(const float* got, const float* ref, int64_t n, float rtol,
                              float atol, const float* mag = nullptr) {
  CloseStats s;
  const double floor = static_cast<double>(atol) / rtol;
  for (int64_t i = 0; i < n; ++i) {
    const double r = ref[i];
    const double d = std::abs(static_cast<double>(got[i]) - r);
    if (!(d <= s.max_abs)) s.max_abs = d;  // also propagates NaN
    if (std::abs(r) >= floor) {
      s.max_rel = std::max(s.max_rel, d / std::abs(r));
    } else {
      s.near_zero++;
    }
    const double m = mag ? std::abs(static_cast<double>(mag[i])) : std::abs(r);
    const double ratio = d / (atol + rtol * m);
    if (!(ratio <= s.worst)) {  // NaN -> recorded as worst
      s.worst = ratio;
      s.worst_index = i;
    }
  }
  return s;
}

inline ::testing::AssertionResult ExpectCloseImpl(const std::string& what, const Tensor& got,
                                                  const NpyArray& ref, float rtol,
                                                  float atol) {
  if (got.numel() != ref.numel())
    return ::testing::AssertionFailure() << what << ": numel " << got.numel()
                                         << " != " << ref.numel();
  const CloseStats s =
      close_stats(got.data<float>(), ref.data.data(), got.numel(), rtol, atol);
  ::testing::Test::RecordProperty(what + ".max_abs", std::to_string(s.max_abs));
  ::testing::Test::RecordProperty(what + ".max_rel", std::to_string(s.max_rel));
  if (s.worst <= 1.0) return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure()
         << what << ": worst element " << s.worst_index << " used " << s.worst
         << "x the tolerance (max_abs=" << s.max_abs << ", max_rel=" << s.max_rel
         << " over |ref|>=" << atol / rtol << ", near_zero=" << s.near_zero
         << ", rtol=" << rtol << ", atol=" << atol << ")";
}

#define MTRT_EXPECT_CLOSE(what, got, ref) \
  EXPECT_TRUE(::mtrt::testing::ExpectCloseImpl(what, got, ref, \
                                               ::mtrt::testing::kDefaultRtol, \
                                               ::mtrt::testing::kDefaultAtol))

}  // namespace mtrt::testing
