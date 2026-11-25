#pragma once

// gtest-free comparison helpers, shared by the CPU gtest suite (via golden.h) and the CUDA
// test binaries (which do not link gtest). See docs/ENGINEERING_DECISIONS.md D-MoE-10.

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace mtrt::testing {

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

}  // namespace mtrt::testing
