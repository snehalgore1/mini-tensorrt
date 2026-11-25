#pragma once

// Tolerance support for whole-MoE-layer comparisons (docs/ENGINEERING_DECISIONS.md D-MoE-10),
// shared by tests/moe_test.cpp (CPU) and backends/cuda/moe_cuda_test.cu (GPU). gtest-free.

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "tests/support/npy.h"

namespace mtrt::testing {

// Per-element size of what was SUMMED to produce each layer output (not the size of the
// result):   mag[t,c] = |x[t,c]| + sum_s w[t,s] * ( sum_i |a[j,i]| |W2t[e,i,c]| + |b2[e,c]| )
// with j = inv[t*k+s], e = idx[t,s]. FP32 summation error scales with sum|term|, and it
// does not shrink when the terms cancel. The clamp case has outputs ~6 built from terms
// ~585 (100x cancellation), so a |ref|-relative tolerance would fail on pure rounding noise.
// Whole-layer comparisons therefore use |d| <= atol + rtol*mag. A real bug (wrong expert or
// wrong column) produces errors on the order of mag itself, i.e. ~1e5x this tolerance.
inline std::vector<float> layer_term_magnitude(const std::string& c) {
  auto G = [&](const char* n) { return load_golden("moe_" + c + "_" + n); };
  NpyArray x = G("x"), w = G("w"), a = G("a"), w2t = G("w2t"), b2 = G("b2");
  NpyArray inv = load_golden_i32("moe_" + c + "_inv"), idx = load_golden_i32("moe_" + c + "_idx");
  const int64_t T = x.shape[0], H = x.shape[1], k = w.shape[1], I = a.shape[1];
  std::vector<float> mag(static_cast<size_t>(T * H));
  for (int64_t t = 0; t < T; ++t)
    for (int64_t col = 0; col < H; ++col) {
      double acc = std::abs(x.data[t * H + col]);
      for (int64_t s = 0; s < k; ++s) {
        const int64_t j = inv.idata[t * k + s], e = idx.idata[t * k + s];
        double dot = std::abs(b2.data[e * H + col]);
        for (int64_t i = 0; i < I; ++i)
          dot += std::abs(a.data[j * I + i]) * std::abs(w2t.data[(e * I + i) * H + col]);
        acc += w.data[t * k + s] * dot;
      }
      mag[static_cast<size_t>(t * H + col)] = static_cast<float>(acc);
    }
  return mag;
}

}  // namespace mtrt::testing
