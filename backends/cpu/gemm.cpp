#include "backends/cpu/gemm.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "backends/cpu/thread_pool.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace mtrt::cpu {

namespace {
constexpr int MR = 4;   // micro-tile rows
constexpr int NR = 4;   // micro-tile cols
constexpr int MC = 128; // cache block: M
constexpr int NC = 128; // cache block: N
constexpr int KC = 256; // cache block: K
}  // namespace

// ---- 0. Naive triple loop (ijk) -------------------------------------------
void gemm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
      C[i * N + j] = acc;
    }
  }
}

// ---- 1. Loop reorder (ikj): unit-stride inner loop over B and C -----------
void gemm_reorder(const float* A, const float* B, float* C, int M, int N, int K) {
  std::fill(C, C + static_cast<long>(M) * N, 0.0f);
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      const float a = A[i * K + k];
      const float* br = &B[k * N];
      float* cr = &C[i * N];
      for (int j = 0; j < N; ++j) cr[j] += a * br[j];
    }
  }
}

// ---- 2. Register blocking: accumulate a 4x4 output tile in registers ------
void gemm_register(const float* A, const float* B, float* C, int M, int N, int K) {
  const int Mm = M / MR * MR;
  const int Nm = N / NR * NR;
  for (int i = 0; i < Mm; i += MR) {
    for (int j = 0; j < Nm; j += NR) {
      float acc[MR][NR] = {{0}};
      for (int k = 0; k < K; ++k) {
        const float a0 = A[(i + 0) * K + k], a1 = A[(i + 1) * K + k];
        const float a2 = A[(i + 2) * K + k], a3 = A[(i + 3) * K + k];
        const float* br = &B[k * N + j];
        const float b0 = br[0], b1 = br[1], b2 = br[2], b3 = br[3];
        acc[0][0] += a0 * b0; acc[0][1] += a0 * b1; acc[0][2] += a0 * b2; acc[0][3] += a0 * b3;
        acc[1][0] += a1 * b0; acc[1][1] += a1 * b1; acc[1][2] += a1 * b2; acc[1][3] += a1 * b3;
        acc[2][0] += a2 * b0; acc[2][1] += a2 * b1; acc[2][2] += a2 * b2; acc[2][3] += a2 * b3;
        acc[3][0] += a3 * b0; acc[3][1] += a3 * b1; acc[3][2] += a3 * b2; acc[3][3] += a3 * b3;
      }
      for (int ii = 0; ii < MR; ++ii)
        for (int jj = 0; jj < NR; ++jj) C[(i + ii) * N + j + jj] = acc[ii][jj];
    }
  }
  // Edge strips (sizes not a multiple of the tile).
  for (int i = Mm; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float s = 0.0f;
      for (int k = 0; k < K; ++k) s += A[i * K + k] * B[k * N + j];
      C[i * N + j] = s;
    }
  for (int i = 0; i < Mm; ++i)
    for (int j = Nm; j < N; ++j) {
      float s = 0.0f;
      for (int k = 0; k < K; ++k) s += A[i * K + k] * B[k * N + j];
      C[i * N + j] = s;
    }
}

// ---- 3. Cache tiling: block M/N/K around a constant-bound 4x4 microkernel --
void gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
  std::fill(C, C + static_cast<long>(M) * N, 0.0f);
  const int Mm = M / MR * MR;  // main region covered by full 4x4 tiles
  const int Nm = N / NR * NR;
  for (int jc = 0; jc < Nm; jc += NC) {
    const int jce = std::min(jc + NC, Nm);
    for (int pc = 0; pc < K; pc += KC) {
      const int kc_end = std::min(pc + KC, K);
      for (int ic = 0; ic < Mm; ic += MC) {
        const int ice = std::min(ic + MC, Mm);
        for (int i = ic; i < ice; i += MR) {
          for (int j = jc; j < jce; j += NR) {
            // Fixed 4x4 tile, constant loop bounds -> register-resident + unrolled.
            float acc[MR][NR] = {{0}};
            for (int k = pc; k < kc_end; ++k) {
              const float a0 = A[(i + 0) * K + k], a1 = A[(i + 1) * K + k];
              const float a2 = A[(i + 2) * K + k], a3 = A[(i + 3) * K + k];
              const float* br = &B[k * N + j];
              const float b0 = br[0], b1 = br[1], b2 = br[2], b3 = br[3];
              acc[0][0] += a0 * b0; acc[0][1] += a0 * b1; acc[0][2] += a0 * b2; acc[0][3] += a0 * b3;
              acc[1][0] += a1 * b0; acc[1][1] += a1 * b1; acc[1][2] += a1 * b2; acc[1][3] += a1 * b3;
              acc[2][0] += a2 * b0; acc[2][1] += a2 * b1; acc[2][2] += a2 * b2; acc[2][3] += a2 * b3;
              acc[3][0] += a3 * b0; acc[3][1] += a3 * b1; acc[3][2] += a3 * b2; acc[3][3] += a3 * b3;
            }
            for (int ii = 0; ii < MR; ++ii)
              for (int jj = 0; jj < NR; ++jj) C[(i + ii) * N + j + jj] += acc[ii][jj];
          }
        }
      }
    }
  }
  // Edge strips (rows >= Mm, or cols >= Nm) computed directly over full K.
  for (int i = Mm; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float s = 0.0f;
      for (int k = 0; k < K; ++k) s += A[i * K + k] * B[k * N + j];
      C[i * N + j] = s;
    }
  for (int i = 0; i < Mm; ++i)
    for (int j = Nm; j < N; ++j) {
      float s = 0.0f;
      for (int k = 0; k < K; ++k) s += A[i * K + k] * B[k * N + j];
      C[i * N + j] = s;
    }
}

// Pack helpers: zero-padded panels (rows of mr / cols of nr) so the microkernel
// is always a full tile with unit-stride access; edges need no special-casing.
namespace {
// packB[panel(jr)*(kc*nr) + k*nr + jj] = B[(pc+k), jc+jr+jj]  (0 if out of range)
void pack_b(const float* B, int N, int pc, int kc, int jc, int nc, int nr,
            std::vector<float>& packB) {
  int idx = 0;
  for (int jr = 0; jr < nc; jr += nr)
    for (int k = 0; k < kc; ++k)
      for (int jj = 0; jj < nr; ++jj)
        packB[idx++] = (jr + jj < nc) ? B[(pc + k) * N + jc + jr + jj] : 0.0f;
}
// packA[panel(ir)*(kc*mr) + k*mr + ii] = A[(ic+ir+ii), pc+k]  (0 if out of range)
void pack_a(const float* A, int K, int ic, int mc, int pc, int kc, int mr,
            std::vector<float>& packA) {
  int idx = 0;
  for (int ir = 0; ir < mc; ir += mr)
    for (int k = 0; k < kc; ++k)
      for (int ii = 0; ii < mr; ++ii)
        packA[idx++] = (ir + ii < mc) ? A[(ic + ir + ii) * K + pc + k] : 0.0f;
}
}  // namespace

// ---- 4. Panel packing: contiguous panels for a 4x4 scalar microkernel -----
void gemm_packed(const float* A, const float* B, float* C, int M, int N, int K) {
  std::fill(C, C + static_cast<long>(M) * N, 0.0f);
  std::vector<float> packA(static_cast<size_t>(MC) * KC);
  std::vector<float> packB(static_cast<size_t>(KC) * NC);
  for (int jc = 0; jc < N; jc += NC) {
    const int nc = std::min(NC, N - jc);
    for (int pc = 0; pc < K; pc += KC) {
      const int kc = std::min(KC, K - pc);
      pack_b(B, N, pc, kc, jc, nc, NR, packB);
      for (int ic = 0; ic < M; ic += MC) {
        const int mc = std::min(MC, M - ic);
        pack_a(A, K, ic, mc, pc, kc, MR, packA);
        for (int ir = 0; ir < mc; ir += MR) {
          const float* pa = &packA[static_cast<size_t>(ir / MR) * kc * MR];
          for (int jr = 0; jr < nc; jr += NR) {
            const float* pb = &packB[static_cast<size_t>(jr / NR) * kc * NR];
            float acc[MR][NR] = {{0}};
            for (int k = 0; k < kc; ++k)
              for (int ii = 0; ii < MR; ++ii)
                for (int jj = 0; jj < NR; ++jj)
                  acc[ii][jj] += pa[k * MR + ii] * pb[k * NR + jj];
            const int mr = std::min(MR, mc - ir);
            const int nr = std::min(NR, nc - jr);
            for (int ii = 0; ii < mr; ++ii)
              for (int jj = 0; jj < nr; ++jj)
                C[(ic + ir + ii) * N + jc + jr + jj] += acc[ii][jj];
          }
        }
      }
    }
  }
}

// ---- 5. NEON microkernel: wide 8x8 tile (16 float32x4 accumulators) --------
// 8 rows x 8 cols = 16 accumulator vectors -> enough independent FMA chains to
// hide latency and saturate the FMA units, unlike a 4x4 tile.
// Parameterized NEON GEMM with runtime cache-block sizes, for the autotuning
// sweep (bench_autotune). gemm_neon calls this with the default MC/NC/KC.
void gemm_neon_blocked(const float* A, const float* B, float* C, int M, int N, int K,
                       int bMC, int bNC, int bKC) {
#if defined(__ARM_NEON)
  constexpr int NMR = 8, NNR = 8;
  std::fill(C, C + static_cast<long>(M) * N, 0.0f);
  std::vector<float> packA(static_cast<size_t>(bMC) * bKC);
  std::vector<float> packB(static_cast<size_t>(bKC) * bNC);
  for (int jc = 0; jc < N; jc += bNC) {
    const int nc = std::min(bNC, N - jc);
    for (int pc = 0; pc < K; pc += bKC) {
      const int kc = std::min(bKC, K - pc);
      pack_b(B, N, pc, kc, jc, nc, NNR, packB);
      for (int ic = 0; ic < M; ic += bMC) {
        const int mc = std::min(bMC, M - ic);
        pack_a(A, K, ic, mc, pc, kc, NMR, packA);
        for (int ir = 0; ir < mc; ir += NMR) {
          const float* pa = &packA[static_cast<size_t>(ir / NMR) * kc * NMR];
          for (int jr = 0; jr < nc; jr += NNR) {
            const float* pb = &packB[static_cast<size_t>(jr / NNR) * kc * NNR];
            float32x4_t c[NMR][2];
            for (int ii = 0; ii < NMR; ++ii) {
              c[ii][0] = vdupq_n_f32(0);
              c[ii][1] = vdupq_n_f32(0);
            }
            for (int k = 0; k < kc; ++k) {
              const float32x4_t b0 = vld1q_f32(&pb[k * NNR]);
              const float32x4_t b1 = vld1q_f32(&pb[k * NNR + 4]);
              const float* a = &pa[k * NMR];
              for (int ii = 0; ii < NMR; ++ii) {
                c[ii][0] = vfmaq_n_f32(c[ii][0], b0, a[ii]);
                c[ii][1] = vfmaq_n_f32(c[ii][1], b1, a[ii]);
              }
            }
            float ct[NMR][NNR];
            for (int ii = 0; ii < NMR; ++ii) {
              vst1q_f32(&ct[ii][0], c[ii][0]);
              vst1q_f32(&ct[ii][4], c[ii][1]);
            }
            const int mr = std::min(NMR, mc - ir);
            const int nr = std::min(NNR, nc - jr);
            for (int ii = 0; ii < mr; ++ii)
              for (int jj = 0; jj < nr; ++jj)
                C[(ic + ir + ii) * N + jc + jr + jj] += ct[ii][jj];
          }
        }
      }
    }
  }
#else
  (void)bMC; (void)bNC; (void)bKC;
  gemm_packed(A, B, C, M, N, K);  // no NEON: fall back to the packed scalar path
#endif
}

void gemm_neon(const float* A, const float* B, float* C, int M, int N, int K) {
  // Autotuned block sizes (bench_autotune): larger NC/KC panels fit the M1 Pro's
  // L2, giving ~12% over the scalar-ladder default (128/128/256). The scalar
  // tiled/packed steps keep the original MC/NC/KC so the documented ladder holds.
  gemm_neon_blocked(A, B, C, M, N, K, /*MC=*/128, /*NC=*/512, /*KC=*/512);
}

// ---- 6. Multithreading: split output rows across the pool -----------------
void gemm_threaded(const float* A, const float* B, float* C, int M, int N, int K,
                   ThreadPool& pool) {
  pool.parallel_for(M, [&](int64_t r0, int64_t r1) {
    const int rows = static_cast<int>(r1 - r0);
    gemm_neon(A + r0 * K, B, C + r0 * N, rows, N, K);
  });
}

namespace {
// 0 naive, 1 neon, 2 auto (default), 3 threaded. Read once from MTRT_MATMUL.
int matmul_mode() {
  static const int mode = [] {
    const char* e = std::getenv("MTRT_MATMUL");
    if (!e) return 2;
    if (!std::strcmp(e, "naive")) return 0;
    if (!std::strcmp(e, "neon")) return 1;
    if (!std::strcmp(e, "threaded")) return 3;
    return 2;
  }();
  return mode;
}
}  // namespace

void gemm_auto(const float* A, const float* B, float* C, int M, int N, int K) {
  const int mode = matmul_mode();
  if (mode == 0) return gemm_naive(A, B, C, M, N, K);
  if (mode == 1) return gemm_neon(A, B, C, M, N, K);

  static ThreadPool pool;  // built once, shared across all MatMul calls
  // Thread only when there is enough work to amortize dispatch and enough rows
  // to split; otherwise the single-core NEON kernel wins.
  const long flops = 2L * M * N * K;
  if (mode == 3 || (M >= 32 && flops >= (1L << 21))) {
    gemm_threaded(A, B, C, M, N, K, pool);
  } else {
    gemm_neon(A, B, C, M, N, K);
  }
}

}  // namespace mtrt::cpu
