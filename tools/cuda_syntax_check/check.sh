#!/bin/bash
# Type-checks the MoE CUDA sources with clang on a machine WITHOUT nvcc (e.g. the Mac), using
# the minimal stub headers in stub/. Catches syntax, type, overload and undeclared-name errors
# in both the host and the device (sm_75) compilation passes. It does NOT compile, link or run
# anything, and the stubs declare only the CUDA/cuBLAS API this code uses -- the real build
# and the correctness tests run on the GPU machine (kaggle/README.md).
#
# Run from anywhere:  tools/cuda_syntax_check/check.sh      (exit 0 = clean)
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
STUB="$ROOT/tools/cuda_syntax_check/stub"
cd "$ROOT"
F=(-std=c++17 -include "$STUB/cuda_runtime.h" -I"$STUB" -I. -Iinclude -Wall -Wextra
   -fsyntax-only '-DMTRT_GOLDENS_DIR="x"')
rc=0
for f in backends/cuda/moe.cu backends/cuda/moe_cuda_test.cu backends/cuda/bench_moe_cuda.cu; do
  for pass in "--cuda-host-only" "--cuda-device-only --cuda-gpu-arch=sm_75"; do
    # shellcheck disable=SC2086
    out=$(clang++ -x cuda $pass -nocudainc -nocudalib "${F[@]}" "$f" 2>&1 | grep -E "error|warning")
    if [ -z "$out" ]; then echo "clean  $f [$pass]"; else echo "FAIL   $f [$pass]"; echo "$out" | head -25; rc=1; fi
  done
done
for f in backends/cuda/moe_layer.cpp; do
  out=$(clang++ "${F[@]}" "$f" 2>&1 | grep -E "error|warning")
  if [ -z "$out" ]; then echo "clean  $f"; else echo "FAIL   $f"; echo "$out" | head -25; rc=1; fi
done
exit $rc
