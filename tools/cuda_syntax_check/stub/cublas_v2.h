#pragma once
// SYNTAX-CHECK STUB ONLY (not the real cuBLAS).
#include "cuda_runtime.h"
struct cublasContext; typedef cublasContext* cublasHandle_t;
enum cublasStatus_t { CUBLAS_STATUS_SUCCESS = 0 };
enum cublasOperation_t { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1 };
enum cublasMath_t { CUBLAS_DEFAULT_MATH = 0 };
extern "C" {
cublasStatus_t cublasCreate(cublasHandle_t*);
cublasStatus_t cublasDestroy(cublasHandle_t);
cublasStatus_t cublasSetStream(cublasHandle_t, cudaStream_t);
cublasStatus_t cublasSetMathMode(cublasHandle_t, cublasMath_t);
cublasStatus_t cublasSetWorkspace(cublasHandle_t, void*, size_t);
cublasStatus_t cublasSgemm(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                           const float*, const float*, int, const float*, int, const float*,
                           float*, int);
}
