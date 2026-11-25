#pragma once
// SYNTAX-CHECK STUB ONLY (not the real CUDA runtime), used by tools/cuda_syntax_check/check.sh:
// just enough declarations to type-check
// MiniTensorRT's MoE CUDA sources with clang -fsyntax-only on a machine without CUDA.
#ifdef __CUDA__
#include <__clang_cuda_builtin_vars.h>
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#else
#define __global__
#define __device__
#define __host__
#define __shared__
#endif
#include <cstddef>
struct dim3 { unsigned x, y, z; __host__ __device__ dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {} };
struct float4 { float x, y, z, w; };
enum cudaError_t { cudaSuccess = 0 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3 };
enum cudaDeviceAttr { cudaDevAttrClockRate = 13, cudaDevAttrMemoryClockRate = 36, cudaDevAttrGlobalMemoryBusWidth = 37, cudaDevAttrMultiProcessorCount = 16 };
struct CUstream_st; typedef CUstream_st* cudaStream_t;
struct CUevent_st; typedef CUevent_st* cudaEvent_t;
#define cudaStreamNonBlocking 0x01
struct cudaDeviceProp { char name[256]; int major, minor; };
extern "C" {
const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaGetLastError();
cudaError_t cudaMalloc(void**, size_t);
cudaError_t cudaMallocHost(void**, size_t);
cudaError_t cudaFree(void*);
cudaError_t cudaFreeHost(void*);
cudaError_t cudaMemset(void*, int, size_t);
cudaError_t cudaMemsetAsync(void*, int, size_t, cudaStream_t);
cudaError_t cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind);
cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t*, unsigned);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaEventCreate(cudaEvent_t*);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t);
cudaError_t cudaEventSynchronize(cudaEvent_t);
cudaError_t cudaEventElapsedTime(float*, cudaEvent_t, cudaEvent_t);
cudaError_t cudaEventDestroy(cudaEvent_t);
cudaError_t cudaGetDevice(int*);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp*, int);
cudaError_t cudaDriverGetVersion(int*);
cudaError_t cudaRuntimeGetVersion(int*);
cudaError_t cudaDeviceGetAttribute(int*, cudaDeviceAttr, int);
cudaError_t cudaMemGetInfo(size_t*, size_t*);
unsigned __cudaPushCallConfiguration(dim3, dim3, size_t = 0, void* = 0);
cudaError_t cudaConfigureCall(dim3, dim3, size_t = 0, cudaStream_t = 0);
}
template <class T> cudaError_t cudaMalloc(T** p, size_t n) { return cudaMalloc((void**)p, n); }
template <class T> cudaError_t cudaMallocHost(T** p, size_t n) { return cudaMallocHost((void**)p, n); }
#ifdef __CUDA__
__device__ void __syncthreads();
__device__ unsigned __ballot_sync(unsigned, int);
__device__ int __popc(unsigned);
__device__ float __shfl_xor_sync(unsigned, float, int, int = 32);
__device__ int __shfl_xor_sync(unsigned, int, int, int = 32);
__device__ float expf(float);
__device__ float sqrtf(float);
__device__ float fminf(float, float);
__device__ float fmaxf(float, float);
#endif
