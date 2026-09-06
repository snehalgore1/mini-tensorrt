# CUDA GPU backend. Only added to the build when MTRT_CUDA=ON (see the root
# CMakeLists); never configured on the macOS/CPU path.
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Target the Colab T4 (sm_75) by default; override with -DCMAKE_CUDA_ARCHITECTURES=.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()
# sm_75 (T4) is still fully supported; silence nvcc's forward-looking deprecation note.
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)

# GPU backend library: kernels (.cu) + the CudaExecutor (.cpp, host code that
# dispatches graph nodes to the kernels). Links the CPU core (mtrt) for the Graph
# IR / model loader it reuses.
add_library(mtrt_cuda STATIC kernels.cu gemm_cuda.cu cuda_executor.cpp)
target_include_directories(mtrt_cuda PUBLIC
  ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR})
target_link_libraries(mtrt_cuda PUBLIC CUDA::cudart CUDA::cublas mtrt)

# M0 smoke test: nvcc + cudart + kernel launch + host<->device copies.
add_executable(cuda_smoke cuda_smoke.cu)
target_include_directories(cuda_smoke PRIVATE
  ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR})
target_link_libraries(cuda_smoke PRIVATE CUDA::cudart)

# M1/M2 kernel tests: GPU kernels vs the committed PyTorch goldens.
add_executable(cuda_test cuda_test.cu)
target_include_directories(cuda_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(cuda_test PRIVATE mtrt_cuda)
target_compile_definitions(cuda_test PRIVATE
  MTRT_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/models/goldens")

# M3 whole-model test: CudaExecutor (GPU) vs the CPU executor on a real graph.
add_executable(cuda_model_test cuda_model_test.cpp)
target_include_directories(cuda_model_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(cuda_model_test PRIVATE mtrt_cuda)
target_compile_definitions(cuda_model_test PRIVATE
  MTRT_MODELS_DIR="${CMAKE_SOURCE_DIR}/models"
  MTRT_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/models/goldens")

# M4 GPU GEMM roofline study: our naive/tiled kernels vs cuBLAS.
add_executable(bench_gemm_cuda bench_gemm_cuda.cu)
target_include_directories(bench_gemm_cuda PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(bench_gemm_cuda PRIVATE mtrt_cuda)
