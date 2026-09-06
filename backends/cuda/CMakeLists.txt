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

# GPU kernel library (device tensor + kernels + cuBLAS matmul), reused by the
# tests now and the CudaExecutor later.
add_library(mtrt_cuda STATIC kernels.cu)
target_include_directories(mtrt_cuda PUBLIC
  ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR})
target_link_libraries(mtrt_cuda PUBLIC CUDA::cudart CUDA::cublas)

# M0 smoke test: nvcc + cudart + kernel launch + host<->device copies.
add_executable(cuda_smoke cuda_smoke.cu)
target_include_directories(cuda_smoke PRIVATE
  ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR})
target_link_libraries(cuda_smoke PRIVATE CUDA::cudart)

# M1 kernel tests: GPU kernels vs the committed PyTorch goldens.
add_executable(cuda_test cuda_test.cu)
target_include_directories(cuda_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(cuda_test PRIVATE mtrt_cuda)
target_compile_definitions(cuda_test PRIVATE
  MTRT_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/models/goldens")
