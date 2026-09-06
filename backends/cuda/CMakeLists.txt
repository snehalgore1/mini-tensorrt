# CUDA GPU backend. Only added to the build when MTRT_CUDA=ON (see the root
# CMakeLists); never configured on the macOS/CPU path.
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# M0 smoke test: proves nvcc compiles, cudart links, a kernel launches, and
# host<->device copies work on the target GPU (run on Colab: ./cuda_smoke).
add_executable(cuda_smoke cuda_smoke.cu)
target_include_directories(cuda_smoke PRIVATE
  ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR})
target_link_libraries(cuda_smoke PRIVATE CUDA::cudart)
