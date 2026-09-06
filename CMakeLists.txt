cmake_minimum_required(VERSION 3.20)
project(MiniTensorRT CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MTRT_SANITIZERS "Enable ASan/UBSan" OFF)
# GPU backend. OFF by default so the macOS/CPU build is completely unaffected; ON
# is used on a CUDA machine (e.g. Colab). All CUDA code lives under backends/cuda/.
option(MTRT_CUDA "Build the CUDA GPU backend" OFF)

include(FetchContent)

# --- dependencies -----------------------------------------------------------
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_Declare(nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
  GIT_SHALLOW TRUE)
set(JSON_BuildTests OFF CACHE INTERNAL "")

FetchContent_MakeAvailable(googletest nlohmann_json)

# --- core library -----------------------------------------------------------
add_library(mtrt STATIC
  src/tensor/tensor.cpp
  src/graph/graph.cpp
  src/ops/registry.cpp
  src/ops/kernels_elementwise.cpp
  src/ops/kernels_matmul.cpp
  src/ops/kernels_activation.cpp
  src/ops/kernels_fused.cpp
  src/ops/kernels_transformer.cpp
  src/ops/kernels_gpt.cpp
  src/memory/memory_planner.cpp
  src/optimizer/fusion.cpp
  src/optimizer/const_fold.cpp
  src/profiler/profiler.cpp
  src/frontend/json/json_loader.cpp
  src/runtime/executor.cpp
  backends/cpu/gemm.cpp
  backends/cpu/thread_pool.cpp)

target_include_directories(mtrt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# Repo root so "backends/cpu/..." headers resolve for the lib, tests, benches.
target_include_directories(mtrt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# Internal cross-file headers under src/ (e.g. src/ops/kernels.h).
target_include_directories(mtrt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(mtrt PRIVATE nlohmann_json::nlohmann_json)
target_compile_options(mtrt PRIVATE -Wall -Wextra)

if(MTRT_SANITIZERS)
  target_compile_options(mtrt PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(mtrt PUBLIC -fsanitize=address,undefined)
endif()

# --- CUDA backend (only when MTRT_CUDA=ON) ----------------------------------
if(MTRT_CUDA)
  enable_language(CUDA)
  add_subdirectory(backends/cuda)
endif()

# --- tools & benchmarks -----------------------------------------------------
add_subdirectory(tools)
add_subdirectory(benchmarks)

enable_testing()
add_subdirectory(tests)
