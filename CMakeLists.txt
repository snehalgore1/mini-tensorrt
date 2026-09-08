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
# ONNX frontend. OFF by default so the core build needs no protobuf; ON adds a
# second real frontend (needs protobuf: `brew install protobuf`). See DESIGN D3.
option(MTRT_ONNX "Build the ONNX frontend (requires protobuf)" OFF)

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
  # MUST run before enable_language(CUDA): that call auto-populates
  # CMAKE_CUDA_ARCHITECTURES with nvcc's default (sm_52 on CUDA 12), which would
  # make the `if(NOT DEFINED)` in backends/cuda/CMakeLists a no-op and compile the
  # FP16 tensor-core kernel (guarded by __CUDA_ARCH__>=700) out to an empty stub.
  # Default to the Colab T4 (sm_75); still overridable with -DCMAKE_CUDA_ARCHITECTURES=.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
  endif()
  enable_language(CUDA)
  add_subdirectory(backends/cuda)
endif()

# --- ONNX frontend (only when MTRT_ONNX=ON) ---------------------------------
# Generates onnx.pb.{h,cc} from the vendored schema with protoc, then builds the
# loader into a separate mtrt_onnx lib so the core `mtrt` (and its consumers) stay
# protobuf-free. The onnx.pb types never escape onnx_loader.cpp (invariant 2).
if(MTRT_ONNX)
  find_package(Protobuf CONFIG REQUIRED)
  message(STATUS "MTRT_ONNX: protobuf ${Protobuf_VERSION}")
  set(ONNX_PB_DIR ${CMAKE_BINARY_DIR}/onnx_gen)
  file(MAKE_DIRECTORY ${ONNX_PB_DIR})
  add_custom_command(
    OUTPUT ${ONNX_PB_DIR}/onnx.pb.cc ${ONNX_PB_DIR}/onnx.pb.h
    COMMAND protobuf::protoc
    ARGS --cpp_out=${ONNX_PB_DIR}
         -I ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx onnx.proto
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx/onnx.proto
    COMMENT "protoc: generating onnx.pb.{h,cc}")

  add_library(mtrt_onnx STATIC
    ${ONNX_PB_DIR}/onnx.pb.cc
    src/frontend/onnx/onnx_loader.cpp)
  # Force protoc to run before onnx_loader.cpp compiles (it includes onnx.pb.h).
  set_source_files_properties(src/frontend/onnx/onnx_loader.cpp PROPERTIES
    OBJECT_DEPENDS ${ONNX_PB_DIR}/onnx.pb.h)
  target_include_directories(mtrt_onnx PRIVATE ${ONNX_PB_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(mtrt_onnx PUBLIC mtrt PRIVATE protobuf::libprotobuf)
  target_compile_definitions(mtrt_onnx PUBLIC MTRT_HAVE_ONNX)
  # protoc-generated code is not warning-clean; don't fail the core's -Werror-ish flags on it.
  target_compile_options(mtrt_onnx PRIVATE -w)
endif()

# --- tools & benchmarks -----------------------------------------------------
add_subdirectory(tools)
add_subdirectory(benchmarks)

enable_testing()
add_subdirectory(tests)
