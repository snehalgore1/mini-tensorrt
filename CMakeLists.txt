cmake_minimum_required(VERSION 3.20)
project(MiniTensorRT CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MTRT_SANITIZERS "Enable ASan/UBSan" OFF)

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
  src/frontend/json/json_loader.cpp
  src/runtime/executor.cpp)

target_include_directories(mtrt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# Internal cross-file headers under src/ (e.g. src/ops/kernels.h).
target_include_directories(mtrt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(mtrt PRIVATE nlohmann_json::nlohmann_json)
target_compile_options(mtrt PRIVATE -Wall -Wextra)

if(MTRT_SANITIZERS)
  target_compile_options(mtrt PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(mtrt PUBLIC -fsanitize=address,undefined)
endif()

# --- tests ------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
