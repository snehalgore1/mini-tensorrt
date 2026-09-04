include(GoogleTest)

add_executable(mtrt_tests
  tensor_test.cpp
  graph_test.cpp
  registry_test.cpp
  kernel_test.cpp
  executor_test.cpp
  golden_kernel_test.cpp
  json_frontend_test.cpp
  model_test.cpp
  memory_test.cpp
  fusion_test.cpp
  const_fold_test.cpp
  transformer_test.cpp)

target_link_libraries(mtrt_tests PRIVATE mtrt GTest::gtest_main)
# So tests can include "tests/support/..." helpers.
target_include_directories(mtrt_tests PRIVATE ${CMAKE_SOURCE_DIR})
# Absolute path to committed golden fixtures, independent of the run cwd.
target_compile_definitions(mtrt_tests PRIVATE
  MTRT_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/models/goldens"
  MTRT_MODELS_DIR="${CMAKE_SOURCE_DIR}/models")
target_compile_options(mtrt_tests PRIVATE -Wall -Wextra)

gtest_discover_tests(mtrt_tests)
