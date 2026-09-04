include(GoogleTest)

add_executable(mtrt_tests
  tensor_test.cpp
  graph_test.cpp
  registry_test.cpp
  kernel_test.cpp
  executor_test.cpp)

target_link_libraries(mtrt_tests PRIVATE mtrt GTest::gtest_main)
target_compile_options(mtrt_tests PRIVATE -Wall -Wextra)

gtest_discover_tests(mtrt_tests)
