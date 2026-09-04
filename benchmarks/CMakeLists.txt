add_executable(bench_model bench_model.cpp)
target_link_libraries(bench_model PRIVATE mtrt)
target_compile_definitions(bench_model PRIVATE
  MTRT_MODELS_DIR="${CMAKE_SOURCE_DIR}/models")
target_compile_options(bench_model PRIVATE -Wall -Wextra)

add_executable(bench_gemm bench_gemm.cpp)
target_link_libraries(bench_gemm PRIVATE mtrt "-framework Accelerate")
target_compile_options(bench_gemm PRIVATE -Wall -Wextra)
