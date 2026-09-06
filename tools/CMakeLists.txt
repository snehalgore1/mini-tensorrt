add_executable(run_gpt2 run_gpt2.cpp)
target_link_libraries(run_gpt2 PRIVATE mtrt)
target_compile_definitions(run_gpt2 PRIVATE
  MTRT_MODELS_DIR="${CMAKE_SOURCE_DIR}/models")
target_compile_options(run_gpt2 PRIVATE -Wall -Wextra)
