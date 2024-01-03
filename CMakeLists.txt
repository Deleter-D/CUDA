cmake_minimum_required(VERSION 3.17)

project(CUDAExample)

enable_language(CUDA)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/executable)
set(TARGET_ARCHITECTURES "50\;60\;70\;80" CACHE STRING INTERNAL)
message(${TARGET_ARCHITECTURES})

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(./utils)
add_subdirectory(00_CUDA_official_documentation)
add_subdirectory(01_programming_model)
add_subdirectory(02_execution_model)
add_subdirectory(03_global_memory)
add_subdirectory(04_shared_and_constant_memory)
add_subdirectory(05_stream_and_concurrence)
add_subdirectory(06_adjusting_instruction-level_primitives)