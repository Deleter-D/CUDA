cmake_minimum_required(VERSION 3.17)

project(CUDAExample)

enable_language(CUDA)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/executable)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(./utils)
add_subdirectory(00_CUDA_official_documentation)
add_subdirectory(01_programming_model)
add_subdirectory(02_execution_model)
add_subdirectory(03_global_memory)