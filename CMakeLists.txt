cmake_minimum_required(VERSION 3.20)
project(ChronosGpuEngine LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(CHRONOS_BUILD_SHELL "Build standalone duckdb_gpu shell" ON)
option(CHRONOS_BUILD_TESTS "Build GPU engine tests" ON)

add_subdirectory(cuda_engine)
add_subdirectory(extension)

if(CHRONOS_BUILD_SHELL AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/gpu_shell/CMakeLists.txt")
    add_subdirectory(gpu_shell)
endif()
