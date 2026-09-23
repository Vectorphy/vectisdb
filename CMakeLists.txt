cmake_minimum_required(VERSION 3.20)
project(VectisDB VERSION 1.0.0 LANGUAGES CXX CUDA)

option(VECTISDB_BUILD_GPU_SHELL "Build the DuckDB-linked VectisDB shell" OFF)
option(VECTISDB_BUILD_TESTS "Build CUDA engine test targets" ON)

if(NOT DEFINED VECTOR_GPU_BUILD_TESTS)
    set(VECTOR_GPU_BUILD_TESTS "${VECTISDB_BUILD_TESTS}" CACHE BOOL "Build CUDA engine tests")
endif()

# The shell adds cuda_engine itself and needs an existing DuckDB core build.
if(VECTISDB_BUILD_GPU_SHELL)
    add_subdirectory(gpu_shell)
else()
    add_subdirectory(cuda_engine)
endif()
