# vector_gpu_offload — out-of-tree DuckDB loadable extension.
#
# Zero core changes: this hooks in purely through OptimizerExtension::Register and
# LogicalExtensionOperator, both public extension-facing APIs. Not added to the root DuckDB
# CMakeLists.txt or extension/CMakeLists.txt — build it standalone against a DuckDB install, following
# the extension-template / extension-ci-tools conventions:
#
#   https://github.com/duckdb/extension-template
#
# Typical wiring (once dropped into an extension-template checkout):
#   1. Point the template's vcpkg/CMake config at DuckDB commit af1b4a9bd2 instead of a released
#      DuckDB version, so headers match the version pinned for VectisDB 1.0.0.
#   2. Add this directory's sources to the template's extension target.
#   3. Link vector_gpu_cuda_engine (see ../cuda_engine/CMakeLists.txt) as an additional static library.

cmake_minimum_required(VERSION 3.20)
project(vector_gpu_offload CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXTENSION_SOURCES
    src/gpu_offload_extension.cpp
    src/table_checker.cpp
    src/gpu_logical_operator.cpp
    src/physical_gpu_execute.cpp
    src/physical_gpu_streaming_projection.cpp
    src/expression_translator.cpp
    src/expression_cost.cpp
    src/vector_converter.cpp
    src/table_scanner.cpp
    src/gpu_batch_accumulator.cpp
    src/vector_gpu_host.cpp
)

# NOTE: this target intentionally does not declare its DuckDB dependency here — extension-template's
# build harness supplies the duckdb::duckdb / duckdb_static target and the DUCKDB_CPP_EXTENSION_ENTRY
# macro plumbing. Add it back once this is dropped into a real template checkout, e.g.:
#
#   add_library(vector_gpu_offload_extension STATIC ${EXTENSION_SOURCES})
#   target_link_libraries(vector_gpu_offload_extension PRIVATE duckdb_static vector_gpu_cuda_engine)
#   target_include_directories(vector_gpu_offload_extension PRIVATE include ../cuda_engine/include)

add_library(vector_gpu_offload_sources_only OBJECT ${EXTENSION_SOURCES})
target_include_directories(vector_gpu_offload_sources_only PRIVATE include ../cuda_engine/include)
