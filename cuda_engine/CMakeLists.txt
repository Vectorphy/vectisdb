# cuda_engine — standalone library, no DuckDB dependency. Buildable and testable on its own against a
# real CUDA install (verified against CUDA Toolkit 13.3 on Windows + an NVIDIA GTX 1650, compute
# capability 7.5). Not wired into the root DuckDB CMakeLists.txt (per the minimum-change goal).
#
# Build + run tests:
#   cmake -S vector-gpu-engine/cuda_engine -B vector-gpu-engine/cuda_engine/build -G Ninja ^
#         -DCMAKE_BUILD_TYPE=Release
#   cmake --build vector-gpu-engine/cuda_engine/build
#   ctest --test-dir vector-gpu-engine/cuda_engine/build --output-on-failure
#
# On Windows: run from a Developer Command Prompt / after vcvarsall.bat x64, or point
# CMAKE_CUDA_HOST_COMPILER at cl.exe explicitly, so nvcc can find the MSVC host compiler. On Linux/WSL2,
# the GCC/Clang host compiler is picked up automatically -- no equivalent step needed.
#
# libcudf/RMM-dependent operators (hash join, group-by) remain source-level stubs — see
# docs/KNOWN_ISSUES.md for why (RAPIDS needs WSL2/Linux; this is where that dependency can now actually be
# wired up). Everything else here (RMM-pool-equivalent allocator, NVRTC code generator, kernel cache,
# Thrust-based filter) is real, builds on both platforms, and is covered by cuda_engine/tests/.

cmake_minimum_required(VERSION 3.20)

# Must be set before project(): forces the /MT (static) MSVC runtime to match duckdb_static.lib's
# runtime setting. Without this, cuda_engine defaults to /MD (CMake's normal MSVC default), and linking
# this library directly into an extension binary alongside duckdb_static fails with LNK2038 runtime
# library mismatches -- hit this for real the first time extension code actually called into cuda_engine
# (TableChecker::HasVramHeadroom -> RmmPool). See docs/EXECUTION_TRACKER.md. Irrelevant on Linux/Clang,
# where there is no MSVC runtime to select -- guarded so CMP0091/CMAKE_MSVC_RUNTIME_LIBRARY are never
# touched off Windows.
# CMAKE_HOST_WIN32, not MSVC: this block runs BEFORE project(), and MSVC is only defined by project()
# once the compiler has been probed -- so the guard was silently false and neither line below took effect.
# Symptom (hit for real once a toolchain reinstall forced a regenerate): this target compiled /MD while
# duckdb_static and vector_gpu_cuda_engine are /MT, and the link died in LNK2038 across every core object.
if(CMAKE_HOST_WIN32)
    cmake_policy(SET CMP0091 NEW)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

project(vector_gpu_cuda_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # 75 = Turing (this dev machine's GTX 1650). Override with -DCMAKE_CUDA_ARCHITECTURES=<arch> for
    # other target hardware.
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

find_package(CUDAToolkit REQUIRED)

if(MSVC)
    # CCCL (Thrust's backend, pulled in transitively via <thrust/...> in gpu_filter.cu) requires the
    # conforming MSVC preprocessor. Without this flag nvcc's host-compiler pass fails with C1189 on any
    # translation unit that includes a CCCL/Thrust header.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/Zc:preprocessor")
else()
    # GCC/Clang host-compiler pass: -fPIC so this static lib can be linked into a shared extension
    # binary, -O3 to match the MSVC /O2-equivalent release performance the Windows build gets by default.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-O3,-fPIC")
endif()

add_library(vector_gpu_cuda_engine STATIC
    src/gpu_engine.cpp
    src/gpu_cost_model.cpp
    src/gpu_executor.cu
    src/gpu_logger.cpp
    src/rmm_pool.cpp
    src/kernel_cache.cpp
    src/code_generator.cpp
    src/pinned_buffer_pool.cpp
    src/gpu_memory_pool.cpp
    src/gpu_stream_pipeline.cu
    src/gpu_column_cache.cpp
    src/operators/gpu_hash_join.cu
    src/operators/gpu_cross_join.cu
    src/operators/gpu_groupby.cu
    src/operators/gpu_filter.cu
)

# gpu_executor.cu uses extended __device__ lambdas (thrust::copy_if predicate).
target_compile_options(vector_gpu_cuda_engine PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_include_directories(vector_gpu_cuda_engine PUBLIC include)
# src/ holds gpu_executor_internal.hpp, the implementation-private interface shared between
# gpu_executor.cu and src/operators/*.cu. PRIVATE: it is deliberately not part of the public API.
target_include_directories(vector_gpu_cuda_engine PRIVATE src)
target_link_libraries(vector_gpu_cuda_engine PUBLIC
    CUDA::cudart
    CUDA::nvrtc
    CUDA::cuda_driver
)
target_compile_features(vector_gpu_cuda_engine PUBLIC cxx_std_17)

option(VECTOR_GPU_BUILD_TESTS "Build cuda_engine unit tests" ON)
if(VECTOR_GPU_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
