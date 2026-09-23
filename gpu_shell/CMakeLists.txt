# duckdb_gpu — DuckDB shell with the VectisDB optimizer extension statically linked.
#
# This is what turns "the routing decision says GPU" into "the query actually executed on the GPU":
# it links the extension sources (optimizer + physical operator + converters) and cuda_engine against
# this repo's already-built duckdb_static(.lib|.a), and registers the optimizer before opening the
# database.
#
# Build instructions and the DuckDB source pin are in ../SETUP.md.
# BUILD_CORE must point to a DuckDB build at the matching source revision, with core_functions and
# parquet linked statically.

cmake_minimum_required(VERSION 3.20)

# Must precede project(): on Windows, duckdb_static.lib and cuda_engine are both built with the static
# (/MT) runtime, so this binary has to match or the link fails with LNK2038 (see
# cuda_engine/CMakeLists.txt). Irrelevant on Linux/Clang -- guarded so CMP0091/CMAKE_MSVC_RUNTIME_LIBRARY
# are never touched off Windows.
# CMAKE_HOST_WIN32, not MSVC: this block runs BEFORE project(), and MSVC is only defined by project()
# once the compiler has been probed -- so the guard was silently false and neither line below took effect.
# Symptom (hit for real once a toolchain reinstall forced a regenerate): this target compiled /MD while
# duckdb_static and vector_gpu_cuda_engine are /MT, and the link died in LNK2038 across every core object.
if(CMAKE_HOST_WIN32)
    cmake_policy(SET CMP0091 NEW)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

project(vectisdb_duckdb_shell LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
endif()

set(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(DUCKDB_SOURCE "${REPO_ROOT}/../duckdb-core" CACHE PATH "Path to the pinned DuckDB source tree")
set(BUILD_CORE "${DUCKDB_SOURCE}/build/reldebug" CACHE PATH "Path to its core_functions + parquet build output")
set(EXT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../extension")

# CMAKE_STATIC_LIBRARY_PREFIX/SUFFIX resolve to ("", ".lib") under MSVC and ("lib", ".a") under
# GCC/Clang, so one archive-path helper covers both platforms.
function(duckdb_core_lib out_var relative_dir name)
    set(${out_var} "${BUILD_CORE}/${relative_dir}/${CMAKE_STATIC_LIBRARY_PREFIX}${name}${CMAKE_STATIC_LIBRARY_SUFFIX}" PARENT_SCOPE)
endfunction()

duckdb_core_lib(DUCKDB_STATIC_LIB src duckdb_static)
if(NOT EXISTS "${DUCKDB_STATIC_LIB}")
    message(FATAL_ERROR "${DUCKDB_STATIC_LIB} not found — build DuckDB into ${BUILD_CORE} first (see header comment above).")
endif()

# cuda_engine (the actual GPU executor) as a subproject, so one build produces everything.
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../cuda_engine" cuda_engine_build)

# vector_gpu_host — the engine as a linkable library as well as an executable.
#
# Everything that used to be compiled straight into duckdb_gpu now lives here, plus vector_gpu_host.cpp,
# which wraps it in a small C++ API (extension/include/vector_gpu_host.hpp). Applications can call
# Session::Query() in-process and share one RmmPool instead of starting a separate process per query.
add_library(vector_gpu_host STATIC
    ${EXT_DIR}/src/vector_gpu_host.cpp
    ${EXT_DIR}/src/gpu_offload_extension.cpp
    ${EXT_DIR}/src/table_checker.cpp
    ${EXT_DIR}/src/gpu_logical_operator.cpp
    ${EXT_DIR}/src/physical_gpu_execute.cpp
    ${EXT_DIR}/src/physical_gpu_streaming_projection.cpp
    ${EXT_DIR}/src/expression_translator.cpp
    ${EXT_DIR}/src/expression_cost.cpp
    ${EXT_DIR}/src/vector_converter.cpp
    ${EXT_DIR}/src/table_scanner.cpp
    ${EXT_DIR}/src/gpu_batch_accumulator.cpp
)

# PUBLIC so anything linking this library inherits the include path for vector_gpu_host.hpp.
target_include_directories(vector_gpu_host PUBLIC ${EXT_DIR}/include)
target_include_directories(vector_gpu_host PRIVATE
    ${EXT_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda_engine/include
    ${DUCKDB_SOURCE}/src/include
    ${DUCKDB_SOURCE}/third_party/fmt/include
    ${DUCKDB_SOURCE}/third_party/re2
    ${DUCKDB_SOURCE}/third_party/hyperloglog
    ${DUCKDB_SOURCE}/third_party/fastpforlib
    ${DUCKDB_SOURCE}/third_party/fsst
    ${DUCKDB_SOURCE}/third_party/miniz
    ${DUCKDB_SOURCE}/third_party/utf8proc/include
    ${DUCKDB_SOURCE}/third_party/concurrentqueue
    ${DUCKDB_SOURCE}/third_party/pcg
    ${DUCKDB_SOURCE}/third_party/tdigest
    ${DUCKDB_SOURCE}/third_party/mbedtls/include
    ${DUCKDB_SOURCE}/third_party/yyjson/include
    ${DUCKDB_SOURCE}/third_party/skiplist
)

add_executable(duckdb_gpu main.cpp)
target_include_directories(duckdb_gpu PRIVATE
    ${EXT_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda_engine/include
    ${DUCKDB_SOURCE}/src/include
    ${DUCKDB_SOURCE}/third_party/fmt/include
    ${DUCKDB_SOURCE}/third_party/re2
    ${DUCKDB_SOURCE}/third_party/hyperloglog
    ${DUCKDB_SOURCE}/third_party/fastpforlib
    ${DUCKDB_SOURCE}/third_party/fsst
    ${DUCKDB_SOURCE}/third_party/miniz
    ${DUCKDB_SOURCE}/third_party/utf8proc/include
    ${DUCKDB_SOURCE}/third_party/concurrentqueue
    ${DUCKDB_SOURCE}/third_party/pcg
    ${DUCKDB_SOURCE}/third_party/tdigest
    ${DUCKDB_SOURCE}/third_party/mbedtls/include
    ${DUCKDB_SOURCE}/third_party/yyjson/include
    ${DUCKDB_SOURCE}/third_party/skiplist
)

# Proves the in-process path: two isolated Sessions in ONE process, with per-query stats.
add_executable(test_host_api test_host_api.cpp)
target_link_libraries(test_host_api PRIVATE vector_gpu_host)

# Proves GpuBatchAccumulator/GpuMemoryPool against REAL DuckDB DataChunks and REAL pinned memory. Needs
# duckdb.hpp directly (to manufacture DataChunks by hand) -- unlike test_host_api, which only ever sees
# SQL strings through vector_gpu_host.hpp's deliberately-DuckDB-free API -- so it gets the same raw
# include set as duckdb_gpu/main.cpp rather than inheriting vector_gpu_host's PUBLIC-only include.
add_executable(test_batch_accumulator test_batch_accumulator.cpp)
target_include_directories(test_batch_accumulator PRIVATE
    ${EXT_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda_engine/include
    ${DUCKDB_SOURCE}/src/include
    ${DUCKDB_SOURCE}/third_party/fmt/include
    ${DUCKDB_SOURCE}/third_party/re2
    ${DUCKDB_SOURCE}/third_party/hyperloglog
    ${DUCKDB_SOURCE}/third_party/fastpforlib
    ${DUCKDB_SOURCE}/third_party/fsst
    ${DUCKDB_SOURCE}/third_party/miniz
    ${DUCKDB_SOURCE}/third_party/utf8proc/include
    ${DUCKDB_SOURCE}/third_party/concurrentqueue
    ${DUCKDB_SOURCE}/third_party/pcg
    ${DUCKDB_SOURCE}/third_party/tdigest
    ${DUCKDB_SOURCE}/third_party/mbedtls/include
    ${DUCKDB_SOURCE}/third_party/yyjson/include
    ${DUCKDB_SOURCE}/third_party/skiplist
)
target_link_libraries(test_batch_accumulator PRIVATE vector_gpu_host)

# GpuBatchAccumulator performance benchmark: 3 batch sizes x 5 execution modes, 10 s budget.
add_executable(bench_batch_accumulator bench_batch_accumulator.cpp)
target_include_directories(bench_batch_accumulator PRIVATE
    ${EXT_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda_engine/include
    ${DUCKDB_SOURCE}/src/include
    ${DUCKDB_SOURCE}/third_party/fmt/include
    ${DUCKDB_SOURCE}/third_party/re2
    ${DUCKDB_SOURCE}/third_party/hyperloglog
    ${DUCKDB_SOURCE}/third_party/fastpforlib
    ${DUCKDB_SOURCE}/third_party/fsst
    ${DUCKDB_SOURCE}/third_party/miniz
    ${DUCKDB_SOURCE}/third_party/utf8proc/include
    ${DUCKDB_SOURCE}/third_party/concurrentqueue
    ${DUCKDB_SOURCE}/third_party/pcg
    ${DUCKDB_SOURCE}/third_party/tdigest
    ${DUCKDB_SOURCE}/third_party/mbedtls/include
    ${DUCKDB_SOURCE}/third_party/yyjson/include
    ${DUCKDB_SOURCE}/third_party/skiplist
)
target_link_libraries(bench_batch_accumulator PRIVATE vector_gpu_host)

# Order matters for the static linker: duckdb_static first, then the third-party archives it needs.
duckdb_core_lib(_lib_ext_loader   extension                    duckdb_generated_extension_loader)
duckdb_core_lib(_lib_core_funcs   extension/core_functions     core_functions_extension)
duckdb_core_lib(_lib_parquet      extension/parquet            parquet_extension)
duckdb_core_lib(_lib_fmt          third_party/fmt              duckdb_fmt)
duckdb_core_lib(_lib_re2          third_party/re2              duckdb_re2)
duckdb_core_lib(_lib_miniz        third_party/miniz             duckdb_miniz)
duckdb_core_lib(_lib_utf8proc     third_party/utf8proc          duckdb_utf8proc)
duckdb_core_lib(_lib_hyperloglog  third_party/hyperloglog       duckdb_hyperloglog)
duckdb_core_lib(_lib_fastpforlib  third_party/fastpforlib       duckdb_fastpforlib)
duckdb_core_lib(_lib_fsst         third_party/fsst              duckdb_fsst)
duckdb_core_lib(_lib_mbedtls      third_party/mbedtls           duckdb_mbedtls)
duckdb_core_lib(_lib_yyjson       third_party/yyjson             duckdb_yyjson)
duckdb_core_lib(_lib_zstd         third_party/zstd              duckdb_zstd)
duckdb_core_lib(_lib_skiplist     third_party/skiplist           duckdb_skiplistlib)

# Supplies ExtensionHelper::LoadAllExtensions / LoadExtension, which duckdb_static references but does
# not define — the build generates one loader per configured extension set. MSVC's linker repeatedly
# scans archives until everything resolves, so listing duckdb_static before the loader that satisfies it
# works there; GNU ld resolves in one left-to-right pass and would leave these symbols undefined, so on
# non-MSVC the whole DuckDB archive set is wrapped in --start-group/--end-group to let ld iterate.
set(_duckdb_core_libs
    "${_lib_ext_loader}"
    "${_lib_core_funcs}"
    "${_lib_parquet}"
    "${DUCKDB_STATIC_LIB}"
    "${_lib_fmt}"
    "${_lib_re2}"
    "${_lib_miniz}"
    "${_lib_utf8proc}"
    "${_lib_hyperloglog}"
    "${_lib_fastpforlib}"
    "${_lib_fsst}"
    "${_lib_mbedtls}"
    "${_lib_yyjson}"
    "${_lib_zstd}"
    "${_lib_skiplist}"
)
if(MSVC)
    target_link_libraries(vector_gpu_host PUBLIC vector_gpu_cuda_engine ${_duckdb_core_libs})
else()
    target_link_libraries(vector_gpu_host PUBLIC
        vector_gpu_cuda_engine
        -Wl,--start-group ${_duckdb_core_libs} -Wl,--end-group
    )
endif()

if(WIN32)
    target_link_libraries(vector_gpu_host PUBLIC ws2_32 rstrtmgr bcrypt)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(vector_gpu_host PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
endif()

target_link_libraries(duckdb_gpu PRIVATE vector_gpu_host)

target_compile_definitions(vector_gpu_host PUBLIC DUCKDB_STATIC_BUILD)
if(MSVC)
    target_compile_options(vector_gpu_host PRIVATE /bigobj /EHsc)
    target_compile_options(duckdb_gpu PRIVATE /bigobj /EHsc)
    target_link_options(duckdb_gpu PRIVATE /STACK:16777216)
    target_compile_options(test_host_api PRIVATE /bigobj /EHsc)
    target_compile_options(test_batch_accumulator PRIVATE /bigobj /EHsc)
    target_compile_options(bench_batch_accumulator PRIVATE /bigobj /EHsc)
endif()

enable_testing()
add_test(NAME test_host_api COMMAND test_host_api)
add_test(NAME test_batch_accumulator COMMAND test_batch_accumulator)
add_test(NAME bench_batch_accumulator COMMAND bench_batch_accumulator)
