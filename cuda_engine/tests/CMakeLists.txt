# One executable per component, matching the file-per-test-suite layout used elsewhere in this scaffold.
# No external test framework dependency — see test_framework.hpp. Each binary's exit code (0 = all
# checks passed) is what `ctest` uses for pass/fail.

function(vector_gpu_add_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE vector_gpu_cuda_engine)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# Pure C++, no GPU required — always safe to run even on a machine without an NVIDIA device.
vector_gpu_add_test(test_kernel_cache test_kernel_cache.cpp)
vector_gpu_add_test(test_gpu_logger test_gpu_logger.cpp)
# The routing half of the cost model (AnalyzePlanShape) -- plan arithmetic only, no device involved.
vector_gpu_add_test(test_plan_shape test_plan_shape.cpp)

# Require a real GPU + CUDA context.
vector_gpu_add_test(test_rmm_pool test_rmm_pool.cpp)
vector_gpu_add_test(test_gpu_memory_pool test_gpu_memory_pool.cpp)
vector_gpu_add_test(test_gpu_filter test_gpu_filter.cu)
vector_gpu_add_test(test_code_generator test_code_generator.cpp)
vector_gpu_add_test(test_gpu_engine test_gpu_engine.cpp)
vector_gpu_add_test(test_gpu_executor test_gpu_executor.cpp)
# The three-stream projection pipeline, against real pinned ring slots and real kernels.
vector_gpu_add_test(test_gpu_stream_pipeline test_gpu_stream_pipeline.cpp)

# Built but deliberately NOT registered with add_test: this is a profiling harness, not a pass/fail check.
# Whether three CUDA streams overlap is a property of the driver and the card, so it is measured with
# `nsys` (gpu_shell/bench_stream_pipeline.sh) rather than asserted. See the file's own header.
add_executable(bench_stream_pipeline bench_stream_pipeline.cpp)
target_link_libraries(bench_stream_pipeline PRIVATE vector_gpu_cuda_engine)
target_include_directories(bench_stream_pipeline PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# Compares the working-set model against real device memory while a plan runs -- see the file header for
# why it measures rather than asserting the model against itself.
vector_gpu_add_test(test_gpu_cost_model test_gpu_cost_model.cpp)
