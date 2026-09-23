.PHONY: all clean test gpu-test shell

BUILD_DIR ?= build
SHELL_BUILD_DIR ?= build/shell
DUCKDB_SOURCE ?= $(CURDIR)/../duckdb-core
BUILD_CORE ?= $(DUCKDB_SOURCE)/build/reldebug
CUDA_ARCH ?= 75

all:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DVECTISDB_BUILD_GPU_SHELL=OFF -DVECTISDB_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH)
	cmake --build $(BUILD_DIR) --parallel

test: all
	ctest --test-dir $(BUILD_DIR) --output-on-failure -R '^(test_kernel_cache|test_gpu_logger|test_plan_shape)$$'

gpu-test: all
	ctest --test-dir $(BUILD_DIR) --output-on-failure

shell:
	cmake -S . -B $(SHELL_BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DVECTISDB_BUILD_GPU_SHELL=ON -DVECTISDB_BUILD_TESTS=OFF -DDUCKDB_SOURCE=$(DUCKDB_SOURCE) -DBUILD_CORE=$(BUILD_CORE) -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH)
	cmake --build $(SHELL_BUILD_DIR) --target duckdb_gpu --parallel

clean:
	cmake -E rm -rf $(BUILD_DIR)
