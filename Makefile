.PHONY: all clean test shell

BUILD_DIR ?= build

all:
	@cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	@cmake --build $(BUILD_DIR) -j4

shell:
	@cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DCHRONOS_BUILD_SHELL=ON
	@cmake --build $(BUILD_DIR) --target duckdb_gpu -j4

test:
	@cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DCHRONOS_BUILD_TESTS=ON
	@cmake --build $(BUILD_DIR) --target test_plan_shape test_gpu_stream_pipeline -j4
	@ctest --test-dir $(BUILD_DIR) --output-on-failure

clean:
	@rm -rf $(BUILD_DIR)
