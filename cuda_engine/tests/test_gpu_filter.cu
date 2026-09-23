#include "test_framework.hpp"
#include "gpu_filter_ops.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <vector>

using namespace vector_gpu;

namespace {

void CheckCuda(cudaError_t status, const char *what) {
	if (status != cudaSuccess) {
		std::cerr << "CUDA error in " << what << ": " << cudaGetErrorString(status) << "\n";
		std::exit(1); // a raw CUDA failure here means the test harness itself is broken, not a test result
	}
}

//! Runs FilterGreaterThanInt32 on `host_input` and returns the filtered host-side result plus count.
//! Handles the H2D/D2H copies so each test case just deals in plain std::vector<int32_t>.
std::vector<int32_t> RunFilter(const std::vector<int32_t> &host_input, int32_t threshold, size_t &out_count) {
	int32_t *device_input = nullptr;
	int32_t *device_output = nullptr;
	auto n = host_input.size();

	if (n > 0) {
		CheckCuda(cudaMalloc(&device_input, n * sizeof(int32_t)), "cudaMalloc input");
		CheckCuda(cudaMalloc(&device_output, n * sizeof(int32_t)), "cudaMalloc output");
		CheckCuda(cudaMemcpy(device_input, host_input.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice),
		         "cudaMemcpy H2D");
	}

	FilterGreaterThanInt32(device_input, n, threshold, device_output, &out_count);

	std::vector<int32_t> host_output(out_count);
	if (out_count > 0) {
		CheckCuda(cudaMemcpy(host_output.data(), device_output, out_count * sizeof(int32_t), cudaMemcpyDeviceToHost),
		         "cudaMemcpy D2H");
	}

	if (device_input) {
		cudaFree(device_input);
	}
	if (device_output) {
		cudaFree(device_output);
	}
	return host_output;
}

std::vector<int32_t> CpuReference(const std::vector<int32_t> &input, int32_t threshold) {
	std::vector<int32_t> result;
	std::copy_if(input.begin(), input.end(), std::back_inserter(result), [&](int32_t v) { return v > threshold; });
	return result;
}

} // namespace

static void TestEmptyInput() {
	size_t out_count = 999; // deliberately non-zero sentinel to prove it gets overwritten to 0
	auto result = RunFilter({}, 5, out_count);
	CHECK(out_count == 0);
	CHECK(result.empty());
}

static void TestAllElementsPass() {
	std::vector<int32_t> input = {10, 20, 30, 40};
	size_t out_count = 0;
	auto result = RunFilter(input, 0, out_count);
	CHECK(out_count == 4);
	CHECK(result == CpuReference(input, 0));
}

static void TestNoElementsPass() {
	std::vector<int32_t> input = {1, 2, 3, 4};
	size_t out_count = 0;
	auto result = RunFilter(input, 100, out_count);
	CHECK(out_count == 0);
	CHECK(result.empty());
}

static void TestSomeElementsPass() {
	std::vector<int32_t> input = {5, -3, 10, 0, 7, -1, 100};
	size_t out_count = 0;
	auto result = RunFilter(input, 4, out_count);
	auto expected = CpuReference(input, 4);
	CHECK(out_count == expected.size());
	CHECK(result == expected);
}

static void TestSingleElementPass() {
	size_t out_count = 0;
	auto result = RunFilter({42}, 10, out_count);
	CHECK(out_count == 1);
	CHECK(result.size() == 1 && result[0] == 42);
}

static void TestSingleElementFail() {
	size_t out_count = 0;
	auto result = RunFilter({5}, 10, out_count);
	CHECK(out_count == 0);
	CHECK(result.empty());
}

static void TestBoundaryValueNotStrictlyGreater() {
	// Predicate is strictly ">", so a value equal to the threshold must NOT pass — an easy off-by-one to
	// get wrong (>= vs >).
	size_t out_count = 0;
	auto result = RunFilter({10}, 10, out_count);
	CHECK(out_count == 0);
}

static void TestLargeArrayMatchesCpuReference() {
	// Exercises multi-block grid dispatch (kBlockSize-sized launches elsewhere; here it's Thrust's own
	// internal grid sizing) with a size that isn't a round multiple of any particular block size.
	const size_t n = 1'000'003;
	std::vector<int32_t> input(n);
	for (size_t i = 0; i < n; i++) {
		input[i] = static_cast<int32_t>((i * 2654435761u) % 1000); // deterministic pseudo-random spread
	}
	size_t out_count = 0;
	auto result = RunFilter(input, 500, out_count);
	auto expected = CpuReference(input, 500);
	CHECK(out_count == expected.size());
	CHECK(result == expected);
}

static void TestNegativeThreshold() {
	std::vector<int32_t> input = {-10, -5, -1, 0, 1, 5};
	size_t out_count = 0;
	auto result = RunFilter(input, -5, out_count);
	auto expected = CpuReference(input, -5);
	CHECK(result == expected);
}

static void TestInt32ExtremesNoOverflow() {
	// Edge case: values at the int32 boundary must compare correctly with no overflow-related surprises
	// in the predicate (a naive `value - threshold > 0` style rewrite, which this implementation does NOT
	// do, would overflow here — worth locking in explicitly).
	std::vector<int32_t> input = {INT32_MIN, INT32_MIN + 1, 0, INT32_MAX - 1, INT32_MAX};
	size_t out_count = 0;
	auto result = RunFilter(input, INT32_MAX - 1, out_count);
	CHECK(out_count == 1);
	CHECK(result.size() == 1 && result[0] == INT32_MAX);
}

static void TestNullOutCountThrows() {
	int32_t dummy_in = 1, dummy_out = 0;
	CHECK_THROWS(FilterGreaterThanInt32(&dummy_in, 1, 0, &dummy_out, nullptr));
}

static void TestNullBuffersWithNonzeroNThrows() {
	size_t out_count = 0;
	CHECK_THROWS(FilterGreaterThanInt32(nullptr, 1, 0, nullptr, &out_count));
}

static void TestLatchedUnrelatedErrorDoesNotPoisonFilter() {
	// Session-7 KI-5 regression: with a stale error latched by an earlier, unrelated CUDA call, a fully
	// valid filter call used to throw a spurious, misattributed exception ("invalid device ordinal")
	// out of Thrust's internals. The filter now clears the stale latch on entry.
	auto latch = cudaMemcpy(nullptr, nullptr, 16, cudaMemcpyDeviceToDevice); // deliberately invalid
	CHECK(latch != cudaSuccess); // the latch is set

	std::vector<int32_t> input = {1, 2, 3, 4, 5, 6, 7, 8};
	size_t out_count = 0;
	auto result = RunFilter(input, 4, out_count); // must succeed despite the stale latch
	CHECK(out_count == 4);
	CHECK(result == CpuReference(input, 4));
	cudaGetLastError(); // leave a clean slate for the next test either way
}

static void TestInPlaceCompactionThrows() {
	// Session-7 regression: input == output is UB per Thrust/CUB (it happened to produce correct
	// results on this GPU, but nothing guarantees that). Exact aliasing is now rejected explicitly.
	int32_t *device_buf = nullptr;
	CheckCuda(cudaMalloc(&device_buf, 8 * sizeof(int32_t)), "cudaMalloc in-place buf");
	std::vector<int32_t> host = {1, 2, 3, 4, 5, 6, 7, 8};
	CheckCuda(cudaMemcpy(device_buf, host.data(), 8 * sizeof(int32_t), cudaMemcpyHostToDevice), "H2D in-place");
	size_t out_count = 0;
	CHECK_THROWS(FilterGreaterThanInt32(device_buf, 8, 4, device_buf, &out_count));
	cudaFree(device_buf);
}

int main() {
	RUN_TEST(TestEmptyInput);
	RUN_TEST(TestAllElementsPass);
	RUN_TEST(TestNoElementsPass);
	RUN_TEST(TestSomeElementsPass);
	RUN_TEST(TestSingleElementPass);
	RUN_TEST(TestSingleElementFail);
	RUN_TEST(TestBoundaryValueNotStrictlyGreater);
	RUN_TEST(TestLargeArrayMatchesCpuReference);
	RUN_TEST(TestNegativeThreshold);
	RUN_TEST(TestInt32ExtremesNoOverflow);
	RUN_TEST(TestNullOutCountThrows);
	RUN_TEST(TestNullBuffersWithNonzeroNThrows);
	RUN_TEST(TestLatchedUnrelatedErrorDoesNotPoisonFilter);
	RUN_TEST(TestInPlaceCompactionThrows);
	TEST_MAIN_EPILOGUE();
}
