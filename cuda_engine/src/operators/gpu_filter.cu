// gpu_filter.cu
//
// FilterGreaterThanInt32: real Thrust-based implementation, see gpu_filter_ops.hpp for the contract.
// This is the Windows-native stand-in for what would be cudf::apply_boolean_mask / a comparison
// binary-op in the full libcudf-based design (see docs/ARCHITECTURE.md and docs/KNOWN_ISSUES.md for why
// libcudf itself isn't used here) — Thrust ships with the CUDA Toolkit and needed no extra setup on this
// machine, unlike RAPIDS.
//
// ExecuteFilterStub below is a legacy per-operator entry point and is not used by the active recursive
// GpuEngine::ExecutePlan dispatcher. The dispatcher handles FILTER nodes through the fused expression
// path. FilterGreaterThanInt32 remains a separate Thrust primitive with its own unit tests.

#include "gpu_filter_ops.hpp"
#include "gpu_engine.hpp"

#include <new>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

namespace vector_gpu {

namespace {
struct GreaterThanPredicate {
	int32_t threshold;
	__host__ __device__ bool operator()(int32_t value) const {
		return value > threshold;
	}
};
} // namespace

void FilterGreaterThanInt32(const int32_t *input, size_t n, int32_t threshold, int32_t *output, size_t *out_count) {
	if (out_count == nullptr) {
		throw std::invalid_argument("FilterGreaterThanInt32: out_count must not be null");
	}
	if (n == 0) {
		*out_count = 0;
		return;
	}
	if (input == nullptr || output == nullptr) {
		throw std::invalid_argument("FilterGreaterThanInt32: input/output must not be null when n > 0");
	}
	// Thrust/CUB forbids overlapping input/output for copy_if (UB). Exact aliasing — the realistic
	// in-place-compaction mistake — is detectable, so reject it. (Partial overlap can't be reliably
	// detected from raw pointers; the header documents the non-overlap requirement.)
	if (input == output) {
		throw std::invalid_argument("FilterGreaterThanInt32: in-place compaction (input == output) is not "
		                            "supported — Thrust/CUB requires non-overlapping input and output");
	}

	// KI-5 fix: consume any error latched by an earlier, unrelated CUDA call. Verified on hardware
	// (session 7): with a stale cudaErrorInvalidValue latched, a fully valid copy_if over valid device
	// buffers threw "invalid device ordinal" out of Thrust's internal setup. Clearing the latch here
	// keeps someone else's error from failing this call; our own failures are still surfaced by the
	// checks below.
	cudaGetLastError();

	thrust::device_ptr<const int32_t> input_ptr(input);
	thrust::device_ptr<int32_t> output_ptr(output);

	thrust::device_ptr<int32_t> end;
	try {
		end = thrust::copy_if(thrust::device, input_ptr, input_ptr + n, output_ptr, GreaterThanPredicate {threshold});
	} catch (const std::bad_alloc &e) {
		// Thrust's temporary-storage allocation failure raises std::bad_alloc, which is NOT derived from
		// std::runtime_error — rethrow as the documented exception type so callers written to the header
		// contract ("Throws std::runtime_error") actually catch it.
		throw std::runtime_error(std::string("FilterGreaterThanInt32: thrust::copy_if temporary allocation "
		                                     "failed (device out of memory): ") +
		                         e.what());
	}
	// (thrust::system_error already derives from std::runtime_error and needs no translation.)

	auto sync_status = cudaGetLastError();
	if (sync_status != cudaSuccess) {
		throw std::runtime_error(std::string("FilterGreaterThanInt32: thrust::copy_if failed: ") +
		                         cudaGetErrorString(sync_status));
	}

	*out_count = static_cast<size_t>(end - output_ptr);
}

bool ExecuteFilterStub(const GpuPlanNode &plan, const std::vector<GpuColumn> &inputs, GpuExecutionResult &out) {
	(void)plan;
	(void)inputs;
	out.success = false;
	out.error_message = "gpu_filter: GpuPlanNode dispatch not implemented yet (LogicalOperator -> GpuPlanNode "
	                    "translation is still a TODO — see docs/EXECUTION_TRACKER.md #16). The underlying "
	                    "primitive, FilterGreaterThanInt32, is implemented and unit-tested independently.";
	return false;
}

} // namespace vector_gpu
