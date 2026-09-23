#include "gpu_engine.hpp"
#include "rmm_pool.hpp"
#include "gpu_logger.hpp"

#include <cuda_runtime.h>

namespace vector_gpu {

namespace {
//! Free VRAM in MiB, or -1 if the driver cannot say.
int64_t FreeVramMiB() {
	size_t free_bytes = 0, total_bytes = 0;
	if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
		return -1;
	}
	return static_cast<int64_t>(free_bytes / (1024 * 1024));
}

//! Below this much free VRAM, WDDM starts paging device allocations to host RAM. It does NOT raise
//! CUDA_ERROR_OUT_OF_MEMORY -- the query still succeeds and still reports as a GPU execution, just up
//! to ~12x slower (measured, session 26). Nothing else in the trace distinguishes that from a genuinely
//! fast run, which is the "claims GPU, isn't faster" symptom in docs/KNOWN_ISSUES.md (session 29).
constexpr int64_t kPagingRiskFreeMiB = 128;
} // namespace

//! The real recursive plan executor (gpu_executor.cu): uploads scanned columns, runs PROJECTION/FILTER
//! as NVRTC-compiled fused kernels plus Thrust compaction, and copies only the final result back.
//! Returns false with out.error_message set for anything it cannot execute — never throws.
bool ExecuteGpuPlan(const GpuPlanNode &plan, const std::vector<GpuColumn> &inputs, const GpuExecuteOptions &options,
                    GpuExecutionResult &out);

GpuEngine &GpuEngine::Instance() {
	static GpuEngine instance;
	return instance;
}

GpuExecutionResult GpuEngine::ExecutePlan(const GpuPlanNode &plan, const std::vector<GpuColumn> &inputs,
                                          const GpuExecuteOptions &options) {
	GpuExecutionResult result;

	// Session-7 fix: the header promises "returns success=false with an explanatory error", but
	// EnsureInitialized throws when CUDA is unavailable (no driver / no device) — which used to escape
	// the engine boundary into a caller written to the contract. Convert to the documented failure shape.
	try {
		RmmPool::Instance().EnsureInitialized();
	} catch (const std::exception &e) {
		result.success = false;
		result.error_message = std::string("GpuEngine::ExecutePlan: GPU memory pool initialization failed "
		                                   "(is a CUDA device available?): ") +
		                       e.what();
		return result;
	}

	// Executes the whole plan TREE (not a single node): ExecuteGpuPlan recurses through
	// SCAN/FILTER/PROJECTION, keeping every intermediate in device memory and copying back only the
	// final result. Unsupported shapes (joins, aggregates, nullable columns) come back as
	// success=false with an explanatory message; the optimizer additionally declines to offload those
	// plans in the first place, so they run on DuckDB's CPU path rather than failing.
	// Sample free VRAM around execution. A plan can clear routing's headroom check, offload, and then
	// quietly page against host RAM -- reporting as an ordinary successful GPU execution the whole time.
	// Logging the low-water mark is what makes a silently-slow success distinguishable after the fact
	// from a genuinely fast one. Cheap (~microseconds) next to any plan worth offloading.
	const auto free_before = FreeVramMiB();
	ExecuteGpuPlan(plan, inputs, options, result);
	const auto free_after = FreeVramMiB();
	if (free_before >= 0 && free_after >= 0 && GpuLogger::Instance().Enabled(LogLevel::INFO)) {
		const bool paging_risk = free_after < kPagingRiskFreeMiB || free_before < kPagingRiskFreeMiB;
		VGPU_LOG(LogLevel::INFO, "vram_watermark",
		         "\"free_before_mib\":" + std::to_string(free_before) + ",\"free_after_mib\":" +
		             std::to_string(free_after) + ",\"paging_risk\":" + (paging_risk ? "true" : "false") +
		             ",\"threshold_mib\":" + std::to_string(kPagingRiskFreeMiB));
	}
	return result;
}

} // namespace vector_gpu
