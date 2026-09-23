// gpu_stream_pipeline.cu
//
// The three-stream, event-ordered projection pipeline declared in include/gpu_stream_pipeline.hpp. Read
// that header first -- it carries the design rationale (why the shape is this narrow, why the streams are
// non-blocking, why every allocation happens in the constructor). This file is the mechanics.
//
// The whole point, per submitted batch:
//
//     cudaMemcpyAsync(..., stream_h2d)            // upload batch N's columns
//     cudaEventRecord(h2d_done, stream_h2d)
//     cudaStreamWaitEvent(stream_exec, h2d_done)  // NOT cudaStreamSynchronize -- the CPU does not wait
//     <fused NVRTC kernels>                          on stream_exec
//     cudaEventRecord(exec_done, stream_exec)
//     cudaStreamWaitEvent(stream_d2h, exec_done)
//     cudaMemcpyAsync(..., stream_d2h)            // download batch N's results
//     cudaEventRecord(d2h_done, stream_d2h)
//
// Every one of those calls returns immediately. The dependency chain lives entirely on the device, so
// batch N+1's upload starts the instant batch N's upload finishes -- it does not wait for batch N's
// kernel, and batch N's download does not hold up batch N+2's upload. The calling thread's only
// interaction with completion is cudaEventQuery (non-blocking) and, when it has genuinely run out of
// other work, one cudaEventSynchronize on the oldest batch.

#include "gpu_stream_pipeline.hpp"

#include "code_generator.hpp"
#include "gpu_executor_internal.hpp" // TypeSize
#include "gpu_logger.hpp"
#include "kernel_cache.hpp"

#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include <algorithm>
#include <array>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

namespace vector_gpu {

namespace {

#define VGPU_CUDA_CHECK(call, what)                                                                                  \
	do {                                                                                                              \
		auto _status = (call);                                                                                       \
		if (_status != cudaSuccess) {                                                                                \
			throw std::runtime_error(std::string("gpu_stream_pipeline: ") + (what) + " failed: " +                   \
			                         cudaGetErrorString(_status));                                                   \
		}                                                                                                             \
	} while (0)

//! The ring must hold one slot per in-flight upload plus one for the producer to fill, or the pipeline
//! can never reach kDepth and the H2D/kernel/D2H of successive batches serialize -- see CanSubmit's slot
//! accounting, and gpu_memory_pool.hpp's kSlotCount comment for the profile that established it.
static_assert(GpuMemoryPool::kSlotCount >= GpuStreamPipeline::kDepth + 1,
              "GpuMemoryPool::kSlotCount must leave the producer a free slot with kDepth uploads in flight");

constexpr unsigned int kUnpackBlock = 256;
constexpr unsigned int kMaxGrid = 65535u;

//! Expands a packed Arrow-style validity bitmap (GpuColumn::validity's format: 1 bit/row, LSB-first,
//! bit=1 valid) into a dense bool array, one byte per row.
//!
//! Deliberately a second copy of gpu_executor.cu's UnpackValidityKernel rather than a shared symbol: that
//! one lives in that file's anonymous namespace, and this file may not call into the rest of the executor
//! anyway -- everything it launches must land on stream_exec (see the header on why the streams are
//! non-blocking), which a shared helper taking no stream argument could not promise. Both are four lines
//! implementing one documented bitmap layout; if that layout ever changes, GpuColumn::validity's own
//! comment is the single source of truth both must follow.
__global__ void UnpackValidityKernel(const uint8_t *packed, bool *out, uint64_t n) {
	for (uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
		out[idx] = (packed[idx / 8] & (uint8_t(1) << (idx % 8))) != 0;
	}
}

//! Process-wide kernel cache + generator for the pipeline path.
//!
//! Separate from gpu_executor.cu's SharedGenerator, on purpose and for a reason that is about correctness,
//! not tidiness. A KernelCache hands out BORROWED handles (see kernel_cache.hpp) that stay valid only
//! until the next Insert into that same cache, and it owns/frees what it evicts. Sharing one cache between
//! the legacy single-stream executor and this pipeline would let one path's compile evict a handle the
//! other path is about to launch. Two caches cost one extra NVRTC compile per expression per path, once
//! per process -- both are static, so a repeated query shape still skips compilation entirely, which is
//! the property the cache exists for.
CodeGenerator &PipelineGenerator() {
	static KernelCache cache(256);
	static CodeGenerator generator(cache);
	return generator;
}

//! The name gpu_executor.cu's ExecuteProjection gives a projection's i-th output -- and therefore the name
//! an operator ABOVE a projection uses to reference it (see CollectBindingNames in
//! extension/src/gpu_offload_extension.cpp, which calls this convention load-bearing for exactly this
//! reason). This pipeline follows it for both its intermediate levels and its final result.
std::string ProjectionOutputName(size_t index) {
	return "col" + std::to_string(index);
}

//! Walks the PROJECTION chain from `plan` down to its SCAN leaf. Returns the SCAN, and fills `levels` with
//! the projections BOTTOM-UP (closest to the scan first), which is execution order. Returns nullptr if the
//! chain is anything other than one-or-more PROJECTIONs ending in a SCAN.
//!
//! A chain, not just a single projection: real DuckDB plans reaching this operator are routinely
//! PROJECTION -> PROJECTION -> GET (the optimizer inserts a narrowing projection over the scan), so
//! accepting only a direct SCAN child would make this operator unreachable on the queries it exists for.
const GpuPlanNode *CollectProjectionChain(const GpuPlanNode &plan, std::vector<const GpuPlanNode *> &levels) {
	std::vector<const GpuPlanNode *> top_down;
	const GpuPlanNode *node = &plan;
	while (node->op_type == GpuOpType::PROJECTION) {
		if (node->children.size() != 1 || !node->children[0] || node->expressions.empty()) {
			return nullptr;
		}
		top_down.push_back(node);
		node = node->children[0].get();
	}
	if (top_down.empty() || node->op_type != GpuOpType::SCAN) {
		return nullptr;
	}
	levels.assign(top_down.rbegin(), top_down.rend());
	return node;
}

//! TypeSize(), but as a query rather than a throw -- Supports() must answer false for a type with no fixed
//! device width, not propagate an exception out of a routing check.
bool HasFixedWidth(GpuValueType type) {
	return type != GpuValueType::DICTIONARY_STRING;
}

template <typename T>
__global__ void AddStateKernel(T *data, uint64_t n, const T *d_running_state) {
	T state = *d_running_state;
	for (uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
		data[idx] += state;
	}
}

template <typename T>
__global__ void UpdateStateKernel(const T *data, uint64_t n, T *d_running_state) {
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		*d_running_state = data[n - 1];
	}
}

template <typename T>
void RunCumulativeSumImpl(void *d_out, uint64_t rows, void *d_state, cudaStream_t stream) {
	T *data = static_cast<T *>(d_out);
	T *state = static_cast<T *>(d_state);
	thrust::inclusive_scan(thrust::cuda::par.on(stream), data, data + rows, data);
	
	auto grid = static_cast<unsigned int>((rows + 255) / 256);
	if (grid > 65535) grid = 65535;
	AddStateKernel<<<grid, 256, 0, stream>>>(data, rows, state);
	UpdateStateKernel<<<1, 1, 0, stream>>>(data, rows, state);
}

void RunCumulativeSum(GpuValueType type, void *d_out, uint64_t rows, void *d_state, cudaStream_t stream) {
	switch (type) {
	case GpuValueType::INT16: RunCumulativeSumImpl<int16_t>(d_out, rows, d_state, stream); break;
	case GpuValueType::INT32: RunCumulativeSumImpl<int32_t>(d_out, rows, d_state, stream); break;
	case GpuValueType::INT64: RunCumulativeSumImpl<int64_t>(d_out, rows, d_state, stream); break;
	case GpuValueType::FLOAT32: RunCumulativeSumImpl<float>(d_out, rows, d_state, stream); break;
	case GpuValueType::FLOAT64: RunCumulativeSumImpl<double>(d_out, rows, d_state, stream); break;
	default: throw std::runtime_error("unsupported type for RUNNING_SUM");
	}
}

} // namespace

//! One projection level's per-stage device storage.
struct LevelBuffers {
	std::vector<void *> d_out;
	std::vector<bool *> d_out_valid;
	//! Device-resident kernel argument arrays, one per expression, uploaded ONCE at construction (the
	//! buffers they point at never move). This is what lets the launch go through
	//! CodeGenerator::LaunchPreloaded and touch no shared scratch -- see that method's header comment for
	//! the cross-stream hazard it avoids.
	std::vector<const void **> d_expr_inputs;
	std::vector<const bool **> d_expr_valid;
};

//! Per-batch device + pinned-host storage. One of these per kDepth; every buffer in it is allocated once,
//! in the pipeline's constructor, sized for max_rows, and reused for every batch that lands in this stage.
//!
//! At namespace scope rather than in the anonymous namespace above: GpuStreamPipelineImpl (declared in the
//! header, so external linkage) holds an array of these, and giving an externally-linked type a member of
//! internal-linkage type is the kind of thing compilers are entitled to complain about.
struct StageBuffers {
	// Input side, one entry per declared input column.
	std::vector<void *> d_data;
	std::vector<bool *> d_valid;     // dense, 1 byte/row -- written only when a batch carries NULLs
	std::vector<uint8_t *> d_packed; // the uploaded Arrow bitmap, unpacked into d_valid on stream_exec
	// One per projection level, bottom-up.
	std::vector<LevelBuffers> levels;
	//! Pinned D2H landing area for the TOP level's outputs: every column's data region, then every
	//! column's dense-validity region, at the offsets the pipeline computed once.
	uint8_t *h_out = nullptr;
	//! Host-side packed bitmaps handed out as GpuColumn::validity, one per output column. Plain heap, not
	//! pinned: nothing copies INTO these from the device, they are filled by a host loop over h_out.
	std::vector<std::vector<uint8_t>> h_packed;

	cudaEvent_t h2d_done = nullptr;
	cudaEvent_t exec_done = nullptr;
	cudaEvent_t d2h_done = nullptr;

	// Per-submission state, reset on every Submit into this stage.
	uint64_t rows = 0;
	bool any_null = false;
	//! The batch's pinned INPUT slot, held only until h2d_done fires -- see GpuStreamBatch's comment for
	//! why it is released that early rather than when the results are read.
	GpuMemoryPoolSlot input_slot;
	bool input_released = true;
};

//! One projection level of the plan, resolved once at construction.
struct PipelineLevel {
	std::vector<GpuExpr> expressions;
	std::vector<GpuValueType> output_types;
	std::vector<size_t> output_sizes;
	//! Per expression: which of the PREVIOUS level's outputs (or, for level 0, which declared input
	//! column) feeds each of its `inputs[i]` slots.
	std::vector<std::vector<int>> expr_inputs;
};

struct GpuStreamPipelineImpl {
	// Declared schema of every submitted batch.
	std::vector<std::string> input_names;
	std::vector<GpuValueType> input_types;
	std::vector<size_t> input_sizes;
	// The projection chain, bottom-up: levels.front() reads the scanned columns, levels.back() produces
	// the result.
	std::vector<PipelineLevel> levels;
	// Offsets within a stage's pinned output buffer, for the top level's columns.
	std::vector<size_t> out_data_offset;
	std::vector<size_t> out_valid_offset;
	size_t h_out_bytes = 0;

	uint64_t max_rows = 0;

	cudaStream_t stream_h2d = nullptr;
	cudaStream_t stream_exec = nullptr;
	cudaStream_t stream_d2h = nullptr;

	std::array<StageBuffers, GpuStreamPipeline::kStageCount> stages;
	std::deque<int> in_flight;    // stage indices, oldest first
	std::vector<int> free_stages; // stage indices available to Submit
	size_t input_slots_held = 0;
	//! Set when a CUDA call inside Submit failed. The streams may then hold half-issued work referencing a
	//! stage's buffers, so no further batch may be submitted -- the caller must destroy the pipeline (the
	//! destructor drains all three streams before freeing anything) and fail the query.
	bool failed = false;

	std::vector<void *> device_allocs; // everything to cudaFree, in one place
	std::vector<std::vector<void *>> d_running_state; // per level, per expression

	const PipelineLevel &top() const {
		return levels.back();
	}

	~GpuStreamPipelineImpl() {
		// Drain BEFORE freeing: an abandoned pipeline (a submission that threw, a query cancelled
		// mid-drain) can still have H2D/kernel/D2H work queued that reads or writes these exact buffers.
		// Best-effort -- at process teardown the runtime may already be gone, and a destructor has nothing
		// to report a failure to.
		for (auto stream : {stream_h2d, stream_exec, stream_d2h}) {
			if (stream != nullptr) {
				cudaStreamSynchronize(stream);
			}
		}
		for (auto &stage : stages) {
			for (auto event : {stage.h2d_done, stage.exec_done, stage.d2h_done}) {
				if (event != nullptr) {
					cudaEventDestroy(event);
				}
			}
			if (stage.h_out != nullptr) {
				cudaFreeHost(stage.h_out);
			}
			// stage.input_slot returns itself to GpuMemoryPool when this stage is destroyed, which happens
			// after this body -- i.e. after the streams above have drained, so no in-flight H2D can still
			// be reading the pinned memory it hands back.
		}
		for (auto *ptr : device_allocs) {
			cudaFree(ptr);
		}
		for (auto stream : {stream_h2d, stream_exec, stream_d2h}) {
			if (stream != nullptr) {
				cudaStreamDestroy(stream);
			}
		}
	}

	void *AllocDevice(size_t bytes) {
		if (bytes == 0) {
			bytes = 1;
		}
		void *ptr = nullptr;
		VGPU_CUDA_CHECK(cudaMalloc(&ptr, bytes), "cudaMalloc for a pipeline stage buffer");
		device_allocs.push_back(ptr);
		return ptr;
	}
};

namespace {

//! Resolves one expression's input column names against the names its level reads from: the declared
//! scan columns for level 0, the previous level's "col<i>" outputs above that. Returns false if any name
//! is unresolvable, which is a routing answer ("this pipeline cannot run that plan"), not an error.
bool ResolveExpressionInputs(const GpuExpr &expr, const std::vector<std::string> &source_names,
                             std::vector<int> &out_indices) {
	out_indices.clear();
	for (auto &name : expr.input_columns) {
		int found = -1;
		for (size_t i = 0; i < source_names.size(); i++) {
			if (source_names[i] == name) {
				found = static_cast<int>(i);
				break;
			}
		}
		if (found < 0) {
			return false;
		}
		out_indices.push_back(found);
	}
	return true;
}

//! The column names a given level reads: the scan's columns at the bottom, the level below's outputs
//! everywhere above it.
std::vector<std::string> SourceNamesForLevel(const std::vector<std::string> &input_names,
                                             const std::vector<const GpuPlanNode *> &levels, size_t level) {
	if (level == 0) {
		return input_names;
	}
	std::vector<std::string> names;
	names.reserve(levels[level - 1]->expressions.size());
	for (size_t i = 0; i < levels[level - 1]->expressions.size(); i++) {
		names.push_back(ProjectionOutputName(i));
	}
	return names;
}

} // namespace

bool GpuStreamPipeline::Supports(const GpuPlanNode &plan) {
	std::vector<const GpuPlanNode *> levels;
	auto *scan = CollectProjectionChain(plan, levels);
	if (scan == nullptr || scan->column_names.empty()) {
		return false;
	}
	std::vector<int> indices;
	for (size_t level = 0; level < levels.size(); level++) {
		auto source_names = SourceNamesForLevel(scan->column_names, levels, level);
		for (auto &expr : levels[level]->expressions) {
			if (!HasFixedWidth(expr.output_type) || expr.generated_cuda_source.empty()) {
				return false;
			}
			if (!ResolveExpressionInputs(expr, source_names, indices)) {
				return false;
			}
		}
	}
	return true;
}

GpuStreamPipeline::GpuStreamPipeline(const GpuPlanNode &plan, const std::vector<std::string> &input_names,
                                     const std::vector<GpuValueType> &input_types, uint64_t max_rows_per_batch)
    : impl_(new GpuStreamPipelineImpl()) {
	// Every failure below throws, and impl_ is already a fully-constructed member by then, so its
	// destructor runs and releases whatever this constructor had allocated up to that point. That is the
	// whole cleanup story -- there is no second path to get wrong.
	std::vector<const GpuPlanNode *> plan_levels;
	auto *scan = CollectProjectionChain(plan, plan_levels);
	if (scan == nullptr) {
		throw std::runtime_error("gpu_stream_pipeline: plan is not a chain of PROJECTIONs over a SCAN");
	}
	if (input_names.size() != input_types.size() || input_names.empty()) {
		throw std::runtime_error("gpu_stream_pipeline: input_names and input_types must be the same "
		                         "non-empty length");
	}
	if (max_rows_per_batch == 0) {
		throw std::runtime_error("gpu_stream_pipeline: max_rows_per_batch must be > 0");
	}

	auto &impl = *impl_;
	impl.max_rows = max_rows_per_batch;
	impl.input_names = input_names;
	impl.input_types = input_types;
	for (auto type : input_types) {
		impl.input_sizes.push_back(TypeSize(type)); // throws for a type with no device representation
	}

	// Resolve every expression's input column names to positions ONCE. Doing it per submission would put
	// a string comparison per column per batch on the critical path, and would also let a mismatch
	// surface mid-scan instead of here, while the caller can still decline the plan.
	for (size_t level = 0; level < plan_levels.size(); level++) {
		auto source_names = SourceNamesForLevel(input_names, plan_levels, level);
		PipelineLevel resolved;
		for (auto &expr : plan_levels[level]->expressions) {
			std::vector<int> indices;
			if (!ResolveExpressionInputs(expr, source_names, indices)) {
				throw std::runtime_error("gpu_stream_pipeline: a projection expression references a column its "
				                         "input does not produce");
			}
			resolved.expressions.push_back(expr);
			resolved.expr_inputs.push_back(std::move(indices));
			resolved.output_types.push_back(expr.output_type);
			resolved.output_sizes.push_back(TypeSize(expr.output_type));
		}
		impl.levels.push_back(std::move(resolved));
	}

	// Pinned output layout, for the top level only: all data regions, then all dense-validity regions.
	size_t offset = 0;
	for (auto size : impl.top().output_sizes) {
		impl.out_data_offset.push_back(offset);
		offset += static_cast<size_t>(max_rows_per_batch) * size;
	}
	for (size_t e = 0; e < impl.top().output_sizes.size(); e++) {
		impl.out_valid_offset.push_back(offset);
		offset += static_cast<size_t>(max_rows_per_batch); // 1 byte/row
	}
	impl.h_out_bytes = offset;

	// Same single-device convention as PinnedBufferPool/GpuMemoryPool, and for the same secondary reason:
	// making the first real Runtime API call from inside this constructor keeps the runtime's exit-time
	// teardown registered before anything this object owns (see gpu_memory_pool.hpp's file header).
	VGPU_CUDA_CHECK(cudaSetDevice(0), "cudaSetDevice(0)");

	// cudaStreamNonBlocking on all three -- see the header. With plain cudaStreamCreate these would each
	// implicitly synchronize with the legacy default stream, hence transitively with each other, and the
	// pipeline would run correctly while overlapping nothing.
	VGPU_CUDA_CHECK(cudaStreamCreateWithFlags(&impl.stream_h2d, cudaStreamNonBlocking), "cudaStreamCreate(h2d)");
	VGPU_CUDA_CHECK(cudaStreamCreateWithFlags(&impl.stream_exec, cudaStreamNonBlocking), "cudaStreamCreate(exec)");
	VGPU_CUDA_CHECK(cudaStreamCreateWithFlags(&impl.stream_d2h, cudaStreamNonBlocking), "cudaStreamCreate(d2h)");

	for (size_t s = 0; s < kStageCount; s++) {
		auto &stage = impl.stages[s];
		// cudaEventDisableTiming: these events exist purely to order streams and to be polled, never to
		// be read with cudaEventElapsedTime. The flag removes the timestamp bookkeeping from every record.
		VGPU_CUDA_CHECK(cudaEventCreateWithFlags(&stage.h2d_done, cudaEventDisableTiming), "cudaEventCreate(h2d)");
		VGPU_CUDA_CHECK(cudaEventCreateWithFlags(&stage.exec_done, cudaEventDisableTiming), "cudaEventCreate(exec)");
		VGPU_CUDA_CHECK(cudaEventCreateWithFlags(&stage.d2h_done, cudaEventDisableTiming), "cudaEventCreate(d2h)");

		for (size_t i = 0; i < input_names.size(); i++) {
			stage.d_data.push_back(impl.AllocDevice(static_cast<size_t>(max_rows_per_batch) * impl.input_sizes[i]));
			stage.d_valid.push_back(static_cast<bool *>(impl.AllocDevice(static_cast<size_t>(max_rows_per_batch))));
			stage.d_packed.push_back(
			    static_cast<uint8_t *>(impl.AllocDevice((static_cast<size_t>(max_rows_per_batch) + 7) / 8)));
		}
		for (auto &level : impl.levels) {
			LevelBuffers buffers;
			for (size_t e = 0; e < level.expressions.size(); e++) {
				buffers.d_out.push_back(
				    impl.AllocDevice(static_cast<size_t>(max_rows_per_batch) * level.output_sizes[e]));
				buffers.d_out_valid.push_back(
				    static_cast<bool *>(impl.AllocDevice(static_cast<size_t>(max_rows_per_batch))));
			}
			stage.levels.push_back(std::move(buffers));
		}
		for (size_t e = 0; e < impl.top().expressions.size(); e++) {
			stage.h_packed.emplace_back((static_cast<size_t>(max_rows_per_batch) + 7) / 8, 0xFFu);
		}

		// Upload the per-expression kernel argument arrays. A synchronous cudaMemcpy is correct HERE and
		// only here: it is construction-time, once, before any stream carries work -- the rule this file
		// follows everywhere else (no synchronizing call between batches) is about the steady state.
		for (size_t level = 0; level < impl.levels.size(); level++) {
			for (auto &indices : impl.levels[level].expr_inputs) {
				std::vector<const void *> host_inputs;
				std::vector<const bool *> host_valid;
				for (auto index : indices) {
					auto i = static_cast<size_t>(index);
					if (level == 0) {
						host_inputs.push_back(stage.d_data[i]);
						host_valid.push_back(stage.d_valid[i]);
					} else {
						host_inputs.push_back(stage.levels[level - 1].d_out[i]);
						host_valid.push_back(stage.levels[level - 1].d_out_valid[i]);
					}
				}
				auto count = host_inputs.empty() ? size_t(1) : host_inputs.size();
				auto *inputs_device = static_cast<const void **>(impl.AllocDevice(count * sizeof(const void *)));
				auto *valid_device = static_cast<const bool **>(impl.AllocDevice(count * sizeof(const bool *)));
				if (!host_inputs.empty()) {
					VGPU_CUDA_CHECK(cudaMemcpy(inputs_device, host_inputs.data(),
					                           host_inputs.size() * sizeof(const void *), cudaMemcpyHostToDevice),
					                "upload of an expression's input pointer array");
					VGPU_CUDA_CHECK(cudaMemcpy(valid_device, host_valid.data(),
					                           host_valid.size() * sizeof(const bool *), cudaMemcpyHostToDevice),
					                "upload of an expression's validity pointer array");
				}
				stage.levels[level].d_expr_inputs.push_back(inputs_device);
				stage.levels[level].d_expr_valid.push_back(valid_device);
			}
		}

		VGPU_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void **>(&stage.h_out), impl.h_out_bytes,
		                              cudaHostAllocPortable),
		                "cudaHostAlloc for a pipeline stage's pinned output buffer");
		impl.free_stages.push_back(static_cast<int>(s));
	}

	// Compile every expression NOW, so an expression NVRTC rejects fails while the caller can still
	// decline the plan cleanly -- not after a scan has already been opened and rows consumed. The handles
	// are deliberately not kept: a cached handle is a borrow valid only until the next Insert (see
	// kernel_cache.hpp), so Submit re-fetches, which on a warm cache is a hash plus a map lookup.
	for (size_t level = 0; level < impl.levels.size(); level++) {
		std::vector<void *> level_states;
		for (size_t e = 0; e < impl.levels[level].expressions.size(); e++) {
			auto &expr = impl.levels[level].expressions[e];
			PipelineGenerator().CompileOrFetch(expr);
			if (expr.window_primitive == GpuWindowPrimitive::RUNNING_SUM) {
				void *ptr = impl.AllocDevice(impl.levels[level].output_sizes[e]);
				VGPU_CUDA_CHECK(cudaMemset(ptr, 0, impl.levels[level].output_sizes[e]), "init running state");
				level_states.push_back(ptr);
			} else {
				level_states.push_back(nullptr);
			}
		}
		impl.d_running_state.push_back(level_states);
	}

	VGPU_LOG(LogLevel::INFO, "stream_pipeline_init",
	         "\"depth\":" + std::to_string(kDepth) + ",\"stages\":" + std::to_string(kStageCount) +
	             ",\"max_rows\":" + std::to_string(max_rows_per_batch) +
	             ",\"inputs\":" + std::to_string(input_names.size()) + ",\"levels\":" +
	             std::to_string(impl.levels.size()) + ",\"outputs\":" +
	             std::to_string(impl.top().expressions.size()));
}

GpuStreamPipeline::~GpuStreamPipeline() = default;

const std::vector<GpuValueType> &GpuStreamPipeline::output_types() const {
	return impl_->top().output_types;
}

size_t GpuStreamPipeline::in_flight() const {
	return impl_->in_flight.size();
}

void GpuStreamPipeline::Poll() {
	auto &impl = *impl_;
	// Oldest first, and stop at the first batch whose upload has not landed: every H2D is issued on
	// stream_h2d, so they complete in submission order and a later one cannot be done before an earlier
	// one. Scanning past the first not-ready entry would be wasted queries, not extra correctness.
	for (auto stage_index : impl.in_flight) {
		auto &stage = impl.stages[static_cast<size_t>(stage_index)];
		if (stage.input_released) {
			continue;
		}
		auto status = cudaEventQuery(stage.h2d_done);
		if (status == cudaErrorNotReady) {
			break;
		}
		if (status != cudaSuccess) {
			impl.failed = true;
			throw std::runtime_error(std::string("gpu_stream_pipeline: cudaEventQuery on the H2D event "
			                                     "failed: ") +
			                         cudaGetErrorString(status));
		}
		// The upload has genuinely completed, so nothing is reading this pinned slot any more: hand it
		// straight back so the producer can fill the next batch into it while this batch's kernel and
		// download are still running.
		stage.input_slot = GpuMemoryPoolSlot();
		stage.input_released = true;
		impl.input_slots_held--;
	}
}

bool GpuStreamPipeline::CanSubmit() {
	Poll();
	auto &impl = *impl_;
	if (impl.failed || impl.in_flight.size() >= kDepth) {
		return false;
	}
	// Not implied by the check above: a stage stays checked out while the consumer still holds its
	// GpuStreamResult, so with a result in hand there can be fewer free stages than kDepth - in_flight.
	if (impl.free_stages.empty()) {
		return false;
	}
	// SLOT ACCOUNTING, and why it is +2 rather than +1. After the submit this is gating, the pipeline will
	// hold input_slots_held + 1 pinned ring slots. The producer then needs ONE more to accumulate the next
	// batch into -- and it takes that slot from inside GpuBatchAccumulator::TakeReadyBatch, which BLOCKS
	// on GpuMemoryPool until its specific next slot in the cyclic rotation frees. Only this class ever
	// frees one, and it only does so from Poll(), which the producer is not calling while it is blocked.
	// So allowing a submit that leaves zero free slots is not a stall, it is a deadlock. Hence: refuse
	// unless the submit still leaves the producer a slot.
	if (impl.input_slots_held + 2 > GpuMemoryPool::kSlotCount) {
		return false;
	}
	return true;
}

void GpuStreamPipeline::Submit(GpuStreamBatch batch) {
	auto &impl = *impl_;
	if (!CanSubmit()) {
		throw std::runtime_error("gpu_stream_pipeline: Submit called when the pipeline cannot accept a batch "
		                         "(check CanSubmit() first)");
	}
	if (batch.row_count == 0) {
		throw std::runtime_error("gpu_stream_pipeline: refusing an empty batch -- a zero-row kernel launch is a "
		                         "caller bug, and an empty batch has no results to hand back");
	}
	if (batch.row_count > impl.max_rows) {
		throw std::runtime_error("gpu_stream_pipeline: batch of " + std::to_string(batch.row_count) +
		                         " rows exceeds the " + std::to_string(impl.max_rows) +
		                         "-row size every stage buffer was allocated for");
	}
	if (batch.columns.size() != impl.input_names.size()) {
		throw std::runtime_error("gpu_stream_pipeline: batch has " + std::to_string(batch.columns.size()) +
		                         " columns but the declared input schema has " +
		                         std::to_string(impl.input_names.size()));
	}
	if (impl.free_stages.empty()) {
		// Unreachable via CanSubmit(), which checks exactly this -- kept because the alternative is
		// reading back() off an empty vector.
		throw std::runtime_error("gpu_stream_pipeline: no free stage available (every stage is either in "
		                         "flight or still checked out by an unreleased GpuStreamResult)");
	}

	auto stage_index = impl.free_stages.back();
	auto &stage = impl.stages[static_cast<size_t>(stage_index)];
	stage.rows = batch.row_count;
	stage.any_null = false;
	for (size_t i = 0; i < batch.columns.size(); i++) {
		auto &column = batch.columns[i];
		// By NAME, not just by position: these columns are consumed positionally from here on, so a
		// producer that reordered them would silently reinterpret one column's bytes as another's.
		if (column.name != impl.input_names[i] || column.type != impl.input_types[i]) {
			throw std::runtime_error("gpu_stream_pipeline: batch column " + std::to_string(i) + " ('" +
			                         column.name + "') does not match the declared schema entry '" +
			                         impl.input_names[i] + "'");
		}
		if (column.row_count != batch.row_count) {
			throw std::runtime_error("gpu_stream_pipeline: column '" + column.name + "' carries " +
			                         std::to_string(column.row_count) + " rows but the batch declares " +
			                         std::to_string(batch.row_count));
		}
		if (column.validity != nullptr) {
			stage.any_null = true;
		}
	}

	try {
		auto rows = static_cast<size_t>(batch.row_count);

		// ---- stream_h2d: upload this batch's columns straight out of the producer's pinned slot.
		for (size_t i = 0; i < batch.columns.size(); i++) {
			auto &column = batch.columns[i];
			VGPU_CUDA_CHECK(cudaMemcpyAsync(stage.d_data[i], column.data, rows * impl.input_sizes[i],
			                                cudaMemcpyHostToDevice, impl.stream_h2d),
			                "H2D upload of a batch column");
			if (column.validity != nullptr) {
				VGPU_CUDA_CHECK(cudaMemcpyAsync(stage.d_packed[i], column.validity, (rows + 7) / 8,
				                                cudaMemcpyHostToDevice, impl.stream_h2d),
				                "H2D upload of a batch column's validity bitmap");
			}
		}
		VGPU_CUDA_CHECK(cudaEventRecord(stage.h2d_done, impl.stream_h2d), "cudaEventRecord(h2d_done)");

		// ---- stream_exec: gate on the upload, then compute. cudaStreamWaitEvent, NOT
		// cudaStreamSynchronize: the dependency is expressed to the device and this thread walks away.
		VGPU_CUDA_CHECK(cudaStreamWaitEvent(impl.stream_exec, stage.h2d_done, 0), "cudaStreamWaitEvent(h2d_done)");
		if (stage.any_null) {
			// Only when this batch actually carries NULLs. A batch with none launches the fused kernels
			// with no validity arguments at all (see below), so none of this runs and the common case
			// costs nothing.
			auto grid = static_cast<unsigned int>((rows + kUnpackBlock - 1) / kUnpackBlock);
			if (grid > kMaxGrid) {
				grid = kMaxGrid; // the kernel's grid-stride loop covers the remainder
			}
			for (size_t i = 0; i < batch.columns.size(); i++) {
				if (batch.columns[i].validity != nullptr) {
					UnpackValidityKernel<<<grid, kUnpackBlock, 0, impl.stream_exec>>>(stage.d_packed[i],
					                                                                 stage.d_valid[i], rows);
				} else {
					// This column has no NULLs in THIS batch, but a sibling does, so the kernel will read
					// every referenced column's validity array. Fill it with "valid" rather than leaving
					// whatever the previous batch in this stage wrote there. cudaMemsetAsync of 1 per byte
					// is exactly `true` for a bool array.
					VGPU_CUDA_CHECK(cudaMemsetAsync(stage.d_valid[i], 1, rows, impl.stream_exec),
					                "cudaMemsetAsync of an all-valid validity array");
				}
			}
		}
		// Levels bottom-up, all on stream_exec, so level L+1's kernels are ordered after level L's without
		// a single host-side wait -- one stream IS the dependency.
		for (size_t level = 0; level < impl.levels.size(); level++) {
			auto &spec = impl.levels[level];
			auto &buffers = stage.levels[level];
			for (size_t e = 0; e < spec.expressions.size(); e++) {
				auto kernel = PipelineGenerator().CompileOrFetch(spec.expressions[e]);
				PipelineGenerator().LaunchPreloaded(kernel, buffers.d_expr_inputs[e],
				                                    static_cast<int>(spec.expr_inputs[e].size()), buffers.d_out[e],
				                                    rows, /*selection=*/nullptr, impl.stream_exec,
				                                    stage.any_null ? buffers.d_expr_valid[e] : nullptr,
				                                    stage.any_null ? buffers.d_out_valid[e] : nullptr);
				if (spec.expressions[e].window_primitive == GpuWindowPrimitive::RUNNING_SUM) {
					void *d_state = impl.d_running_state[level][e];
					RunCumulativeSum(spec.output_types[e], buffers.d_out[e], rows, d_state, impl.stream_exec);
				}
			}
		}
		VGPU_CUDA_CHECK(cudaEventRecord(stage.exec_done, impl.stream_exec), "cudaEventRecord(exec_done)");

		// ---- stream_d2h: gate on the kernels, then download the top level into pinned host memory.
		VGPU_CUDA_CHECK(cudaStreamWaitEvent(impl.stream_d2h, stage.exec_done, 0),
		                "cudaStreamWaitEvent(exec_done)");
		auto &top_buffers = stage.levels.back();
		for (size_t e = 0; e < impl.top().expressions.size(); e++) {
			VGPU_CUDA_CHECK(cudaMemcpyAsync(stage.h_out + impl.out_data_offset[e], top_buffers.d_out[e],
			                                rows * impl.top().output_sizes[e], cudaMemcpyDeviceToHost,
			                                impl.stream_d2h),
			                "D2H copy of a result column");
			if (stage.any_null) {
				VGPU_CUDA_CHECK(cudaMemcpyAsync(stage.h_out + impl.out_valid_offset[e], top_buffers.d_out_valid[e],
				                                rows, cudaMemcpyDeviceToHost, impl.stream_d2h),
				                "D2H copy of a result column's validity");
			}
		}
		VGPU_CUDA_CHECK(cudaEventRecord(stage.d2h_done, impl.stream_d2h), "cudaEventRecord(d2h_done)");
	} catch (...) {
		// Half-issued work referencing this stage's buffers may already be queued on one or more streams,
		// so the stage cannot be reused and no further batch may be submitted. The pipeline is now
		// terminal; its destructor drains the streams before freeing anything.
		impl.failed = true;
		throw;
	}

	impl.free_stages.pop_back();
	stage.input_slot = std::move(batch.slot);
	stage.input_released = false;
	impl.input_slots_held++;
	impl.in_flight.push_back(stage_index);
	VGPU_LOG(LogLevel::DEBUG, "stream_submit",
	         "\"stage\":" + std::to_string(stage_index) + ",\"rows\":" + std::to_string(batch.row_count) +
	             ",\"in_flight\":" + std::to_string(impl.in_flight.size()) + ",\"nullable\":" +
	             (stage.any_null ? std::string("true") : std::string("false")));
}

namespace {

//! Builds the caller-visible result for a stage whose D2H is known to have completed, and pops it off the
//! in-flight queue. Everything here is host work over already-landed pinned memory.
void TakeCompletedStage(GpuStreamPipelineImpl &impl, int stage_index, std::vector<GpuColumn> &columns,
                        uint64_t &row_count) {
	auto &stage = impl.stages[static_cast<size_t>(stage_index)];
	auto &top = impl.top();
	columns.clear();
	columns.reserve(top.expressions.size());
	for (size_t e = 0; e < top.expressions.size(); e++) {
		GpuColumn column;
		column.name = ProjectionOutputName(e);
		column.type = top.output_types[e];
		column.data = stage.h_out + impl.out_data_offset[e];
		column.row_count = stage.rows;
		if (stage.any_null) {
			// Pack the dense per-row bytes the kernel wrote into the Arrow bitmap GpuColumn documents.
			// Host-side and exactly once, mirroring ExecuteGpuPlan's own final pack -- packing on the
			// device would need threads to share a byte.
			auto &packed = stage.h_packed[e];
			std::fill(packed.begin(), packed.end(), 0xFFu);
			const auto *dense = reinterpret_cast<const bool *>(stage.h_out + impl.out_valid_offset[e]);
			for (uint64_t r = 0; r < stage.rows; r++) {
				if (!dense[r]) {
					packed[r / 8] &= ~(uint8_t(1) << (r % 8));
				}
			}
			column.validity = packed.data();
		}
		columns.push_back(std::move(column));
	}
	row_count = stage.rows;

	// The D2H finished, so the H2D certainly did (they are chained through exec_done). If Poll has not
	// reclaimed this batch's input slot yet, do it now rather than holding it for the result's lifetime.
	if (!stage.input_released) {
		stage.input_slot = GpuMemoryPoolSlot();
		stage.input_released = true;
		impl.input_slots_held--;
	}
	impl.in_flight.pop_front();
}

} // namespace

bool GpuStreamPipeline::TryTake(GpuStreamResult &out) {
	auto &impl = *impl_;
	Poll();
	if (impl.in_flight.empty()) {
		return false;
	}
	auto stage_index = impl.in_flight.front();
	auto &stage = impl.stages[static_cast<size_t>(stage_index)];
	auto status = cudaEventQuery(stage.d2h_done);
	if (status == cudaErrorNotReady) {
		return false;
	}
	if (status != cudaSuccess) {
		impl.failed = true;
		throw std::runtime_error(std::string("gpu_stream_pipeline: cudaEventQuery on the D2H event failed: ") +
		                         cudaGetErrorString(status));
	}
	// Release whatever `out` was holding FIRST: that returns its stage to the free list, and this call is
	// about to hand it a different one.
	out.Reset();
	TakeCompletedStage(impl, stage_index, out.columns_, out.row_count_);
	out.pipeline_ = this;
	out.stage_ = stage_index;
	return true;
}

bool GpuStreamPipeline::TakeBlocking(GpuStreamResult &out) {
	auto &impl = *impl_;
	if (impl.in_flight.empty()) {
		return false;
	}
	auto stage_index = impl.in_flight.front();
	auto &stage = impl.stages[static_cast<size_t>(stage_index)];
	// The ONLY blocking call in this class. Waiting on the D2H event, not on the stream, so it wakes as
	// soon as THIS batch has landed even if later batches are still queued behind it on stream_d2h.
	auto status = cudaEventSynchronize(stage.d2h_done);
	if (status != cudaSuccess) {
		impl.failed = true;
		throw std::runtime_error(std::string("gpu_stream_pipeline: cudaEventSynchronize on the D2H event "
		                                     "failed: ") +
		                         cudaGetErrorString(status));
	}
	out.Reset();
	TakeCompletedStage(impl, stage_index, out.columns_, out.row_count_);
	out.pipeline_ = this;
	out.stage_ = stage_index;
	return true;
}

void GpuStreamPipeline::ReleaseStage(int stage) {
	if (stage < 0) {
		return;
	}
	impl_->free_stages.push_back(stage);
}

GpuStreamResult::GpuStreamResult(GpuStreamResult &&other) noexcept
    : pipeline_(other.pipeline_), stage_(other.stage_), columns_(std::move(other.columns_)),
      row_count_(other.row_count_) {
	other.pipeline_ = nullptr;
	other.stage_ = -1;
	other.row_count_ = 0;
}

GpuStreamResult &GpuStreamResult::operator=(GpuStreamResult &&other) noexcept {
	if (this != &other) {
		Reset();
		pipeline_ = other.pipeline_;
		stage_ = other.stage_;
		columns_ = std::move(other.columns_);
		row_count_ = other.row_count_;
		other.pipeline_ = nullptr;
		other.stage_ = -1;
		other.row_count_ = 0;
	}
	return *this;
}

void GpuStreamResult::Reset() {
	if (pipeline_ != nullptr) {
		pipeline_->ReleaseStage(stage_);
	}
	pipeline_ = nullptr;
	stage_ = -1;
	columns_.clear();
	row_count_ = 0;
}

GpuStreamResult::~GpuStreamResult() {
	Reset();
}

} // namespace vector_gpu
