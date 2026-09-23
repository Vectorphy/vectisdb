#include "physical_gpu_streaming_projection.hpp"

#include "gpu_batch_accumulator.hpp"
#include "gpu_logger.hpp"
#include "gpu_offload_extension.hpp"
#include "gpu_stream_pipeline.hpp"
#include "rmm_pool.hpp"
#include "vector_converter.hpp"

#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/physical_operator_states.hpp"

#include <chrono>
#include <deque>
#include <mutex>

namespace vector_gpu {

using namespace duckdb;

namespace {

//! Upper bound on the batch size this operator asks the accumulator for.
//!
//! Bigger is better up to a point, and the point is not slot capacity -- it is device memory. The pipeline
//! allocates GpuStreamPipeline::kStageCount sets of buffers sized for a whole batch, so this number
//! multiplies by four on the card. At 1,048,576 rows a two-column DOUBLE projection sits around 180 MB of
//! VRAM, comfortably inside the footprint routing already approved for this plan
//! (PhysicalGpuExecute::MAX_CHUNK_ROWS is 2,097,152 rows for ONE chunk), while making each batch's upload
//! and kernel milliseconds rather than microseconds -- which is what has to be true before three CUDA
//! streams can overlap at all on a driver whose per-flush latency is ~100 microseconds.
//!
//! A multiple of STANDARD_VECTOR_SIZE (2048 * 512), as GpuBatchAccumulator requires. (old arch)
constexpr idx_t kMaxStreamBatchRows = 350224384; // 334 vector blocks (~2.80 GB VRAM with doubles)

//! Serializes streaming-projection QUERIES against each other, process-wide.
//!
//! Not about thread safety within one query -- `ParallelOperator() == false` already makes DuckDB run this
//! operator's whole pipeline on one thread. It is about GpuMemoryPool, whose AcquireNext() is strictly
//! CYCLIC: it hands out slot 0, then 1, then 2, then 3, then 0 again, and BLOCKS until that specific index
//! frees. One producer/consumer pair rotating through it is exactly the discipline the ring was designed
//! for (see gpu_memory_pool.hpp). Two independent pairs interleaving their acquisitions are not: each can
//! end up waiting on the index the other is holding, and neither is running the code that would release
//! it. That is a hang, not a slowdown.
//!
//! Taken with try_lock, and a second concurrent streaming query FAILS rather than waiting. Waiting looks
//! friendlier and is worse: DuckDB runs query pipelines on a shared task-thread pool, so a task parked on
//! this lease occupies a thread the query holding the lease may need to finish -- with `SET threads=1`
//! that is a hang, and a hung database is a far worse outcome than one query returning a clear error the
//! caller can retry. The error is a NotImplementedException, which fails only that query (see
//! docs/KNOWN_ISSUES.md on why it must never be an InternalException).
//!
//! The honest fix is per-query pinned slot pools rather than one process-wide ring, which is a change to
//! GpuMemoryPool's contract and belongs in its own session (docs/TODO.md).
std::mutex &StreamingPipelineLease() {
	static std::mutex lease;
	return lease;
}

} // namespace

//! Where this operator's rows are coming from right now.
enum class GpuStreamMode : uint8_t {
	//! Steady state: input chunks are flattened into pinned memory and executed on the GPU.
	GPU,
	//! The bailout has fired. No new input is looked at until every batch already on the device has been
	//! handed back, because those rows are EARLIER in the input than anything the CPU will now produce.
	DRAINING,
	//! Everything from here runs on the CPU, one input chunk at a time. The GPU resources are already
	//! released by the time this state is reached.
	CPU,
};

class GpuStreamProjectionGlobalState : public GlobalOperatorState {
public:
	std::mutex lock;
	std::unique_lock<std::mutex> lease;
	bool prepared = false;
	GpuStreamMode mode = GpuStreamMode::GPU;

	unique_ptr<GpuBatchAccumulator> accumulator;
	unique_ptr<GpuStreamPipeline> pipeline;

	GpuStreamResult result;
	idx_t emitted = 0;

	std::deque<unique_ptr<DataChunk>> cpu_pending;
	idx_t input_since_flush = 0;

	// Diagnostics only (VECTOR_GPU_LOG / --gpu-trace).
	std::chrono::steady_clock::time_point started;
	uint64_t rows_in = 0;
	uint64_t rows_gpu = 0;
	uint64_t rows_cpu = 0;
	uint64_t batches = 0;
	bool reported = false;

	~GpuStreamProjectionGlobalState() override {
		result.Reset();
		pipeline.reset();
	}
};

class GpuStreamProjectionLocalState : public OperatorState {
public:
	GpuStreamProjectionLocalState(ExecutionContext &context, const std::vector<GpuProjectionLevel> &levels) {
		for (auto &level : levels) {
			executors.push_back(make_uniq<ExpressionExecutor>(context.client, level.expressions));
		}
		for (idx_t level = 0; level + 1 < levels.size(); level++) {
			auto intermediate = make_uniq<DataChunk>();
			intermediate->Initialize(Allocator::DefaultAllocator(), levels[level].types);
			intermediates.push_back(std::move(intermediate));
		}
	}

	bool input_consumed = false;
	vector<unique_ptr<ExpressionExecutor>> executors;
	vector<unique_ptr<DataChunk>> intermediates;
};

bool PhysicalGpuStreamingProjection::Supports(const GpuPlanNode &plan) {
	return GpuStreamPipeline::Supports(plan);
}

PhysicalGpuStreamingProjection::PhysicalGpuStreamingProjection(PhysicalPlan &physical_plan, vector<LogicalType> types,
                                                               std::shared_ptr<GpuPlanNode> gpu_plan_p,
                                                               std::vector<std::string> input_names_p,
                                                               std::vector<GpuProjectionLevel> cpu_levels_p,
                                                               idx_t estimated_cardinality,
                                                               std::vector<idx_t> active_indices_p,
                                                               ExecutionMode execution_mode_p)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, std::move(types), estimated_cardinality),
      gpu_plan(std::move(gpu_plan_p)), execution_mode(execution_mode_p), input_names(std::move(input_names_p)),
      cpu_levels(std::move(cpu_levels_p)), active_indices(std::move(active_indices_p)) {
	if (gpu_plan && gpu_plan->execution_mode != execution_mode) {
		gpu_plan->execution_mode = execution_mode;
	}
}

InsertionOrderPreservingMap<string> PhysicalGpuStreamingProjection::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["ExecutionMode"] = (execution_mode == ExecutionMode::PIPELINE_STREAM_UNCACHED)
	                              ? "PIPELINE_STREAM_UNCACHED"
	                              : "PIPELINE_STREAM_CACHED";
	string names;
	for (auto &name : input_names) {
		if (!names.empty()) {
			names += ", ";
		}
		names += name;
	}
	result["Input"] = names;
	result["Levels"] = to_string(cpu_levels.size());
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

const vector<LogicalType> &PhysicalGpuStreamingProjection::InputTypes() const {
	return (*children.begin()).get().GetTypes();
}

unique_ptr<GlobalOperatorState> PhysicalGpuStreamingProjection::GetGlobalOperatorState(ClientContext &context) const {
	return make_uniq<GpuStreamProjectionGlobalState>();
}

unique_ptr<OperatorState> PhysicalGpuStreamingProjection::GetOperatorState(ExecutionContext &context) const {
	return make_uniq<GpuStreamProjectionLocalState>(context, cpu_levels);
}

void PhysicalGpuStreamingProjection::Prepare(ClientContext &context, GpuStreamProjectionGlobalState &state) const {
	// Claimed before anything else, so no two queries ever hold pinned ring slots at the same time.
	state.lease = std::unique_lock<std::mutex>(StreamingPipelineLease(), std::try_to_lock);
	if (!state.lease.owns_lock()) {
		throw NotImplementedException(
		    "PhysicalGpuStreamingProjection: another GPU streaming query is already running in this process, "
		    "and the pinned ring buffer they would share hands out slots in a fixed rotation that only one "
		    "producer at a time can follow safely -- retry this query, or unset VECTOR_GPU_STREAM_PIPELINE "
		    "to use the default GPU operator, which has no such restriction");
	}

	auto &child_types = InputTypes();
	std::vector<LogicalType> accumulator_types;
	if (!active_indices.empty()) {
		for (auto idx : active_indices) {
			accumulator_types.push_back(child_types[idx]);
		}
	} else {
		accumulator_types = std::vector<LogicalType>(child_types.begin(), child_types.end());
	}
	try {
		// As large as one pinned ring slot holds, capped so the pipeline's four sets of device buffers stay
		// well inside what routing approved. Never below the default: a very wide column set can push
		// RowsThatFitInOneSlot down, and at that point the accumulator's own constructor is the authority
		// on whether it fits at all (it throws, and this operator fails the query rather than guessing).
		idx_t batch_rows = 0;
		const char *batch_rows_env = std::getenv("VECTOR_GPU_STREAM_BATCH_ROWS");
		if (batch_rows_env && batch_rows_env[0] != '\0') {
			try {
				batch_rows = std::stoull(batch_rows_env);
			} catch (...) {
			}
		}
		if (batch_rows == 0) {
			batch_rows = MinValue<idx_t>(kMaxStreamBatchRows, GpuBatchAccumulator::RowsThatFitInOneSlot(accumulator_types));
		}
		batch_rows = (batch_rows / STANDARD_VECTOR_SIZE) * STANDARD_VECTOR_SIZE;
		if (batch_rows == 0) {
			batch_rows = STANDARD_VECTOR_SIZE;
		}
		state.accumulator = make_uniq<GpuBatchAccumulator>(input_names, accumulator_types, batch_rows);
		state.pipeline =
		    make_uniq<GpuStreamPipeline>(*gpu_plan, input_names, state.accumulator->column_types(), batch_rows);
	} catch (const std::exception &e) {
		// NOT InternalException: that marks the whole DatabaseInstance fatally invalidated and fails every
		// later query on the connection. See docs/KNOWN_ISSUES.md.
		throw NotImplementedException("PhysicalGpuStreamingProjection: could not start GPU streaming execution: %s",
		                              e.what());
	}
	if (state.pipeline->output_types().size() != types.size()) {
		throw InvalidInputException("PhysicalGpuStreamingProjection: plan produces %llu columns but the operator "
		                            "has %llu output types",
		                            (unsigned long long)state.pipeline->output_types().size(),
		                            (unsigned long long)types.size());
	}
	VGPU_LOG(vector_gpu::LogLevel::INFO, "stream_project_start",
	         "\"columns\":" + std::to_string(input_names.size()) + ",\"levels\":" + std::to_string(cpu_levels.size()) +
	             ",\"batch_rows\":" + std::to_string(state.accumulator->batch_rows()) +
	             ",\"vram_total_pool\":" + std::to_string(RmmPool::Instance().TotalPoolBytes()) +
	             ",\"vram_allocated\":" + std::to_string(RmmPool::Instance().AllocatedBytes()));
	state.started = std::chrono::steady_clock::now();
	state.prepared = true;
}

void PhysicalGpuStreamingProjection::SubmitAccumulated(GpuStreamProjectionGlobalState &state) const {
	if (!state.accumulator || state.accumulator->row_count() == 0) {
		return;
	}
	auto batch = state.accumulator->TakeReadyBatch();
	GpuStreamBatch submission;
	submission.slot = std::move(batch.slot);
	submission.columns = std::move(batch.columns);
	submission.row_count = batch.row_count;
	state.rows_gpu += submission.row_count;
	state.batches++;
	state.input_since_flush = 0;
	state.pipeline->Submit(std::move(submission));
}

bool PhysicalGpuStreamingProjection::EmitFromResult(DataChunk &chunk, GpuStreamProjectionGlobalState &state) const {
	auto total = idx_t(state.result.row_count());
	if (state.emitted >= total) {
		return false;
	}
	auto count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, total - state.emitted);
	chunk.Reset();
	for (idx_t col = 0; col < state.result.columns().size(); col++) {
		// Straight out of the pipeline's PINNED landing buffer into a DuckDB Vector. No intermediate
		// ColumnDataCollection -- the same reasoning PhysicalGpuExecute records for its own emit path: it
		// was a full second copy of every output row and bought nothing.
		ConvertGpuColumnRangeToVector(state.result.columns()[col], state.emitted, count, chunk.data[col]);
	}
	chunk.SetCardinality(count);
	state.emitted += count;
	return true;
}

void PhysicalGpuStreamingProjection::ExecuteOnCpu(DataChunk &input, DataChunk &chunk,
                                                 GpuStreamProjectionGlobalState &gstate,
                                                 GpuStreamProjectionLocalState &lstate) const {
	reference<DataChunk> source = input;
	for (idx_t level = 0; level < cpu_levels.size(); level++) {
		const bool last = (level + 1 == cpu_levels.size());
		auto &destination = last ? chunk : *lstate.intermediates[level];
		if (!last) {
			destination.Reset();
		}
		lstate.executors[level]->Execute(source.get(), destination);
		source = destination;
	}
	gstate.rows_cpu += chunk.size();
}

void PhysicalGpuStreamingProjection::MoveAccumulatedToCpu(GpuStreamProjectionGlobalState &state) const {
	if (state.accumulator && state.accumulator->row_count() > 0) {
		auto &input_types = InputTypes();
		auto batch = state.accumulator->TakeReadyBatch();
		for (idx_t offset = 0; offset < idx_t(batch.row_count); offset += STANDARD_VECTOR_SIZE) {
			auto count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, idx_t(batch.row_count) - offset);
			auto pending = make_uniq<DataChunk>();
			pending->Initialize(Allocator::DefaultAllocator(), input_types);
			if (!active_indices.empty() && active_indices.size() < input_types.size()) {
				for (idx_t col = 0; col < batch.columns.size(); col++) {
					ConvertGpuColumnRangeToVector(batch.columns[col], offset, count, pending->data[active_indices[col]]);
				}
			} else {
				for (idx_t col = 0; col < batch.columns.size(); col++) {
					ConvertGpuColumnRangeToVector(batch.columns[col], offset, count, pending->data[col]);
				}
			}
			pending->SetCardinality(count);
			state.cpu_pending.push_back(std::move(pending));
		}
		// `batch` (and the pinned slot its columns point into) dies here -- every byte was copied above.
	}
	state.accumulator.reset();
}

bool PhysicalGpuStreamingProjection::ShouldBailOut(GpuStreamProjectionGlobalState &state) const {
	if (state.mode != GpuStreamMode::GPU || !state.accumulator) {
		return false;
	}
	// Only ask once a whole batch window of INPUT rows has gone past. Below that the filter's selectivity
	// is simply unknown, and bailing out on a query that is merely SMALL would give up the GPU for a
	// workload that was never going to fill a batch in the first place.
	if (state.input_since_flush < state.accumulator->batch_rows()) {
		return false;
	}
	return state.accumulator->row_count() < MIN_GPU_BATCH_ROWS;
}

void PhysicalGpuStreamingProjection::BeginBailout(GpuStreamProjectionGlobalState &state, const char *reason) const {
	VGPU_LOG(vector_gpu::LogLevel::INFO, "stream_project_bailout",
	         "\"reason\":" + GpuLogger::Quote(reason) + ",\"kept\":" +
	             std::to_string(state.accumulator ? uint64_t(state.accumulator->row_count()) : uint64_t(0)) +
	             ",\"input_since_flush\":" + std::to_string(state.input_since_flush) +
	             ",\"threshold\":" + std::to_string(MIN_GPU_BATCH_ROWS));
	MoveAccumulatedToCpu(state);
	state.mode = GpuStreamMode::DRAINING;
}

bool PhysicalGpuStreamingProjection::MakeRoomToSubmit(DataChunk &chunk, GpuStreamProjectionGlobalState &state) const {
	if (state.pipeline->CanSubmit()) {
		return true;
	}
	// Full. The only thing that frees a stage is taking the oldest finished batch -- and once taken, its
	// rows must go out before anything else, so the caller has to return HAVE_MORE_OUTPUT and try again.
	state.result.Reset();
	if (state.pipeline->TryTake(state.result) || state.pipeline->TakeBlocking(state.result)) {
		state.emitted = 0;
		EmitFromResult(chunk, state);
		return false;
	}
	// Nothing in flight and still no room: only reachable if a previous submission left the pipeline
	// terminal, in which case Submit itself reports it with a real message.
	return state.pipeline->CanSubmit();
}

OperatorResultType PhysicalGpuStreamingProjection::Execute(ExecutionContext &context, DataChunk &input,
                                                           DataChunk &chunk, GlobalOperatorState &gstate_p,
                                                           OperatorState &state_p) const {
	auto &gstate = gstate_p.Cast<GpuStreamProjectionGlobalState>();
	auto &lstate = state_p.Cast<GpuStreamProjectionLocalState>();

	std::lock_guard<std::mutex> guard(gstate.lock);
	if (!gstate.prepared) {
		Prepare(context.client, gstate);
	}

	try {
		// (1) Rows still pending from the batch currently being handed back always go first: results leave
		//     the pipeline in submission order, and that order is the query's row order.
		if (EmitFromResult(chunk, gstate)) {
			return OperatorResultType::HAVE_MORE_OUTPUT;
		}

		// (2) Bailout in progress. Refuse to look at new input until every batch already on the device has
		//     been emitted -- those rows are EARLIER in the input than anything the CPU is about to
		//     produce, and a projection must not reorder its output.
		if (gstate.mode == GpuStreamMode::DRAINING) {
			gstate.result.Reset();
			if (gstate.pipeline->in_flight() > 0) {
				gstate.pipeline->TakeBlocking(gstate.result);
				gstate.emitted = 0;
				if (EmitFromResult(chunk, gstate)) {
					return OperatorResultType::HAVE_MORE_OUTPUT;
				}
			}
			if (!gstate.cpu_pending.empty()) {
				auto pending = std::move(gstate.cpu_pending.front());
				gstate.cpu_pending.pop_front();
				ExecuteOnCpu(*pending, chunk, gstate, lstate);
				return OperatorResultType::HAVE_MORE_OUTPUT;
			}
			// Everything the GPU owed has been handed over. Release the device and pinned resources now
			// rather than at the end of the query -- the ring slot in particular is what another query
			// would be waiting on.
			gstate.pipeline.reset();
			gstate.accumulator.reset();
			gstate.lease.unlock();
			gstate.mode = GpuStreamMode::CPU;
			VGPU_LOG(vector_gpu::LogLevel::INFO, "stream_project_cpu_mode",
			         "\"rows_gpu\":" + std::to_string(gstate.rows_gpu));
		}

		// (3) CPU tail: one input chunk in, one output chunk out.
		if (gstate.mode == GpuStreamMode::CPU) {
			if (lstate.input_consumed) {
				// This chunk was already taken into pinned memory before the bailout fired; projecting it
				// again here would duplicate its rows.
				lstate.input_consumed = false;
				chunk.SetCardinality(0);
				return OperatorResultType::NEED_MORE_INPUT;
			}
			ExecuteOnCpu(input, chunk, gstate, lstate);
			gstate.rows_in += input.size();
			return OperatorResultType::NEED_MORE_INPUT;
		}

		// (4) GPU steady state. Take this chunk's SURVIVING rows into pinned memory, exactly once.
		if (!lstate.input_consumed) {
			// A filtered chunk is an arbitrary size, so the active batch no longer fills to exactly
			// batch_rows the way an unfiltered 2,048-row scan chunk did. Flush early rather than let Append
			// refuse to straddle two ring slots.
			if (gstate.accumulator->HasReadyBatch() ||
			    gstate.accumulator->row_count() + input.size() > gstate.accumulator->batch_rows()) {
				if (!MakeRoomToSubmit(chunk, gstate)) {
					return OperatorResultType::HAVE_MORE_OUTPUT;
				}
				SubmitAccumulated(gstate);
			}
			// GpuBatchAccumulator::Append resolves the chunk through Vector::ToUnifiedFormat, so a chunk
			// DuckDB's PhysicalFilter sliced with a SelectionVector contributes only its SURVIVING rows --
			// the selection is applied on the way INTO pinned memory, and filtered-out rows never exist on
			// this side of PCIe at all.
			if (!active_indices.empty() && active_indices.size() < input.ColumnCount()) {
				vector<LogicalType> act_types;
				for (auto idx : active_indices) {
					act_types.push_back(input.data[idx].GetType());
				}
				DataChunk active_chunk;
				active_chunk.InitializeEmpty(act_types);
				for (idx_t i = 0; i < active_indices.size(); i++) {
					active_chunk.data[i].Reference(input.data[active_indices[i]]);
				}
				active_chunk.SetCardinality(input.size());
				gstate.accumulator->Append(active_chunk, nullptr, active_chunk.size());
			} else {
				gstate.accumulator->Append(input, nullptr, input.size());
			}
			gstate.rows_in += input.size();
			gstate.input_since_flush += input.size();
			lstate.input_consumed = true;

			if (ShouldBailOut(gstate)) {
				BeginBailout(gstate, "filter selectivity below the GPU batch threshold");
				return OperatorResultType::HAVE_MORE_OUTPUT;
			}
			if (gstate.accumulator->HasReadyBatch() && gstate.pipeline->CanSubmit()) {
				SubmitAccumulated(gstate);
			}
		}

		// (5) Hand back anything the device has already finished, without waiting for it.
		gstate.result.Reset();
		if (gstate.pipeline->TryTake(gstate.result)) {
			gstate.emitted = 0;
			if (EmitFromResult(chunk, gstate)) {
				return OperatorResultType::HAVE_MORE_OUTPUT;
			}
		}
		lstate.input_consumed = false;
		chunk.SetCardinality(0);
		return OperatorResultType::NEED_MORE_INPUT;
	} catch (const NotImplementedException &) {
		throw;
	} catch (const std::exception &e) {
		// A CUDA failure mid-stream fails THIS query. It deliberately does not fall back to the CPU: rows
		// from earlier batches have already been emitted, and re-running the rest on a different engine
		// would mix two engines' arithmetic in one result set. See docs/STREAMING_INGEST_DESIGN.md's
		// "clean fallback mechanism" note -- refusals belong at routing time, not here.
		throw NotImplementedException("PhysicalGpuStreamingProjection: GPU streaming execution failed: %s", e.what());
	}
}

OperatorFinalizeResultType PhysicalGpuStreamingProjection::FinalExecute(ExecutionContext &context, DataChunk &chunk,
                                                                        GlobalOperatorState &gstate_p,
                                                                        OperatorState &state_p) const {
	auto &gstate = gstate_p.Cast<GpuStreamProjectionGlobalState>();
	auto &lstate = state_p.Cast<GpuStreamProjectionLocalState>();

	std::lock_guard<std::mutex> guard(gstate.lock);
	if (!gstate.prepared) {
		// No input ever arrived (an empty table, or a filter that matched nothing), so nothing was ever
		// prepared and there is nothing to flush.
		chunk.SetCardinality(0);
		return OperatorFinalizeResultType::FINISHED;
	}

	try {
		if (EmitFromResult(chunk, gstate)) {
			return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
		}

		// The input is over, so whatever the accumulator still holds is the tail of the scan. Below the GPU
		// threshold it goes to the CPU for exactly the reason the mid-query bailout exists: a few hundred
		// rows do not repay an upload, a launch and a download.
		if (gstate.mode == GpuStreamMode::GPU && gstate.accumulator) {
			auto tail = gstate.accumulator->row_count();
			if (tail > 0 && tail < MIN_GPU_BATCH_ROWS) {
				BeginBailout(gstate, "final partial batch below the GPU batch threshold");
			} else {
				if (tail > 0) {
					if (!MakeRoomToSubmit(chunk, gstate)) {
						return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
					}
					SubmitAccumulated(gstate);
				}
				gstate.mode = GpuStreamMode::DRAINING;
			}
		}

		if (gstate.mode == GpuStreamMode::DRAINING) {
			gstate.result.Reset();
			if (gstate.pipeline && gstate.pipeline->in_flight() > 0) {
				gstate.pipeline->TakeBlocking(gstate.result);
				gstate.emitted = 0;
				if (EmitFromResult(chunk, gstate)) {
					return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
				}
			}
			if (!gstate.cpu_pending.empty()) {
				auto pending = std::move(gstate.cpu_pending.front());
				gstate.cpu_pending.pop_front();
				ExecuteOnCpu(*pending, chunk, gstate, lstate);
				return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
			}
			gstate.pipeline.reset();
			gstate.accumulator.reset();
			if (gstate.lease.owns_lock()) {
				gstate.lease.unlock();
			}
			gstate.mode = GpuStreamMode::CPU;
		}
	} catch (const NotImplementedException &) {
		throw;
	} catch (const std::exception &e) {
		throw NotImplementedException("PhysicalGpuStreamingProjection: GPU streaming execution failed: %s", e.what());
	}

	if (!gstate.reported) {
		gstate.reported = true;
		auto total_us = static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - gstate.started)
		        .count());
		// Scan time is DuckDB's own now, measured by DuckDB's own profiler on the child operators, so this
		// path reports zero for it rather than inventing a number: everything it can honestly attribute to
		// itself is upload + kernel + download + emit, which overlap and cannot be split further.
		RecordGpuPhaseTimings(0, total_us, 0, gstate.rows_gpu);
		VGPU_LOG(vector_gpu::LogLevel::INFO, "stream_project_done",
		         "\"batches\":" + std::to_string(gstate.batches) + ",\"rows_in\":" + std::to_string(gstate.rows_in) +
		             ",\"rows_gpu\":" + std::to_string(gstate.rows_gpu) +
		             ",\"rows_cpu\":" + std::to_string(gstate.rows_cpu) + ",\"us\":" + std::to_string(total_us));
	}
	// MUST clear the chunk before reporting FINISHED. DuckDB reuses the same DataChunk across calls, so
	// returning without touching it leaves the previous call's rows and cardinality in place and the
	// consumer counts them twice. (The same trap PhysicalGpuExecute documents.)
	chunk.SetCardinality(0);
	return OperatorFinalizeResultType::FINISHED;
}

} // namespace vector_gpu
