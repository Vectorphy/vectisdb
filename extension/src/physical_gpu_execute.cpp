#include "physical_gpu_execute.hpp"
#include "gpu_column_cache.hpp"
#include "gpu_offload_extension.hpp"
#include "gpu_logger.hpp"
#include "table_scanner.hpp"
#include "vector_converter.hpp"

#include "duckdb/common/mutex.hpp"
#include "duckdb/execution/physical_operator_states.hpp"

#include <chrono>
#include <cstring>
#include <future>

namespace vector_gpu {

using namespace duckdb;

namespace {
//! Recursively walks gpu_plan collecting real, scanned data for every SCAN leaf found, flattening the
//! results into one inputs list. The recursive executor resolves each SCAN leaf from this collection,
//! so its order must match the plan traversal order (including both sides of a join).
//! FILTER/PROJECTION/AGGREGATE/JOIN nodes themselves have no data to gather (their input comes entirely
//! from their SCAN descendants), so only SCAN nodes are handled specially here.
//!
//! WHOLE-INPUT path only (GROUP_BY_AGGREGATE/HASH_JOIN/CROSS_PRODUCT): these plans can have more than one
//! SCAN leaf (a join's two sides), which is exactly why they still use this eager, everything-at-once
//! collector rather than StreamingTableScanner -- that class handles one table at a time, matching the
//! chunked path's single-SCAN-leaf shape (see FindScanLeaf below and IsRowIndependent's doc comment).
void CollectScanInputs(ClientContext &context, const GpuPlanNode &node, std::vector<GpuColumn> &inputs,
                       std::vector<std::vector<uint8_t>> &owned_buffers) {
	if (node.op_type == GpuOpType::SCAN) {
		auto scanned = ScanTableToGpuColumns(context, node.table, node.column_names, owned_buffers);
		inputs.insert(inputs.end(), std::make_move_iterator(scanned.begin()), std::make_move_iterator(scanned.end()));
		return;
	}
	for (auto &child : node.children) {
		CollectScanInputs(context, *child, inputs, owned_buffers);
	}
}

//! Every SCAN leaf of `node`, in the same traversal order CollectScanInputs gathers their data in.
void CollectScanLeaves(const GpuPlanNode &node, std::vector<const GpuPlanNode *> &out) {
	if (node.op_type == GpuOpType::SCAN) {
		out.push_back(&node);
		return;
	}
	for (auto &child : node.children) {
		CollectScanLeaves(*child, out);
	}
}

//! CHUNKED (row-independent) path only. Locates the single SCAN leaf a row-independent plan is built
//! from -- SCAN/FILTER/PROJECTION never combine two relations (see IsRowIndependent), so there is always
//! exactly one, unlike the whole-input path's CollectScanInputs, which must handle a join's two.
//! Returns nullptr if none is found; the caller treats that as a routing/translation invariant violation
//! (it should not happen for a plan that passed IsRowIndependent and the "never offload a plan that
//! computes nothing" rule, but neither of those is re-verified here, so this is a return value to check,
//! not an assertion to trust blindly).
const GpuPlanNode *FindScanLeaf(const GpuPlanNode &node) {
	if (node.op_type == GpuOpType::SCAN) {
		return &node;
	}
	for (auto &child : node.children) {
		if (auto *found = FindScanLeaf(*child)) {
			return found;
		}
	}
	return nullptr;
}

//! True when each output row depends only on its own input row, so the input can be executed in
//! independent chunks and the concatenated results equal the whole-input result. SCAN/PROJECTION are
//! row-wise; FILTER drops rows but never combines them. GROUP_BY_AGGREGATE, HASH_JOIN and CROSS_PRODUCT
//! all need cross-row state (partial aggregates to merge, a complete build side to probe, every right row
//! for every left row), so chunking them would silently change answers -- they stay whole-input.
//!
//! CROSS_PRODUCT is doubly disqualified: chunking slices every scanned column by the SAME row range, and
//! this operator's two inputs have independent row counts, so a chunk would truncate both sides at once
//! and lose most of the pairs.
// IsRowIndependent now lives in cuda_engine (src/gpu_cost_model.cpp, declared in gpu_engine.hpp). It was
// duplicated here and in the routing layer's VRAM sizing, which is a correctness hazard rather than
// mere duplication: if routing believes a plan will be chunked and execution does not, routing approves
// a plan sized against 2M rows that then allocates for the entire input.

} // namespace

PhysicalGpuExecute::PhysicalGpuExecute(PhysicalPlan &physical_plan, vector<LogicalType> types,
                                        std::shared_ptr<GpuPlanNode> gpu_plan_p, idx_t estimated_cardinality,
                                        ExecutionMode execution_mode_p)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, std::move(types), estimated_cardinality),
      gpu_plan(std::move(gpu_plan_p)), execution_mode(execution_mode_p) {
	if (gpu_plan && gpu_plan->execution_mode != execution_mode) {
		gpu_plan->execution_mode = execution_mode;
	}
}

//! Global state holds the lazily-materialized result plus the scan cursor and a mutex guarding first-time
//! materialization (multiple threads may call GetDataInternal before materialization completes).
class GpuExecuteGlobalState : public GlobalSourceState {
public:
	explicit GpuExecuteGlobalState(const vector<LogicalType> &types) {
	}

	mutex materialize_lock;
	bool prepared = false;
	bool chunked = false;

	//! WHOLE-INPUT path only (GROUP BY / JOIN, or VECTOR_GPU_NO_CHUNK): unchanged from before the
	//! session-30 streaming redesign. Populated once by ScanInputsOnce and passed to ExecuteChunk as-is,
	//! with no slicing -- these plans need the whole table at once.
	std::vector<std::vector<uint8_t>> owned_buffers;
	std::vector<GpuColumn> inputs;
	//! Scan cost, attributed to the (only) chunk on this path.
	uint64_t scan_us = 0;

	//! CHUNKED (row-independent) path only. Drives a resumable, ramping-batch-size scan (see
	//! StreamingTableScanner); each batch is independently owned, so handing one to a background compute
	//! task while scanning the next never risks the two aliasing memory.
	unique_ptr<StreamingTableScanner> scanner;
	idx_t next_batch_rows = 0;
	//! A batch ScanNextBatch has already scanned but whose GPU execution KickOffCompute has not yet
	//! started. At most one at a time: the pipeline is depth-2 (one batch computing, one batch scanned
	//! and waiting), matching PhysicalGpuExecute::MAX_CHUNK_ROWS's invariant that only one GPU chunk is
	//! ever resident on the device -- this does not add a second one, it only overlaps HOST-side scanning
	//! with the single in-flight chunk's GPU execution.
	unique_ptr<ScannedRowBatch> pending_batch;

	//! The current result being drained. With the chunked path this is one batch's worth.
	vector_gpu::GpuExecutionResult result;
	//! Rows of `result` already handed to DuckDB.
	idx_t emitted = 0;
	//! CHUNKED path, CACHE-REPLAY sub-path (GpuColumnCache, docs/GPU_RESIDENT_CACHE_DESIGN.md). Non-empty
	//! when every column this plan's SCAN needs was already device-resident: the row count of each cached
	//! chunk, in scan order. The table is then never scanned at all -- chunk i is executed by handing the
	//! engine GpuExecuteOptions::cache_chunk_index = i. Empty means "populate, don't replay".
	std::vector<uint64_t> cache_chunk_rows;
	//! Index of the next chunk of `cache_chunk_rows` to dispatch.
	idx_t next_cache_chunk = 0;

	//! CHUNKED path, CACHE-POPULATE sub-path. True between GpuColumnCache::BeginStaging and the matching
	//! CommitStaging/AbortStaging. The destructor is the backstop for a query that ends without draining
	//! (a LIMIT upstream, an exception) -- without it the staged chunks stay resident for the life of the
	//! process, invisible and unreclaimable.
	bool staging_open = false;
	GpuTableRef staging_table;
	//! Rows handed to GPU execution so far. CommitStaging refuses to publish a column whose staged chunks
	//! do not sum to exactly this, which is what makes a cache entry mean "the COMPLETE column".
	uint64_t staged_rows = 0;

	//! The batch currently executing on a background thread, started by KickOffCompute. std::future
	//! carries an exception across the thread boundary, so a failure in the producer surfaces at get() on
	//! the consumer -- it is not swallowed. ~future for a launch::async task blocks, so the worker cannot
	//! outlive this state and touch freed buffers.
	std::future<vector_gpu::GpuExecutionResult> pending;

	~GpuExecuteGlobalState() override {
		// Join the in-flight chunk BEFORE touching the cache: that background task may still be inside
		// GpuColumnCache::StageChunk, and AbortStaging frees exactly the buffers its H2D is writing into.
		// Relying on ~future to join is too late -- a destructor BODY runs before its members are
		// destroyed, so `pending` would still be alive here. Exceptions from a failed chunk are swallowed
		// deliberately: this is cleanup, and the original failure has already propagated to DuckDB.
		if (pending.valid()) {
			try {
				pending.get();
			} catch (...) {
			}
		}
		if (staging_open) {
			GpuColumnCache::Instance().AbortStaging(staging_table);
		}
	}
};

//! Deliberately small: the point is time-to-first-compute. 64Ki rows of 2 INT64 columns is ~1 MB, which
//! uploads in well under a millisecond, so the first kernel starts almost immediately instead of after
//! the whole table has been scanned and copied.
static constexpr idx_t kFirstChunkRows = 65536;
//! Doubling stops here so a large scan settles into efficient full-size launches rather than paying
//! per-chunk launch and allocation overhead indefinitely. Defined in the header (and therefore visible
//! to TableChecker) so routing sizes VRAM against the same chunk execution actually uses.
static constexpr idx_t kMaxChunkRows = PhysicalGpuExecute::MAX_CHUNK_ROWS;

class GpuExecuteLocalState : public LocalSourceState {};

unique_ptr<GlobalSourceState> PhysicalGpuExecute::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<GpuExecuteGlobalState>(types);
}

unique_ptr<LocalSourceState> PhysicalGpuExecute::GetLocalSourceState(ExecutionContext &context,
                                                                     GlobalSourceState &gstate) const {
	return make_uniq<GpuExecuteLocalState>();
}

void PhysicalGpuExecute::ExecuteOnce(ClientContext &context, vector_gpu::GpuExecutionResult &out_result) const {
	// Whole-input path, kept because a plan that is not row-independent (GROUP BY, JOIN) must see every
	// row at once, and because this is a documented external test hook (see the header) even though
	// GetDataInternal itself no longer calls it.
	GpuExecuteGlobalState scratch {types};
	ScanInputsOnce(context, scratch);
	out_result = ExecuteChunk(scratch.inputs, scratch.scan_us);
}

void PhysicalGpuExecute::ScanInputsOnce(ClientContext &context, GpuExecuteGlobalState &gstate) const {
	// gpu_plan's SCAN leaves get genuinely scanned from DuckDB storage (table_scanner.cpp) and converted
	// to GpuColumns (vector_converter.cpp). `owned_buffers` must outlive the ExecutePlan call, which is
	// why it lives in the global state rather than on the stack.
	auto scan_start = std::chrono::steady_clock::now();
	try {
		CollectScanInputs(context, *gpu_plan, gstate.inputs, gstate.owned_buffers);
	} catch (const std::exception &e) {
		// A missing table/column (schema changed between planning and execution) or an unsupported column
		// type slipping past the Table Checker -- a genuine runtime failure, but still must not be
		// InternalException (see the standing rule below) or it takes down the whole connection.
		throw InvalidInputException("PhysicalGpuExecute: failed to gather input data for GPU execution: %s",
		                            e.what());
	}
	gstate.scan_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                           std::chrono::steady_clock::now() - scan_start)
	                                           .count());
}

uint64_t PhysicalGpuExecute::CacheResidentRowCount(const GpuPlanNode &plan) const {
	std::vector<const GpuPlanNode *> scans;
	CollectScanLeaves(plan, scans);
	if (scans.empty()) {
		return 0;
	}
	uint64_t total_rows = 0;
	for (auto *scan : scans) {
		// ReplayableChunkRows, not a per-column presence test: it returns non-empty only when every
		// requested column is COMMITTED and their chunk boundaries agree, which is the property that makes
		// "cached" mean "the complete table, consistently" rather than "some of it" (session 31's bug).
		auto chunk_rows = GpuColumnCache::Instance().ReplayableChunkRows(scan->table, scan->column_names);
		if (chunk_rows.empty()) {
			return 0; // all-or-nothing; see the header for why a partial hit is not served
		}
		for (auto rows : chunk_rows) {
			total_rows += rows;
		}
	}
	return total_rows;
}

vector_gpu::GpuExecutionResult PhysicalGpuExecute::ExecuteChunk(const std::vector<GpuColumn> &inputs,
                                                                uint64_t scan_us,
                                                                const vector_gpu::GpuExecuteOptions &options,
                                                                uint64_t input_rows) const {
	// Takes inputs by const reference and mutates nothing shared: this runs on the prefetch thread as
	// well as the consumer thread, so touching the global state here would be a data race.
	auto execute_start = std::chrono::steady_clock::now();
	auto result = GpuEngine::Instance().ExecutePlan(*gpu_plan, inputs, options);
	auto execute_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                            std::chrono::steady_clock::now() - execute_start)
	                                            .count());
	// `input_rows` covers the cache-replay path, where the rows come from device memory and `inputs` is
	// empty by design -- reporting zero there would make a cache hit look like it processed nothing.
	auto rows = inputs.empty() ? input_rows : inputs.front().row_count;
	VGPU_LOG(vector_gpu::LogLevel::DEBUG, "chunk",
	         "\"rows\":" + std::to_string(rows) + ",\"us\":" + std::to_string(execute_us));
	RecordGpuPhaseTimings(scan_us, execute_us, 0, rows);

	if (!result.success) {
		// IMPORTANT: not InternalException -- that marks the whole DatabaseInstance fatally invalidated,
		// failing every later query on the connection. See docs/KNOWN_ISSUES.md.
		throw NotImplementedException("PhysicalGpuExecute: GPU execution failed: %s", result.error_message);
	}
	if (result.columns.size() != types.size()) {
		throw InvalidInputException("PhysicalGpuExecute: GPU returned %llu columns but the operator has %llu "
		                            "output types",
		                            (unsigned long long)result.columns.size(), (unsigned long long)types.size());
	}
	return result;
}

void PhysicalGpuExecute::ScanNextBatch(GpuExecuteGlobalState &gstate) const {
	if (gstate.scanner->exhausted()) {
		return; // nothing left to scan; pending_batch stays whatever it already was (null, once
		        // KickOffCompute has consumed the last real batch)
	}
	auto rows_target = gstate.next_batch_rows;
	// Ramp: the first batch is deliberately tiny so a kernel starts almost immediately; later batches
	// grow so a long scan is not dominated by per-batch launch/scan overhead.
	gstate.next_batch_rows = MinValue<idx_t>(gstate.next_batch_rows * 2, kMaxChunkRows);

	auto batch = gstate.scanner->NextBatch(rows_target);
	VGPU_LOG(vector_gpu::LogLevel::DEBUG, "stream_chunk",
	         "\"rows\":" + std::to_string(batch.row_count) + ",\"scan_us\":" + std::to_string(batch.scan_us) +
	             ",\"exhausted\":" + (batch.exhausted ? std::string("true") : std::string("false")));
	gstate.pending_batch = make_uniq<ScannedRowBatch>(std::move(batch));
}

void PhysicalGpuExecute::KickOffCompute(GpuExecuteGlobalState &gstate) const {
	if (!gstate.pending_batch || gstate.pending_batch->row_count == 0) {
		return; // nothing scanned yet, or the scanner's trailing empty batch on exhaustion -- no task to run
	}
	auto batch = std::move(gstate.pending_batch);
	VGPU_LOG(vector_gpu::LogLevel::DEBUG, "prefetch", "\"rows\":" + std::to_string(batch->row_count));
	// Counted here, not in ScanNextBatch: only batches that actually EXECUTE reach ExecuteScan and get
	// staged, so this is the number CommitStaging must see each staged column add up to.
	vector_gpu::GpuExecuteOptions options;
	options.stage_scan_to_cache = gstate.staging_open;
	if (gstate.staging_open) {
		gstate.staged_rows += batch->row_count;
	}
	// launch::async, not deferred: the whole point is that this runs WHILE the caller scans the batch
	// after this one, and while DuckDB drains the PREVIOUS chunk. `batch` is moved into the task's own
	// storage, so it (and the buffers it owns, which batch->columns points into) outlives this function
	// regardless of what gstate.pending_batch holds by the time the task actually runs. Only one such task
	// is ever in flight at a time (the caller always awaits `pending` via .get() before calling this
	// again), so there is no risk of two chunks executing on the GPU concurrently -- see
	// ExecContext::stream's comment in gpu_executor_internal.hpp for why that invariant matters beyond
	// just this file.
	gstate.pending = std::async(std::launch::async, [this, options, batch = std::move(batch)]() mutable {
		return ExecuteChunk(batch->columns, batch->scan_us, options);
	});
}

void PhysicalGpuExecute::KickOffCachedChunk(GpuExecuteGlobalState &gstate) const {
	if (gstate.next_cache_chunk >= gstate.cache_chunk_rows.size()) {
		return; // every cached chunk dispatched -- gstate.pending stays invalid, ending the drain loop
	}
	auto index = gstate.next_cache_chunk++;
	auto rows = gstate.cache_chunk_rows[index];
	VGPU_LOG(vector_gpu::LogLevel::DEBUG, "cache_chunk",
	         "\"chunk\":" + std::to_string(index) + ",\"rows\":" + std::to_string(rows));
	// Same one-in-flight discipline as KickOffCompute: exactly one chunk is on the device at a time, so
	// the routing decision that approved this plan for one MAX_CHUNK_ROWS chunk still holds. The rows
	// themselves are already resident, so there is nothing to scan or upload -- this task is pure compute.
	vector_gpu::GpuExecuteOptions options;
	options.cache_chunk_index = static_cast<int64_t>(index);
	gstate.pending = std::async(std::launch::async, [this, options, rows]() {
		return ExecuteChunk({}, 0, options, rows);
	});
}

void PhysicalGpuExecute::FinishStaging(GpuExecuteGlobalState &gstate) const {
	if (!gstate.staging_open) {
		return;
	}
	gstate.staging_open = false;
	// Commit ONLY on a scan that reached the end of the table. Anything else (an upstream LIMIT that
	// stopped pulling, a scanner that never opened) leaves a partial column set, and publishing that is
	// precisely the silent wrong answer this whole staging mechanism exists to prevent.
	if (gstate.scanner && gstate.scanner->exhausted()) {
		VGPU_LOG(vector_gpu::LogLevel::INFO, "cache_populate",
		         "\"table\":" + GpuLogger::Quote(gstate.staging_table.ToString()) + ",\"rows\":" +
		             std::to_string(gstate.staged_rows));
		GpuColumnCache::Instance().CommitStaging(gstate.staging_table, gstate.staged_rows);
	} else {
		GpuColumnCache::Instance().AbortStaging(gstate.staging_table);
	}
}

SourceResultType PhysicalGpuExecute::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                                     OperatorSourceInput &input) const {
	auto &gstate = input.global_state.Cast<GpuExecuteGlobalState>();
	lock_guard<mutex> guard(gstate.materialize_lock);

	if (!gstate.prepared) {
		// Chunking is only valid when each output row depends solely on its own input row. GROUP BY and
		// JOIN need every row at once, so they keep the whole-input path.
		// VECTOR_GPU_NO_CHUNK=1 forces the whole-input path -- an A/B switch for isolating whether a
		// wrong answer comes from chunking or from the operator itself.
		static const bool chunking_disabled = [] {
			const char *v = std::getenv("VECTOR_GPU_NO_CHUNK");
			return v != nullptr && std::strcmp(v, "0") != 0;
		}();
		gstate.chunked = !chunking_disabled && IsRowIndependent(*gpu_plan);

		const bool uncached_mode = (execution_mode == ExecutionMode::PIPELINE_STREAM_UNCACHED ||
		                            (gpu_plan && gpu_plan->execution_mode == ExecutionMode::PIPELINE_STREAM_UNCACHED));

		if (gstate.chunked) {
			auto *scan_node = FindScanLeaf(*gpu_plan);
			if (scan_node == nullptr) {
				throw InvalidInputException(
				    "PhysicalGpuExecute: row-independent plan has no SCAN leaf to stream from");
			}

			// GPU-resident column cache (branch gpu-resident-cache, docs/GPU_RESIDENT_CACHE_DESIGN.md):
			// If uncached_mode is active (e.g. auto-bypassed by planner threshold), bypass cache replay
			// and bypass cache staging to avoid memory pressure / eviction thrashing.
			std::vector<uint64_t> cache_chunk_rows;
			if (!uncached_mode) {
				cache_chunk_rows =
				    GpuColumnCache::Instance().ReplayableChunkRows(scan_node->table, scan_node->column_names);
			} else {
				VGPU_LOG(vector_gpu::LogLevel::INFO, "cache",
				         "\"status\":\"auto_bypassed\",\"mode\":\"PIPELINE_STREAM_UNCACHED\","
				         "\"table\":" + GpuLogger::Quote(scan_node->table.ToString()));
			}

			if (!cache_chunk_rows.empty()) {
				uint64_t cached_rows = 0;
				for (auto rows : cache_chunk_rows) {
					cached_rows += rows;
				}
				// `rows` here is the number to compare against the table's true row count when checking
				// that a cache hit served the WHOLE table -- the single most useful line in the log for
				// catching a regression of the session-31 truncation bug.
				VGPU_LOG(vector_gpu::LogLevel::INFO, "cache_full_hit",
				         "\"table\":" + GpuLogger::Quote(scan_node->table.ToString()) + ",\"chunks\":" +
				             std::to_string(cache_chunk_rows.size()) + ",\"rows\":" + std::to_string(cached_rows) +
				             ",\"generation\":" +
				             std::to_string(GpuColumnCache::Instance().TableGeneration(scan_node->table)));
				gstate.cache_chunk_rows = std::move(cache_chunk_rows);
				KickOffCachedChunk(gstate); // chunk 0; the drain loop below pulls it and dispatches the rest
			} else {
				try {
					gstate.scanner = make_uniq<StreamingTableScanner>(context.client, scan_node->table,
					                                                  scan_node->column_names);
				} catch (const std::exception &e) {
					throw InvalidInputException(
					    "PhysicalGpuExecute: failed to open streaming scan for GPU execution: %s", e.what());
				}
				// Open the staging slot BEFORE the first batch executes: ExecuteScan only offers a chunk
				// to the cache when one is open. Closed by FinishStaging when the drain finishes, or by
				// ~GpuExecuteGlobalState if this query ends without draining.
				if (!uncached_mode) {
					GpuColumnCache::Instance().BeginStaging(scan_node->table);
					gstate.staging_open = true;
					gstate.staging_table = scan_node->table;
				} else {
					gstate.staging_open = false;
				}
				gstate.next_batch_rows = kFirstChunkRows;
				// Prime the pipeline: scan batch 0 (nothing to overlap with yet -- this is the
				// unavoidable time-to-first-byte cost, same as before), kick off its GPU execution, then
				// IMMEDIATELY scan batch 1. That second scan runs WHILE batch 0 computes on the
				// background thread instead of only after it finishes, which is the actual point of
				// streaming rather than just chunking -- see docs/STREAMING_INGEST_DESIGN.md for the full
				// before/after picture. Each batch, once executed, is STAGED into GpuColumnCache (see
				// gpu_executor.cu's ExecuteScan) -- so a table starts UNCACHED, accumulates chunk by
				// chunk as this first query streams through it, becomes readable only when the scan
				// completes, and a SECOND query against the same table replays it above instead.
				ScanNextBatch(gstate);
				KickOffCompute(gstate);
				ScanNextBatch(gstate);
			}
		} else {
			// WHOLE-INPUT path (GROUP BY / JOIN / CROSS PRODUCT). Ask the cache BEFORE asking DuckDB for
			// data -- the scan is the expensive half, and a resident table makes it entirely unnecessary.
			auto cached_rows = uncached_mode ? 0 : CacheResidentRowCount(*gpu_plan);
			if (cached_rows > 0) {
				VGPU_LOG(vector_gpu::LogLevel::INFO, "cache_full_hit",
				         "\"path\":\"whole_input\",\"rows\":" + std::to_string(cached_rows));
				vector_gpu::GpuExecuteOptions options;
				options.serve_whole_scan_from_cache = true;
				gstate.result = ExecuteChunk({}, 0, options, cached_rows);
			} else {
				if (uncached_mode) {
					VGPU_LOG(vector_gpu::LogLevel::INFO, "cache",
					         "\"status\":\"auto_bypassed\",\"mode\":\"PIPELINE_STREAM_UNCACHED\","
					         "\"path\":\"whole_input\"");
				} else {
					VGPU_LOG(vector_gpu::LogLevel::INFO, "cache_miss",
					         "\"path\":\"whole_input\",\"reason\":\"table not fully resident; this path does not "
					         "populate the cache\"");
				}
				ScanInputsOnce(context.client, gstate);
				gstate.result = ExecuteChunk(gstate.inputs, gstate.scan_us);
				gstate.scan_us = 0;
			}
		}
		gstate.prepared = true;
	}

	// Drain the current result; when it is exhausted, wait for the next chunk (already computing) and
	// kick off the one after that. A loop rather than an if, because a FILTER chunk can legitimately
	// produce zero rows.
	while (gstate.emitted >=
	       (gstate.result.columns.empty() ? idx_t(0) : idx_t(gstate.result.columns.front().row_count))) {
		if (!gstate.chunked || !gstate.pending.valid()) {
			// Every chunk has been executed and drained, so the scan (if there was one) ran to
			// completion -- the one and only point at which a staged column set may be published. Done
			// here rather than when the scanner reports exhaustion, because a chunk's H2D into the cache's
			// buffers is only guaranteed complete once its future has been get()'d.
			FinishStaging(gstate);
			// MUST clear the chunk before reporting FINISHED. DuckDB reuses the same DataChunk across
			// calls, so returning without touching it leaves the PREVIOUS call's rows and cardinality in
			// place, and the consumer counts them a second time. The pre-chunking code never hit this
			// because ColumnDataCollection::Scan zeroes the chunk itself when it runs out.
			// Symptom when missing: SUM stayed correct (the stale rows repeat real values) while COUNT(*)
			// over-reported -- 1,385,095 against a true 999,495.
			chunk.SetCardinality(0);
			return SourceResultType::FINISHED;
		}
		// Waits only if the producer has not finished yet -- often it already has, since scanning the
		// NEXT batch (below) took real time too. Any exception the producer threw is rethrown here, on
		// the thread DuckDB can actually see.
		gstate.result = gstate.pending.get();
		gstate.emitted = 0;
		if (!gstate.cache_chunk_rows.empty()) {
			// Cache replay: nothing to scan, just dispatch the next resident chunk.
			KickOffCachedChunk(gstate);
		} else {
			// Kick off the batch that was scanned WHILE we were draining/waiting above, then immediately
			// start scanning the one after THAT -- keeps every subsequent chunk's scan overlapping the
			// previous chunk's GPU execution, not just the first one.
			KickOffCompute(gstate);
			ScanNextBatch(gstate);
		}
	}

	auto total_rows = idx_t(gstate.result.columns.front().row_count);
	auto count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, total_rows - gstate.emitted);
	chunk.Reset();
	for (idx_t col = 0; col < gstate.result.columns.size(); col++) {
		ConvertGpuColumnRangeToVector(gstate.result.columns[col], gstate.emitted, count, chunk.data[col]);
	}
	chunk.SetCardinality(count);
	gstate.emitted += count;
	VGPU_LOG(vector_gpu::LogLevel::DEBUG, "emit",
	         "\"count\":" + std::to_string(count) + ",\"emitted\":" + std::to_string(gstate.emitted) +
	             ",\"total\":" + std::to_string(total_rows) + ",\"cols\":" + std::to_string(gstate.result.columns.size()));
	return SourceResultType::HAVE_MORE_OUTPUT;
}

} // namespace vector_gpu
