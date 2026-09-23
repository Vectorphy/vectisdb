#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "gpu_engine.hpp"

namespace vector_gpu {

//! Streams results of a GPU-executed subplan back into DuckDB as ordinary DataChunks. Behaves like a
//! source operator (PhysicalColumnDataScan): on first pull it materializes the entire result by calling
//! into cuda_engine, then streams it out chunk-by-chunk. DuckDB's executor sees only a normal
//! PhysicalOperator producing standard DataChunks — it has no awareness that NVIDIA silicon was involved,
//! per the "black box" design in docs/ARCHITECTURE.md.
class PhysicalGpuExecute : public duckdb::PhysicalOperator {
public:
	static constexpr const duckdb::PhysicalOperatorType TYPE = duckdb::PhysicalOperatorType::EXTENSION;

	//! Largest input chunk a row-independent plan is executed in. Public because routing must size VRAM
	//! against the same number execution will actually use: each chunk builds and destroys its own
	//! DeviceArena and only one prefetch is ever in flight, so peak device memory for a chunked plan is
	//! one chunk, not the whole input. TableChecker::HasVramHeadroomForPlan passes this to
	//! EstimateWorkingSet. If the two ever disagree, routing approves a plan execution cannot run.
	static constexpr duckdb::idx_t MAX_CHUNK_ROWS = 2u * 1024u * 1024u;

	PhysicalGpuExecute(duckdb::PhysicalPlan &physical_plan, duckdb::vector<duckdb::LogicalType> types,
	                    std::shared_ptr<GpuPlanNode> gpu_plan, duckdb::idx_t estimated_cardinality,
	                    ExecutionMode execution_mode = ExecutionMode::PIPELINE_STREAM_CACHED);

	std::shared_ptr<GpuPlanNode> gpu_plan;
	ExecutionMode execution_mode = ExecutionMode::PIPELINE_STREAM_CACHED;

public:
	duckdb::string GetName() const override {
		return "GPU_EXECUTE";
	}
	bool IsSource() const override {
		return true;
	}
	bool ParallelSource() const override {
		return false; // TODO(phase 1): support partitioned/parallel scan-out of the materialized result
	}

	duckdb::unique_ptr<duckdb::GlobalSourceState> GetGlobalSourceState(duckdb::ClientContext &context) const override;
	duckdb::unique_ptr<duckdb::LocalSourceState> GetLocalSourceState(duckdb::ExecutionContext &context,
	                                                                 duckdb::GlobalSourceState &gstate) const override;
	duckdb::SourceResultType GetDataInternal(duckdb::ExecutionContext &context, duckdb::DataChunk &chunk,
	                                         duckdb::OperatorSourceInput &input) const override;

private:
	//! Called once, lazily, guarded by the global state's mutex.
	//! Runs the plan on the GPU once and hands back the raw device-copied result. The result is then
	//! emitted to DuckDB one STANDARD_VECTOR_SIZE chunk at a time straight from these buffers — it is
	//! deliberately NOT packed into a ColumnDataCollection first. That intermediate was a full second
	//! copy of every output row (measured at 6.8 ms for a 1M-row join, plus the peak memory), and it
	//! bought nothing: DuckDB consumes the result chunk-wise regardless.
	void ExecuteOnce(duckdb::ClientContext &context, vector_gpu::GpuExecutionResult &result) const;

	//! WHOLE-INPUT path only (GROUP BY / JOIN, or VECTOR_GPU_NO_CHUNK): scans every SCAN leaf into the
	//! global state's host buffers, once per query, via ScanTableToGpuColumns. Unchanged by the session-30
	//! streaming redesign -- these plans need every row at once and don't benefit from batching it.
	void ScanInputsOnce(duckdb::ClientContext &context, class GpuExecuteGlobalState &gstate) const;

	//! WHOLE-INPUT path only. Rows this plan can be served with WITHOUT scanning anything: non-zero only
	//! when EVERY one of its SCAN leaves is already device-resident in full, in which case the caller runs
	//! the plan with GpuExecuteOptions::serve_whole_scan_from_cache and never opens a table.
	//!
	//! All-or-nothing across the leaves, deliberately. A join with one cached side and one uncached one
	//! would have to interleave cached columns with scanned ones in the flat `inputs` list ExecuteScan
	//! consumes positionally -- a real feature, but one whose failure mode is columns silently paired with
	//! the wrong data. Until that is worth building, a partial hit scans normally and repopulates.
	uint64_t CacheResidentRowCount(const GpuPlanNode &plan) const;

	//! Runs the plan over `inputs` -- a complete, independently-owned set of columns: either the whole
	//! table (whole-input path) or one already-scanned streamed batch (chunked path). No slicing happens
	//! here; each caller already hands over exactly the rows to execute. Throws NotImplementedException
	//! if the engine declines the plan.
	//!
	//! `options` carries the GPU-resident column cache wiring (populate this chunk into the cache, or
	//! serve it FROM the cache -- see gpu_engine.hpp). On the cache-replay path `inputs` is empty and
	//! `input_rows` is how many rows that cached chunk holds, so the phase timings still report real
	//! throughput instead of zero.
	vector_gpu::GpuExecutionResult ExecuteChunk(const std::vector<vector_gpu::GpuColumn> &inputs, uint64_t scan_us,
	                                            const vector_gpu::GpuExecuteOptions &options = {},
	                                            uint64_t input_rows = 0) const;

	//! CHUNKED (row-independent) path only. Scans the NEXT batch synchronously, on the calling thread,
	//! into gstate.pending_batch -- ready for KickOffCompute. A no-op once the scanner is exhausted (see
	//! StreamingTableScanner). This is intentionally NOT backgrounded: driving DuckDB's table-scan state
	//! from a second thread is not a documented-safe contract, unlike GPU execution (see
	//! table_scanner.hpp's StreamingTableScanner doc comment and docs/STREAMING_INGEST_DESIGN.md).
	void ScanNextBatch(class GpuExecuteGlobalState &gstate) const;

	//! CHUNKED path only. Starts gstate.pending_batch's GPU execution on a background thread -- so it
	//! runs WHILE the caller goes on to call ScanNextBatch for the batch after that (letting scanning
	//! genuinely overlap GPU execution, not merely follow it) and while DuckDB drains the PREVIOUS chunk's
	//! already-computed result. Consumes (moves out of) pending_batch. No-op if pending_batch is null or
	//! holds zero rows (the scanner's trailing empty batch on exhaustion).
	void KickOffCompute(class GpuExecuteGlobalState &gstate) const;

	//! CHUNKED path, CACHE-REPLAY sub-path. Starts the next already-device-resident chunk's GPU execution
	//! on a background thread, exactly like KickOffCompute does for a freshly scanned batch -- so the
	//! drain loop below is identical for both, and only the source of the rows differs. No-op once every
	//! cached chunk has been dispatched, which is what ends the drain loop (gstate.pending goes invalid).
	void KickOffCachedChunk(class GpuExecuteGlobalState &gstate) const;

	//! CHUNKED path, CACHE-POPULATE sub-path. Closes the staging slot opened for this query: COMMITS it
	//! (making the table's columns servable by a later query) only if the scan genuinely ran to
	//! completion, ABORTS it otherwise. Idempotent, and a no-op when no slot was opened.
	void FinishStaging(class GpuExecuteGlobalState &gstate) const;
};

} // namespace vector_gpu
