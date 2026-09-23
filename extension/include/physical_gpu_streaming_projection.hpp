#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/expression.hpp"
#include "gpu_engine.hpp"

#include <memory>
#include <string>
#include <vector>

namespace vector_gpu {

//! One projection level of the offloaded chain, kept in the form the CPU can evaluate it in.
//!
//! The GPU form of these expressions is already inside `gpu_plan` as generated CUDA. This is the ORIGINAL
//! bound form, retained so the operator can finish a query on the CPU when the GPU stops being worth it
//! (see the bailout in the .cpp). Bottom-up, like the pipeline's own levels: `expressions` read the
//! previous level's output, or the child operator's chunk at level 0.
struct GpuProjectionLevel {
	duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions;
	//! Types this level's expressions produce, in order. The top level's must equal the operator's own
	//! output types.
	duckdb::vector<duckdb::LogicalType> types;
};

//! Runs a chain of projections on the GPU over rows DuckDB's own operators produced, through
//! GpuStreamPipeline's three CUDA streams.
//!
//! THIS IS AN OPERATOR, NOT A SOURCE, AND THAT IS THE WHOLE POINT. Every earlier GPU operator in this
//! engine replaced the scan as well as the work above it, and then re-opened the table BY NAME to read it
//! (ScanTableToGpuColumns / StreamingTableScanner). That is what forced `WHERE` clauses onto the CPU:
//! DuckDB pushes a predicate down into the scan AND prunes row groups it can prove wholly satisfy it,
//! folding their counts into constants elsewhere in the plan -- so a by-name re-scan returns rows the
//! surrounding plan has already accounted for. `SELECT COUNT(*) FROM t WHERE k > 100` reported 1,385,095
//! against a true 999,495 (docs/TODO.md P2 item 6, pinned by `pushdown/count-star-not-double-counted`).
//!
//! Consuming DuckDB's own output removes the hazard at the root rather than working around it. The scan,
//! the pushed-down predicate, the row-group pruning and any separate `PhysicalFilter` all stay exactly
//! where DuckDB put them, on the CPU, and this operator sees only the rows that survived. Nothing is
//! re-scanned, so nothing can be double-counted, and only surviving rows ever cross PCIe.
//!
//! SHAPE. `children[0]` is whatever DuckDB planned below the projections -- typically
//! `PhysicalFilter <- PhysicalTableScan`, or just the scan when the predicate was pushed into it. The
//! offloaded part is the projection chain above that; see GpuStreamPipeline::Supports.
//!
//! SINGLE-THREADED (`ParallelOperator() == false`), which makes DuckDB run the whole pipeline
//! sequentially. Not merely unimplemented: the pinned input slots come from GpuMemoryPool's fixed CYCLIC
//! rotation, so two producers drawing from it concurrently can each end up waiting on a slot the other
//! holds.
class PhysicalGpuStreamingProjection : public duckdb::PhysicalOperator {
public:
	static constexpr const duckdb::PhysicalOperatorType TYPE = duckdb::PhysicalOperatorType::EXTENSION;

	//! Below this many rows in a batch, running it on the GPU is not worth the transfer: bailout territory.
	//! See ShouldBailOut in the .cpp for exactly when it fires and what it costs.
	static constexpr duckdb::idx_t MIN_GPU_BATCH_ROWS = 10000;

	//! True when this operator can run `plan` at all. Checked by the caller BEFORE constructing one --
	//! there is no fallback once a physical plan has been committed, so an unsupported shape must never
	//! reach here.
	static bool Supports(const GpuPlanNode &plan);

	//! `input_names` names the columns `children[0]` produces, in its output order, using the same names
	//! the plan's expressions reference them by. `cpu_levels` is the same projection chain in bound form,
	//! bottom-up, for the CPU bailout path.
	PhysicalGpuStreamingProjection(duckdb::PhysicalPlan &physical_plan, duckdb::vector<duckdb::LogicalType> types,
	                               std::shared_ptr<GpuPlanNode> gpu_plan, std::vector<std::string> input_names,
	                               std::vector<GpuProjectionLevel> cpu_levels,
	                               duckdb::idx_t estimated_cardinality,
	                               std::vector<duckdb::idx_t> active_indices = {},
	                               ExecutionMode execution_mode = ExecutionMode::PIPELINE_STREAM_CACHED);

	std::shared_ptr<GpuPlanNode> gpu_plan;
	ExecutionMode execution_mode = ExecutionMode::PIPELINE_STREAM_CACHED;
	//! Names of `children[0]`'s output columns, in order.
	std::vector<std::string> input_names;
	//! The projection chain in bound form, bottom-up. Mutable only in the sense that ExpressionExecutors
	//! are built from it; the expressions themselves are never modified.
	std::vector<GpuProjectionLevel> cpu_levels;
	//! Active column indices in children[0]'s output chunk (empty = all columns).
	std::vector<duckdb::idx_t> active_indices;

public:
	duckdb::string GetName() const override {
		return "GPU_STREAM_PROJECT";
	}
	duckdb::InsertionOrderPreservingMap<duckdb::string> ParamsToString() const override;

	bool ParallelOperator() const override {
		return true;
	}
	//! The tail of the input never fills a batch, so it is only flushed once DuckDB says the input is
	//! done. Without this the last partial batch would simply never be executed.
	bool RequiresFinalExecute() const override {
		return true;
	}

	duckdb::unique_ptr<duckdb::GlobalOperatorState> GetGlobalOperatorState(duckdb::ClientContext &context) const override;
	duckdb::unique_ptr<duckdb::OperatorState> GetOperatorState(duckdb::ExecutionContext &context) const override;
	duckdb::OperatorResultType Execute(duckdb::ExecutionContext &context, duckdb::DataChunk &input,
	                                   duckdb::DataChunk &chunk, duckdb::GlobalOperatorState &gstate,
	                                   duckdb::OperatorState &state) const override;
	duckdb::OperatorFinalizeResultType FinalExecute(duckdb::ExecutionContext &context, duckdb::DataChunk &chunk,
	                                                duckdb::GlobalOperatorState &gstate,
	                                                duckdb::OperatorState &state) const override;

private:
	//! `children[0]`'s output types -- the schema of every chunk this operator is handed.
	const duckdb::vector<duckdb::LogicalType> &InputTypes() const;

	//! Builds the accumulator and the pipeline on first use. Throws a NON-fatal DuckDB exception (never
	//! InternalException -- see docs/KNOWN_ISSUES.md) if either cannot be created.
	void Prepare(duckdb::ClientContext &context, class GpuStreamProjectionGlobalState &state) const;

	//! Hands the accumulator's current rows to the pipeline. No-op when it holds none.
	void SubmitAccumulated(class GpuStreamProjectionGlobalState &state) const;

	//! Emits up to STANDARD_VECTOR_SIZE rows of the result currently being drained. Returns false when
	//! there was nothing to emit.
	bool EmitFromResult(duckdb::DataChunk &chunk, class GpuStreamProjectionGlobalState &state) const;

	//! Runs the whole projection chain on the CPU, `input` -> `chunk`, level by level. The bailout path.
	void ExecuteOnCpu(duckdb::DataChunk &input, duckdb::DataChunk &chunk,
	                  class GpuStreamProjectionGlobalState &gstate, class GpuStreamProjectionLocalState &lstate) const;

	//! True when the CPU filter above has turned out to be aggressive enough that batches will never fill:
	//! a whole batch window of INPUT rows has gone by and fewer than MIN_GPU_BATCH_ROWS of them survived.
	bool ShouldBailOut(class GpuStreamProjectionGlobalState &state) const;

	//! Copies whatever is still in pinned memory back out into ordinary DataChunks for the CPU to finish,
	//! releases the accumulator (and its ring slot), and switches to draining. Only ever called with fewer
	//! than MIN_GPU_BATCH_ROWS rows accumulated, so the copy is bounded at five chunks.
	void BeginBailout(class GpuStreamProjectionGlobalState &state, const char *reason) const;
	void MoveAccumulatedToCpu(class GpuStreamProjectionGlobalState &state) const;

	//! True when a batch can be handed to the pipeline right now. When it cannot, takes the oldest finished
	//! batch (which is what frees a stage) and emits it into `chunk` -- so a false means "return
	//! HAVE_MORE_OUTPUT and try again", not "something went wrong".
	bool MakeRoomToSubmit(duckdb::DataChunk &chunk, class GpuStreamProjectionGlobalState &state) const;
};

} // namespace vector_gpu
