#pragma once

#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "gpu_engine.hpp"                             // vector_gpu::GpuPlanNode
#include "physical_gpu_streaming_projection.hpp"      // GpuProjectionLevel

#include <string>
#include <vector>

namespace vector_gpu {

//! Replaces a Table-Checker-approved logical subtree. Holds a pre-translated GpuPlanNode plus a reference
//! to the original subtree (kept only for ToString/EXPLAIN readability and for extracting bound column
//! types when building PhysicalGpuExecute — not re-executed).
struct GpuExecuteOperator : public duckdb::LogicalExtensionOperator {
public:
	GpuExecuteOperator(std::shared_ptr<GpuPlanNode> plan, duckdb::vector<duckdb::LogicalType> result_types,
	                    duckdb::unique_ptr<duckdb::LogicalOperator> original_subtree,
	                    ExecutionMode execution_mode = ExecutionMode::PIPELINE_STREAM_CACHED);

	std::shared_ptr<GpuPlanNode> gpu_plan;
	duckdb::vector<duckdb::LogicalType> result_types;
	ExecutionMode execution_mode = ExecutionMode::PIPELINE_STREAM_CACHED;
	//! Kept for diagnostics/fallback only; PhysicalGpuExecute does not execute this.
	duckdb::unique_ptr<duckdb::LogicalOperator> original_subtree;
	//! The replaced subtree's column bindings, captured at construction. Returned verbatim by
	//! GetColumnBindings so operators above this node keep resolving their expressions — essential now
	//! that an inner subtree (not just the plan root) can be offloaded.
	duckdb::vector<duckdb::ColumnBinding> original_bindings;

public:
	duckdb::PhysicalOperator &CreatePlan(duckdb::ClientContext &context,
	                                    duckdb::PhysicalPlanGenerator &planner) override;
	duckdb::string GetExtensionName() const override {
		return "vector_gpu_execute";
	}
	duckdb::string GetName() const override {
		return "GPU_EXECUTE";
	}
	duckdb::vector<duckdb::ColumnBinding> GetColumnBindings() override;
	void ResolveTypes() override;
};

//! Replaces ONLY a chain of projections, keeping the operator below them (the scan, and any filter over
//! it) as a real CHILD that DuckDB executes normally. The split that makes `WHERE` offloadable.
//!
//! Every other GPU operator in this engine replaces the scan too, and then re-opens the table by name --
//! which is precisely why filtered scans had to be declined: DuckDB pushes the predicate into the scan AND
//! prunes row groups it proves wholly satisfy it, folding their counts into constants elsewhere in the
//! plan, so a by-name re-scan double-counts (docs/TODO.md P2 item 6). Here the scan, the predicate and the
//! pruning all stay with DuckDB and this node consumes whatever survives, so there is nothing to
//! double-count and only surviving rows are ever moved.
struct GpuStreamProjectionOperator : public duckdb::LogicalExtensionOperator {
public:
	GpuStreamProjectionOperator(std::shared_ptr<GpuPlanNode> plan, duckdb::vector<duckdb::LogicalType> result_types,
	                            std::vector<std::string> input_names, std::vector<GpuProjectionLevel> levels,
	                            duckdb::vector<duckdb::ColumnBinding> original_bindings,
	                            std::vector<duckdb::idx_t> active_indices = {},
	                            ExecutionMode execution_mode = ExecutionMode::PIPELINE_STREAM_CACHED);

	std::shared_ptr<GpuPlanNode> gpu_plan;
	duckdb::vector<duckdb::LogicalType> result_types;
	ExecutionMode execution_mode = ExecutionMode::PIPELINE_STREAM_CACHED;
	//! Names of `children[0]`'s output columns, in its output order -- the same names the lowered
	//! expressions reference them by.
	std::vector<std::string> input_names;
	//! The replaced projections in bound form, bottom-up, carried through to the physical operator so it
	//! can finish the query on the CPU if the GPU stops being worth it.
	std::vector<GpuProjectionLevel> levels;
	//! The bindings of the projection chain's ROOT, which anything above this node still refers to.
	duckdb::vector<duckdb::ColumnBinding> original_bindings;
	//! Indices into children[0]'s output chunk for the active columns. Empty means all columns are active.
	std::vector<duckdb::idx_t> active_indices;

public:
	duckdb::PhysicalOperator &CreatePlan(duckdb::ClientContext &context,
	                                     duckdb::PhysicalPlanGenerator &planner) override;
	duckdb::string GetExtensionName() const override {
		return "vector_gpu_stream_projection";
	}
	duckdb::string GetName() const override {
		return "GPU_STREAM_PROJECT";
	}
	duckdb::vector<duckdb::ColumnBinding> GetColumnBindings() override;
	void ResolveTypes() override;
};

} // namespace vector_gpu
