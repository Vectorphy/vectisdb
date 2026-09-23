#include "gpu_logical_operator.hpp"
#include "physical_gpu_execute.hpp"
#include "physical_gpu_streaming_projection.hpp"

#include "duckdb/execution/physical_plan_generator.hpp"

namespace vector_gpu {

using namespace duckdb;

GpuExecuteOperator::GpuExecuteOperator(std::shared_ptr<GpuPlanNode> plan, vector<LogicalType> result_types_p,
                                        unique_ptr<LogicalOperator> original_subtree_p,
                                        ExecutionMode execution_mode_p)
    : gpu_plan(std::move(plan)), result_types(std::move(result_types_p)),
      execution_mode(execution_mode_p), original_subtree(std::move(original_subtree_p)) {
	types = result_types;
	if (gpu_plan) {
		gpu_plan->execution_mode = execution_mode;
	}
	// Capture the bindings of the subtree being replaced. Any operator ABOVE this one still refers to
	// the original subtree's bindings in its expressions, so this node has to keep answering to them —
	// see GetColumnBindings.
	if (original_subtree) {
		original_bindings = original_subtree->GetColumnBindings();
	}
}

void GpuExecuteOperator::ResolveTypes() {
	types = result_types;
}

vector<ColumnBinding> GpuExecuteOperator::GetColumnBindings() {
	// Must reproduce the REPLACED subtree's bindings exactly. This node can sit anywhere in the plan
	// (the optimizer offloads the largest GPU-executable subtree, not just the root), and every
	// expression in the operators above it was bound against the original subtree's column bindings.
	// Returning fresh TableIndex(0) bindings instead — as an earlier version did — is invisible while
	// this node is the root, but the moment anything sits on top of it (a LIMIT, an ORDER BY, an
	// aggregate) planning fails with "Failed to bind column reference".
	if (!original_bindings.empty()) {
		return original_bindings;
	}
	// Only reachable if no original subtree was supplied (defensive; the optimizer always supplies one).
	vector<ColumnBinding> bindings;
	for (idx_t i = 0; i < result_types.size(); i++) {
		bindings.emplace_back(TableIndex(0), ProjectionIndex(i));
	}
	return bindings;
}

PhysicalOperator &GpuExecuteOperator::CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) {
	// This operator replaced its whole subtree, scan included, so the physical form is the source-shaped
	// PhysicalGpuExecute that re-scans by name. The streaming pipeline is reached through
	// GpuStreamProjectionOperator instead -- it needs a real child to consume, which is exactly what makes
	// it able to sit above a filter (see that class, and TrySplitOffload).
	return planner.Make<PhysicalGpuExecute>(result_types, gpu_plan, estimated_cardinality, execution_mode);
}

GpuStreamProjectionOperator::GpuStreamProjectionOperator(std::shared_ptr<GpuPlanNode> plan,
                                                         vector<LogicalType> result_types_p,
                                                         std::vector<std::string> input_names_p,
                                                         std::vector<GpuProjectionLevel> levels_p,
                                                         vector<ColumnBinding> original_bindings_p,
                                                         std::vector<duckdb::idx_t> active_indices_p,
                                                         ExecutionMode execution_mode_p)
    : gpu_plan(std::move(plan)), result_types(std::move(result_types_p)), execution_mode(execution_mode_p),
      input_names(std::move(input_names_p)), levels(std::move(levels_p)),
      original_bindings(std::move(original_bindings_p)), active_indices(std::move(active_indices_p)) {
	types = result_types;
	if (gpu_plan) {
		gpu_plan->execution_mode = execution_mode;
	}
}

void GpuStreamProjectionOperator::ResolveTypes() {
	types = result_types;
}

vector<ColumnBinding> GpuStreamProjectionOperator::GetColumnBindings() {
	// The replaced projection chain's ROOT bindings, verbatim -- every expression in the operators above
	// this node was bound against them. Same rule, and the same failure mode when it is not followed
	// ("Failed to bind column reference"), as GpuExecuteOperator::GetColumnBindings above.
	return original_bindings;
}

PhysicalOperator &GpuStreamProjectionOperator::CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) {
	// The child is planned by DuckDB exactly as it would have been without us: a PhysicalTableScan
	// carrying whatever predicate was pushed into it, with a PhysicalFilter above it when the optimizer
	// left one standing. That is the entire mechanism by which a WHERE clause becomes offloadable -- the
	// filtering stays where DuckDB put it, and the GPU sees only what survives.
	auto &child_plan = planner.CreatePlan(*children[0]);
	auto &op = planner.Make<PhysicalGpuStreamingProjection>(result_types, gpu_plan, input_names, std::move(levels),
	                                                        estimated_cardinality, active_indices, execution_mode);
	op.children.push_back(child_plan);
	return op;
}

} // namespace vector_gpu
