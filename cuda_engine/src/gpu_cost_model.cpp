// gpu_cost_model.cpp
//
// Two questions about a GpuPlanNode tree, both answered at ROUTING time so a bad plan becomes a silent
// CPU fallback instead of a failed or slow query (docs/CONTEXT.md gotcha 2):
//
//   EstimateWorkingSet  -- will it FIT?     Peak device bytes for one execution.
//   AnalyzePlanShape    -- will it PAY?     Weighted arithmetic derived per row of data moved.
//
// The two share this file because they share their inputs and their failure mode: both are hand-derived
// models of what the operators next door actually do, and both fail as a wrong routing decision rather
// than a wrong answer.
//
// WHY THIS FILE EXISTS AT ALL, rather than the one-line "rows x output width" the extension used to
// compute for itself. Measured in session 26: at 64M cross-product pairs that formula predicted 1464 MiB
// against an actual 3446 MiB -- 2.35x low -- because it sizes an operator's OUTPUT and every operator's
// real cost is its INTERNAL temporaries. A global AVG over 64M rows allocates ~2 GiB of permutation,
// sort-key, head-flag and group-id buffers in order to produce 8 bytes. No amount of tuning the VRAM
// budget fraction fixes an input that is wrong by 135%.
//
// AND WHY IT LIVES IN cuda_engine RATHER THAN IN THE EXTENSION. Every number below was read off a
// specific ctx.arena.Alloc call in the operator next door. Kept on the far side of the boundary it would
// go stale the first time somebody adds an allocation to gpu_groupby.cu, and the failure would surface as
// a wrong routing decision under memory pressure -- the least debuggable symptom this engine has. Here,
// the model and the allocations are edited together, and tests/test_gpu_cost_model.cpp checks the
// prediction against a real cudaMemGetInfo delta so drift fails a test instead of a query.
//
// EVERYTHING SATURATES. n*m is exactly the quantity that overflows a 64-bit row count, and an estimate
// that wraps reads as "fits comfortably" -- the one direction this must never fail in.

#include "gpu_engine.hpp"

#include <cstdint>
#include <limits>

namespace vector_gpu {

namespace {

constexpr uint64_t kMax = (std::numeric_limits<uint64_t>::max)();

uint64_t SatMul(uint64_t a, uint64_t b) {
	if (a == 0 || b == 0) {
		return 0;
	}
	if (a > kMax / b) {
		return kMax;
	}
	return a * b;
}

uint64_t SatAdd(uint64_t a, uint64_t b) {
	return (a > kMax - b) ? kMax : a + b;
}

//! Byte width of one output row. Falls back to 8 when the translator left it unset: a zero width would
//! make an unbounded row count cost nothing, which is the wrong direction to be wrong in.
uint64_t RowWidth(const GpuPlanNode &node) {
	return node.estimated_row_width_bytes == 0 ? sizeof(double) : node.estimated_row_width_bytes;
}

uint64_t NodeRows(const GpuPlanNode &node, uint64_t cap) {
	auto rows = node.estimated_rows;
	return (cap != 0 && rows > cap) ? cap : rows;
}

//! Rows entering `node` from its i-th child, which is NOT node.estimated_rows for any operator that
//! changes cardinality. For GROUP_BY_AGGREGATE in particular, node.estimated_rows is DuckDB's GROUP
//! count -- using it here would size a 64M-row sort against 5 groups.
uint64_t ChildRows(const GpuPlanNode &node, size_t index, uint64_t cap) {
	if (index >= node.children.size()) {
		return 0;
	}
	return NodeRows(*node.children[index], cap);
}

//! Device bytes this ONE node allocates, excluding its children. Each term cites the allocation it
//! models; if you change an operator's allocations, change the matching term here.
uint64_t NodeBytes(const GpuPlanNode &node, uint64_t cap) {
	const auto width = RowWidth(node);
	const auto out_rows = NodeRows(node, cap);

	switch (node.op_type) {
	case GpuOpType::SCAN:
		// gpu_executor.cu ExecuteScan: one device buffer per scanned column.
		return SatMul(out_rows, width);

	case GpuOpType::PROJECTION:
		// gpu_executor.cu ExecuteProjection: one output buffer per expression. No temporaries -- the
		// fused kernel reads its inputs in place, through the selection vector when there is one.
		return SatMul(out_rows, width);

	case GpuOpType::CROSS_PRODUCT:
		// gpu_cross_join.cu: output columns only. The pair indices are computed in-kernel from the row
		// number (idx / m, idx % m) precisely so there are no n*m index vectors to hold.
		return SatMul(out_rows, width);

	case GpuOpType::FILTER: {
		// gpu_executor.cu ExecuteFilter: one bool mask per predicate, a uint64 compaction buffer, and a
		// uint32 selection vector. Carried columns are NOT copied -- that is what late materialization
		// buys -- so a filter costs nothing proportional to its output width.
		const auto in_rows = ChildRows(node, 0, cap);
		const uint64_t predicates = node.expressions.empty() ? 1 : node.expressions.size();
		const uint64_t per_row = predicates /* bool masks */ + sizeof(uint64_t) /* indices */
		                         + sizeof(uint32_t) /* selection, bounded by input rows */;
		return SatMul(in_rows, per_row);
	}

	case GpuOpType::GROUP_BY_AGGREGATE: {
		// gpu_groupby.cu: permutation, sort_key, head and group_id are allocated unconditionally, plus
		// one widened key buffer per key column -- all uint64 over the INPUT rows. This is the term the
		// old estimate missed entirely.
		const auto in_rows = ChildRows(node, 0, cap);
		const uint64_t buffers = 4 + node.group_keys.size();
		auto bytes = SatMul(in_rows, SatMul(buffers, sizeof(uint64_t)));
		// Thrust's stable_sort_by_key allocates its own double-buffer OUTSIDE our arena (one key array
		// plus one value array), so it is invisible at the Alloc sites but very much present on the
		// device. Measured: without this term the model came in at 0.82 of observed peak for a 4M-row
		// single-key group-by -- under-predicting, which is the one direction that matters. No sort pass
		// runs when there are no keys (the global-aggregate case), so this is conditional.
		if (!node.group_keys.empty()) {
			bytes = SatAdd(bytes, SatMul(in_rows, 2 * sizeof(uint64_t)));
		}
		// Per-group output: group_rows, one column per aggregate, and AVG's sum/count scratch.
		auto per_group = SatAdd(width, sizeof(uint64_t));
		for (auto &aggregate : node.aggregates) {
			if (aggregate.kind == GpuAggregateKind::AVG) {
				per_group = SatAdd(per_group, sizeof(double) + sizeof(int64_t));
			}
		}
		return SatAdd(bytes, SatMul(out_rows, per_group));
	}

	case GpuOpType::HASH_JOIN: {
		// gpu_hash_join.cu: left gets widened/lower/upper/counts/offsets (5 x uint64), right gets
		// widened/permutation (2 x uint64), and the expansion writes two uint64 index vectors over the
		// OUTPUT rows before gathering the output columns.
		const auto left_rows = ChildRows(node, 0, cap);
		const auto right_rows = ChildRows(node, 1, cap);
		auto bytes = SatMul(left_rows, 5 * sizeof(uint64_t));
		// Right side: widened + permutation, plus Thrust's own sort_by_key double-buffer (see the
		// group-by note above -- same hidden allocation, same reason it is not at an Alloc site).
		bytes = SatAdd(bytes, SatMul(right_rows, 4 * sizeof(uint64_t)));
		return SatAdd(bytes, SatMul(out_rows, SatAdd(2 * sizeof(uint64_t), width)));
	}
	}
	return SatMul(out_rows, width);
}

const char *OperatorName(GpuOpType op_type) {
	switch (op_type) {
	case GpuOpType::SCAN:
		return "SCAN";
	case GpuOpType::FILTER:
		return "FILTER";
	case GpuOpType::PROJECTION:
		return "PROJECTION";
	case GpuOpType::HASH_JOIN:
		return "HASH_JOIN";
	case GpuOpType::GROUP_BY_AGGREGATE:
		return "GROUP_BY_AGGREGATE";
	case GpuOpType::CROSS_PRODUCT:
		return "CROSS_PRODUCT";
	}
	return "UNKNOWN";
}

void Accumulate(const GpuPlanNode &node, uint64_t cap, GpuWorkingSetEstimate &out) {
	for (auto &child : node.children) {
		Accumulate(*child, cap, out);
	}
	auto bytes = NodeBytes(node, cap);
	out.total_bytes = SatAdd(out.total_bytes, bytes);
	if (bytes > out.largest_bytes) {
		out.largest_bytes = bytes;
		out.largest_operator = OperatorName(node.op_type);
	}
}

} // namespace

bool IsRowIndependent(const GpuPlanNode &node) {
	if (node.op_type == GpuOpType::GROUP_BY_AGGREGATE || node.op_type == GpuOpType::HASH_JOIN ||
	    node.op_type == GpuOpType::CROSS_PRODUCT) {
		return false;
	}
	for (auto &child : node.children) {
		if (!IsRowIndependent(*child)) {
			return false;
		}
	}
	return true;
}

namespace {

//! Weighted ops one row through `node`'s own expressions costs, times the rows it evaluates them over.
//!
//! The row count is not always node.estimated_rows. A FILTER evaluates its predicates over every row
//! ENTERING it and only then drops some, so charging it for its output would under-count a selective
//! filter by exactly its selectivity -- the same mistake NodeBytes avoids by reading ChildRows.
double NodeOps(const GpuPlanNode &node) {
	double per_row = 0.0;
	for (auto &expr : node.expressions) {
		per_row += expr.estimated_ops_per_row;
	}
	if (per_row == 0.0) {
		return 0.0;
	}
	const auto rows = node.op_type == GpuOpType::FILTER ? ChildRows(node, 0, 0) : node.estimated_rows;
	return per_row * static_cast<double>(rows);
}

void AccumulateShape(const GpuPlanNode &node, GpuPlanShape &shape) {
	for (auto &child : node.children) {
		AccumulateShape(*child, shape);
	}
	if (node.op_type == GpuOpType::SCAN) {
		shape.scanned_rows = SatAdd(shape.scanned_rows, node.estimated_rows);
		shape.scanned_bytes = SatAdd(shape.scanned_bytes, SatMul(node.estimated_rows, RowWidth(node)));
	}
	if (node.estimated_rows > shape.max_rows) {
		shape.max_rows = node.estimated_rows;
	}
	shape.total_ops += NodeOps(node);
}

GpuWorkDensity ClassifyDensity(const GpuPlanShape &shape) {
	// A plan that prices NOTHING is unmeasured, not cheap, and the difference matters: a GROUP BY over a
	// cross product does enormous work per scanned row -- 9k rows in, 64M pairs sorted and reduced -- and
	// carries not one GpuExpr to count it with. Scoring that as INSUFFICIENT would decline a measured
	// 1.90x win (session 27). Handing it to the caller as INCONCLUSIVE leaves exactly the pre-existing
	// row-amplification behaviour in place for the shapes this model cannot see.
	if (shape.total_ops == 0.0) {
		return GpuWorkDensity::INCONCLUSIVE;
	}
	if (shape.ops_per_scanned_row >= kOffloadOpsPerRow) {
		return GpuWorkDensity::SUFFICIENT;
	}
	if (shape.ops_per_scanned_row < kMinOpsPerScannedRow) {
		return GpuWorkDensity::INSUFFICIENT;
	}
	return GpuWorkDensity::INCONCLUSIVE;
}

} // namespace

GpuPlanShape AnalyzePlanShape(const GpuPlanNode &plan) {
	GpuPlanShape shape;
	AccumulateShape(plan, shape);
	// A plan that scans nothing has no data cost to amortise, so nothing is being traded off. Reporting
	// 0 would read as "worst possible" and decline it.
	shape.amplification = shape.scanned_rows == 0
	                          ? static_cast<double>(shape.max_rows)
	                          : static_cast<double>(shape.max_rows) / static_cast<double>(shape.scanned_rows);
	// Same reasoning for the density metrics: nothing scanned means nothing to amortise, so all of the
	// plan's arithmetic is pure gain. Dividing by zero here would produce a NaN that compares false
	// against BOTH thresholds and lands the plan in INCONCLUSIVE by accident.
	shape.ops_per_scanned_row =
	    shape.scanned_rows == 0 ? shape.total_ops : shape.total_ops / static_cast<double>(shape.scanned_rows);
	shape.ops_per_scanned_byte =
	    shape.scanned_bytes == 0 ? shape.total_ops : shape.total_ops / static_cast<double>(shape.scanned_bytes);
	shape.density = ClassifyDensity(shape);
	return shape;
}

GpuWorkingSetEstimate EstimateWorkingSet(const GpuPlanNode &plan, uint64_t max_chunk_rows) {
	GpuWorkingSetEstimate estimate;
	// The cap applies only where the executor genuinely chunks. Applying it to a GROUP BY or a join would
	// approve a plan sized against 2M rows that then allocates for the entire input.
	const bool chunked = max_chunk_rows != 0 && IsRowIndependent(plan);
	estimate.chunk_capped = chunked;
	Accumulate(plan, chunked ? max_chunk_rows : 0, estimate);
	return estimate;
}

uint64_t EstimateColumnsSize(const GpuPlanNode &plan) {
	uint64_t total = 0;
	if (plan.op_type == GpuOpType::SCAN) {
		uint64_t width = RowWidth(plan);
		if (width == 0 && !plan.column_names.empty()) {
			width = plan.column_names.size() * sizeof(double);
		}
		total = SatMul(plan.estimated_rows, width);
	}
	for (const auto &child : plan.children) {
		if (child) {
			total = SatAdd(total, EstimateColumnsSize(*child));
		}
	}
	return total;
}

ExecutionMode DetermineExecutionMode(const GpuPlanNode &plan, uint64_t vram_budget, double threshold) {
	uint64_t estimated_columns_size = EstimateColumnsSize(plan);
	ExecutionMode execution_mode = ExecutionMode::PIPELINE_STREAM_CACHED;
	if (vram_budget > 0 && estimated_columns_size > vram_budget * threshold) {
		execution_mode = ExecutionMode::PIPELINE_STREAM_UNCACHED;
	}
	return execution_mode;
}

} // namespace vector_gpu

