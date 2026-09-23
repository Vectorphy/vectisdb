#include "duckdb.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_insert.hpp"
#include "duckdb/planner/operator/logical_update.hpp"
#include "duckdb/planner/operator/logical_delete.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/column_binding_map.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"

#include "table_checker.hpp"
#include "gpu_logical_operator.hpp"
#include "expression_cost.hpp"
#include "expression_translator.hpp"
#include "code_generator.hpp"
#include "kernel_cache.hpp"
#include "gpu_column_cache.hpp"
#include "gpu_offload_extension.hpp"
#include "gpu_logger.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>

using namespace duckdb;

namespace vector_gpu {

//! Offload tracing (see gpu_offload_extension.hpp). std::atomic because the optimizer callback runs on
//! whichever thread is planning a query.
static std::atomic<bool> g_gpu_offload_trace {false};
static std::atomic<uint64_t> g_gpu_offload_count {0};
//! Runtime switch for the Pipeline Stream split-plan path. Initialized from the
//! VECTOR_GPU_STREAM_PIPELINE environment variable on first use (inside StreamSplitEnabled),
//! then kept as a mutable atomic so `SET gpu_stream_pipeline` can change it without a restart.
//! Default: true (Pipeline Stream is the canonical execution mode as of session 49).
static std::atomic<bool> g_stream_split_enabled {true};

void SetGpuOffloadTrace(bool enabled) {
	g_gpu_offload_trace = enabled;
}

uint64_t GetGpuOffloadCount() {
	return g_gpu_offload_count.load();
}

//! Phase timers (see gpu_offload_extension.hpp). Atomic for the same reason as the offload counter:
//! materialization runs on whichever thread pulls from the source operator.
static std::atomic<uint64_t> g_gpu_scan_us {0};
static std::atomic<uint64_t> g_gpu_execute_us {0};
static std::atomic<uint64_t> g_gpu_pack_us {0};
static std::atomic<uint64_t> g_gpu_input_rows {0};
static std::atomic<uint64_t> g_gpu_optimize_us {0};

void RecordGpuPhaseTimings(uint64_t scan_us, uint64_t execute_us, uint64_t pack_us, uint64_t input_rows) {
	if (!g_gpu_offload_trace) {
		return; // costs nothing unless the user asked for tracing
	}
	g_gpu_scan_us += scan_us;
	g_gpu_execute_us += execute_us;
	g_gpu_pack_us += pack_us;
	g_gpu_input_rows += input_rows;
}

GpuPhaseTimings GetGpuPhaseTimings() {
	GpuPhaseTimings timings;
	timings.scan_us = g_gpu_scan_us.load();
	timings.execute_us = g_gpu_execute_us.load();
	timings.pack_us = g_gpu_pack_us.load();
	timings.input_rows = g_gpu_input_rows.load();
	timings.optimize_us = g_gpu_optimize_us.load();
	return timings;
}

//! Resolves a join/group-by key expression to the column name it refers to. Only bare column references
//! are supported (the overwhelmingly common case for equi-join/group-by keys); anything else (an
//! expression computed on the fly, e.g. `a.x + 1 = b.y`) throws GpuUnsupportedExpression, which the
//! caller (Optimize, below) catches to fall back to CPU execution.
static std::string ResolveKeyColumnName(const Expression &expr, const ColumnBindingNames &bindings) {
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
		throw GpuUnsupportedExpression("join/group-by key is not a bare column reference: " + expr.ToString());
	}
	auto &colref = expr.Cast<BoundColumnRefExpression>();
	auto &binding = colref.Binding();
	std::pair<idx_t, idx_t> key {binding.table_index.index, binding.column_index.GetIndex()};
	auto it = bindings.find(key);
	if (it == bindings.end()) {
		throw GpuUnsupportedExpression("join/group-by key column not resolvable to a scanned column");
	}
	return it->second;
}

//! Walks the whole approved subtree bottom-up, collecting (table_index, projection_index) -> column name
//! for every operator that introduces NEW output bindings a parent might reference -- not just
//! LogicalGet. A LogicalFilter/LogicalProjection directly over a scan only ever references the scan's
//! own bindings, but a LogicalProjection sitting *above* a LogicalAggregate or LogicalComparisonJoin
//! references THAT operator's own output bindings (e.g. LogicalAggregate's group_index/aggregate_index),
//! which are unrelated to the original table's column numbering. Discovered the hard way: an initial
//! version of this function only walked LogicalGet, and `SELECT a, COUNT(*) FROM t GROUP BY a` failed to
//! translate (the outer projection couldn't resolve `a`'s post-aggregate binding) even though the
//! group-by key itself resolved fine -- see docs/EXECUTION_TRACKER.md.
//!
//! Must run bottom-up (children before parent): resolving an aggregate's group-output binding needs its
//! child's bindings (the original scan) to already be populated, since the group expression is itself
//! usually a plain column reference into that child.
static void CollectBindingNames(LogicalOperator &op, ColumnBindingNames &names) {
	for (auto &child : op.children) {
		CollectBindingNames(*child, names);
	}

	if (op.type == LogicalOperatorType::LOGICAL_GET) {
		auto &get = op.Cast<LogicalGet>();
		auto &column_ids = get.GetColumnIds();
		for (idx_t i = 0; i < column_ids.size(); i++) {
			std::pair<idx_t, idx_t> key {get.table_index.index, i};
			names[key] = get.GetColumnName(column_ids[i]).GetIdentifierName();
		}
	} else if (op.type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &aggregate = op.Cast<LogicalAggregate>();
		// Detect whether TranslateToGpuPlan will use the computed-key slow path for this aggregate.
		// If any key is not a bare column ref the translator injects a synthetic PROJECTION child
		// and names ALL group key outputs as "col<i>" -- regardless of whether individual keys are
		// bare or computed. An outer operator referencing these bindings must use those same "col<i>"
		// names or it will fail to find its input column at execution time.
		bool any_computed_key = false;
		for (idx_t i = 0; i < aggregate.groups.size(); i++) {
			try {
				ResolveKeyColumnName(*aggregate.groups[i], names);
			} catch (const GpuUnsupportedExpression &) {
				any_computed_key = true;
				break;
			}
		}
		for (idx_t i = 0; i < aggregate.groups.size(); i++) {
			std::string name;
			if (any_computed_key) {
				// TranslateToGpuPlan will emit all group keys through a synthetic PROJECTION child,
				// giving each key the output name "col<i>" (the PROJECTION convention, see
				// GpuNodeOutputNames). Use the same names here so outer-operator references resolve.
				name = "col" + std::to_string(i);
			} else {
				try {
					// Bare column ref (the common case): reuse the child's column name directly.
					name = ResolveKeyColumnName(*aggregate.groups[i], names);
				} catch (const GpuUnsupportedExpression &) {
					// Should not be reachable: any_computed_key would be true if this throws.
					name = "group_" + std::to_string(i);
				}
			}
			names[{aggregate.group_index.index, i}] = name;
		}
		for (idx_t i = 0; i < op.expressions.size(); i++) {
			// Aggregate outputs (SUM(x), COUNT(*), ...) have no single source column to name them after.
			// This name is LOAD-BEARING, not a placeholder: cuda_engine/src/operators/gpu_groupby.cu names
			// its i-th aggregate output column with exactly this string, so a parent operator (e.g. a
			// projection over the aggregate's result) resolves to the column the engine actually emits.
			// Changing the format here without changing it there silently breaks that lookup.
			names[{aggregate.aggregate_index.index, i}] = "agg_" + std::to_string(i);
		}
	} else if (op.type == LogicalOperatorType::LOGICAL_PROJECTION) {
		// A projection renames everything it outputs: cuda_engine's ExecuteProjection emits its i-th
		// expression as "col<i>". Without this, an operator ABOVE a projection (typically an aggregate
		// grouping on a projected key -- DuckDB inserts such a projection to narrow group keys) cannot
		// resolve its own inputs and the whole subtree falls back to CPU.
		//
		// LOAD-BEARING, exactly like the "agg_<i>" names above: this string must match what
		// cuda_engine/src/gpu_executor.cu's ExecuteProjection actually names its outputs.
		auto &projection = op.Cast<LogicalProjection>();
		for (idx_t i = 0; i < op.expressions.size(); i++) {
			names[{projection.table_index.index, i}] = "col" + std::to_string(i);
		}
	}
	// LogicalComparisonJoin needs no special handling here (unlike LogicalAggregate above): per
	// LogicalJoin::GetColumnBindings() (src/planner/operator/logical_join.cpp), only MARK joins
	// introduce a genuinely new output table index (mark_index); every other join type -- including
	// INNER, the only one TranslateToGpuPlan accepts (see the JoinType::INNER check there) -- simply
	// concatenates its children's own bindings unchanged. So a parent operator referencing an INNER
	// join's output is really referencing the original scan's bindings straight through, which are
	// already collected from the children above. This was actually verified, not assumed: an earlier
	// version of this function (and TranslateToGpuPlan) didn't restrict join_type at all, and no test
	// hit a missing-binding failure for the join case the way GROUP BY did for LogicalAggregate.
}

//! Mirrors expression_translator.cpp's (anonymous-namespace) MapLogicalType and table_checker.cpp's
//! IsSupportedType -- intentionally duplicated close to its own use, matching the pattern those files
//! document. Throws GpuUnsupportedExpression so an unsupported aggregate type (notably HUGEINT, which is
//! what DuckDB's SUM over an integer column returns) falls back to CPU instead of failing the query.
static GpuValueType MapAggregateType(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::DECIMAL:
		return type.InternalType() == PhysicalType::INT16 ? GpuValueType::INT16 :
		       type.InternalType() == PhysicalType::INT32 ? GpuValueType::INT32 :
		       type.InternalType() == PhysicalType::INT64 ? GpuValueType::INT64 :
		       type.InternalType() == PhysicalType::INT128 ? GpuValueType::HUGEINT :
		       throw GpuUnsupportedExpression("unsupported decimal width");
	case LogicalTypeId::SMALLINT:
		return GpuValueType::INT16;
	case LogicalTypeId::INTEGER:
		return GpuValueType::INT32;
	case LogicalTypeId::BIGINT:
		return GpuValueType::INT64;
	case LogicalTypeId::HUGEINT:
		return GpuValueType::HUGEINT;
	case LogicalTypeId::FLOAT:
		return GpuValueType::FLOAT32;
	case LogicalTypeId::DOUBLE:
		return GpuValueType::FLOAT64;
	case LogicalTypeId::BOOLEAN:
		return GpuValueType::BOOLEAN;
	default:
		throw GpuUnsupportedExpression("unsupported aggregate type for GPU execution: " + type.ToString());
	}
}

//! Lowers one BoundAggregateExpression into the engine's executable GpuAggregate spec. Only the
//! reductions cuda_engine/src/operators/gpu_groupby.cu actually implements are accepted; every other
//! shape throws GpuUnsupportedExpression, which the caller turns into a silent CPU fallback.
//!
//! The restrictions here deliberately mirror gpu_groupby.cu's own validation (see its file header for
//! why each exists). Checking at translation time is what keeps an unsupported aggregate from being
//! offloaded and then failing the query; the engine-side copy is the backstop for a hand-built plan.
static GpuAggregate TranslateAggregate(const Expression &expr, const ColumnBindingNames &bindings) {
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) {
		throw GpuUnsupportedExpression("aggregate list entry is not a bound aggregate: " + expr.ToString());
	}
	auto &aggregate = expr.Cast<BoundAggregateExpression>();
	if (aggregate.IsDistinct()) {
		throw GpuUnsupportedExpression("DISTINCT aggregates have no GPU implementation");
	}
	if (aggregate.GetFilter()) {
		throw GpuUnsupportedExpression("FILTER-qualified aggregates have no GPU implementation");
	}
	if (aggregate.GetOrderBys()) {
		throw GpuUnsupportedExpression("ordered aggregates have no GPU implementation");
	}

	auto &function_name = aggregate.Function().GetName();
	auto &children = aggregate.GetChildren();

	GpuAggregate result;
	result.output_type = MapAggregateType(expr.GetReturnType());

	if (function_name == "count_star") {
		if (!children.empty()) {
			throw GpuUnsupportedExpression("count_star with arguments is not a shape the GPU path models");
		}
		if (result.output_type != GpuValueType::INT64) {
			throw GpuUnsupportedExpression("COUNT(*) is expected to return BIGINT");
		}
		result.kind = GpuAggregateKind::COUNT_STAR;
		return result;
	}
	if (children.size() != 1) {
		throw GpuUnsupportedExpression("GPU aggregates take exactly one argument: " + expr.ToString());
	}
	result.input_column = ResolveKeyColumnName(*children[0], bindings);
	auto input_type = MapAggregateType(children[0]->GetReturnType());

	if (function_name == "count") {
		if (result.output_type != GpuValueType::INT64) {
			throw GpuUnsupportedExpression("COUNT is expected to return BIGINT");
		}
		result.kind = GpuAggregateKind::COUNT;
		return result;
	}
	if (function_name == "sum") {
		if (result.output_type == GpuValueType::FLOAT64) {
			if (input_type != GpuValueType::FLOAT32 && input_type != GpuValueType::FLOAT64) {
				throw GpuUnsupportedExpression("GPU SUM over floats supports only FLOAT/DOUBLE input");
			}
		} else if (result.output_type == GpuValueType::HUGEINT) {
			if (input_type != GpuValueType::INT16 && input_type != GpuValueType::INT32 && input_type != GpuValueType::INT64 && input_type != GpuValueType::HUGEINT) {
				throw GpuUnsupportedExpression("GPU SUM over integers supports only INT16/32/64/128 input");
			}
		} else {
			throw GpuUnsupportedExpression("GPU SUM supports only DOUBLE or HUGEINT output");
		}
		result.kind = GpuAggregateKind::SUM;
		return result;
	}
	if (function_name == "avg") {
		// DuckDB's AVG returns DOUBLE for every numeric input, so the only shape to accept is
		// numeric-in / DOUBLE-out. The kernel accumulates in double and divides by the group count.
		if (result.output_type != GpuValueType::FLOAT64) {
			throw GpuUnsupportedExpression("GPU AVG is expected to return DOUBLE");
		}
		if (input_type != GpuValueType::INT16 && input_type != GpuValueType::INT32 && 
		    input_type != GpuValueType::INT64 && input_type != GpuValueType::HUGEINT &&
		    input_type != GpuValueType::FLOAT32 && input_type != GpuValueType::FLOAT64) {
			throw GpuUnsupportedExpression("GPU AVG supports numeric columns only");
		}
		result.kind = GpuAggregateKind::AVG;
		return result;
	}
	if (function_name == "min" || function_name == "max") {
		if (input_type != GpuValueType::INT16 && input_type != GpuValueType::INT32 && 
		    input_type != GpuValueType::INT64 && input_type != GpuValueType::HUGEINT &&
		    input_type != GpuValueType::BOOLEAN) {
			// DuckDB orders NaN above every other value; the kernel's comparator does not.
			throw GpuUnsupportedExpression("GPU MIN/MAX supports only integer/boolean columns");
		}
		if (result.output_type != input_type) {
			throw GpuUnsupportedExpression("GPU MIN/MAX expects its result type to match its input");
		}
		result.kind = function_name == "min" ? GpuAggregateKind::MIN : GpuAggregateKind::MAX;
		return result;
	}
	throw GpuUnsupportedExpression("aggregate function '" + function_name.GetIdentifierName() +
	                               "' has no GPU implementation");
}

static GpuExpr TranslateToGpuExpr(const Expression &expr, const ColumnBindingNames &bindings) {
	auto translated = TranslateExpression(expr, bindings);
	GpuExpr result;
	result.generated_cuda_source = WrapAsOutputStatement(translated);
	result.input_columns = translated.input_columns;
	result.output_type = translated.output_type;
	result.source_hash = KernelCache::HashSource(result.generated_cuda_source);
	// Counted HERE, from the bound tree, because this is the last point at which one exists: downstream
	// the expression is CUDA source text. See expression_cost.hpp.
	result.estimated_ops_per_row = EstimateExpressionOpsPerRow(expr);
	return result;
}

//! Recursively lowers an approved LogicalOperator subtree into a GpuPlanNode tree.
//!
//! Real for LOGICAL_GET/LOGICAL_FILTER/LOGICAL_PROJECTION (translated via expression_translator.cpp,
//! which covers column refs, numeric/boolean constants, comparisons, and +,-,*,/ arithmetic) and for
//! LOGICAL_AGGREGATE_AND_GROUP_BY (lowered to an executable GpuAggregate list by TranslateAggregate
//! above, and run by cuda_engine/src/operators/gpu_groupby.cu).
//! LOGICAL_COMPARISON_JOIN is likewise real: INNER-only, one integer/boolean equality key, lowered with
//! an explicit projection-map-aware output column list and run by
//! cuda_engine/src/operators/gpu_hash_join.cu.
//!
//! Throws GpuUnsupportedExpression for anything this pass can't translate (complex predicates, UDFs,
//! non-column-reference join/group keys, ...). The caller (Optimize, below) catches this and falls back
//! to CPU execution -- exactly like a Table Checker rejection, just discovered one level deeper.
static std::shared_ptr<GpuPlanNode> TranslateToGpuPlan(LogicalOperator &op, const ColumnBindingNames &bindings) {
	auto node = std::make_shared<GpuPlanNode>();
	//! Set to true by cases that translate and wire their own children (e.g., the GROUP_BY computed-key
	//! path, which inserts a synthetic PROJECTION child between the logical child and the group-by node).
	//! Prevents the post-switch child loop from adding duplicates.
	bool skip_children = false;

	switch (op.type) {
	case LogicalOperatorType::LOGICAL_GET: {
		auto &get = op.Cast<LogicalGet>();

		// CORRECTNESS GUARD. GpuOpType::SCAN carries only a table name and a column list, and
		// ScanTableToGpuColumns reads every row of those columns. DuckDB, however, routinely pushes a
		// WHERE clause down INTO the scan (EXPLAIN shows it as "Filters: ..." on the Seq Scan) and drops
		// the separate LogicalFilter node. Translating such a Get as a plain SCAN would therefore feed
		// the GPU every row of the table while the query asked for a filtered subset — a silent
		// wrong-answer bug, not a performance issue. Until SCAN can carry pushed-down predicates, any
		// Get that has them (statically or via a dynamic filter from a join) must fall back to CPU.
		//
		// This was masked in testing only because a pushed-down filter also lowers the Get's estimated
		// cardinality below TableChecker's 100k offload threshold; on a larger table the same plan would
		// clear the threshold and return wrong rows.
		// CORRECTNESS GUARD -- and NOT merely "SCAN cannot carry a predicate".
		//
		// Session 21 implemented pushed-down filters properly: TableFilter::ToExpression reconstructs the
		// predicate, which lowers cleanly into a FILTER node above the SCAN, and the rows it produced were
		// exactly right. It still had to be reverted, because applying the predicate is not the whole job.
		//
		// DuckDB derives MORE than a predicate from a pushed-down filter. For
		//   SELECT COUNT(*) FROM t WHERE k > 100
		// the optimized plan is:
		//   Projection "+"(385600, #0)  <- constant folded in
		//     Aggregate count_star()
		//       Seq Scan (Filters: k > 100)
		// DuckDB pruned the row groups it proved ENTIRELY satisfy the predicate, folded their row count
		// into that constant, and left the scan to return only the remainder. ScanTableToGpuColumns
		// re-opens the table BY NAME and reads every row, so our operator correctly returned all 999,495
		// matching rows -- and DuckDB then added its 385,600, reporting 1,385,095.
		//
		// So the hazard is not the predicate; it is that the surrounding plan has already been rewritten
		// on the assumption that this scan returns a PRUNED subset. Nothing in the LogicalGet says which
		// row groups were pruned or what was folded away, so there is no cheap way to tell a safe case
		// from an unsafe one. Until GpuOpType::SCAN can reproduce DuckDB's own pruning decisions, a
		// filtered Get must stay on the CPU.
		//
		// Pinned by `pushdown/count-star-not-double-counted` in gpu_shell/verify_gpu_vs_cpu.sh.
		//
		// NO LONGER THE LAST WORD ON FILTERED SCANS (session 40). A filtered query is not declined outright
		// any more: GpuOffloadOptimizer::TrySplitOffload keeps the Get -- predicate, pruning and all -- as
		// a CPU child and offloads only the projections above it, so the rows reaching the GPU are the ones
		// DuckDB itself produced. Nothing is re-scanned there, so nothing can be double-counted.
		//
		// The refusal below stays exactly as strict for THIS path, because this path is the one that
		// re-opens the table by name. Removing it here would not enable anything the split does not
		// already enable; it would only restore a wrong answer that has a pinned regression test.
		if (get.table_filters.HasFilters()) {
			throw GpuUnsupportedExpression("scan has pushed-down table filters, and this path re-scans the "
			                               "table by name; DuckDB may also have pruned row groups and folded "
			                               "their counts, which a re-scan would double-count (a split plan "
			                               "handles this shape instead -- see TrySplitOffload)");
		}
		// Dynamic filters stay refused: they are populated by a join as it EXECUTES, so there is
		// nothing to translate at plan time.
		if (get.dynamic_filters && get.dynamic_filters->HasFilters()) {
			throw GpuUnsupportedExpression("scan has dynamically-generated filters, whose contents are "
			                               "only known at run time");
		}
		node->op_type = GpuOpType::SCAN;
		// Only a real catalog table can be offloaded: ScanTableToGpuColumns re-opens the source BY NAME
		// through the catalog, so a table FUNCTION (range(), read_csv_auto(), a CTE, ...) has nothing to
		// look up. Falling back to get.function.name here — as an earlier version did — produced a SCAN
		// node naming a function, which then failed at execution with "table 'range' not found" and took
		// the whole query down instead of quietly running on CPU. Found by a real query:
		//   COPY (SELECT ... FROM range(1, 20000001) t(i)) TO 'big.csv'
		auto table_entry = get.GetTable();
		if (!table_entry) {
			throw GpuUnsupportedExpression("scan source is a table function, not a catalog table, so it "
			                               "cannot be re-scanned by name for GPU execution");
		}
		// All three components, not just the name: the scanner re-opens the table by this identity and the
		// GPU-resident column cache keys entries by it. A bare name means two tables called `t` in different
		// schemas or attached databases resolve to -- and cache as -- the same thing. See GpuTableRef.
		node->table.catalog = table_entry->ParentCatalog().GetName().GetIdentifierName();
		node->table.schema = table_entry->ParentSchema().name.GetIdentifierName();
		node->table.table = table_entry->name.GetIdentifierName();
		auto &column_ids = get.GetColumnIds();
		// Virtual columns (rowid, the empty column) name nothing the catalog can re-open -- see the
		// matching refusal in TableChecker::ShouldOffloadRecursive for how a plan acquires one and what it
		// used to cost. Repeated here because this function is reachable independently of that check, and
		// because a GpuUnsupportedExpression is the cheapest possible way to say "not this plan".
		for (auto &column_id : column_ids) {
			if (column_id.IsVirtualColumn()) {
				throw GpuUnsupportedExpression("scan reads virtual column '" +
				                               get.GetColumnName(column_id).GetIdentifierName() +
				                               "', which has no catalog column to re-scan by name");
			}
		}
		// Emit columns in the Get's real OUTPUT order. With projection pushdown, LogicalGet outputs
		// only the projection_ids subset (each id indexing column_ids) rather than every column_id —
		// see LogicalGet::GetColumnBindings. Scanning column_ids blindly would hand the executor the
		// wrong columns, in the wrong order, whenever the offloaded subtree's root is the scan or a
		// filter (where output order is the result order) instead of a projection.
		// Binding resolution is unaffected: CollectBindingNames keys on the same index space.
		if (get.projection_ids.empty()) {
			for (idx_t i = 0; i < column_ids.size(); i++) {
				node->column_names.push_back(get.GetColumnName(column_ids[i]).GetIdentifierName());
			}
		} else {
			for (auto proj_id : get.projection_ids) {
				auto index = proj_id.GetIndex();
				if (index >= column_ids.size()) {
					throw GpuUnsupportedExpression("scan projection id out of range");
				}
				node->column_names.push_back(get.GetColumnName(column_ids[index]).GetIdentifierName());
			}
		}

		break;
	}
	case LogicalOperatorType::LOGICAL_FILTER: {
		node->op_type = GpuOpType::FILTER;
		for (auto &expr : op.expressions) {
			node->expressions.push_back(TranslateToGpuExpr(*expr, bindings));
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_PROJECTION: {
		node->op_type = GpuOpType::PROJECTION;
		for (auto &expr : op.expressions) {
			node->expressions.push_back(TranslateToGpuExpr(*expr, bindings));
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
		auto &join = op.Cast<LogicalComparisonJoin>();
		if (join.join_type != JoinType::INNER) {
			// Correctness fix, not just a translation gap: LEFT/RIGHT/OUTER/SINGLE have different
			// null-handling semantics, and SEMI/ANTI/MARK/RIGHT_SEMI/RIGHT_ANTI have entirely different
			// *output shapes* (e.g. SEMI returns only left-side rows with a match, no right-side columns
			// at all) that a plain build_keys/probe_keys GpuOpType::HASH_JOIN node cannot represent. Before
			// this check, TableChecker's operator whitelist alone would have let any of these reach here
			// and be silently mistranslated as an ordinary equi-join shape. This guard preserves join
			// semantics because the executor supports only the INNER equi-join shape represented here.
			throw GpuUnsupportedExpression("hash join translation only supports JoinType::INNER");
		}
		node->op_type = GpuOpType::HASH_JOIN;
		if (join.conditions.size() != 1) {
			// gpu_hash_join.cu matches on one widened key value; a composite key needs the whole key
			// tuple collapsed to a single comparable value, which it does not attempt.
			throw GpuUnsupportedExpression("GPU join supports exactly one equality condition");
		}
		for (auto &condition : join.conditions) {
			if (!condition.IsComparison() || condition.GetComparisonType() != ExpressionType::COMPARE_EQUAL) {
				// A real hash join only implements equi-join keys -- anything else (range joins, etc.)
				// needs a different join strategy this pass doesn't attempt to lower.
				throw GpuUnsupportedExpression("hash join requires an equality condition");
			}
			auto &left_type = condition.GetLHS().GetReturnType();
			auto &right_type = condition.GetRHS().GetReturnType();
			// Keys are matched by raw bit pattern (see gpu_hash_join.cu): floats would make -0.0 != +0.0
			// and NaN != NaN, and a mixed width would widen INT32 -1 and INT64 -1 to different values,
			// silently dropping negative matches.
			switch (left_type.id()) {
			case LogicalTypeId::INTEGER:
			case LogicalTypeId::BIGINT:
			case LogicalTypeId::BOOLEAN:
				break;
			default:
				throw GpuUnsupportedExpression("GPU join supports only integer/boolean keys, not " +
				                               left_type.ToString());
			}
			if (left_type.id() != right_type.id()) {
				throw GpuUnsupportedExpression("GPU join requires both key columns to have the same type");
			}
			node->build_keys.push_back(ResolveKeyColumnName(condition.GetLHS(), bindings));
			node->probe_keys.push_back(ResolveKeyColumnName(condition.GetRHS(), bindings));
		}

		// CORRECTNESS GUARD, same class of bug as LogicalGet's projection_ids above. A join's output is
		// NOT simply every left column followed by every right column: LogicalJoin::ResolveTypes maps
		// both children's types through left_projection_map/right_projection_map, and our optimizer
		// extension runs AFTER the built-in optimizers (column_lifetime_analyzer populates those maps),
		// so they are routinely non-empty by the time we see the plan. Deriving the output list from
		// GetColumnBindings() -- which applies exactly the same maps -- is what keeps the columns the
		// engine emits lined up with the `types` the result is packed against.
		for (auto &binding : op.GetColumnBindings()) {
			std::pair<idx_t, idx_t> key {binding.table_index.index, binding.column_index.GetIndex()};
			auto it = bindings.find(key);
			if (it == bindings.end()) {
				throw GpuUnsupportedExpression("join output column not resolvable to a scanned column");
			}
			node->output_columns.push_back(it->second);
		}
		// Columns are selected by name on the engine side, so a name produced by both inputs (a self-join,
		// or two tables that both have an `id`) could silently resolve to the wrong side's data.
		for (size_t i = 0; i < node->output_columns.size(); i++) {
			for (size_t j = i + 1; j < node->output_columns.size(); j++) {
				if (node->output_columns[i] == node->output_columns[j]) {
					throw GpuUnsupportedExpression("GPU join output has duplicate column name '" +
					                               node->output_columns[i] +
					                               "', which cannot be resolved unambiguously");
				}
			}
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT: {
		node->op_type = GpuOpType::CROSS_PRODUCT;
		if (op.children.size() != 2) {
			throw GpuUnsupportedExpression("cross product does not have exactly two inputs");
		}
		// Same derivation as the join above, and for the same reason: the engine's output columns are
		// packed straight into a chunk of this operator's `types`, so the list must come from whatever
		// GetColumnBindings() reports. LogicalCrossProduct (a LogicalUnconditionalJoin) has no projection
		// maps, so this is both children's bindings concatenated -- but deriving it rather than assuming it
		// keeps the two two-input operators on one rule.
		for (auto &binding : op.GetColumnBindings()) {
			std::pair<idx_t, idx_t> key {binding.table_index.index, binding.column_index.GetIndex()};
			auto it = bindings.find(key);
			if (it == bindings.end()) {
				throw GpuUnsupportedExpression("cross product output column not resolvable to a scanned column");
			}
			node->output_columns.push_back(it->second);
		}
		// Columns are selected by name on the engine side, so a name produced by both inputs (a self cross
		// join, or two tables that both have an `id`) could silently resolve to the wrong side's data.
		for (size_t i = 0; i < node->output_columns.size(); i++) {
			for (size_t j = i + 1; j < node->output_columns.size(); j++) {
				if (node->output_columns[i] == node->output_columns[j]) {
					throw GpuUnsupportedExpression("GPU cross product output has duplicate column name '" +
					                               node->output_columns[i] +
					                               "', which cannot be resolved unambiguously");
				}
			}
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
		auto &aggregate = op.Cast<LogicalAggregate>();
		node->op_type = GpuOpType::GROUP_BY_AGGREGATE;
		if (aggregate.grouping_sets.size() > 1 || !aggregate.grouping_functions.empty()) {
			// ROLLUP/CUBE/GROUPING SETS compute several grouping levels in one operator and append extra
			// GROUPING() output columns; the GPU node models exactly one grouping and no such columns,
			// so the output would not line up with the operator's own `types`.
			throw GpuUnsupportedExpression("grouping sets / ROLLUP / CUBE have no GPU implementation");
		}

		// --- Phase 1: classify each group key as bare (column reference only) or computed. ----------
		//
		// Integer/boolean keys only: gpu_groupby.cu groups by raw bit pattern, under which -0.0 and
		// +0.0 fall in different groups and NaN never matches itself -- both differ from DuckDB.
		// HUGEINT is likewise excluded: WidenKeyKernel casts to uint64, which is too narrow for 128 bits.
		// SMALLINT/INT16 is now accepted alongside INTEGER/BIGINT/BOOLEAN.
		struct KeyInfo {
			bool is_bare;
			std::string resolved_name; // populated when is_bare == true
			GpuExpr gpu_expr;          // populated when is_bare == false
		};
		std::vector<KeyInfo> key_infos;
		bool has_computed_keys = false;

		for (idx_t i = 0; i < aggregate.groups.size(); i++) {
			auto &group_expr = *aggregate.groups[i];
			switch (group_expr.GetReturnType().id()) {
			case LogicalTypeId::SMALLINT:
			case LogicalTypeId::INTEGER:
			case LogicalTypeId::BIGINT:
			case LogicalTypeId::BOOLEAN:
				break;
			default:
				throw GpuUnsupportedExpression("GPU group-by supports only integer/boolean group keys, not " +
				                               group_expr.GetReturnType().ToString());
			}
			KeyInfo info;
			try {
				info.resolved_name = ResolveKeyColumnName(group_expr, bindings);
				info.is_bare = true;
			} catch (const GpuUnsupportedExpression &) {
				// Not a bare column reference -- attempt to translate it as a scalar expression. Lossless
				// casts and integer arithmetic are handled by TranslateToGpuExpr; non-lossless casts
				// (e.g. BIGINT->INTEGER) and unsupported functions throw GpuUnsupportedExpression and
				// propagate up, causing a CPU fallback for the whole plan.
				info.is_bare = false;
				info.gpu_expr = TranslateToGpuExpr(group_expr, bindings);
				has_computed_keys = true;
			}
			key_infos.push_back(std::move(info));
		}

		if (!has_computed_keys) {
			// --- Fast path: all bare column references, no projection needed. --------------------
			for (auto &info : key_infos) {
				node->group_keys.push_back(info.resolved_name);
			}
			for (auto &agg_expr : op.expressions) {
				// aggregate_exprs stays as the human-readable descriptor; `aggregates` is the executable
				// form the engine dispatches on. The two are parallel by construction.
				node->aggregate_exprs.push_back(agg_expr->ToString());
				node->aggregates.push_back(TranslateAggregate(*agg_expr, bindings));
			}
		} else {
			// --- Slow path: at least one computed key. Inject a synthetic PROJECTION child. -------
			//
			// The engine's GROUP_BY_AGGREGATE kernel resolves group keys by NAME from its child's
			// output columns. It has no expression evaluator of its own. To allow a computed key
			// (e.g. CAST(k AS SMALLINT) inserted by the Perfect Hash Group By optimizer), we materialise
			// every key into a named column via a PROJECTION node inserted between the scan child and
			// the GROUP_BY node, then reference it by the projection's output name.
			//
			// The projection emits:
			//   col0 .. col<K-1>  : the K group key expressions (bare or computed)
			//   col<K> .. col<N>  : pass-throughs for each distinct aggregate input column
			// All output names are the standard PROJECTION convention used by GpuNodeOutputNames.

			auto proj_node = std::make_shared<GpuPlanNode>();
			proj_node->op_type = GpuOpType::PROJECTION;

			// Step A: emit one projection expression per group key.
			for (idx_t i = 0; i < key_infos.size(); i++) {
				auto &info = key_infos[i];
				if (info.is_bare) {
					// Bare ref: translate the expression (produces inputs[j][row]) so it passes
					// through the column into a stable col<i> slot in the projection output.
					proj_node->expressions.push_back(
					    TranslateToGpuExpr(*aggregate.groups[i], bindings));
				} else {
					proj_node->expressions.push_back(info.gpu_expr);
				}
				node->group_keys.push_back("col" + std::to_string(i));
			}

			// Step B: collect the distinct aggregate input columns and add pass-through expressions.
			// Maps original column name -> projection slot index, so we don't emit duplicates.
			std::map<std::string, size_t> agg_col_to_slot;
			size_t next_slot = key_infos.size();
			for (auto &agg_expr_ptr : op.expressions) {
				if (agg_expr_ptr->GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) {
					continue; // TranslateAggregate will produce a clear error below
				}
				auto &agg = agg_expr_ptr->Cast<BoundAggregateExpression>();
				auto &agg_children = agg.GetChildren();
				if (agg_children.empty()) {
					continue; // COUNT_STAR: no input column needed
				}
				// If the aggregate argument is not a bare column ref (e.g. SUM(k+1)), we skip it here
				// and let TranslateAggregate fail below -- the exception propagates as a CPU fallback.
				try {
					auto col_name = ResolveKeyColumnName(*agg_children[0], bindings);
					if (agg_col_to_slot.find(col_name) == agg_col_to_slot.end()) {
						proj_node->expressions.push_back(
						    TranslateToGpuExpr(*agg_children[0], bindings));
						agg_col_to_slot[col_name] = next_slot++;
					}
				} catch (const GpuUnsupportedExpression &) {
					// Non-bare aggregate argument: let TranslateAggregate report the error.
				}
			}

			// Step C: build updated bindings for the aggregate translation pass that reflect
			// the projection's output names (col<k>, col<k+1>, ...) instead of the original
			// scan column names. Only the bindings that appear in agg_col_to_slot are remapped;
			// everything else is unchanged (CountStar / unsupported shapes stay as-is).
			ColumnBindingNames agg_bindings = bindings;
			for (auto &kv : bindings) {
				auto it = agg_col_to_slot.find(kv.second);
				if (it != agg_col_to_slot.end()) {
					agg_bindings[kv.first] = "col" + std::to_string(it->second);
				}
			}

			// Step D: translate aggregates against the updated bindings.
			for (auto &agg_expr : op.expressions) {
				node->aggregate_exprs.push_back(agg_expr->ToString());
				node->aggregates.push_back(TranslateAggregate(*agg_expr, agg_bindings));
			}

			// Step E: translate children into the PROJECTION node (not into the GROUP_BY node), then
			// wire the projection as the group-by's single child. Set skip_children so the post-switch
			// loop does not re-process children and add them a second time to node->children.
			for (auto &child : op.children) {
				proj_node->children.push_back(TranslateToGpuPlan(*child, bindings));
			}
			if (!proj_node->children.empty()) {
				proj_node->estimated_rows = proj_node->children[0]->estimated_rows;
			}
			// Row width for the projection: sum of output column sizes (key slots + pass-through slots).
			idx_t proj_width = 0;
			for (auto &proj_expr : proj_node->expressions) {
				switch (proj_expr.output_type) {
				case GpuValueType::BOOLEAN: proj_width += 1; break;
				case GpuValueType::INT16:   proj_width += 2; break;
				case GpuValueType::INT32: case GpuValueType::FLOAT32: proj_width += 4; break;
				default:                    proj_width += 8; break;
				}
			}
			proj_node->estimated_row_width_bytes = static_cast<uint32_t>(proj_width);

			node->children.push_back(proj_node);
			skip_children = true;
		}
		break;
	}
	default:
		throw GpuUnsupportedExpression("unsupported operator in GPU plan: " + LogicalOperatorToString(op.type));
	}

	if (!skip_children) {
		for (auto &child : op.children) {
			node->children.push_back(TranslateToGpuPlan(*child, bindings));
		}
	}

	// COST MODEL INPUTS (see GpuPlanNode::estimated_rows). Filled in after the children exist so a
	// cross product can be sized on the product of its inputs rather than on DuckDB's estimate for
	// itself -- n*m is exact and cheap here, and it is the one cardinality large enough that trusting an
	// estimate would matter.
	node->estimated_rows = op.estimated_cardinality;
	idx_t row_width = 0;
	for (auto &type : op.types) {
		row_width += GetTypeIdSize(type.InternalType());
	}
	node->estimated_row_width_bytes = static_cast<uint32_t>(row_width);
	if (node->op_type == GpuOpType::CROSS_PRODUCT && node->children.size() == 2) {
		auto left = node->children[0]->estimated_rows;
		auto right = node->children[1]->estimated_rows;
		// Saturating: n*m is exactly what wraps a 64-bit count, and a wrapped value reads as "small".
		node->estimated_rows =
		    (left != 0 && right > (std::numeric_limits<uint64_t>::max)() / left)
		        ? (std::numeric_limits<uint64_t>::max)()
		        : left * right;
	}
	return node;
}

//! The output column names a translated node produces, mirroring exactly what the engine emits (see
//! cuda_engine/src/gpu_executor.cu and src/operators/). Kept deliberately in lockstep with those: if an
//! operator's output naming changes there, it must change here too.
static std::vector<std::string> GpuNodeOutputNames(const GpuPlanNode &node) {
	switch (node.op_type) {
	case GpuOpType::SCAN:
		return node.column_names;
	case GpuOpType::FILTER:
		// A filter compacts rows but carries its child's columns through unchanged.
		return node.children.empty() ? std::vector<std::string>() : GpuNodeOutputNames(*node.children[0]);
	case GpuOpType::PROJECTION: {
		std::vector<std::string> names;
		for (idx_t i = 0; i < node.expressions.size(); i++) {
			names.push_back("col" + std::to_string(i));
		}
		return names;
	}
	case GpuOpType::GROUP_BY_AGGREGATE: {
		auto names = node.group_keys;
		for (idx_t i = 0; i < node.aggregates.size(); i++) {
			names.push_back("agg_" + std::to_string(i));
		}
		return names;
	}
	case GpuOpType::HASH_JOIN:
	case GpuOpType::CROSS_PRODUCT:
		return node.output_columns;
	}
	return {};
}

//! True only for plans the GPU executor (cuda_engine/src/gpu_executor.cu) genuinely runs end-to-end.
//! Every GpuOpType now has a real implementation, so this is a blanket accept — it is kept (rather than
//! deleted) as the single, obvious choke point to narrow again if an operator is ever added ahead of its
//! kernel, which is exactly the state SCAN/FILTER/PROJECTION, GROUP_BY_AGGREGATE and HASH_JOIN each
//! passed through.
//!
//! A node whose specific shape the kernel can't run (a computed group key, AVG, a float group/join key,
//! a composite join key, DISTINCT, ...) never reaches this check: TranslateToGpuPlan throws
//! GpuUnsupportedExpression during translation, which is already a CPU fallback.
//! False for a plan that would move data to the device without computing anything on it.
//!
//! A tree whose ROOT is a SCAN uploads every column and then copies it straight back unchanged — the
//! result of a SCAN is its input. That is strictly worse than not offloading: pure H2D + D2H cost for
//! zero work. It happens more often than it looks, because TryOffload recurses: when an aggregate or
//! projection above the scan is refused, the bare scan underneath is still "offloadable" on its own.
//!
//! Found with the activity logger (session 13), which showed exactly this for
//! `SELECT k, COUNT(*) FROM t GROUP BY k` -- a __cast projection made the GROUP BY decline, leaving a
//! trace of `offload op=GET`, `h2d 8000000 bytes`, `d2h 8000000 bytes` and no kernel at all.
//!
//! A root CROSS_PRODUCT is the same trade, only worse. It computes no per-row value either -- it copies
//! each input row n or m times -- so offloading one whose result goes straight back to the host pays H2D
//! plus a D2H of the FULL n*m expansion for zero arithmetic, while DuckDB's CPU cross product streams the
//! same rows without materializing them at all. A cross product earns its place only when something
//! above it (a projection, filter or aggregate) consumes those pairs ON THE DEVICE, which is precisely
//! the case where the root is that operator and not the cross product.
//!
//! Only the root matters: SCAN and CROSS_PRODUCT nodes *below* a real operator are feeding computation,
//! which is fine.
static bool PerformsGpuComputation(const GpuPlanNode &node) {
	return node.op_type != GpuOpType::SCAN && node.op_type != GpuOpType::CROSS_PRODUCT;
}

static bool IsExecutableOnGpu(const GpuPlanNode &node) {
	switch (node.op_type) {
	case GpuOpType::SCAN:
	case GpuOpType::FILTER:
	case GpuOpType::PROJECTION:
	case GpuOpType::GROUP_BY_AGGREGATE:
	case GpuOpType::HASH_JOIN:
	case GpuOpType::CROSS_PRODUCT:
		break;
	default:
		return false;
	}
	if (node.op_type == GpuOpType::HASH_JOIN || node.op_type == GpuOpType::CROSS_PRODUCT) {
		// The engine selects a two-input operator's output columns BY NAME from its two inputs, so a name
		// produced by both sides (a self-join; two tables that both have an `id`; or two projections, which
		// both name their outputs col0, col1, ...) cannot be resolved to the right side's data.
		//
		// This has to be caught HERE rather than only in the engine. Checking `output_columns` for
		// duplicates at translation time is not enough: DuckDB's left/right projection maps routinely trim
		// the output to just one of the two same-named columns, so the output list looks unambiguous while
		// the INPUTS are still ambiguous. Found by a real self-join query, which was offloaded and then
		// failed at execution ("output column 'cid' is produced by both join inputs") instead of quietly
		// falling back to CPU — a failed query rather than a slower one.
		if (node.children.size() != 2) {
			return false;
		}
		auto left_names = GpuNodeOutputNames(*node.children[0]);
		auto right_names = GpuNodeOutputNames(*node.children[1]);
		for (auto &left_name : left_names) {
			for (auto &right_name : right_names) {
				if (left_name == right_name) {
					return false;
				}
			}
		}
	}
	for (auto &child : node.children) {
		if (!IsExecutableOnGpu(*child)) {
			return false;
		}
	}
	return true;
}

//! Returns true when the split-plan path (PhysicalGpuStreamingProjection) is active.
//! On by default since session 49: Pipeline Stream (Cached) with FP64 Strict Math is the
//! canonical execution mode. Overridable in two ways without recompilation:
//!   - env var   : VECTOR_GPU_STREAM_PIPELINE=0  (applied once at process start)
//!   - DuckDB SET: SET gpu_stream_pipeline = false (takes effect on the next query)
//! The CPU fallback inside PhysicalGpuStreamingProjection is always present.
static bool StreamSplitEnabled() {
	// Apply the VECTOR_GPU_STREAM_PIPELINE environment variable exactly once. After that
	// g_stream_split_enabled is the live switch; SET gpu_stream_pipeline writes to it and
	// takes effect immediately on the next query, without requiring a restart.
	static const bool env_applied = [] {
		const char *value = std::getenv("VECTOR_GPU_STREAM_PIPELINE");
		if (value != nullptr) {
			const bool from_env = std::strcmp(value, "0") != 0 &&
			                      std::strcmp(value, "false") != 0 &&
			                      std::strcmp(value, "FALSE") != 0;
			g_stream_split_enabled.store(from_env, std::memory_order_relaxed);
		}
		return true; // dummy; only the side-effect matters
	}();
	(void)env_applied;
	return g_stream_split_enabled.load(std::memory_order_relaxed);
}

//! Resolve a lifted projection's BoundColumnRefExpressions to positional BoundReferenceExpressions into
//! that level's input schema. ColumnBindingResolver normally does this for every projection, but it only
//! visits an operator's `expressions`/children -- the projection chain TrySplitOffload lifts out lives in
//! GpuStreamProjectionOperator::levels, which it never sees. PhysicalGpuStreamingProjection then feeds
//! those expressions straight into an ExpressionExecutor for its mid-query bailout, and BOUND_COLUMN_REF
//! is not executable ("expression of unknown type"). `input_index` maps each input column binding to its
//! ordinal in the level's input chunk.
static void ResolveSplitLevelRefs(duckdb::unique_ptr<Expression> &expr,
                                  const column_binding_map_t<idx_t> &input_index) {
	if (expr->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
		auto &col = expr->Cast<BoundColumnRefExpression>();
		auto found = input_index.find(col.Binding());
		if (found == input_index.end()) {
			throw GpuUnsupportedExpression("split projection references a column that is not in its input schema");
		}
		expr = make_uniq<BoundReferenceExpression>(col.GetAlias(), col.GetReturnType(), found->second);
		return;
	}
	ExpressionIterator::EnumerateChildren(
	    *expr, [&](duckdb::unique_ptr<Expression> &child) { ResolveSplitLevelRefs(child, input_index); });
}

static void CollectReferencedColumnBindings(const Expression &expr, column_binding_set_t &referenced) {
	if (expr.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
		auto &col = expr.Cast<BoundColumnRefExpression>();
		referenced.insert(col.Binding());
		return;
	}
	ExpressionIterator::EnumerateChildren(
	    expr, [&](const Expression &child) { CollectReferencedColumnBindings(child, referenced); });
}

class GpuOffloadOptimizer : public OptimizerExtension {
public:
	GpuOffloadOptimizer() {
		optimize_function = Optimize;
	}

	//! SPLIT OFFLOAD: replace only a chain of PROJECTIONs, leaving the scan (and any filter over it) as a
	//! real child that DuckDB executes on the CPU.
	//!
	//! This is what makes a `WHERE` clause offloadable at all. The whole-subtree path below cannot take
	//! one, and the reason is a correctness guard, not a missing feature: it re-opens the table BY NAME
	//! and reads every row, while DuckDB has already pushed the predicate into the scan AND pruned the row
	//! groups it proved wholly satisfy it, folding their counts into constants elsewhere in the plan. The
	//! measured consequence was `SELECT COUNT(*) FROM t WHERE k > 100` reporting 1,385,095 against a true
	//! 999,495 (session 21, reverted; pinned by `pushdown/count-star-not-double-counted`).
	//!
	//! Splitting the plan removes the hazard rather than working around it. Nothing is re-scanned, so
	//! there is nothing to double-count: the scan, the pushed-down predicate, the row-group pruning and
	//! any standing PhysicalFilter all stay with DuckDB, and PhysicalGpuStreamingProjection consumes the
	//! chunks that come out -- filtered rows never cross PCIe because they are already gone before this
	//! operator sees them (GpuBatchAccumulator resolves each chunk's SelectionVector into pinned memory).
	//!
	//! Deliberately narrow, because the blast radius of getting routing wrong is every query:
	//!   - the offloaded part is a chain of PROJECTIONs, which is the one shape
	//!     PhysicalGpuStreamingProjection runs (see GpuStreamPipeline::Supports);
	//!   - the retained child is a GET, or a FILTER over a GET, and nothing else. Consuming any operator's
	//!     output would be equally SAFE, but the cost model reads the child's estimated cardinality as
	//!     "rows scanned", which only means what it says for a scan.
	//! Returns true when `node` was replaced.
	static bool TrySplitOffload(ClientContext &context, unique_ptr<LogicalOperator> &node) {
		if (!StreamSplitEnabled() || node->type != LogicalOperatorType::LOGICAL_PROJECTION) {
			return false;
		}
		// Walk the projection chain down to the operator that will stay on the CPU.
		std::vector<LogicalOperator *> chain; // top-down
		LogicalOperator *cursor = node.get();
		while (cursor->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			if (cursor->children.size() != 1 || cursor->expressions.empty()) {
				return false;
			}
			chain.push_back(cursor);
			cursor = cursor->children[0].get();
		}
		auto &boundary = *cursor;
		// Partial-plan offloading: accept ANY non-PROJECTION operator as the CPU boundary. The chain
		// walk above already consumed all contiguous LOGICAL_PROJECTIONs, so encountering one here
		// should not be reachable -- decline it defensively. Every other operator (LOGICAL_GET,
		// LOGICAL_FILTER, LOGICAL_AGGREGATE_AND_GROUP_BY, LOGICAL_LIMIT, LOGICAL_ORDER, LOGICAL_SORT,
		// LOGICAL_WINDOW, etc.) is a valid handoff point: it runs entirely on DuckDB's own CPU
		// operators and this streaming projection sees only the rows it produces.
		//
		// Correctness: the original GET/FILTER(GET)-only restriction existed because the whole-subtree
		// re-scan path re-opens the table by name and is sensitive to pushed-down predicates and
		// row-group pruning. The SPLIT path CONSUMES DuckDB's output; it never re-opens anything, so
		// that hazard does not apply.
		if (boundary.type == LogicalOperatorType::LOGICAL_PROJECTION) {
			return false; // unreachable: the chain walk above consumed all projections
		}
		// GpuBatchAccumulator can only hold GPU-supported numeric/boolean types. If the boundary
		// emits a column the accumulator cannot pack (VARCHAR, DATE, TIMESTAMP, BLOB, etc.) the
		// split cannot proceed for this boundary.
		for (auto &btype : boundary.types) {
			try {
				MapAggregateType(btype);
			} catch (const GpuUnsupportedExpression &) {
				return false;
			}
		}

		// Same absolute floor the whole-subtree path applies, asked of the rows the CHILD will produce:
		// below it there is not enough work to repay a transfer however dense the arithmetic is.
		if (boundary.estimated_cardinality < TableChecker::MIN_ROW_COUNT_THRESHOLD) {
			return false;
		}

		// Name the child's output columns. The lowered expressions reference them by these names, and the
		// physical operator declares the same list as its input schema, so the two cannot disagree about
		// which column is which.
		ColumnBindingNames bindings;
		CollectBindingNames(*node, bindings);
		auto boundary_bindings = boundary.GetColumnBindings();
		if (boundary_bindings.size() != boundary.types.size()) {
			return false;
		}

		// Identify active columns participating in the projection expressions to enable column-selective
		// offload. Only active columns are streamed, uploaded across PCIe, and processed on GPU.
		column_binding_set_t referenced_bindings;
		for (auto *proj : chain) {
			for (auto &expr : proj->expressions) {
				CollectReferencedColumnBindings(*expr, referenced_bindings);
			}
		}

		std::vector<idx_t> active_indices;
		for (idx_t i = 0; i < boundary_bindings.size(); i++) {
			if (referenced_bindings.find(boundary_bindings[i]) != referenced_bindings.end()) {
				active_indices.push_back(i);
			}
		}

		const bool selective = !active_indices.empty() && active_indices.size() < boundary_bindings.size();
		if (!selective) {
			active_indices.clear();
			for (idx_t i = 0; i < boundary_bindings.size(); i++) {
				active_indices.push_back(i);
			}
		}

		std::vector<std::string> input_names;
		for (auto idx : active_indices) {
			auto &binding = boundary_bindings[idx];
			auto found = bindings.find({binding.table_index.index, binding.column_index.GetIndex()});
			if (found == bindings.end()) {
				return false;
			}
			input_names.push_back(found->second);
		}
		// Unique, because everything downstream resolves columns BY NAME. Two child columns sharing a name
		// would silently feed one column's bytes to an expression that asked for the other.
		for (size_t i = 0; i < input_names.size(); i++) {
			for (size_t j = i + 1; j < input_names.size(); j++) {
				if (input_names[i] == input_names[j]) {
					return false;
				}
			}
		}

		// Lower the projections, bottom-up, and describe the child's output as the plan's SCAN leaf.
		auto scan = std::make_shared<GpuPlanNode>();
		scan->op_type = GpuOpType::SCAN;
		scan->column_names = input_names;
		scan->estimated_rows = boundary.estimated_cardinality;
		// NOTE the empty GpuTableRef, and leave it empty. On this path the SCAN node DESCRIBES an input
		// schema; it is not a table anything may re-open or cache. The rows arriving here are a FILTERED
		// subset, so publishing them under a table identity -- which is what a populated ref would invite
		// the GPU-resident column cache to do -- would make a later query read a fraction of its table and
		// call it complete.
		std::shared_ptr<GpuPlanNode> plan = scan;
		std::string decline_reason;
		try {
			uint32_t row_width = 0;
			for (auto idx : active_indices) {
				auto &type = boundary.types[idx];
				MapAggregateType(type); // whitelist check; throws for anything the accumulator cannot hold
				row_width += static_cast<uint32_t>(GetTypeIdSize(type.InternalType()));
			}
			scan->estimated_row_width_bytes = row_width;

			for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
				auto projection = std::make_shared<GpuPlanNode>();
				projection->op_type = GpuOpType::PROJECTION;
				projection->children.push_back(plan);
				projection->estimated_rows = (*it)->estimated_cardinality;
				for (auto &expr : (*it)->expressions) {
					projection->expressions.push_back(TranslateToGpuExpr(*expr, bindings));
				}
				plan = projection;
			}
		} catch (const GpuUnsupportedExpression &e) {
			VGPU_LOG(vector_gpu::LogLevel::INFO, "decline",
			         "\"op\":\"SPLIT_PROJECTION\",\"reason\":" + vector_gpu::GpuLogger::Quote(e.what()));
			return false;
		}

		if (!PhysicalGpuStreamingProjection::Supports(*plan) || !PerformsGpuComputation(*plan) ||
		    GpuNodeOutputNames(*plan).size() != node->types.size()) {
			return false;
		}
		// The same guards the whole-subtree path applies to a lowered plan, in the same order and for the
		// same reasons (cheapest first; the VRAM one creates a CUDA context, so it goes last).
		if (!TableChecker::HasSufficientAmplification(*plan, &decline_reason) ||
		    !TableChecker::HasSafeNullHandling(context, *plan, &decline_reason) ||
		    !TableChecker::HasVramHeadroomForPlan(*plan, &decline_reason)) {
			VGPU_LOG(vector_gpu::LogLevel::INFO, "decline",
			         "\"op\":\"SPLIT_PROJECTION\",\"reason\":" + vector_gpu::GpuLogger::Quote(decline_reason));
			return false;
		}

		// The CPU-bailout expressions for each lifted projection, bottom-up. Built from COPIES so a failure
		// to resolve leaves the chain (and therefore the CPU plan we fall back to) untouched.
		//
		// levels[0] is the bottom projection: its input schema is `boundary`'s output. levels[i>0] reads
		// the level below it -- chain[chain.size() - i] (chain is top-down). GetColumnBindings must be read
		// before any expressions move; LogicalProjection derives its binding count from expressions.size().
		std::vector<duckdb::vector<ColumnBinding>> chain_bindings;
		for (auto *proj : chain) {
			chain_bindings.push_back(proj->GetColumnBindings());
		}
		std::vector<GpuProjectionLevel> levels;
		try {
			for (idx_t li = 0; li < chain.size(); li++) {
				auto *proj = chain[chain.size() - 1 - li]; // bottom-up
				const duckdb::vector<ColumnBinding> &input_bindings =
				    (li == 0) ? boundary_bindings : chain_bindings[chain.size() - li];
				column_binding_map_t<idx_t> input_index;
				for (idx_t i = 0; i < input_bindings.size(); i++) {
					input_index[input_bindings[i]] = i;
				}
				GpuProjectionLevel level;
				level.types = proj->types;
				for (auto &proj_expr : proj->expressions) {
					auto resolved = proj_expr->Copy();
					ResolveSplitLevelRefs(resolved, input_index);
					level.expressions.push_back(std::move(resolved));
				}
				levels.push_back(std::move(level));
			}
		} catch (const GpuUnsupportedExpression &e) {
			VGPU_LOG(vector_gpu::LogLevel::INFO, "decline",
			         "\"op\":\"SPLIT_PROJECTION\",\"reason\":" + vector_gpu::GpuLogger::Quote(e.what()));
			return false;
		}

		uint64_t split_columns_size = 0;
		uint64_t split_vram_budget = 0;
		ExecutionMode split_execution_mode =
		    TableChecker::DetermineExecutionMode(*plan, &split_columns_size, &split_vram_budget);
		plan->execution_mode = split_execution_mode;

		auto result_types = node->types;
		auto estimated_cardinality = node->estimated_cardinality;
		auto original_bindings = node->GetColumnBindings();
		auto levels_count = levels.size();
		// Detach the retained child from the bottom projection before the chain is destroyed with `node`.
		auto retained = std::move(chain.back()->children[0]);
		auto retained_type = LogicalOperatorToString(retained->type);
		auto split_op = make_uniq<GpuStreamProjectionOperator>(
		    plan, result_types, input_names, std::move(levels), original_bindings,
		    selective ? active_indices : std::vector<idx_t>{}, split_execution_mode);
		split_op->children.push_back(std::move(retained));
		split_op->SetEstimatedCardinality(estimated_cardinality);
		node = std::move(split_op);
		if (g_gpu_offload_trace) {
			g_gpu_offload_count++;
		}
		if (split_execution_mode == ExecutionMode::PIPELINE_STREAM_UNCACHED) {
			VGPU_LOG(vector_gpu::LogLevel::INFO, "planner",
			         "\"heuristic\":\"auto_bypass_cache\",\"mode\":\"PIPELINE_STREAM_UNCACHED\","
			         "\"columns_size\":" +
			             std::to_string(split_columns_size) +
			             ",\"budget\":" + std::to_string(split_vram_budget));
		}
		VGPU_LOG(vector_gpu::LogLevel::INFO, "offload",
		         "\"op\":\"SPLIT_PROJECTION\",\"rows\":" + std::to_string(estimated_cardinality) +
		             ",\"cpu_child\":" + vector_gpu::GpuLogger::Quote(retained_type) +
		             ",\"levels\":" + std::to_string(levels_count) +
		             ",\"mode\":" +
		             (split_execution_mode == ExecutionMode::PIPELINE_STREAM_UNCACHED
		                  ? "\"PIPELINE_STREAM_UNCACHED\""
		                  : "\"PIPELINE_STREAM_CACHED\""));
		return true;
	}

	//! Walks the plan tree and calls TrySplitOffload at every node to find PROJECTION chains buried
	//! under operators the whole-subtree path cannot take (LIMIT, ORDER_BY, WINDOW, AGGREGATE used
	//! as a CPU boundary, subquery boundaries, etc.). This is the "partial-plan" pass that enables
	//! offloading through any CPU-opaque boundary, not just GET/FILTER(GET).
	//!
	//! Visits the current node BEFORE its children so an outer projection chain is split before an
	//! inner one. After a successful split `node` is replaced with a GpuStreamProjectionOperator;
	//! we still recurse into its retained CPU boundary child (now at node->children[0]) to find
	//! further eligible inner chains (e.g. a heavy PROJECTION feeding the AGGREGATE boundary).
	static bool TryPartialOffload(ClientContext &context, unique_ptr<LogicalOperator> &node) {
		if (!StreamSplitEnabled()) {
			return false;
		}
		// Try to split at this node first. If it succeeds node is replaced in-place.
		const bool split_here = TrySplitOffload(context, node);
		// Always recurse into children even after a successful split: after a split,
		// node->children[0] is the retained CPU subtree and may contain further eligible chains.
		bool any = split_here;
		for (auto &child : node->children) {
			if (TryPartialOffload(context, child)) {
				any = true;
			}
		}
		return any;
	}

	//! Attempts to replace `node` (in place) with a GpuExecuteOperator. Returns true on success.
	//! On failure, recurses into the children so a GPU-executable SUBTREE still gets offloaded even
	//! when an ancestor isn't offloadable. This matters constantly in practice: almost every real
	//! query has a LIMIT / ORDER BY / aggregate on top of the scan+projection work, and checking only
	//! the root would leave all of it on the CPU. With subtree offload, the heavy per-row work runs on
	//! the GPU and the small operator on top finishes on the CPU.
	static bool TryOffload(ClientContext &context, unique_ptr<LogicalOperator> &node) {
		if (StreamSplitEnabled() && TrySplitOffload(context, node)) {
			return true;
		}
		// The `decline` records here are the whole reason this logger exists: "why did my query not run on
		// the GPU?" is otherwise only answerable by bisecting the routing rules by hand.
		std::string decline_reason;
		// check_vram=false: the VRAM query creates the CUDA context (~93 ms, measured), and most plans die
		// at translation below — paying for a context that is then discarded. Asked again after translation.
		const bool checker_approved =
		    TableChecker::ShouldOffload(context, *node, &decline_reason, /* check_vram = */ false);
		if (!checker_approved) {
			VGPU_LOG(vector_gpu::LogLevel::INFO, "decline",
			         "\"op\":" + vector_gpu::GpuLogger::Quote(LogicalOperatorToString(node->type)) +
			             ",\"rows\":" + std::to_string(node->estimated_cardinality) +
			             ",\"reason\":" + vector_gpu::GpuLogger::Quote(decline_reason));
		}
		if (checker_approved) {
			ColumnBindingNames bindings;
			CollectBindingNames(*node, bindings);
			std::shared_ptr<GpuPlanNode> gpu_plan;
			bool translated = true;
			try {
				gpu_plan = TranslateToGpuPlan(*node, bindings);
			} catch (const GpuUnsupportedExpression &e) {
				VGPU_LOG(vector_gpu::LogLevel::INFO, "decline",
				         "\"op\":" + vector_gpu::GpuLogger::Quote(LogicalOperatorToString(node->type)) +
				             ",\"reason\":" + vector_gpu::GpuLogger::Quote(e.what()));
				// Approved on operator/type shape, but an actual expression can't be lowered (a UDF, a
				// non-equi join, a computed group key, ...). Fall through to the children.
				translated = false;
			}
			if (translated && IsExecutableOnGpu(*gpu_plan) && !PerformsGpuComputation(*gpu_plan)) {
				VGPU_LOG(vector_gpu::LogLevel::INFO, "decline",
				         "\"op\":" + vector_gpu::GpuLogger::Quote(LogicalOperatorToString(node->type)) +
				             ",\"reason\":\"plan computes nothing on GPU (bare scan): uploading and copying back "
				             "unchanged costs more than running on CPU\"");
			}
			// ARITY GUARD. The engine's result columns are packed straight into a DuckDB chunk of this
			// operator's `types`, so the two must agree exactly. If they do not, PhysicalGpuExecute throws
			// at EXECUTION time -- which fails the query instead of falling back, the one outcome this
			// design exists to avoid. Checking here turns a hard failure into an ordinary CPU fallback.
			//
			// Real case: DuckDB rewrites AVG(v) into sum(v) plus a dividing projection, so the offloaded
			// subtree's output arity stopped matching the operator's declared types and
			// `SELECT k, AVG(v), COUNT(*) ... GROUP BY k` died with "GPU returned 3 columns but the
			// operator has 4 output types".
			if (translated && GpuNodeOutputNames(*gpu_plan).size() != node->types.size()) {
				VGPU_LOG(vector_gpu::LogLevel::INFO, "decline",
				         "\"op\":" + vector_gpu::GpuLogger::Quote(LogicalOperatorToString(node->type)) +
				             ",\"reason\":\"translated plan outputs " +
				             std::to_string(GpuNodeOutputNames(*gpu_plan).size()) +
				             " columns but the operator declares " + std::to_string(node->types.size()) +
				             "\"");
				translated = false;
			}

			// Deferred VRAM check: only now, with a lowered plan we know the engine can run, is it worth
			// creating a CUDA context to ask whether the data fits.
			//
			// Sized against the LOWERED PLAN, not the LogicalOperator. The GpuPlanNode tree is what
			// actually allocates -- it knows each operator's internal temporaries and whether the plan
			// will be chunked, neither of which is visible from op.types x estimated_cardinality. Sizing
			// off the logical operator is what made this check 2.35x optimistic (session 26).
			// WORK-AMPLIFICATION GUARD. Checked before the VRAM question because it needs no CUDA context
			// and rejects the common case: a plan that does one row of work per row scanned cannot win,
			// at any size. Measured session 27 -- a projection over 1M-64M rows loses 2.8-3.4x and gets
			// worse with scale, because the scan cost grows linearly while the CPU stays in cache.
			if (translated && IsExecutableOnGpu(*gpu_plan) && PerformsGpuComputation(*gpu_plan) &&
			    !TableChecker::HasSufficientAmplification(*gpu_plan, &decline_reason)) {
				VGPU_LOG(vector_gpu::LogLevel::INFO, "decline",
				         "\"op\":" + vector_gpu::GpuLogger::Quote(LogicalOperatorToString(node->type)) +
				             ",\"reason\":" + vector_gpu::GpuLogger::Quote(decline_reason));
				translated = false;
			}

			// NULL-HANDLING GUARD (session 30). Checked here, alongside the VRAM question below, for the
			// same reason: only the LOWERED plan knows whether it ended up row-independent
			// (SCAN/FILTER/PROJECTION), and that answer is exactly what decides whether a nullable column
			// is safe. Cheap (catalog/statistics lookups, no CUDA context), so it runs before the VRAM
			// check rather than after, matching the existing amplification-before-VRAM ordering above.
			if (translated && IsExecutableOnGpu(*gpu_plan) && PerformsGpuComputation(*gpu_plan) &&
			    !TableChecker::HasSafeNullHandling(context, *gpu_plan, &decline_reason)) {
				VGPU_LOG(vector_gpu::LogLevel::INFO, "decline",
				         "\"op\":" + vector_gpu::GpuLogger::Quote(LogicalOperatorToString(node->type)) +
				             ",\"reason\":" + vector_gpu::GpuLogger::Quote(decline_reason));
				translated = false;
			}

			if (translated && IsExecutableOnGpu(*gpu_plan) && PerformsGpuComputation(*gpu_plan) &&
			    !TableChecker::HasVramHeadroomForPlan(*gpu_plan, &decline_reason)) {
				VGPU_LOG(vector_gpu::LogLevel::INFO, "decline",
				         "\"op\":" + vector_gpu::GpuLogger::Quote(LogicalOperatorToString(node->type)) +
				             ",\"reason\":" + vector_gpu::GpuLogger::Quote(decline_reason));
			} else if (translated && IsExecutableOnGpu(*gpu_plan) && PerformsGpuComputation(*gpu_plan)) {
				uint64_t offload_columns_size = 0;
				uint64_t offload_vram_budget = 0;
				ExecutionMode execution_mode =
				    TableChecker::DetermineExecutionMode(*gpu_plan, &offload_columns_size, &offload_vram_budget);
				gpu_plan->execution_mode = execution_mode;

				auto result_types = node->types;
				auto estimated_cardinality = node->estimated_cardinality;
				auto logged_op = LogicalOperatorToString(node->type); // `node` is moved below
				auto gpu_op = make_uniq<GpuExecuteOperator>(std::move(gpu_plan), result_types, std::move(node),
				                                            execution_mode);
				gpu_op->SetEstimatedCardinality(estimated_cardinality);
				node = std::move(gpu_op);
				if (g_gpu_offload_trace) {
					g_gpu_offload_count++;
				}
				if (execution_mode == ExecutionMode::PIPELINE_STREAM_UNCACHED) {
					VGPU_LOG(vector_gpu::LogLevel::INFO, "planner",
					         "\"heuristic\":\"auto_bypass_cache\",\"mode\":\"PIPELINE_STREAM_UNCACHED\","
					         "\"columns_size\":" +
					             std::to_string(offload_columns_size) +
					             ",\"budget\":" + std::to_string(offload_vram_budget));
				}
				VGPU_LOG(vector_gpu::LogLevel::INFO, "offload",
				         "\"op\":" + vector_gpu::GpuLogger::Quote(logged_op) +
				             ",\"rows\":" + std::to_string(estimated_cardinality) +
				             ",\"mode\":" +
				             (execution_mode == ExecutionMode::PIPELINE_STREAM_UNCACHED
				                  ? "\"PIPELINE_STREAM_UNCACHED\""
				                  : "\"PIPELINE_STREAM_CACHED\""));
				return true;
			}
		}
		// Nothing could take the whole subtree. Run the partial-plan pass: TryPartialOffload walks
		// every descendant looking for PROJECTION chains to split off, including chains buried under
		// LIMIT, ORDER_BY, AGGREGATE-as-boundary, or any other operator the whole-subtree path cannot
		// take. Ordered after the whole-subtree attempt so plans that ARE fully offloadable still go
		// through the existing path unchanged -- no routing regression vs. session 49 benchmarks.
		//
		// TryPartialOffload recurses into every node it visits (including retained CPU subtrees after
		// a split), so we only fall through to the explicit child loop when it found nothing, to
		// preserve whole-subtree TryOffload semantics on each child independently.
		if (TryPartialOffload(context, node)) {
			return true;
		}

		bool any = false;
		for (auto &child : node->children) {
			if (TryOffload(context, child)) {
				any = true;
			}
		}
		return any;
	}

	//! Walks the WHOLE plan (not just the nodes TryOffload recurses into -- a write-shaped plan has no
	//! LOGICAL_GET/FILTER/PROJECTION at its root for TryOffload to find, so this is its own full
	//! traversal) looking for write/DDL operators, invalidating the GPU-resident column cache
	//! (gpu_column_cache.hpp, branch gpu-resident-cache) for whatever they target BEFORE the statement
	//! executes.
	//!
	//! Verified against real DuckDB source, not assumed: Optimizer::Optimize() has no statement-type
	//! gate (RequireOptimizer() defaults true; only LogicalPrepare/Pragma/Execute/Explain/Get/
	//! CreateSecret override it, none of them writes) and always runs before PhysicalPlanGenerator::Plan()
	//! -- itself always before execution, always before commit. So this can never observe a write later
	//! than the write itself. See docs/GPU_RESIDENT_CACHE_DESIGN.md for the full verification, including
	//! the one confirmed gap (DuckDB's low-level InternalAppender path bypasses the optimizer entirely --
	//! unreachable from SQL, only used by dbgen()/dsdgen() in this tree) and a narrower, accepted one: a
	//! self-referential `INSERT INTO t SELECT ... FROM t` invalidates t's cache HERE, before executing,
	//! but if that same statement's own SELECT half re-populates the cache from t's pre-statement
	//! snapshot, the cache is stale again the instant the INSERT's rows commit -- there is no post-write
	//! hook to catch that. Narrow (needs the read half to also be independently GPU-cache-eligible) and
	//! documented rather than silently possible.
	//!
	//! Exact per-table invalidation for INSERT/UPDATE/DELETE (each confirmed to carry a
	//! `TableCatalogEntry &table` member directly on the node -- verified field-level, not inferred).
	//! Everything else that writes or restructures a table (DDL, ATTACH/DETACH/VACUUM, MERGE_INTO) clears
	//! the WHOLE cache: rare, non-hot-path operators, where a full clear is trivially correct without
	//! needing to verify several more struct shapes (AlterInfo, DropInfo, ...) for little benefit.
	//! The cache identity of a table being written to. MUST be built exactly the way the translator builds
	//! a SCAN node's identity (see TranslateToGpuPlan), or an invalidation looks for a key nothing was ever
	//! stored under and silently fails to drop the entry it exists to drop.
	//!
	//! .GetIdentifierName() on every component, not an implicit/other string conversion: Identifier compares
	//! and is looked up case-INsensitively inside DuckDB's own catalog, but the cache's map keys are plain
	//! case-sensitive std::strings. Taking the raw catalog-stored name on BOTH sides is what keeps "Bench"
	//! in a query and "bench" in the catalog resolving to the same cache entry.
	static GpuTableRef CacheKeyFor(const TableCatalogEntry &table) {
		GpuTableRef ref;
		ref.catalog = table.ParentCatalog().GetName().GetIdentifierName();
		ref.schema = table.ParentSchema().name.GetIdentifierName();
		ref.table = table.name.GetIdentifierName();
		return ref;
	}

	static void InvalidateCacheForWritePlan(const LogicalOperator &node) {
		switch (node.type) {
		case LogicalOperatorType::LOGICAL_INSERT:
			GpuColumnCache::Instance().InvalidateTable(CacheKeyFor(node.Cast<LogicalInsert>().table));
			break;
		case LogicalOperatorType::LOGICAL_UPDATE:
			GpuColumnCache::Instance().InvalidateTable(CacheKeyFor(node.Cast<LogicalUpdate>().table));
			break;
		case LogicalOperatorType::LOGICAL_DELETE:
			GpuColumnCache::Instance().InvalidateTable(CacheKeyFor(node.Cast<LogicalDelete>().table));
			break;
		case LogicalOperatorType::LOGICAL_MERGE_INTO:
		case LogicalOperatorType::LOGICAL_ALTER:
		case LogicalOperatorType::LOGICAL_CREATE_TABLE:
		case LogicalOperatorType::LOGICAL_CREATE_INDEX:
		case LogicalOperatorType::LOGICAL_DROP:
		case LogicalOperatorType::LOGICAL_ATTACH:
		case LogicalOperatorType::LOGICAL_DETACH:
		case LogicalOperatorType::LOGICAL_VACUUM:
		case LogicalOperatorType::LOGICAL_COPY_DATABASE:
			GpuColumnCache::Instance().InvalidateAll();
			break;
		default:
			break;
		}
		for (auto &child : node.children) {
			InvalidateCacheForWritePlan(*child);
		}
	}

	static void Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
		// No match anywhere simply leaves the plan untouched — DuckDB's native CPU execution runs
		// unmodified. That abstention is the entire fallback mechanism.
		//
		// Timed (only while tracing is on) because this cost is paid on EVERY query the extension sees,
		// offloaded or not, and it was unaccounted for in the phase breakdown — see docs/TODO.md.
		auto start = std::chrono::steady_clock::now();
		InvalidateCacheForWritePlan(*plan);
		TryOffload(input.context, plan);
		if (g_gpu_offload_trace) {
			g_gpu_optimize_us += static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start)
			        .count());
		}
	}
};

void RegisterGpuOffloadOptimizer(DBConfig &config) {
	OptimizerExtension::Register(config, GpuOffloadOptimizer());

	// Tuning knobs, mirroring the routing criteria in docs/ARCHITECTURE.md.
	config.AddExtensionOption("gpu_min_row_threshold", "minimum estimated row count to consider GPU offload",
	                          LogicalType::UBIGINT, Value::UBIGINT(TableChecker::MIN_ROW_COUNT_THRESHOLD));
	config.AddExtensionOption("gpu_vram_pool_fraction", "fraction of device VRAM reserved by the RMM pool",
	                          LogicalType::DOUBLE, Value::DOUBLE(0.85));
	// --- Default execution mode: Pipeline Stream (Cached), FP64 Strict Math -------------------
	// gpu_stream_pipeline is ON by default (session 49). The env var VECTOR_GPU_STREAM_PIPELINE
	// is applied once at first use; this option allows runtime changes via SET without a restart.
	config.AddExtensionOption("gpu_stream_pipeline",
	                          "enable Pipeline Stream (Cached) execution mode (default: on since session 49)",
	                          LogicalType::BOOLEAN, Value::BOOLEAN(true),
	                          [](ClientContext &context, SetScope scope, Value &parameter) {
		                          // Write directly to the mutable atomic; takes effect on the next query.
		                          g_stream_split_enabled.store(BooleanValue::Get(parameter),
		                                                       std::memory_order_relaxed);
	                          });
	// FP64 Strict Math is the default (env vars unset = strict FP64). These options make the
	// defaults explicit and queryable via SHOW / RESET, matching the readable-config principle.
	config.AddExtensionOption("gpu_fast_math",
	                          "enable NVRTC --use_fast_math (default: off = FP64 strict math)",
	                          LogicalType::BOOLEAN, Value::BOOLEAN(false),
	                          [](ClientContext &context, SetScope scope, Value &parameter) {
		                          SetGpuFastMathEnabled(BooleanValue::Get(parameter));
	                          });
	config.AddExtensionOption("gpu_precision_relaxation",
	                          "enable FP32 relaxation for analytical transcendentals (default: off = FP64 strict)",
	                          LogicalType::BOOLEAN, Value::BOOLEAN(false),
	                          [](ClientContext &context, SetScope scope, Value &parameter) {
		                          SetGpuFp32RelaxationEnabled(BooleanValue::Get(parameter));
	                          });
}

} // namespace vector_gpu

// The loadable-extension entry point. Compiled only for a real .duckdb_extension build; the
// statically-linked duckdb_gpu binary (VectisDB/gpu_shell) instead calls
// vector_gpu::RegisterGpuOffloadOptimizer(config) directly before opening the database, so it must not
// pull in the extension-loader macro machinery.
#ifdef VECTOR_GPU_LOADABLE_EXTENSION
extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(vector_gpu_offload, loader) {
	auto &db = loader.GetDatabaseInstance();
	auto &config = DBConfig::GetConfig(db);
	vector_gpu::RegisterGpuOffloadOptimizer(config);
}
}
#endif
