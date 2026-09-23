#include "table_checker.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/parser/qualified_name.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"

#include "gpu_column_cache.hpp"
#include "physical_gpu_execute.hpp"
#include "rmm_pool.hpp"

#include <algorithm>
#include <limits>
#include <string>

namespace vector_gpu {

using namespace duckdb;

bool TableChecker::IsSupportedOperator(const LogicalOperator &op) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_GET:
	case LogicalOperatorType::LOGICAL_FILTER:
	case LogicalOperatorType::LOGICAL_PROJECTION:
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		return true;
	default:
		// Anything else (correlated subqueries, window functions, recursive CTEs, ...) is an immediate
		// CPU fallback per the routing criteria in the architecture doc.
		return false;
	}
}

namespace {
//! Mirrors expression_translator.cpp's MapLogicalType exactly -- these two lists must stay in sync, or
//! the Table Checker could approve a plan whose expressions then fail translation (which is handled
//! safely via the GpuUnsupportedExpression fallback in gpu_offload_extension.cpp, but is wasted work: the
//! whole point of this check is to reject that case earlier, before an optimizer pass is attempted).
bool IsSupportedType(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::DECIMAL:
		return type.InternalType() == PhysicalType::INT16 || type.InternalType() == PhysicalType::INT32 ||
		       type.InternalType() == PhysicalType::INT64 || type.InternalType() == PhysicalType::INT128;
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::BOOLEAN:
		return true;
	default:
		// Notably excludes VARCHAR/dictionary-encoded strings for now: the architecture doc's dictionary
		// encoding path (docs/ARCHITECTURE.md) isn't implemented yet, only numeric/boolean expressions
		// are (see expression_translator.cpp) -- so a query touching string columns should fall back to
		// CPU rather than be approved and then fail translation.
		return false;
	}
}
} // namespace

bool TableChecker::HasSupportedTypes(const LogicalOperator &op, std::string *bad_type) {
	for (auto &type : op.types) {
		if (!IsSupportedType(type)) {
			if (bad_type) {
				*bad_type = type.ToString();
			}
			return false;
		}
	}
	return true;
}

namespace {
//! Resolves one column, by NAME, to its statistics -- the same resolution table_scanner.cpp already
//! performs for the same table_name/column_names pair (LogicalIndex -> StorageIndex), extended one step
//! further to ask GetStatistics the question this function exists for. True if NULLs cannot be ruled out.
bool ColumnMightBeNull(ClientContext &context, TableCatalogEntry &table, const std::string &column_name) {
	// LogicalIndex has no default constructor, so it's declared (via auto) only inside the try -- same
	// shape table_scanner.cpp's own by-name resolution already uses for the identical reason.
	try {
		// GetColumnIndex takes Identifier by non-const reference, so it needs a named lvalue -- MSVC
		// accepts binding the temporary directly as a non-conforming extension, but GCC/Clang reject it.
		Identifier column_identifier(column_name);
		auto logical_idx = table.GetColumnIndex(column_identifier);
		auto storage_idx = table.GetStorageIndex(ColumnIndex(logical_idx.index));
		if (!storage_idx.HasPrimaryIndex()) {
			return true; // nested/struct or otherwise unresolvable column -- can't prove anything, decline
		}
		auto stats = table.GetStatistics(context, storage_idx.GetPrimaryIndex());
		if (!stats || stats->CanHaveNull()) {
			// No statistics available, or statistics say NULLs are possible (also
			// BaseStatistics::CreateUnknown's answer, i.e. "unknown" already reads as "might be null" --
			// the fail-safe direction is free, not something this function has to special-case).
			return true;
		}
		return false;
	} catch (const std::exception &) {
		return true; // column no longer exists (schema changed since translation) -- decline
	}
}

//! Walks `node`'s SCAN leaves, resolving each by table name (re-opening the catalog entry per leaf --
//! cheap catalog lookups, not a real scan) and checking every one of its columns via ColumnMightBeNull.
//! True (decline) the moment any leaf or any column can't be proven null-free.
bool PlanMightProduceNulls(ClientContext &context, const GpuPlanNode &node) {
	if (node.op_type == GpuOpType::SCAN) {
		optional_ptr<TableCatalogEntry> table;
		try {
			// Fully qualified, same as the scanner: an unqualified lookup here would ask about whichever
			// table the CURRENT search path resolves to, not the one the plan is actually going to read.
			table = &Catalog::GetEntry<TableCatalogEntry>(
			    context, QualifiedName(Identifier(node.table.catalog), Identifier(node.table.schema),
			                           Identifier(node.table.table)));
		} catch (const std::exception &) {
			return true; // table no longer exists (schema changed since translation) -- decline
		}
		for (auto &column_name : node.column_names) {
			if (ColumnMightBeNull(context, *table, column_name)) {
				return true;
			}
		}
		return false;
	}
	for (auto &child : node.children) {
		if (PlanMightProduceNulls(context, *child)) {
			return true;
		}
	}
	return false;
}
} // namespace

bool TableChecker::HasSafeNullHandling(ClientContext &context, const GpuPlanNode &plan, std::string *reason) {
	if (IsRowIndependent(plan)) {
		// Session 30: the fused expression/filter kernel handles nulls correctly for SCAN/FILTER/
		// PROJECTION plans (SQL NULL-propagation semantics) -- unconditionally safe regardless of what
		// the SCAN leaves' statistics say.
		return true;
	}
	if (plan.op_type == GpuOpType::GROUP_BY_AGGREGATE) {
		// GROUP_BY_AGGREGATE now has null-aware aggregation kernels.
		return true;
	}
	if (PlanMightProduceNulls(context, plan)) {
		if (reason) {
			*reason = "plan may produce NULL values, which HASH_JOIN/CROSS_PRODUCT do "
			         "not model";
		}
		return false;
	}
	return true;
}

idx_t TableChecker::EstimateCardinality(const LogicalOperator &op) {
	// TODO(phase 1): pull real cardinality estimates from catalog statistics (LogicalGet's table
	// filters / TableCatalogEntry cardinality) instead of op.estimated_cardinality, which may not be
	// populated at optimizer-extension time for every operator type.
	return op.estimated_cardinality;
}

idx_t TableChecker::MaxSubtreeCardinality(const LogicalOperator &op) {
	auto largest = EstimateCardinality(op);
	for (auto &child : op.children) {
		largest = (std::max)(largest, MaxSubtreeCardinality(*child));
	}
	return largest;
}

namespace {
// The per-operator output-width estimate that used to live here is gone. Sizing a plan by its output
// widths was 2.35x optimistic because it ignored every operator's internal temporaries; the replacement
// is EstimateWorkingSet in cuda_engine/src/gpu_cost_model.cpp, which is derived from the operators'
// actual allocation sites and lives next to them so it cannot drift. See HasVramHeadroomForPlan.

//! Saturating multiply: an astronomically large cardinality estimate (e.g. a multi-way cross join)
//! times even a small row width can wrap around 64-bit unsigned overflow, turning "clearly too large to
//! fit in VRAM" into some small wrapped-around number that would incorrectly pass the headroom check.
//! Clamping to idx_t's max instead guarantees an overflow always reads as "doesn't fit" rather than
//! silently reading as "fits great."
idx_t SaturatingMultiply(idx_t a, idx_t b) {
	if (a != 0 && b > (std::numeric_limits<idx_t>::max)() / a) {
		return (std::numeric_limits<idx_t>::max)();
	}
	return a * b;
}
} // namespace

bool TableChecker::HasVramHeadroom(idx_t estimated_bytes) {
	// RmmPool::EnsureInitialized is idempotent (no-op after the first successful call, see
	// rmm_pool.cpp) -- safe to call on every check rather than requiring some separate startup hook.
	// If CUDA itself is unavailable (no driver, no device) this throws; in that situation there is
	// clearly no GPU to offload to, so treat that as "no headroom" rather than letting the exception
	// propagate out of a routing decision and abort the query that triggered it.
	try {
		vector_gpu::RmmPool::Instance().EnsureInitialized();
	} catch (const std::exception &) {
		return false;
	}
	return vector_gpu::RmmPool::Instance().HasHeadroomFor(estimated_bytes);
}

bool TableChecker::ContainsCrossProduct(const LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT) {
		return true;
	}
	for (auto &child : op.children) {
		if (ContainsCrossProduct(*child)) {
			return true;
		}
	}
	return false;
}

bool TableChecker::ShouldOffload(ClientContext &context, LogicalOperator &op, std::string *reason,
                                 bool check_vram) {
	// Decided once for the whole subtree, then carried down: the exemption has to apply to operators
	// ABOVE the cross product (the aggregate consuming the pairs) as well as below it (the small inputs).
	return ShouldOffloadRecursive(context, op, reason, check_vram, ContainsCrossProduct(op));
}

bool TableChecker::ShouldOffloadRecursive(ClientContext &context, LogicalOperator &op, std::string *reason,
                                          bool check_vram, bool cross_product_subtree) {
	if (!IsSupportedOperator(op)) {
		if (reason) {
			*reason = "operator " + LogicalOperatorToString(op.type) + " is not in the supported set";
		}
		return false;
	}
	// A SCAN THAT READS NO REAL COLUMN cannot be re-opened by name, so it must be refused HERE.
	//
	// When nothing above a LogicalGet references any of its columns -- `SELECT 1 FROM t`,
	// `SELECT 42 AS x FROM t`, `EXISTS (SELECT * FROM t)` -- DuckDB's RemoveUnusedColumns substitutes a
	// VIRTUAL column (rowid, or the empty column) so the scan still produces one row per stored row. That
	// is a storage-level construct with no entry in the table's column list, and this engine's scanners
	// re-open a table BY COLUMN NAME through the catalog.
	//
	// Left to run, the plan was approved, translated into a SCAN naming "rowid", and only failed when
	// StreamingTableScanner tried to bind that name -- at EXECUTION time, which fails the QUERY instead of
	// falling back:
	//   Invalid Input Error: PhysicalGpuExecute: failed to open streaming scan for GPU execution:
	//   StreamingTableScanner: column 'rowid' not found on table 'db.main.t'
	// Reproduced on `SELECT 1 FROM t` and `SELECT 42 AS x FROM t`. Refusing at routing time turns it back
	// into an ordinary silent CPU fallback, which is what every other unsupported shape here does.
	if (op.type == LogicalOperatorType::LOGICAL_GET) {
		auto &get = op.Cast<LogicalGet>();
		for (auto &column_id : get.GetColumnIds()) {
			if (column_id.IsVirtualColumn()) {
				if (reason) {
					*reason = "scan reads virtual column '" +
					          get.GetColumnName(column_id).GetIdentifierName() +
					          "' (no column of the table is referenced), which cannot be re-scanned by name";
				}
				return false;
			}
		}
	}
	// A CROSS PRODUCT EXPLODES ITS INPUT, so it is guarded on the way down rather than by the ordinary
	// per-operator rules. n*m is computed from the children's own estimates instead of the operator's
	// estimated_cardinality: an oversized cross product is the one case here that can ask for more memory
	// than the device has, and a refusal must happen HERE (silent CPU fallback) rather than in the engine
	// (failed query). Saturating, because n*m is exactly what wraps a 64-bit count.
	if (op.type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT) {
		if (op.children.size() != 2) {
			if (reason) {
				*reason = "cross product does not have exactly two inputs";
			}
			return false;
		}
		auto output_rows =
		    SaturatingMultiply(EstimateCardinality(*op.children[0]), EstimateCardinality(*op.children[1]));
		if (output_rows > MAX_CROSS_PRODUCT_OUTPUT_ROWS) {
			if (reason) {
				*reason = "cross product of " + std::to_string(EstimateCardinality(*op.children[0])) + " x " +
				          std::to_string(EstimateCardinality(*op.children[1])) + " rows would produce " +
				          std::to_string(output_rows) + " rows, above the " +
				          std::to_string(MAX_CROSS_PRODUCT_OUTPUT_ROWS) + "-row GPU limit";
			}
			return false;
		}
	}
	for (auto &child : op.children) {
		// A child's reason is the real reason this operator is refused, so let it propagate rather than
		// reporting the parent generically.
		if (!ShouldOffloadRecursive(context, *child, reason, check_vram, cross_product_subtree)) {
			return false;
		}
	}
	std::string bad_type;
	if (!HasSupportedTypes(op, &bad_type)) {
		if (reason) {
			*reason = "operator " + LogicalOperatorToString(op.type) + " has column type " + bad_type +
			          ", outside INTEGER/BIGINT/FLOAT/DOUBLE/BOOLEAN";
		}
		return false;
	}
	// Nullability is checked post-translation now (HasSafeNullHandling, called from gpu_offload_extension
	// alongside HasVramHeadroomForPlan) -- only the LOWERED plan knows whether it ended up row-independent,
	// which as of session 30 is exactly what decides whether a nullable column is actually safe. See
	// table_checker.hpp's doc comment on HasSafeNullHandling for why this moved from here.
	// HIGHLY SELECTIVE SCANS STAY ON CPU. Uploading every row to discard almost all of them is a net
	// loss: the transfer is paid in full for a small fraction of the result. Judged from DuckDB's own
	// estimates, which is all that is available at planning time -- an estimate being wrong here costs
	// performance, never correctness, because the fallback produces identical results either way.
	if (op.type == LogicalOperatorType::LOGICAL_FILTER && !op.children.empty()) {
		auto input_rows = EstimateCardinality(*op.children[0]);
		auto output_rows = EstimateCardinality(op);
		if (input_rows > 0 &&
		    static_cast<double>(output_rows) < static_cast<double>(input_rows) * MIN_FILTER_SELECTIVITY) {
			if (reason) {
				*reason = "filter keeps only " + std::to_string(output_rows) + " of " +
				          std::to_string(input_rows) +
				          " rows; uploading all of them to discard most is slower than filtering on CPU";
			}
			return false;
		}
	}

	// The threshold asks "is there enough WORK here to be worth moving data to the device?", so it must
	// be judged on the largest row count FLOWING THROUGH the operator, not on its output alone.
	//
	// Using the output alone made every cardinality-reducing operator unreachable: a global
	// `SELECT SUM(x) FROM t` over 20M rows outputs ONE row, so an AGGREGATE was always compared as
	// "estimated 1 rows ... below the 100000-row offload threshold" and declined no matter how much
	// data it read. Group-by and aggregate plans could therefore never be offloaded at all — their
	// kernels were unreachable by construction rather than by policy.
	//
	// Over the WHOLE SUBTREE, not just the immediate children. Looking one level down implemented the
	// sentence above only for an aggregate sitting directly on a scan; the moment anything intervened,
	// the reducing operator's estimate was compared against the NEXT reducing operator's estimate rather
	// than against the data actually being read. `SELECT SUM(v) FROM t WHERE ...` plans as
	// AGGREGATE(1) -> FILTER(estimated small) -> GET(70M): both the aggregate and the filter were judged
	// on numbers that shrink with each level, and the 70M-row scan underneath -- the entire reason the
	// query is worth offloading -- was never the number compared.
	auto cardinality = EstimateCardinality(op);
	auto work_rows = MaxSubtreeCardinality(op);
	// The cross product node itself is always judged (on its n*m output); everything else in a
	// cross-product subtree is exempt. See ShouldOffloadRecursive's declaration for why.
	const bool apply_threshold =
	    !cross_product_subtree || op.type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT;
	if (apply_threshold && work_rows < MIN_ROW_COUNT_THRESHOLD) {
		if (reason) {
			*reason = "estimated " + std::to_string(work_rows) + " rows of work for " +
			          LogicalOperatorToString(op.type) + " (output " + std::to_string(cardinality) +
			          ") is below the " + std::to_string(MIN_ROW_COUNT_THRESHOLD) + "-row offload threshold";
		}
		return false;
	}
	// NOTE: `check_vram` is accepted and ignored. The VRAM question can no longer be answered from a
	// LogicalOperator -- it needs the lowered GpuPlanNode to know each operator's temporaries and whether
	// the plan will be chunked -- so the caller must ask HasVramHeadroomForPlan after translation. That
	// is also the cheaper order: the VRAM query creates the CUDA context (~93 ms), and most plans die in
	// translation. gpu_offload_extension.cpp already does exactly this and passes check_vram = false.
	(void)check_vram;
	return true;
}

namespace {

//! VECTOR_GPU_MIN_AMPLIFICATION, parsed once, or -1 when it is unset/malformed.
//!
//! Its presence, not just its value, is load-bearing now: setting it selects the LEGACY pure-row
//! amplification rule wholesale, so the experiment paths documented against it (bench_matrix.sh,
//! verify_gpu_vs_cpu.sh, both of which set it to 0 to force declined shapes onto the GPU) keep behaving
//! exactly as they did before the expression-aware model existed.
double AmplificationOverride() {
	static const double value = [] {
		if (const char *env = std::getenv("VECTOR_GPU_MIN_AMPLIFICATION")) {
			try {
				auto parsed = std::stod(env);
				if (parsed >= 0.0) {
					return parsed;
				}
			} catch (const std::exception &) {
				// Malformed value: fall through to the default rather than fail a routing decision.
			}
		}
		return -1.0;
	}();
	return value;
}

std::string DescribeDensity(const GpuPlanShape &shape) {
	return "plan performs " + std::to_string(shape.total_ops) + " weighted operations over " +
	       std::to_string(shape.scanned_rows) + " scanned rows (" + std::to_string(shape.scanned_bytes) +
	       " bytes) = " + std::to_string(shape.ops_per_scanned_row) + " ops/row, " +
	       std::to_string(shape.ops_per_scanned_byte) + " ops/byte";
}

} // namespace

bool TableChecker::HasSufficientAmplification(const GpuPlanNode &plan, std::string *reason) {
	auto shape = AnalyzePlanShape(plan);

	const auto override_threshold = AmplificationOverride();
	if (override_threshold >= 0.0) {
		// Explicit override: run the legacy rule and nothing else, exactly as before.
		if (shape.amplification < override_threshold) {
			if (reason) {
				*reason = "plan derives " + std::to_string(shape.max_rows) + " rows of work from " +
				          std::to_string(shape.scanned_rows) + " scanned rows (amplification " +
				          std::to_string(shape.amplification) + "), below the " +
				          std::to_string(override_threshold) +
				          " set by VECTOR_GPU_MIN_AMPLIFICATION";
			}
			return false;
		}
		return true;
	}

	switch (shape.density) {
	case GpuWorkDensity::SUFFICIENT:
		// Dense enough that the arithmetic dominates the transfer. This is the case the row-amplification
		// metric could not see at all: a heavy-math projection amplifies by 1.0 and was declined, despite
		// being the only shape measured faster than the CPU (session 32).
		return true;
	case GpuWorkDensity::INSUFFICIENT:
		if (reason) {
			*reason = DescribeDensity(shape) + ", below the " + std::to_string(kMinOpsPerScannedRow) +
			          " ops/row needed for the transfer to pay for itself";
		}
		return false;
	case GpuWorkDensity::INCONCLUSIVE:
		break;
	}

	// Between the thresholds the density model has no measured opinion, so the older metric decides.
	// It still sees work no expression carries -- a cross product's n*m expansion, a group-by's sort --
	// which is exactly what the plans in this band tend to be made of.
	if (shape.amplification < MIN_WORK_AMPLIFICATION) {
		if (reason) {
			*reason = DescribeDensity(shape) + ", which is neither dense enough to offload on its own (" +
			          std::to_string(kOffloadOpsPerRow) + " ops/row) nor amplifying enough to carry it (" +
			          std::to_string(shape.amplification) + " rows of work per scanned row, against " +
			          std::to_string(MIN_WORK_AMPLIFICATION) + ")";
		}
		return false;
	}
	return true;
}

bool TableChecker::HasVramHeadroomForPlan(const GpuPlanNode &plan, std::string *reason) {
	// One estimate for the WHOLE plan (see the header): summed across operators, including each one's
	// internal temporaries, and capped at a single chunk when the executor will chunk it.
	auto estimate = EstimateWorkingSet(plan, PhysicalGpuExecute::MAX_CHUNK_ROWS);

	// GPU-resident column cache (branch gpu-resident-cache): GpuColumnCache holds real, permanent
	// cudaMalloc'd memory now, drawn from the SAME physical VRAM this plan's transient DeviceArena
	// allocations compete for -- even though the two are tracked through separate accounting (RmmPool's
	// frozen-at-first-use budget here, the cache's own budget in gpu_column_cache.cpp). Added to the
	// plan's own demand, not subtracted from the budget: RmmPool's budget number is fixed at
	// process-start and never shrinks on its own, so the cache's footprint has to show up on the DEMAND
	// side of this comparison for growing the cache to ever reduce what a later plan is approved for.
	// Without this, routing could approve a plan whose transient working set, added to what the cache
	// already holds permanently, exceeds the card -- the same class of mistake sessions 26/27 already
	// spent real effort fixing for the non-cache case.
	auto cache_bytes = GpuColumnCache::Instance().CurrentBytes();
	auto total_demand = estimate.total_bytes + cache_bytes;
	if (!HasVramHeadroom(total_demand)) {
		if (reason) {
			// Report the BUDGET alongside the demand. Without it "exceeds available VRAM" is unactionable:
			// the budget is a fraction of memory that was free when the CUDA context came up, so it is
			// neither the card's capacity nor what nvidia-smi shows now, and the gap between those three
			// numbers is exactly what a person needs to see to decide whether to raise the fraction.
			*reason = "plan needs about " + std::to_string(estimate.total_bytes / (1024 * 1024)) +
			          " MiB of device memory" + (estimate.chunk_capped ? " per chunk" : "") +
			          " (largest single operator: " + std::string(estimate.largest_operator) + " at " +
			          std::to_string(estimate.largest_bytes / (1024 * 1024)) +
			          " MiB), plus " + std::to_string(cache_bytes / (1024 * 1024)) +
			          " MiB already held by the GPU-resident column cache, against a budget of " +
			          std::to_string(RmmPool::Instance().TotalPoolBytes() / (1024 * 1024)) + " MiB";
		}
		return false;
	}
	return true;
}

duckdb::idx_t TableChecker::EstimateColumnsSize(const GpuPlanNode &plan) {
	return static_cast<duckdb::idx_t>(vector_gpu::EstimateColumnsSize(plan));
}

ExecutionMode TableChecker::DetermineExecutionMode(const GpuPlanNode &plan, uint64_t *out_columns_size,
                                                   uint64_t *out_vram_budget) {
	try {
		vector_gpu::RmmPool::Instance().EnsureInitialized();
	} catch (...) {}
	uint64_t vram_budget = vector_gpu::RmmPool::Instance().TotalPoolBytes();
	uint64_t estimated_columns_size = vector_gpu::EstimateColumnsSize(plan);
	if (out_columns_size) {
		*out_columns_size = estimated_columns_size;
	}
	if (out_vram_budget) {
		*out_vram_budget = vram_budget;
	}
	ExecutionMode execution_mode = ExecutionMode::PIPELINE_STREAM_CACHED;
	if (vram_budget > 0 && estimated_columns_size > vram_budget * 0.75) {
		execution_mode = ExecutionMode::PIPELINE_STREAM_UNCACHED;
	}
	return execution_mode;
}

} // namespace vector_gpu
