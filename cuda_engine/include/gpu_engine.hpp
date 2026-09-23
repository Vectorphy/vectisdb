#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// This header has NO DuckDB dependency on purpose: cuda_engine is a standalone library, callable from the
// extension but not required to link against duckdb.hpp. The extension is responsible for converting to
// and from GpuPlanNode / GpuColumn at its boundary (see extension/src/physical_gpu_execute.cpp).

namespace vector_gpu {

enum class GpuOpType : uint8_t {
	SCAN,
	FILTER,
	PROJECTION,
	HASH_JOIN,
	GROUP_BY_AGGREGATE,
	CROSS_PRODUCT,
};

enum class ExecutionMode : uint8_t {
	PIPELINE_STREAM_CACHED = 0,
	PIPELINE_STREAM_UNCACHED = 1,
	BATCH_DISPATCH = 2,
};

enum class GpuValueType : uint8_t {
	INT16,
	INT32,
	INT64,
	FLOAT32,
	FLOAT64,
	BOOLEAN,
	DICTIONARY_STRING, // integer index into GpuColumn::string_dictionary
	HUGEINT,           // 128-bit integer
};

struct alignas(16) GpuHugeInt {
	uint64_t lower;
	int64_t upper;
	
	bool operator==(const GpuHugeInt& other) const {
		return lower == other.lower && upper == other.upper;
	}
};

enum class GpuWindowPrimitive {
	NONE,
	RUNNING_SUM,
};

//! A single fused scalar expression, pre-lowered to a CUDA C++ string by the code generator.
//! `source_hash` is the SHA-256 of the originating expression subtree, used as the kernel-cache key.
struct GpuExpr {
	std::string source_hash;
	//! A full statement writing the expression's result to `out[idx]`, e.g.
	//! "((double*)out)[idx] = ((const double*)inputs[0])[idx] * ((const double*)inputs[1])[idx];" --
	//! ready to hand to CodeGenerator::WrapGridStrideLoop as-is.
	std::string generated_cuda_source;
	//! Column name feeding inputs[i] inside `generated_cuda_source`, in order. Populated by
	//! extension/src/expression_translator.cpp -- the piece that actually knows which DuckDB columns the
	//! expression referenced; this struct just carries that mapping across the extension/engine boundary.
	std::vector<std::string> input_columns;
	//! The GPU type `out[idx]` is written as.
	GpuValueType output_type = GpuValueType::FLOAT64;
	//! Window primitive if this expression represents a window function.
	GpuWindowPrimitive window_primitive = GpuWindowPrimitive::NONE;
	//! Weighted arithmetic operations this expression performs per row, as counted by
	//! extension/src/expression_cost.cpp (ExpressionCostVisitor) while the bound expression tree was
	//! still available. Consumed by AnalyzePlanShape, which is what decides whether the plan carries
	//! enough compute to be worth moving its data.
	//!
	//! It has to be carried rather than recomputed: by the time a GpuExpr reaches the engine the
	//! expression is a CUDA source STRING, and counting `sin(` occurrences in generated text is not a
	//! cost model. 0.0 means "nobody measured this one" -- treated as free, so an unmeasured expression
	//! can never talk routing INTO an offload it has not justified.
	double estimated_ops_per_row = 0.0;
};

//! The reductions GROUP_BY_AGGREGATE can execute. Deliberately a small closed set: each maps onto one
//! Thrust `reduce_by_key` pass in operators/gpu_groupby.cu, and anything outside it is refused at
//! translation time so the query falls back to DuckDB's CPU path.
enum class GpuAggregateKind : uint8_t {
	COUNT_STAR, // COUNT(*) -- no input column
	COUNT,      // COUNT(col)
	SUM,
	MIN,
	MAX,
	AVG, // sum / count, both computed per group; always produces FLOAT64
};

//! One aggregate computed by a GROUP_BY_AGGREGATE node. Unlike `GpuPlanNode::aggregate_exprs` (a
//! human-readable descriptor), this is the executable form the engine actually dispatches on.
struct GpuAggregate {
	GpuAggregateKind kind = GpuAggregateKind::COUNT_STAR;
	//! Column feeding the aggregate, named as the child operator produces it. Empty for COUNT_STAR.
	std::string input_column;
	//! Type the output column is materialized as. Must match DuckDB's own return type for the aggregate,
	//! since the result is packed straight back into a DuckDB Vector of that type: COUNT/COUNT(*) ->
	//! INT64, MIN/MAX -> the input type, SUM -> FLOAT64.
	GpuValueType output_type = GpuValueType::INT64;
};

//! The full identity of a scanned table: which attached database, which schema, which table.
//!
//! All three, never just the name. `TableCatalogEntry::name` is the BARE table name, so two tables called
//! `t` in different schemas (or different ATTACHed databases) are indistinguishable by it -- and the
//! GPU-resident column cache keys entries by table identity, so sharing a key means one table's rows being
//! served for the other's query. That is not hypothetical: caching `main.t`, then `USE s`, then querying
//! `t` returned `main.t`'s data where DuckDB returns `s.t`'s.
//!
//! It is also what the extension layer re-opens the table by (table_scanner.cpp). Resolving the bare name
//! through the CURRENT search path -- what it used to do -- makes the scan depend on session state the
//! plan does not carry, so the same plan could read two different tables at two different times.
struct GpuTableRef {
	std::string catalog; // attached database name, e.g. "memory"
	std::string schema;  // e.g. "main"
	std::string table;

	bool operator==(const GpuTableRef &other) const {
		return catalog == other.catalog && schema == other.schema && table == other.table;
	}

	//! "catalog.schema.table", for LOGS AND ERROR MESSAGES ONLY. Deliberately not used as a map key:
	//! quoted identifiers may contain dots, so two distinct tables can render to the same string.
	std::string ToString() const {
		return catalog + "." + schema + "." + table;
	}
};

//! Minimal internal stand-in for what the original design intended Substrait to carry. See
//! docs/ARCHITECTURE.md for why Substrait itself is not used here.
struct GpuPlanNode {
	//! Default-initialized (session-7 fix): a default-constructed node used to dispatch ExecutePlan's
	//! switch on an uninitialized byte. SCAN is the safe default — its dispatch path fails cleanly.
	GpuOpType op_type = GpuOpType::SCAN;
	ExecutionMode execution_mode = ExecutionMode::PIPELINE_STREAM_CACHED;
	std::vector<std::shared_ptr<GpuPlanNode>> children;

	// COST MODEL INPUTS. Populated by the translator (extension/src/gpu_offload_extension.cpp) from the
	// LogicalOperator's own estimates; consumed by EstimateWorkingSet below. They describe the plan, not
	// the execution, so they are advisory: a wrong estimate costs a routing decision, never correctness.
	//
	// NOTE for GROUP_BY_AGGREGATE: DuckDB's estimated_cardinality for an aggregate is its GROUP count,
	// not its input row count. The cost model therefore reads input sizes from `children`, never from the
	// aggregate's own `estimated_rows`. See EstimateWorkingSet.
	uint64_t estimated_rows = 0;
	//! Sum of this operator's OUTPUT column widths in bytes, per row.
	uint32_t estimated_row_width_bytes = 0;

	// SCAN
	//! Fully qualified -- see GpuTableRef for why the bare name is not enough.
	GpuTableRef table;
	std::vector<std::string> column_names;

	// FILTER / PROJECTION
	std::vector<GpuExpr> expressions;

	// HASH_JOIN
	//! Equi-join key naming a column of children[0]'s output (the condition's left-hand side).
	std::vector<std::string> build_keys;
	//! Equi-join key naming a column of children[1]'s output (the condition's right-hand side).
	//! build_keys[i] is matched against probe_keys[i].
	std::vector<std::string> probe_keys;
	//! Exact output column list, in order. NOT simply "every left column then every right column":
	//! LogicalJoin::ResolveTypes applies left_projection_map/right_projection_map, so DuckDB's join
	//! output is a mapped SUBSET of the two children's columns. The translator derives this from
	//! LogicalOperator::GetColumnBindings() (which applies the same maps), so the emitted columns line up
	//! with the operator's own `types` that the result is packed against.
	//! CROSS_PRODUCT uses this same field, resolved the same way. LogicalCrossProduct has no projection
	//! maps, so there it is simply both children's bindings concatenated -- but deriving it from
	//! GetColumnBindings() keeps the two operators on one rule.
	std::vector<std::string> output_columns;

	// GROUP_BY_AGGREGATE
	std::vector<std::string> group_keys;
	//! Human-readable descriptors only (Expression::ToString()), kept for diagnostics. `aggregates` below
	//! is what the engine executes; the two are parallel and must stay the same length.
	std::vector<std::string> aggregate_exprs;
	//! Output columns are emitted group keys first, then these aggregates -- matching the order
	//! LogicalAggregate::ResolveTypes() builds its `types` in, which is what the result is packed against.
	std::vector<GpuAggregate> aggregates;
};

//! Host-side column buffer handed across the extension/engine boundary.
//!
//! `data` is a PLAIN (pageable, not pinned) host buffer -- ordinary heap memory owned by the extension
//! layer (table_scanner.cpp's GpuColumnAccumulator / StreamingTableScanner's per-batch buffers). An
//! earlier version of this comment claimed the caller had already cudaHostRegister'd it before reaching
//! ExecutePlan; that was never actually implemented (grep-verified: no cudaHostRegister/cudaMallocHost
//! call existed anywhere in this codebase before session 30) -- describing an intended future contract as
//! present-tense fact, not a discovered regression. As of session 30, pinning happens INTERNALLY and
//! transiently instead: gpu_executor.cu's ExecuteScan copies each column's bytes into a pooled pinned
//! staging buffer (PinnedBufferPool, cuda_engine-only, never exposed across this boundary -- see
//! docs/STREAMING_INGEST_DESIGN.md) immediately before the async H2D upload. `data` itself, as received
//! here, is still ordinary pageable memory; do not assume otherwise when writing a new caller.
struct GpuColumn {
	std::string name;
	GpuValueType type;
	const void *data = nullptr;   // plain (pageable) host buffer, row-major, tightly packed
	// Optional null bitmap, nullptr if the column has no nulls. Layout when present: 1 bit per row,
	// bit-within-byte = row_index % 8 (LSB-first, i.e. row 0 is bit 0x01 of validity[0]), byte index =
	// row_index / 8. Bit set (1) = valid/non-null, bit clear (0) = null -- matches Arrow's validity
	// bitmap convention (extension/src/vector_converter.cpp produces/consumes exactly this layout).
	// Consumed by the fused expression/filter kernels (session 30) for a row-independent plan
	// (SCAN/FILTER/PROJECTION); still refused outright for GROUP_BY_AGGREGATE/HASH_JOIN/CROSS_PRODUCT,
	// whose kernels have no null semantics -- see gpu_executor.cu's ExecuteScan.
	const uint8_t *validity = nullptr;
	uint64_t row_count = 0;
	std::vector<std::string> string_dictionary; // only populated for DICTIONARY_STRING columns
};

struct GpuExecutionResult {
	bool success = false;
	std::string error_message;
	std::vector<GpuColumn> columns;
	// Owns the backing storage for `columns` above (host-side, post D2H copy) so the caller doesn't need
	// to know cuda_engine's internal buffer lifetime rules.
	std::vector<std::vector<uint8_t>> owned_buffers;
};

//! Predicted PEAK device memory for one execution of a plan, and enough detail to explain a refusal.
//!
//! Peak, not sum-of-outputs: DeviceArena frees nothing until the whole plan finishes (see
//! src/gpu_executor_internal.hpp), so every operator's buffers are live simultaneously and the total is
//! the sum over all nodes -- including each operator's INTERNAL temporaries, which dominate for
//! GROUP_BY_AGGREGATE and HASH_JOIN.
struct GpuWorkingSetEstimate {
	//! Saturates at UINT64_MAX rather than wrapping; an overflowing estimate must read as "does not fit".
	uint64_t total_bytes = 0;
	//! True when the row counts were capped because the plan is row-independent and will be chunked.
	bool chunk_capped = false;
	//! Largest single contributor, for the decline reason -- "which operator do I need to shrink?" is the
	//! only actionable form of "it did not fit".
	const char *largest_operator = "";
	uint64_t largest_bytes = 0;
};

//! True when each output row depends only on its own input row, so the caller may execute the plan in
//! independent chunks and concatenate the results. SCAN/PROJECTION are row-wise and FILTER only drops
//! rows; GROUP_BY_AGGREGATE, HASH_JOIN and CROSS_PRODUCT all combine rows across the input and must see
//! it whole.
//!
//! Lives here, beside the cost model, because the two must agree: sizing a plan against a chunk that the
//! executor will not actually use would approve a plan that then allocates for the full input.
bool IsRowIndependent(const GpuPlanNode &node);

//! What AnalyzePlanShape concluded about a plan's compute density. Three states, not two, because the
//! measured evidence only supports a verdict at the ends: see kMinOpsPerScannedRow / kOffloadOpsPerRow.
enum class GpuWorkDensity : uint8_t {
	//! Below kMinOpsPerScannedRow, having priced at least one expression. Too little arithmetic per row of
	//! data moved for the transfer to be repaid at any scale -- decline regardless of any other metric.
	INSUFFICIENT,
	//! Between the two thresholds, OR carrying no priced expressions at all. Either way the model has no
	//! measured opinion and the caller decides (today: by falling back to the legacy row-amplification
	//! metric). Note the second case carefully -- a GROUP BY over a cross product does enormous work and
	//! carries not one GpuExpr to count it with, so "priced nothing" means unmeasured, never cheap.
	INCONCLUSIVE,
	//! At or above kOffloadOpsPerRow. Dense enough that the arithmetic, not the data movement, dominates
	//! -- offload it.
	SUFFICIENT,
};

//! Below this many weighted ops per scanned row, decline outright. `SELECT a + b FROM t` sits at exactly
//! 1.0 and is a measured loss at every scale tested (0.75-0.85x of the CPU even with a warm GPU-resident
//! cache, session 32).
constexpr double kMinOpsPerScannedRow = 3.0;
//! At or above this, offload without consulting any other heuristic. `sin(a)+cos(b)+sqrt(c)+ln(...)+
//! exp(...)` scores ~132 and is the one shape measured FASTER than the CPU (1.14x at 20M rows, 1.21x at
//! 70M, 1.65-1.79x at 50M -- session 32).
//!
//! Honest about the gap: the nearest measured LOSER (`a*b+sqrt(c)`, 0.88x) scores 7 and the nearest
//! measured WINNER scores ~132, so anything between 8 and 131 is unmeasured. 10.0 sits just above the
//! known losers rather than in the middle of the gap, because the harm is asymmetric -- wrongly
//! declining costs nothing, wrongly offloading costs 2-3x.
constexpr double kOffloadOpsPerRow = 10.0;

//! How much work a plan derives from the data it has to move, which is what decides whether offloading
//! it can pay at all. See AnalyzePlanShape.
struct GpuPlanShape {
	//! Rows that must be read out of DuckDB storage and packed into host buffers -- summed over SCAN
	//! leaves. This is the cost that does not shrink no matter how fast the device is.
	uint64_t scanned_rows = 0;
	//! Bytes behind those rows (rows x scanned row width). The physically meaningful denominator: PCIe
	//! and the host scan are paid per BYTE, and two plans reading the same row count through different
	//! column widths do not cost the same.
	uint64_t scanned_bytes = 0;
	//! Largest row count anywhere in the plan.
	uint64_t max_rows = 0;
	//! Total weighted arithmetic the plan performs: summed over every node as (that node's per-row
	//! expression cost) x (the rows it evaluates over). Structural amplification is therefore already
	//! inside this number -- a cross product evaluating a 1-op expression over 20M pairs counts 20M ops,
	//! not 1.
	double total_ops = 0.0;
	//! total_ops / scanned_rows: weighted operations derived per row of data moved. THE routing metric.
	double ops_per_scanned_row = 0.0;
	//! total_ops / scanned_bytes. Reported rather than thresholded -- it is the number to quote when
	//! explaining a decline, since it is comparable across schemas in a way the per-row figure is not.
	double ops_per_scanned_byte = 0.0;
	//! The verdict ops_per_scanned_row implies.
	GpuWorkDensity density = GpuWorkDensity::INSUFFICIENT;
	//! DEPRECATED -- max_rows / scanned_rows, the pure row-amplification metric this model replaces.
	//!
	//! It counts ROWS of work and cannot tell `a + b` from `sin(a)+cos(b)+exp(c)`: both amplify by 1.0,
	//! one loses to the CPU and one beats it. That blind spot is why every heavy-math projection needed
	//! VECTOR_GPU_MIN_AMPLIFICATION=0 to route at all (see gpu_shell/bench_matrix.sh, which sets it for
	//! exactly this reason). Still computed, and still consulted for INCONCLUSIVE plans, because it is
	//! the only signal that sees work no expression carries -- a cross product's n*m expansion, a
	//! group-by's sort.
	double amplification = 0.0;
};

//! Measures the work-to-data ratio of a plan: how much arithmetic it derives per row it has to move.
//!
//! Offloading pays when a lot of compute comes off a little data. Session 27 established the data half
//! by holding the work fixed at 20M cross-product pairs and varying only the scan volume -- GPU time
//! barely moved (447-499 ms) while the verdict flipped from 1.83x slower to 1.16x faster, which is what
//! identifies the SCAN rather than the compute as the thing that decides.
//!
//! What that model could not see is the numerator. It proxied "work" by ROW COUNT, so every row cost the
//! same whatever it computed, and a plan doing one row of work per row scanned was declined even when
//! that row ran five transcendentals. Session 32 measured the consequence directly: with routing forced
//! open, `sin(a)+cos(b)+sqrt(c)+ln(abs(a)+1)+exp(b/100)` over 20-70M rows was the ONLY shape that beat
//! DuckDB's CPU path (1.14-1.21x), and it was declined by default. This function now weighs the
//! arithmetic itself (see GpuExpr::estimated_ops_per_row), so density decides and row count only breaks
//! ties.
GpuPlanShape AnalyzePlanShape(const GpuPlanNode &plan);

//! Estimates peak device memory for `plan`. When the plan is row-independent, every node's row count is
//! capped at `max_chunk_rows` -- chunked execution never has more than one chunk resident, so a 50M-row
//! projection is sized against the chunk, not the table. Pass 0 to disable the cap.
//!
//! Derived from the actual ctx.arena.Alloc call sites in each operator, NOT from output widths alone --
//! that was the 2.35x underestimate recorded in docs/CHANGELOG.md session 26 (a global AVG over 64M rows
//! allocates ~2 GiB of permutation/sort/scan buffers to produce 8 bytes of output). Keep it in lockstep
//! with those operators; tests/test_gpu_cost_model.cpp fails if it drifts.
GpuWorkingSetEstimate EstimateWorkingSet(const GpuPlanNode &plan, uint64_t max_chunk_rows);

//! Sums the total estimated byte size of all scanned columns across all SCAN leaves in `plan`.
uint64_t EstimateColumnsSize(const GpuPlanNode &plan);

//! Determines execution mode based on the auto-bypass cache threshold heuristic:
//! if (estimated_columns_size > vram_budget * threshold) {
//!     execution_mode = ExecutionMode::PIPELINE_STREAM_UNCACHED;
//! }
ExecutionMode DetermineExecutionMode(const GpuPlanNode &plan, uint64_t vram_budget, double threshold = 0.75);

//! How one execution interacts with the GPU-resident column cache (GpuColumnCache, see
//! docs/GPU_RESIDENT_CACHE_DESIGN.md). Both fields default to "the cache is not involved", so a caller
//! that knows nothing about it — the whole-input GROUP BY/JOIN path, the unit tests — executes exactly as
//! it did before the cache existed.
//!
//! The two fields are mutually exclusive: one populates the cache from a scan, the other reads back what
//! a previous execution populated.
struct GpuExecuteOptions {
	//! Hand every column each SCAN leaf uploads to GpuColumnCache as the next chunk of that table's OPEN
	//! staging slot (the caller must have called GpuColumnCache::BeginStaging first, and must call
	//! CommitStaging or AbortStaging when the scan ends). Nothing staged is visible to a later query until
	//! that commit. A chunk the cache declines stays owned by this execution's DeviceArena.
	bool stage_scan_to_cache = false;
	//! >= 0: serve every SCAN leaf from chunk `cache_chunk_index` of its already-committed cache entry
	//! instead of from `inputs`, which must then be empty. This is the point of the cache — the host-side
	//! scan (~74% of GPU-path time, docs/SESSION_30_BENCHMARK_RESULTS.md) does not run at all.
	//!
	//! Chunk-at-a-time rather than whole-column-at-once so a cache hit executes with the same per-chunk
	//! device footprint routing sized the plan against (PhysicalGpuExecute::MAX_CHUNK_ROWS); serving the
	//! whole table as one chunk would run a 20M-row plan inside a 2M-row approval.
	int64_t cache_chunk_index = -1;
	//! Serve every SCAN leaf from the cache in FULL -- every chunk of every column, concatenated on the
	//! device into one contiguous buffer per column -- instead of from `inputs`, which must then be empty.
	//!
	//! For the WHOLE-INPUT operators (GROUP_BY_AGGREGATE, HASH_JOIN, CROSS_PRODUCT), which cannot use
	//! cache_chunk_index above: a GROUP BY sorts across its entire input, so replaying chunk-by-chunk
	//! would produce per-chunk aggregates, not the query's answer. Those plans were the one shape still
	//! paying the full host scan and H2D on every run -- measured at 0.04x of the CPU on a 70M-row
	//! `SELECT SUM(a)` (docs/SESSION_32_CACHE_BENCHMARK.md).
	//!
	//! The concatenation is device-to-device, so nothing crosses PCIe and nothing is scanned on the host.
	//! It costs one extra full-column buffer in the plan's arena for the duration -- which is exactly the
	//! SCAN term EstimateWorkingSet already charges the plan for, so the routing decision that approved
	//! this plan already covers it (GpuColumnCache::CurrentBytes() covers the cache's own copy).
	bool serve_whole_scan_from_cache = false;
};

//! Top-level entrypoint called by PhysicalGpuExecute. Stateless from the caller's perspective — all
//! persistent state (RMM pool, kernel cache) lives behind this call, initialized lazily on first use.
class GpuEngine {
public:
	static GpuEngine &Instance();

	//! Executes `plan` against `inputs` and returns materialized host-side columns. Every GpuOpType has a
	//! real implementation: SCAN/FILTER/PROJECTION in src/gpu_executor.cu, the rest under src/operators/.
	//! A shape an operator cannot run exactly (nullable input, a float join key, an oversized cross
	//! product, ...) returns success=false with an explanatory error rather than garbage.
	GpuExecutionResult ExecutePlan(const GpuPlanNode &plan, const std::vector<GpuColumn> &inputs,
	                               const GpuExecuteOptions &options = {});

private:
	GpuEngine() = default;
};

} // namespace vector_gpu
