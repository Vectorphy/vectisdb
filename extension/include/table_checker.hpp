#pragma once

#include <string>

#include "duckdb/planner/logical_operator.hpp"
#include "gpu_engine.hpp"

namespace vector_gpu {

//! Cost-based routing: decides whether a logical plan (sub)tree is worth offloading to the GPU.
//! Placeholder thresholds — see docs/KNOWN_ISSUES.md. Never mutates the plan; the caller (optimizer
//! callback) is responsible for swapping the subtree once this returns true.
class TableChecker {
public:
	//! Minimum estimated row count below which GPU execution is rejected (PCIe transfer latency exceeds
	//! the time a CPU spends processing the same data out of L1/L2 cache).
	static constexpr duckdb::idx_t MIN_ROW_COUNT_THRESHOLD = 100000;

	//! ABSURDITY BACKSTOP on a cross product's estimated output rows, NOT the real limit.
	//!
	//! The real limit is bytes, applied by HasVramHeadroomForPlan against the working-set model once the
	//! plan is lowered. This row count exists only because ShouldOffload runs BEFORE translation and
	//! before any CUDA context exists, and something has to reject 10^10 rows without paying ~93 ms to
	//! create a context and ask about VRAM.
	//!
	//! Measured session 26: it never binds in practice. A pairwise query carrying 7 DOUBLE columns
	//! through the cross product costs ~104 bytes/pair once projection and group-by temporaries are
	//! counted, so a 4 GiB device runs out at ~36M pairs -- 7.5x below this cap. Treat a decline citing
	//! this constant as "the estimate was astronomically large", never as "this is the device's capacity".
	static constexpr duckdb::idx_t MAX_CROSS_PRODUCT_OUTPUT_ROWS = 1ull << 32;

	//! Minimum ROW-amplification a plan must derive before offloading it can pay.
	//!
	//! NO LONGER THE PRIMARY RULE. The expression-aware density model (AnalyzePlanShape ->
	//! GpuPlanShape::density, thresholds kMinOpsPerScannedRow / kOffloadOpsPerRow) decides first, and
	//! this constant applies only to the band where that model has no measured opinion, plus to runs
	//! that select the legacy rule explicitly with VECTOR_GPU_MIN_AMPLIFICATION.
	//!
	//! Why it was demoted: it counts ROWS of work, so `a + b` and `sin(a)+cos(b)+exp(c/100)` are
	//! indistinguishable to it -- both amplify by 1.0. Session 32 measured the first at 0.75-0.85x of
	//! the CPU and the second at 1.14-1.79x. It was declining the only shape this engine wins on.
	//!
	//! Retained rather than deleted because it sees work that no expression carries: a cross product's
	//! n*m expansion and a group-by's sort do real work per row that no GpuExpr counts, and the numbers
	//! below were measured on exactly that shape.
	//!
	//! MEASURED, not chosen. Session 27 held the work constant at 20M cross-product pairs and varied only
	//! how many rows had to be scanned to produce them:
	//!
	//!   scanned    amplification    GPU vs CPU
	//!   1,000,020            20     1.83x SLOWER
	//!     100,200           200     1.13x SLOWER
	//!      12,000         1,667     1.01x  (break-even)
	//!       8,944         2,236     1.16x FASTER
	//!
	//! GPU time barely moved across that sweep (447-499 ms) because the compute was identical; what
	//! changed was the scan. Break-even sits near 1,600. 1,000 is below it deliberately: the harm is
	//! asymmetric -- wrongly declining costs nothing (the CPU path is the baseline), wrongly offloading
	//! costs 2-3x -- but a threshold set at break-even would also reject genuine winners whose
	//! expressions are heavier than the sweep's. Every measured LOSS is below this line and every
	//! measured WIN above it.
	//!
	//! Consequence worth stating plainly: ordinary GROUP BY and JOIN plans amplify by ~1, carry no
	//! expression cost of their own, and so are still declined by both rules. Their kernels remain
	//! correct and unit-tested, but nothing routes to them until the scan path gets cheaper
	//! (docs/TODO.md P1 item 4, the persistent device-side column cache). That is the honest state --
	//! offloading them measured 2-2.6x slower than not.
	//!
	//! Setting VECTOR_GPU_MIN_AMPLIFICATION selects this rule INSTEAD OF the density model, with the
	//! given value in place of this constant -- the escape hatch for exercising declined paths
	//! deliberately.
	static constexpr double MIN_WORK_AMPLIFICATION = 1000.0;

	//! True when the plan is worth moving its data for: dense enough arithmetically, or (in the band
	//! where density is inconclusive) amplifying enough in rows. Named for the metric it used to apply
	//! exclusively; see MIN_WORK_AMPLIFICATION and GpuPlanShape for the rule it applies now.
	static bool HasSufficientAmplification(const GpuPlanNode &plan, std::string *reason = nullptr);

	//! Returns true if `op` (and its children) are entirely composed of GPU-supported operators, and the
	//! estimated cardinality / VRAM pressure make offloading worthwhile.
	//! `reason`, when non-null, receives a specific explanation on rejection ("operator ORDER_BY not
	//! supported", "estimated 40000 rows below threshold 100000", ...). This exists because the generic
	//! "it was rejected" answer is not actionable: the whole point of the activity log's `decline` record
	//! is telling a user WHICH rule refused their query.
	//! `check_vram` exists so the caller can run the cheap, CUDA-free checks first and only ask about
	//! VRAM once it knows the plan can actually be lowered. Querying VRAM CREATES THE CUDA CONTEXT
	//! (~93 ms, measured), and a plan that passes every structural check here can still fail later in
	//! TranslateToGpuPlan — paying for a context that is then thrown away. Pass false, then call
	//! HasVramHeadroomForPlan() after translation succeeds.
	static bool ShouldOffload(duckdb::ClientContext &context, duckdb::LogicalOperator &op,
	                          std::string *reason = nullptr, bool check_vram = true);

	//! The VRAM half of ShouldOffload, split out so it can run last. The only routing call that touches
	//! CUDA.
	//!
	//! Takes the LOWERED PLAN, not the LogicalOperator, and asks the engine's own working-set model
	//! (EstimateWorkingSet) what one execution will actually allocate. Two things follow from that which
	//! the previous logical-operator version could not express:
	//!
	//!   - Internal temporaries are counted. GROUP_BY_AGGREGATE allocates 8 x (4 + key_count) bytes per
	//!     INPUT row for its permutation/sort/head/group-id buffers -- ~2 GiB for a 64M-row global AVG
	//!     that outputs 8 bytes. Sizing on output width alone was 2.35x optimistic (session 26).
	//!   - The total is SUMMED across the plan, not tested node by node. DeviceArena frees nothing until
	//!     the plan ends, so every operator's buffers are live at once; checking each node separately
	//!     approved 2.3 GB and 1.15 GB against a ~3.0 GB budget.
	//!
	//! Row-independent plans are sized against one chunk (PhysicalGpuExecute::MAX_CHUNK_ROWS), because
	//! chunked execution never holds more than one chunk on the device.
	static bool HasVramHeadroomForPlan(const GpuPlanNode &plan, std::string *reason = nullptr);

	//! True when `plan` is safe to run given whatever nullability its SCAN leaves might carry -- the VRAM
	//! half of routing's counterpart for null-handling, checked at the same stage
	//! (HasVramHeadroomForPlan) and for the same reason: only the LOWERED plan knows whether it ended up
	//! row-independent (SCAN/FILTER/PROJECTION), which is exactly what decides the answer.
	//!
	//! Session 30: the fused expression/filter kernel now threads validity through with SQL
	//! NULL-propagation semantics for row-independent plans, so those are unconditionally safe here
	//! regardless of what their SCAN leaves' statistics say. GROUP_BY_AGGREGATE/HASH_JOIN/CROSS_PRODUCT
	//! still have no null semantics in their Thrust-based kernels (unchanged, not touched this session),
	//! so those are declined unless every SCAN leaf's columns can be PROVEN not to produce NULLs --
	//! "declined" is the fail-safe default, "approved" requires positive evidence, never the reverse.
	//!
	//! An earlier version of this check (session 29's MightProduceNulls) ran pre-translation, directly
	//! from LOGICAL_GET, and declined unconditionally -- correct at the time (nothing yet handled nulls),
	//! but it would have silently kept blocking row-independent plans after session 30 made them safe,
	//! since a pre-translation check cannot know IsRowIndependent's answer. Moved here, post-translation,
	//! so it can ask the right question instead of the same question for every shape.
	static bool HasSafeNullHandling(duckdb::ClientContext &context, const GpuPlanNode &plan, std::string *reason = nullptr);

	//! Determines the execution mode (e.g. PIPELINE_STREAM_CACHED vs PIPELINE_STREAM_UNCACHED)
	//! using the auto-bypass cache threshold heuristic:
	//! if (estimated_columns_size > vram_budget * 0.75) execution_mode = PIPELINE_STREAM_UNCACHED;
	static ExecutionMode DetermineExecutionMode(const GpuPlanNode &plan, uint64_t *out_columns_size = nullptr,
	                                           uint64_t *out_vram_budget = nullptr);

	//! Sizing utility: estimates the total byte size of columns scanned by `plan`.
	static duckdb::idx_t EstimateColumnsSize(const GpuPlanNode &plan);

private:
	//! ShouldOffload's actual recursion.
	//!
	//! `cross_product_subtree` says the subtree being judged contains a LOGICAL_CROSS_PRODUCT, which
	//! suspends MIN_ROW_COUNT_THRESHOLD for every operator in it EXCEPT the cross product itself. The
	//! threshold asks "are there enough rows here to be worth a PCIe round trip?", and a cross product is
	//! the one operator whose answer cannot be read off any single node: its inputs are deliberately small
	//! (a few thousand rows), its output is n*m, and an aggregate above it reports its GROUP count -- one
	//! row for `SELECT MAX(...) FROM a, b`. Judging any of those three on its own row count declines
	//! exactly the arithmetic-bound shape a discrete GPU wins on. The cross product node itself is still
	//! judged, on n*m, so a 100x100 cross join stays on the CPU where it belongs.
	//!
	//! Deliberately narrow: for a plan with no cross product in it this flag is false everywhere and the
	//! routing decision is bit-for-bit what it was before. The general version of this problem -- an
	//! aggregate's threshold being tested against its group count rather than its input rows -- is a
	//! separate item (docs/TODO.md P0 2) that changes routing for plans this one cannot reach.
	static bool ShouldOffloadRecursive(duckdb::ClientContext &context, duckdb::LogicalOperator &op,
	                                   std::string *reason, bool check_vram, bool cross_product_subtree);

	//! True if `op` or anything beneath it is a LOGICAL_CROSS_PRODUCT.
	static bool ContainsCrossProduct(const duckdb::LogicalOperator &op);

	static bool IsSupportedOperator(const duckdb::LogicalOperator &op);

	//! Minimum fraction of its input rows a FILTER must keep for offloading to be worthwhile.
	//!
	//! A GPU filter has to upload EVERY row to decide which ones survive. If a predicate keeps 2% of the
	//! table, that is 100% of the PCIe cost to throw away 98% of the data -- the CPU does the same work
	//! with better cache locality and hands back a fraction of the rows. 0.10 (keep >= 10%) is a
	//! deliberately conservative line: it only declines cases that are clearly losing, rather than trying
	//! to model the crossover precisely.
	static constexpr double MIN_FILTER_SELECTIVITY = 0.10;
	static bool HasSupportedTypes(const duckdb::LogicalOperator &op, std::string *bad_type = nullptr);
	static duckdb::idx_t EstimateCardinality(const duckdb::LogicalOperator &op);
	//! Largest estimated cardinality anywhere in `op`'s subtree, including `op` itself. This -- not the
	//! operator's own output -- is what MIN_ROW_COUNT_THRESHOLD is judged against, because every
	//! cardinality-REDUCING operator (aggregate, selective filter) reports an output far smaller than the
	//! data it had to read, and it is the reading that the threshold exists to price.
	static duckdb::idx_t MaxSubtreeCardinality(const duckdb::LogicalOperator &op);
	static bool HasVramHeadroom(duckdb::idx_t estimated_bytes);
};

} // namespace vector_gpu
