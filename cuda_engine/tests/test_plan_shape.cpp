#include "test_framework.hpp"
#include "gpu_engine.hpp"

#include <memory>
#include <string>
#include <vector>

// AnalyzePlanShape -- the "will it PAY?" half of the cost model. Pure arithmetic over a plan tree, so
// unlike test_gpu_cost_model.cpp this needs no GPU and runs anywhere.
//
// The cases below reflect query shapes measured for GPU-vs-CPU behavior (20M and 70M rows, warm cache), so the
// assertions check the model against measurements rather than against itself. The expression weights are
// the ones ExpressionCostVisitor assigns; they are reproduced here as literals on purpose -- if somebody
// retunes a weight, this file should fail and make them re-check it against those measurements.

using namespace vector_gpu;

namespace {

constexpr double kAddSub = 1.0;
constexpr double kMulDiv = 2.0;
constexpr double kSqrt = 4.0;
constexpr double kTranscendental = 30.0;

std::shared_ptr<GpuPlanNode> MakeScan(const std::vector<std::string> &columns, uint64_t rows, uint32_t width) {
	auto scan = std::make_shared<GpuPlanNode>();
	scan->op_type = GpuOpType::SCAN;
	scan->table = {"memory", "main", "bench"};
	scan->column_names = columns;
	scan->estimated_rows = rows;
	scan->estimated_row_width_bytes = width;
	return scan;
}

GpuExpr MakeExpr(double ops_per_row) {
	GpuExpr expr;
	expr.generated_cuda_source = "((double*)out)[idx] = 0.0;";
	expr.output_type = GpuValueType::FLOAT64;
	expr.estimated_ops_per_row = ops_per_row;
	return expr;
}

//! A projection of `expression_costs` over `rows` scanned rows of `scanned_columns` DOUBLE columns.
std::shared_ptr<GpuPlanNode> MakeProjection(const std::vector<double> &expression_costs, uint64_t rows,
                                            uint32_t scanned_columns) {
	auto projection = std::make_shared<GpuPlanNode>();
	projection->op_type = GpuOpType::PROJECTION;
	for (auto cost : expression_costs) {
		projection->expressions.push_back(MakeExpr(cost));
	}
	projection->children.push_back(MakeScan({"a", "b", "c"}, rows, scanned_columns * 8));
	projection->estimated_rows = rows;
	projection->estimated_row_width_bytes = static_cast<uint32_t>(expression_costs.size() * 8);
	return projection;
}

} // namespace

//! proj-cheap: `SELECT a + b FROM bench`. Measured 0.75x (20M) / 0.85x (70M) of the CPU -- a loss at
//! every scale, and the acceptance case for declining a plan that amplifies by exactly 1.0.
static void TestCheapProjectionIsDeclinedOnDensity() {
	auto plan = MakeProjection({kAddSub}, 20000000, 2);
	auto shape = AnalyzePlanShape(*plan);

	CHECK(shape.scanned_rows == 20000000);
	CHECK(shape.scanned_bytes == 20000000ull * 16);
	CHECK(shape.ops_per_scanned_row == 1.0);
	CHECK(shape.density == GpuWorkDensity::INSUFFICIENT);
	// The metric it replaces cannot tell this plan from the heavy one below: both amplify by 1.0.
	CHECK(shape.amplification == 1.0);
}

//! proj-heavy: `sin(a) + cos(b) + sqrt(c) + ln(abs(a) + 1) + exp(b / 100)`. Measured 1.14x (20M) /
//! 1.21x (70M) FASTER than the CPU -- the only shape in the suite that wins, and the one the old rule
//! declined. Routing it must not need VECTOR_GPU_MIN_AMPLIFICATION.
static void TestHeavyMathProjectionOffloadsOnDensityAlone() {
	// sin + cos + sqrt + ln + abs + (abs+1) + exp + (b/100), plus the four additions joining them.
	const double heavy = 3 * kTranscendental + kSqrt + kTranscendental + kAddSub + kAddSub + kMulDiv + 4 * kAddSub;
	auto plan = MakeProjection({heavy}, 20000000, 3);
	auto shape = AnalyzePlanShape(*plan);

	CHECK(shape.ops_per_scanned_row == heavy);
	CHECK(shape.ops_per_scanned_row >= kOffloadOpsPerRow);
	CHECK(shape.density == GpuWorkDensity::SUFFICIENT);
	// Same amplification as the cheap projection above -- which is the entire point.
	CHECK(shape.amplification == 1.0);
	// 132 ops over 24 scanned bytes.
	CHECK(shape.ops_per_scanned_byte > 5.0);
}

//! The two measured near-misses, which calibrate SQRT: both sit in the band between the thresholds, so
//! neither offloads on density alone. proj-baseline measured 0.88x and proj-wide 0.61x.
static void TestModeratelyDenseProjectionsAreInconclusive() {
	auto baseline = AnalyzePlanShape(*MakeProjection({kMulDiv + kSqrt + kAddSub}, 20000000, 3)); // a*b + sqrt(c)
	CHECK(baseline.ops_per_scanned_row == 7.0);
	CHECK(baseline.density == GpuWorkDensity::INCONCLUSIVE);

	// proj-wide: four separate output expressions, summed because one kernel evaluates them all per row.
	auto wide = AnalyzePlanShape(*MakeProjection({kMulDiv, kAddSub, kSqrt, kAddSub}, 20000000, 3));
	CHECK(wide.ops_per_scanned_row == 8.0);
	CHECK(wide.density == GpuWorkDensity::INCONCLUSIVE);
}

//! A filter evaluates its predicate over its INPUT rows, not its output. Charging it for the rows it
//! keeps would under-count a selective filter by exactly its selectivity.
static void TestFilterIsChargedOverItsInputRows() {
	auto filter = std::make_shared<GpuPlanNode>();
	filter->op_type = GpuOpType::FILTER;
	filter->expressions.push_back(MakeExpr(kAddSub + kAddSub)); // a + b > 50
	filter->children.push_back(MakeScan({"a", "b"}, 20000000, 16));
	filter->estimated_rows = 200000; // 1% selectivity
	filter->estimated_row_width_bytes = 16;

	auto shape = AnalyzePlanShape(*filter);
	CHECK(shape.total_ops == 2.0 * 20000000.0);
	CHECK(shape.ops_per_scanned_row == 2.0);
}

//! Structural amplification has to survive the change of metric: a cross product that turns 9k scanned
//! rows into 20M pairs was MEASURED as a win (session 27), and it earns that on expansion rather than on
//! arithmetic. Counting total ops rather than per-row cost is what keeps it offloadable.
static void TestCrossProductEarnsDensityThroughExpansion() {
	auto cross = std::make_shared<GpuPlanNode>();
	cross->op_type = GpuOpType::CROSS_PRODUCT;
	cross->children.push_back(MakeScan({"av"}, 4472, 8));
	cross->children.push_back(MakeScan({"bv"}, 4472, 8));
	cross->estimated_rows = 4472ull * 4472ull;
	cross->estimated_row_width_bytes = 16;

	auto projection = std::make_shared<GpuPlanNode>();
	projection->op_type = GpuOpType::PROJECTION;
	projection->expressions.push_back(MakeExpr(kAddSub)); // one cheap op per PAIR
	projection->children.push_back(cross);
	projection->estimated_rows = cross->estimated_rows;
	projection->estimated_row_width_bytes = 8;

	auto shape = AnalyzePlanShape(*projection);
	CHECK(shape.scanned_rows == 8944);
	CHECK(shape.ops_per_scanned_row > 2000.0);
	CHECK(shape.density == GpuWorkDensity::SUFFICIENT);
}

//! A plan carrying no priced expressions at all -- an ordinary GROUP BY, whose work lives in a sort no
//! GpuExpr describes -- must come back INCONCLUSIVE, not INSUFFICIENT. "The model priced nothing" is not
//! "the plan is cheap", and conflating them declines whatever the legacy metric used to approve.
static void TestPlanWithNoPricedExpressionsIsUnmeasuredRatherThanCheap() {
	auto group_by = std::make_shared<GpuPlanNode>();
	group_by->op_type = GpuOpType::GROUP_BY_AGGREGATE;
	group_by->group_keys = {"c"};
	group_by->children.push_back(MakeScan({"a", "c"}, 20000000, 16));
	group_by->estimated_rows = 1000;
	group_by->estimated_row_width_bytes = 16;

	auto shape = AnalyzePlanShape(*group_by);
	CHECK(shape.total_ops == 0.0);
	CHECK(shape.ops_per_scanned_row == 0.0);
	CHECK(shape.density == GpuWorkDensity::INCONCLUSIVE);
	// ...and the metric that then decides still declines it, so the outcome is unchanged. Note it is the
	// 20M-row SCAN, not the 1000-row output, that sets max_rows -- an aggregate amplifies by 1.0, which is
	// three orders of magnitude under MIN_WORK_AMPLIFICATION.
	CHECK(shape.max_rows == 20000000);
	CHECK(shape.amplification == 1.0);
}

//! The shape that makes the rule above load-bearing: a global aggregate over a cross product. 8,944 rows
//! in, 20M pairs sorted and reduced, measured 1.90x FASTER than the CPU (session 27) -- and not one
//! priced expression anywhere in it. It must survive on row amplification alone.
static void TestAggregateOverCrossProductSurvivesWithNoPricedExpressions() {
	auto cross = std::make_shared<GpuPlanNode>();
	cross->op_type = GpuOpType::CROSS_PRODUCT;
	cross->children.push_back(MakeScan({"av"}, 4472, 8));
	cross->children.push_back(MakeScan({"bv"}, 4472, 8));
	cross->estimated_rows = 4472ull * 4472ull;
	cross->estimated_row_width_bytes = 16;

	auto aggregate = std::make_shared<GpuPlanNode>();
	aggregate->op_type = GpuOpType::GROUP_BY_AGGREGATE;
	aggregate->children.push_back(cross);
	aggregate->estimated_rows = 1;
	aggregate->estimated_row_width_bytes = 8;

	auto shape = AnalyzePlanShape(*aggregate);
	CHECK(shape.total_ops == 0.0);
	CHECK(shape.density == GpuWorkDensity::INCONCLUSIVE);
	// 20M pairs of work from 8,944 scanned rows: the legacy metric sees this one clearly.
	CHECK(shape.amplification > 2000.0);
}

//! Nothing scanned means nothing to amortise, so the ratios must not divide by zero. A NaN here would
//! compare false against BOTH thresholds and land a plan in INCONCLUSIVE by accident.
static void TestPlanThatScansNothingIsNotADivideByZero() {
	auto projection = std::make_shared<GpuPlanNode>();
	projection->op_type = GpuOpType::PROJECTION;
	projection->expressions.push_back(MakeExpr(kTranscendental));
	projection->estimated_rows = 1000000;
	projection->estimated_row_width_bytes = 8;

	auto shape = AnalyzePlanShape(*projection);
	CHECK(shape.scanned_rows == 0);
	CHECK(shape.scanned_bytes == 0);
	CHECK(shape.ops_per_scanned_row == shape.total_ops);
	CHECK(shape.ops_per_scanned_row == shape.ops_per_scanned_row); // not NaN
	CHECK(shape.density == GpuWorkDensity::SUFFICIENT);
}

//! An expression nobody priced (estimated_ops_per_row left at its 0.0 default -- a translator path that
//! forgot the visitor) must read as "no evidence of density", never as permission to offload. It falls
//! back to the legacy metric, which for a projection is 1.0 and declines: exactly the behaviour that
//! existed before this model, which is the right thing for a plan the model cannot see.
static void TestUnpricedExpressionsDoNotTalkRoutingIntoOffloading() {
	auto plan = MakeProjection({0.0, 0.0}, 20000000, 2);
	auto shape = AnalyzePlanShape(*plan);
	CHECK(shape.total_ops == 0.0);
	CHECK(shape.density != GpuWorkDensity::SUFFICIENT);
	CHECK(shape.amplification == 1.0);
}

static void TestAutoBypassCacheThreshold() {
	// Budget: 3.2 GB (~3,435,973,836 bytes). 75% threshold is ~2.57 GB (~2,576,980,377 bytes).
	uint64_t vram_budget = 3435973836ULL;

	// Case 1: 40M rows x 5 double columns (40 bytes/row) = 1.6 GB columns (< 75% of budget).
	auto plan_40m = MakeProjection({1.0}, 40000000ULL, 5);
	uint64_t size_40m = EstimateColumnsSize(*plan_40m);
	CHECK(size_40m == 40000000ULL * 40ULL);
	CHECK(DetermineExecutionMode(*plan_40m, vram_budget, 0.75) == ExecutionMode::PIPELINE_STREAM_CACHED);

	// Case 2: 150M rows x 5 double columns (40 bytes/row) = 6.0 GB columns (> 75% of budget).
	auto plan_150m = MakeProjection({1.0}, 150000000ULL, 5);
	uint64_t size_150m = EstimateColumnsSize(*plan_150m);
	CHECK(size_150m == 150000000ULL * 40ULL);
	CHECK(DetermineExecutionMode(*plan_150m, vram_budget, 0.75) == ExecutionMode::PIPELINE_STREAM_UNCACHED);

	// Case 3: Exactly around boundary:
	// 75% of 3435973836 is 2576980377.
	// Plan with 2.5 GB columns <= threshold -> CACHED
	auto plan_under = MakeScan({"col0"}, 312500000ULL, 8); // 2.5 GB
	CHECK(DetermineExecutionMode(*plan_under, vram_budget, 0.75) == ExecutionMode::PIPELINE_STREAM_CACHED);
	// Plan with 2.8 GB columns > threshold -> UNCACHED
	auto plan_over = MakeScan({"col0"}, 350000000ULL, 8); // 2.8 GB
	CHECK(DetermineExecutionMode(*plan_over, vram_budget, 0.75) == ExecutionMode::PIPELINE_STREAM_UNCACHED);
}

int main() {
	RUN_TEST(TestCheapProjectionIsDeclinedOnDensity);
	RUN_TEST(TestHeavyMathProjectionOffloadsOnDensityAlone);
	RUN_TEST(TestModeratelyDenseProjectionsAreInconclusive);
	RUN_TEST(TestFilterIsChargedOverItsInputRows);
	RUN_TEST(TestCrossProductEarnsDensityThroughExpansion);
	RUN_TEST(TestPlanWithNoPricedExpressionsIsUnmeasuredRatherThanCheap);
	RUN_TEST(TestAggregateOverCrossProductSurvivesWithNoPricedExpressions);
	RUN_TEST(TestPlanThatScansNothingIsNotADivideByZero);
	RUN_TEST(TestUnpricedExpressionsDoNotTalkRoutingIntoOffloading);
	RUN_TEST(TestAutoBypassCacheThreshold);
	TEST_MAIN_EPILOGUE();
}

