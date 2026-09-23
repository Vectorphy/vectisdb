#include "test_framework.hpp"
#include "gpu_engine.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

// Keeps EstimateWorkingSet honest against the operators it models.
//
// The model is hand-derived from each operator's ctx.arena.Alloc call sites, so it can silently go stale
// the moment somebody adds an allocation. The consequence of staleness is the worst symptom this engine
// has -- a routing decision that approves a plan which then thrashes or fails under memory pressure --
// and it is invisible to every other test here, because they all check RESULTS and the results stay
// correct right up until the device runs out.
//
// So this suite does not check the model against itself. It runs each plan on a real GPU while a second
// thread samples cudaMemGetInfo, and compares the prediction against the observed peak.
//
// The tolerance is deliberately wide (see kLowerRatio/kUpperRatio). Thrust's sort and scan allocate
// their own temporary storage OUTSIDE our arena, other processes on the device move the baseline
// around, and the CUDA context itself is counted in neither. This is a drift alarm, not a precision
// instrument: it fires when an operator's allocations change shape, which is exactly when the model
// needs a human.

using namespace vector_gpu;

namespace {

//! Model may under-predict by this factor before we complain (Thrust's own scratch is not in our arena).
constexpr double kLowerRatio = 0.40;
//! ...and may over-predict by this much. Over-prediction is the safe direction, but a wild overestimate
//! declines plans that would have fit, so it is still worth catching.
constexpr double kUpperRatio = 4.0;

template <typename T>
GpuColumn MakeColumn(const std::string &name, GpuValueType type, const std::vector<T> &values,
                     std::vector<std::vector<uint8_t>> &owned) {
	owned.emplace_back(values.size() * sizeof(T));
	if (!values.empty()) {
		std::memcpy(owned.back().data(), values.data(), values.size() * sizeof(T));
	}
	GpuColumn column;
	column.name = name;
	column.type = type;
	column.data = owned.back().empty() ? nullptr : owned.back().data();
	column.row_count = values.size();
	return column;
}

std::shared_ptr<GpuPlanNode> MakeScan(const std::vector<std::string> &columns, uint64_t rows, uint32_t width) {
	auto scan = std::make_shared<GpuPlanNode>();
	scan->op_type = GpuOpType::SCAN;
	scan->table = {"memory", "main", "t"};
	scan->column_names = columns;
	scan->estimated_rows = rows;
	scan->estimated_row_width_bytes = width;
	return scan;
}

GpuExpr MakeExpr(const std::string &body, const std::vector<std::string> &inputs, GpuValueType out_type) {
	GpuExpr expr;
	expr.generated_cuda_source = body;
	expr.input_columns = inputs;
	expr.output_type = out_type;
	return expr;
}

//! Runs `plan` while sampling device memory, and returns the peak bytes attributable to it.
//! Returns 0 if the sampler never saw a drop (a plan too small or too fast to observe).
uint64_t MeasurePeakBytes(const GpuPlanNode &plan, const std::vector<GpuColumn> &inputs, bool &executed) {
	// Touch the device first so the CUDA context exists and is NOT counted as part of the plan.
	cudaFree(nullptr);
	size_t baseline_free = 0, total = 0;
	cudaMemGetInfo(&baseline_free, &total);

	std::atomic<bool> running {true};
	std::atomic<uint64_t> min_free {static_cast<uint64_t>(baseline_free)};
	std::thread sampler([&] {
		while (running.load()) {
			size_t free_now = 0, total_now = 0;
			if (cudaMemGetInfo(&free_now, &total_now) == cudaSuccess) {
				auto observed = static_cast<uint64_t>(free_now);
				auto current = min_free.load();
				while (observed < current && !min_free.compare_exchange_weak(current, observed)) {
				}
			}
			std::this_thread::sleep_for(std::chrono::microseconds(500));
		}
	});

	auto result = GpuEngine::Instance().ExecutePlan(plan, inputs);
	running = false;
	sampler.join();
	executed = result.success;

	auto low = min_free.load();
	return low >= baseline_free ? 0 : static_cast<uint64_t>(baseline_free) - low;
}

void ReportAndCheck(const char *label, const GpuPlanNode &plan, const std::vector<GpuColumn> &inputs) {
	bool executed = false;
	auto measured = MeasurePeakBytes(plan, inputs, executed);
	auto predicted = EstimateWorkingSet(plan, /* max_chunk_rows = */ 0).total_bytes;

	CHECK(executed == true);
	if (!executed) {
		return;
	}
	printf("    %-22s predicted %6llu MiB   measured %6llu MiB   ratio %.2f\n", label,
	       (unsigned long long)(predicted / (1024 * 1024)), (unsigned long long)(measured / (1024 * 1024)),
	       measured == 0 ? 0.0 : double(predicted) / double(measured));

	if (measured == 0) {
		// Too small or too fast to observe -- assert only that the model produced something sane.
		CHECK(predicted > 0);
		return;
	}
	auto ratio = double(predicted) / double(measured);
	CHECK(ratio >= kLowerRatio);
	CHECK(ratio <= kUpperRatio);
}

} // namespace

static void TestProjectionWorkingSetMatchesDevice() {
	const uint64_t n = 4u * 1000u * 1000u;
	std::vector<double> v(n, 1.5);
	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("v", GpuValueType::FLOAT64, v, owned)};

	auto projection = std::make_shared<GpuPlanNode>();
	projection->op_type = GpuOpType::PROJECTION;
	projection->expressions.push_back(
	    MakeExpr("((double*)out)[idx] = (((const double*)inputs[0])[row]) * 2.0;", {"v"}, GpuValueType::FLOAT64));
	projection->children.push_back(MakeScan({"v"}, n, 8));
	projection->estimated_rows = n;
	projection->estimated_row_width_bytes = 8;

	ReportAndCheck("projection", *projection, inputs);
}

static void TestGroupByWorkingSetMatchesDevice() {
	// The shape the old estimate got most wrong: tiny output, large per-input-row temporaries.
	const uint64_t n = 4u * 1000u * 1000u;
	std::vector<int32_t> k(n);
	std::vector<double> v(n, 2.0);
	for (uint64_t i = 0; i < n; i++) {
		k[i] = static_cast<int32_t>(i % 1000);
	}
	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("k", GpuValueType::INT32, k, owned),
	                               MakeColumn("v", GpuValueType::FLOAT64, v, owned)};

	GpuAggregate sum;
	sum.kind = GpuAggregateKind::SUM;
	sum.input_column = "v";
	sum.output_type = GpuValueType::FLOAT64;

	auto group_by = std::make_shared<GpuPlanNode>();
	group_by->op_type = GpuOpType::GROUP_BY_AGGREGATE;
	group_by->group_keys = {"k"};
	group_by->aggregates = {sum};
	group_by->children.push_back(MakeScan({"k", "v"}, n, 12));
	group_by->estimated_rows = 1000; // GROUPS, not input rows -- the model must not size the sort on this
	group_by->estimated_row_width_bytes = 12;

	ReportAndCheck("group_by", *group_by, inputs);
}

static void TestCrossProductWorkingSetMatchesDevice() {
	const uint64_t side = 2000;
	std::vector<double> a(side), b(side);
	for (uint64_t i = 0; i < side; i++) {
		a[i] = double(i);
		b[i] = double(i) * 0.5;
	}
	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("a", GpuValueType::FLOAT64, a, owned),
	                               MakeColumn("b", GpuValueType::FLOAT64, b, owned)};

	auto cross = std::make_shared<GpuPlanNode>();
	cross->op_type = GpuOpType::CROSS_PRODUCT;
	cross->output_columns = {"a", "b"};
	cross->children.push_back(MakeScan({"a"}, side, 8));
	cross->children.push_back(MakeScan({"b"}, side, 8));
	cross->estimated_rows = side * side;
	cross->estimated_row_width_bytes = 16;

	ReportAndCheck("cross_product", *cross, inputs);
}

static void TestChunkCapAppliesOnlyToRowIndependentPlans() {
	// A projection is chunkable, so a 50M-row plan must be sized against one 2M chunk...
	auto projection = std::make_shared<GpuPlanNode>();
	projection->op_type = GpuOpType::PROJECTION;
	projection->expressions.push_back(MakeExpr("x", {"v"}, GpuValueType::FLOAT64));
	projection->children.push_back(MakeScan({"v"}, 50000000, 8));
	projection->estimated_rows = 50000000;
	projection->estimated_row_width_bytes = 8;

	auto uncapped = EstimateWorkingSet(*projection, 0);
	auto capped = EstimateWorkingSet(*projection, 2u * 1024u * 1024u);
	CHECK(uncapped.chunk_capped == false);
	CHECK(capped.chunk_capped == true);
	CHECK(capped.total_bytes < uncapped.total_bytes / 10);

	// ...but the same input under a GROUP BY is not chunkable, so the cap must NOT apply. Sizing a
	// non-chunkable plan against a chunk is the failure mode this guards: routing would approve a plan
	// that then allocates for the entire input.
	auto group_by = std::make_shared<GpuPlanNode>();
	group_by->op_type = GpuOpType::GROUP_BY_AGGREGATE;
	group_by->group_keys = {"k"};
	group_by->children.push_back(MakeScan({"k", "v"}, 50000000, 12));
	group_by->estimated_rows = 10;
	group_by->estimated_row_width_bytes = 12;

	auto grouped = EstimateWorkingSet(*group_by, 2u * 1024u * 1024u);
	CHECK(grouped.chunk_capped == false);
	CHECK(IsRowIndependent(*group_by) == false);
	CHECK(IsRowIndependent(*projection) == true);
	// 50M input rows x 8 bytes x (4 + 1 key) buffers = ~2 GiB, none of which is visible from the
	// aggregate's own 10-row output estimate.
	CHECK(grouped.total_bytes > 2000ull * 1024 * 1024);
}

static void TestOversizedCrossProductSaturatesRatherThanWrapping() {
	// 10^10 pairs: the product overflows nothing at 64 bits, but the BYTE count would wrap a naive
	// multiply and read as "fits comfortably" -- the one direction this must never fail in.
	auto cross = std::make_shared<GpuPlanNode>();
	cross->op_type = GpuOpType::CROSS_PRODUCT;
	cross->children.push_back(MakeScan({"a"}, 100000, 8));
	cross->children.push_back(MakeScan({"b"}, 100000, 8));
	cross->estimated_rows = 10000000000ull;
	cross->estimated_row_width_bytes = 16;

	auto estimate = EstimateWorkingSet(*cross, 0);
	CHECK(estimate.total_bytes >= 10000000000ull * 16);
	CHECK(std::string(estimate.largest_operator) == "CROSS_PRODUCT");

	// And an estimate large enough to saturate must stay saturated, never wrap to something small.
	auto huge = std::make_shared<GpuPlanNode>();
	huge->op_type = GpuOpType::CROSS_PRODUCT;
	huge->children.push_back(MakeScan({"a"}, 1, 8));
	huge->children.push_back(MakeScan({"b"}, 1, 8));
	huge->estimated_rows = ~0ull;
	huge->estimated_row_width_bytes = 16;
	CHECK(EstimateWorkingSet(*huge, 0).total_bytes == ~0ull);
}

int main() {
	RUN_TEST(TestProjectionWorkingSetMatchesDevice);
	RUN_TEST(TestGroupByWorkingSetMatchesDevice);
	RUN_TEST(TestCrossProductWorkingSetMatchesDevice);
	RUN_TEST(TestChunkCapAppliesOnlyToRowIndependentPlans);
	RUN_TEST(TestOversizedCrossProductSaturatesRatherThanWrapping);
	TEST_MAIN_EPILOGUE();
}
