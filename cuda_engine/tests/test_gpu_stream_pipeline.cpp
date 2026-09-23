#include "test_framework.hpp"

#include "gpu_memory_pool.hpp"
#include "gpu_stream_pipeline.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// Proves GpuStreamPipeline end to end against a REAL GPU and REAL pinned ring slots: batches uploaded on
// stream_h2d, fused NVRTC kernels on stream_exec gated by an event, results downloaded on stream_d2h gated
// by another, and handed back FIFO into pinned host memory.
//
// What this file can and cannot prove. It CAN prove the results are bit-identical to a CPU reference for
// every batch, that batches come back in submission order, that NULLs propagate, that the pinned input
// slot is handed back to GpuMemoryPool early enough for a producer to keep filling, and that more than one
// batch is genuinely in flight at once. It CANNOT prove the three stages physically overlap on the device
// -- that is a profiler question, not an assertion, and the answer lives in an `nsys` timeline (see
// gpu_shell/bench_stream_pipeline.sh). A test that claimed otherwise from wall-clock timing would be
// measuring scheduler noise on a shared dev machine.
//
// Requires a real GPU.

using namespace vector_gpu;

namespace {

constexpr uint64_t kMaxRows = 8192;

std::shared_ptr<GpuPlanNode> MakeScan(const std::vector<std::string> &columns) {
	auto scan = std::make_shared<GpuPlanNode>();
	scan->op_type = GpuOpType::SCAN;
	scan->table = {"memory", "main", "t"};
	scan->column_names = columns;
	return scan;
}

GpuExpr MakeExpr(const std::string &body, const std::vector<std::string> &inputs, GpuValueType out_type) {
	GpuExpr expr;
	expr.generated_cuda_source = body;
	expr.input_columns = inputs;
	expr.output_type = out_type;
	return expr;
}

//! SCAN(a, b) -> PROJECTION(a * 2 + b, a - b), the shape the pipeline exists for. Input reads use [row]
//! and the output write uses [idx], exactly as expression_translator.cpp emits them.
std::shared_ptr<GpuPlanNode> MakeProjectionPlan() {
	auto projection = std::make_shared<GpuPlanNode>();
	projection->op_type = GpuOpType::PROJECTION;
	projection->children.push_back(MakeScan({"a", "b"}));
	projection->expressions.push_back(
	    MakeExpr("((double*)out)[idx] = ((const double*)inputs[0])[row] * 2.0 + ((const double*)inputs[1])[row];",
	             {"a", "b"}, GpuValueType::FLOAT64));
	projection->expressions.push_back(
	    MakeExpr("((double*)out)[idx] = ((const double*)inputs[0])[row] - ((const double*)inputs[1])[row];",
	             {"a", "b"}, GpuValueType::FLOAT64));
	return projection;
}

//! PROJECTION(col0 * 3, col1) over PROJECTION(a + b, a - b) over SCAN(a, b) -- the two-level shape a real
//! DuckDB plan produces (a narrowing/computed projection over the scan, another above it). The upper
//! level's expressions reference the lower level's outputs by the "col<i>" names ExecuteProjection and
//! CollectBindingNames both use.
std::shared_ptr<GpuPlanNode> MakeChainedProjectionPlan() {
	auto inner = std::make_shared<GpuPlanNode>();
	inner->op_type = GpuOpType::PROJECTION;
	inner->children.push_back(MakeScan({"a", "b"}));
	inner->expressions.push_back(
	    MakeExpr("((double*)out)[idx] = ((const double*)inputs[0])[row] + ((const double*)inputs[1])[row];",
	             {"a", "b"}, GpuValueType::FLOAT64));
	inner->expressions.push_back(
	    MakeExpr("((double*)out)[idx] = ((const double*)inputs[0])[row] - ((const double*)inputs[1])[row];",
	             {"a", "b"}, GpuValueType::FLOAT64));

	auto outer = std::make_shared<GpuPlanNode>();
	outer->op_type = GpuOpType::PROJECTION;
	outer->children.push_back(inner);
	outer->expressions.push_back(
	    MakeExpr("((double*)out)[idx] = ((const double*)inputs[0])[row] * 3.0;", {"col0"}, GpuValueType::FLOAT64));
	outer->expressions.push_back(
	    MakeExpr("((double*)out)[idx] = ((const double*)inputs[0])[row];", {"col1"}, GpuValueType::FLOAT64));
	return outer;
}

//! Lays two FLOAT64 columns (and, optionally, their packed validity bitmaps) out inside the caller's
//! already-acquired pinned ring slot and hands back the batch that describes them -- the same thing
//! GpuBatchAccumulator does for real DataChunks, reduced to what this test needs. The GpuColumn pointers
//! ALIAS INTO the slot; nothing here is a separate heap buffer, which is the property that makes the H2D a
//! genuine pinned-source transfer.
//!
//! The slot comes IN rather than being acquired here, because GpuMemoryPool hands slots out in a fixed
//! CYCLIC rotation (0, 1, 2, 0, ...) and blocks on one specific index -- so a caller must follow the
//! producer discipline GpuBatchAccumulator::TakeReadyBatch uses (hand over the slot you were filling, then
//! immediately acquire the next one) or it will eventually wait on a slot the rotation has not reached.
GpuStreamBatch MakeBatch(GpuMemoryPoolSlot slot, const std::vector<double> &a, const std::vector<double> &b,
                         const std::vector<uint8_t> *a_validity = nullptr) {
	GpuStreamBatch batch;
	batch.slot = std::move(slot);
	batch.row_count = a.size();

	auto *base = static_cast<uint8_t *>(batch.slot.data());
	size_t offset = 0;
	auto data_bytes = a.size() * sizeof(double);
	std::memcpy(base + offset, a.data(), data_bytes);
	GpuColumn column_a;
	column_a.name = "a";
	column_a.type = GpuValueType::FLOAT64;
	column_a.data = base + offset;
	column_a.row_count = a.size();
	offset += data_bytes;

	std::memcpy(base + offset, b.data(), data_bytes);
	GpuColumn column_b;
	column_b.name = "b";
	column_b.type = GpuValueType::FLOAT64;
	column_b.data = base + offset;
	column_b.row_count = b.size();
	offset += data_bytes;

	if (a_validity != nullptr) {
		std::memcpy(base + offset, a_validity->data(), a_validity->size());
		column_a.validity = base + offset;
		offset += a_validity->size();
	}

	batch.columns.push_back(std::move(column_a));
	batch.columns.push_back(std::move(column_b));
	return batch;
}

std::vector<double> ReadDoubles(const GpuColumn &column) {
	std::vector<double> values(column.row_count);
	if (column.row_count > 0) {
		std::memcpy(values.data(), column.data, column.row_count * sizeof(double));
	}
	return values;
}

bool ValidAt(const GpuColumn &column, uint64_t row) {
	if (column.validity == nullptr) {
		return true;
	}
	return (column.validity[row / 8] & (uint8_t(1) << (row % 8))) != 0;
}

} // namespace

static void TestSupportsAcceptsOnlyProjectionOverScan() {
	CHECK(GpuStreamPipeline::Supports(*MakeProjectionPlan()));

	// A bare SCAN computes nothing to pipeline.
	CHECK(!GpuStreamPipeline::Supports(*MakeScan({"a"})));

	// A FILTER's output row count is not known until the kernel has run, which is exactly what this
	// pipeline's pre-sized, pre-issued D2H cannot accommodate.
	auto filter = std::make_shared<GpuPlanNode>();
	filter->op_type = GpuOpType::FILTER;
	filter->children.push_back(MakeScan({"a"}));
	filter->expressions.push_back(
	    MakeExpr("((bool*)out)[idx] = ((const double*)inputs[0])[row] > 0.0;", {"a"}, GpuValueType::BOOLEAN));
	CHECK(!GpuStreamPipeline::Supports(*filter));

	// PROJECTION over FILTER over SCAN: a FILTER anywhere in the chain disqualifies it, even though every
	// individual operator here is supported by the engine at large.
	auto projection_over_filter = std::make_shared<GpuPlanNode>();
	projection_over_filter->op_type = GpuOpType::PROJECTION;
	projection_over_filter->children.push_back(filter);
	projection_over_filter->expressions.push_back(
	    MakeExpr("((double*)out)[idx] = ((const double*)inputs[0])[row];", {"a"}, GpuValueType::FLOAT64));
	CHECK(!GpuStreamPipeline::Supports(*projection_over_filter));

	// A chain of projections over a SCAN, on the other hand, is the common real-plan shape and IS run here.
	CHECK(GpuStreamPipeline::Supports(*MakeChainedProjectionPlan()));

	// An expression referencing a column the SCAN does not produce would resolve to nothing at launch
	// time; refuse it while the caller can still pick another operator.
	auto unresolvable = std::make_shared<GpuPlanNode>();
	unresolvable->op_type = GpuOpType::PROJECTION;
	unresolvable->children.push_back(MakeScan({"a"}));
	unresolvable->expressions.push_back(
	    MakeExpr("((double*)out)[idx] = ((const double*)inputs[0])[row];", {"missing"}, GpuValueType::FLOAT64));
	CHECK(!GpuStreamPipeline::Supports(*unresolvable));
}

static void TestConstructorRejectsMismatchedSchema() {
	auto plan = MakeProjectionPlan();
	// Declared schema missing a column the projection references: caught at construction, before any row
	// is consumed, which is the whole point of resolving the mapping up front.
	CHECK_THROWS(GpuStreamPipeline(*plan, {"a"}, {GpuValueType::FLOAT64}, kMaxRows));
	CHECK_THROWS(GpuStreamPipeline(*plan, {"a", "b"}, {GpuValueType::FLOAT64}, kMaxRows));
	CHECK_THROWS(GpuStreamPipeline(*plan, {"a", "b"}, {GpuValueType::FLOAT64, GpuValueType::FLOAT64}, 0));
}

//! The real thing: push several batches through, drain FIFO, and compare every row against a CPU
//! reference. Also records the deepest the pipeline ever got, since a pipeline that never has more than
//! one batch in flight is not a pipeline.
static void TestStreamsManyBatchesCorrectlyAndOverlaps() {
	GpuMemoryPool::Instance().ResetCursorForTesting();
	auto plan = MakeProjectionPlan();
	GpuStreamPipeline pipeline(*plan, {"a", "b"}, {GpuValueType::FLOAT64, GpuValueType::FLOAT64}, kMaxRows);
	CHECK(pipeline.output_types().size() == 2);

	const int batch_count = 9;
	const uint64_t rows_per_batch = 5000;

	std::vector<double> expected_sum;
	std::vector<double> expected_diff;
	std::vector<double> actual_sum;
	std::vector<double> actual_diff;

	auto make_values = [&](int batch, std::vector<double> &a, std::vector<double> &b) {
		a.resize(rows_per_batch);
		b.resize(rows_per_batch);
		for (uint64_t i = 0; i < rows_per_batch; i++) {
			a[i] = static_cast<double>(batch) * 1000.0 + static_cast<double>(i) * 0.5;
			b[i] = static_cast<double>(i) * 0.25 - 3.0;
			expected_sum.push_back(a[i] * 2.0 + b[i]);
			expected_diff.push_back(a[i] - b[i]);
		}
	};

	auto collect = [&](const GpuStreamResult &result) {
		CHECK(result.columns().size() == 2);
		CHECK(result.columns()[0].name == "col0");
		CHECK(result.columns()[1].name == "col1");
		auto sum = ReadDoubles(result.columns()[0]);
		auto diff = ReadDoubles(result.columns()[1]);
		actual_sum.insert(actual_sum.end(), sum.begin(), sum.end());
		actual_diff.insert(actual_diff.end(), diff.begin(), diff.end());
	};

	int submitted = 0;
	size_t max_in_flight = 0;
	GpuStreamResult result;
	// Stands in for the accumulator's currently-filling slot: handed to Submit, then immediately replaced
	// with the next slot in the rotation. See MakeBatch for why the discipline matters.
	auto producer_slot = GpuMemoryPool::Instance().AcquireNext();
	while (submitted < batch_count || pipeline.in_flight() > 0) {
		if (submitted < batch_count && pipeline.CanSubmit()) {
			std::vector<double> a, b;
			make_values(submitted, a, b);
			pipeline.Submit(MakeBatch(std::move(producer_slot), a, b));
			producer_slot = GpuMemoryPool::Instance().AcquireNext();
			submitted++;
			max_in_flight = std::max(max_in_flight, pipeline.in_flight());
			continue;
		}
		// Prefer the non-blocking take -- this is the path a real consumer spends its time in, and it is
		// what keeps the calling thread free while the device works.
		if (!pipeline.TryTake(result) && !pipeline.TakeBlocking(result)) {
			break; // nothing in flight and nothing left to submit
		}
		collect(result);
		result.Reset(); // hand the stage back so the next Submit has one
	}

	CHECK(submitted == batch_count);
	CHECK(pipeline.in_flight() == 0);
	CHECK(actual_sum.size() == expected_sum.size());
	CHECK(actual_diff.size() == expected_diff.size());
	// More than one batch resident at once -- the property that separates a pipeline from a loop. Not a
	// claim about device-side overlap (see the file header), just that the queue genuinely runs deep.
	CHECK(max_in_flight >= 2);

	bool sums_match = actual_sum.size() == expected_sum.size();
	bool diffs_match = actual_diff.size() == expected_diff.size();
	for (size_t i = 0; sums_match && i < expected_sum.size(); i++) {
		// Bit-exact, not approximate: the fused kernel is compiled with --fmad=false precisely so its
		// arithmetic rounds the same way the host's does (see code_generator.cpp). A tolerance here would
		// hide the regression that flag exists to prevent.
		sums_match = actual_sum[i] == expected_sum[i];
	}
	for (size_t i = 0; diffs_match && i < expected_diff.size(); i++) {
		diffs_match = actual_diff[i] == expected_diff[i];
	}
	CHECK(sums_match);
	CHECK(diffs_match);
	// FIFO: the checks above only pass if every batch came back in submission order, since the reference
	// vectors were built in that order and each batch's values are distinct per batch index.
}

//! A batch whose input carries NULLs: the packed bitmap is uploaded, unpacked on stream_exec, threaded
//! through the fused kernel, and packed again host-side on the way out.
//! A two-level projection chain: the level above reads the level below's "col<i>" outputs, both levels run
//! on stream_exec in order, and only the top level is downloaded. This is the shape real DuckDB plans
//! actually reach this operator with, so it is not an exotic extra case -- it is the common one.
static void TestChainedProjectionLevels() {
	GpuMemoryPool::Instance().ResetCursorForTesting();
	auto plan = MakeChainedProjectionPlan();
	CHECK(GpuStreamPipeline::Supports(*plan));
	GpuStreamPipeline pipeline(*plan, {"a", "b"}, {GpuValueType::FLOAT64, GpuValueType::FLOAT64}, kMaxRows);
	CHECK(pipeline.output_types().size() == 2);

	const uint64_t rows = 4096;
	std::vector<double> a(rows), b(rows);
	for (uint64_t i = 0; i < rows; i++) {
		a[i] = static_cast<double>(i) * 0.125;
		b[i] = static_cast<double>(i % 17) - 8.0;
	}

	CHECK(pipeline.CanSubmit());
	pipeline.Submit(MakeBatch(GpuMemoryPool::Instance().AcquireNext(), a, b));
	GpuStreamResult result;
	CHECK(pipeline.TakeBlocking(result));
	CHECK(result.row_count() == rows);

	auto scaled = ReadDoubles(result.columns()[0]);
	auto passthrough = ReadDoubles(result.columns()[1]);
	bool matches = scaled.size() == rows && passthrough.size() == rows;
	for (uint64_t i = 0; matches && i < rows; i++) {
		matches = scaled[i] == (a[i] + b[i]) * 3.0 && passthrough[i] == a[i] - b[i];
	}
	CHECK(matches);
}

static void TestNullPropagation() {
	GpuMemoryPool::Instance().ResetCursorForTesting();
	auto plan = MakeProjectionPlan();
	GpuStreamPipeline pipeline(*plan, {"a", "b"}, {GpuValueType::FLOAT64, GpuValueType::FLOAT64}, kMaxRows);

	const uint64_t rows = 1000;
	std::vector<double> a(rows), b(rows);
	for (uint64_t i = 0; i < rows; i++) {
		a[i] = static_cast<double>(i);
		b[i] = static_cast<double>(i) * 0.5;
	}
	// Every 7th row of `a` is NULL; `b` has none, which exercises the mixed case (one column unpacked from
	// a bitmap, the other filled with "all valid" so the kernel can read both uniformly).
	std::vector<uint8_t> validity((rows + 7) / 8, 0xFFu);
	for (uint64_t i = 0; i < rows; i += 7) {
		validity[i / 8] &= ~(uint8_t(1) << (i % 8));
	}

	CHECK(pipeline.CanSubmit());
	pipeline.Submit(MakeBatch(GpuMemoryPool::Instance().AcquireNext(), a, b, &validity));
	GpuStreamResult result;
	CHECK(pipeline.TakeBlocking(result));
	CHECK(result.row_count() == rows);

	bool nulls_match = true;
	bool values_match = true;
	auto sum = ReadDoubles(result.columns()[0]);
	for (uint64_t i = 0; i < rows; i++) {
		bool expected_valid = (i % 7) != 0;
		if (ValidAt(result.columns()[0], i) != expected_valid) {
			nulls_match = false;
		}
		// A NULL row's VALUE is unspecified (the kernel still computes over whatever bytes were there), so
		// only non-null rows are compared -- same contract the executor's own null handling has.
		if (expected_valid && sum[i] != a[i] * 2.0 + b[i]) {
			values_match = false;
		}
	}
	CHECK(nulls_match);
	CHECK(values_match);
	// The second expression references the same nullable column, so its nullness must agree.
	CHECK(result.columns()[1].validity != nullptr);
}

//! The pinned ring slot a batch was uploaded from must come back to GpuMemoryPool once that batch's H2D
//! COMPLETES -- not when its results are read. Otherwise the producer, which needs a slot to fill the next
//! batch into, waits on a pool whose slots are all held by in-flight work, and nothing frees one until the
//! producer submits again. That is a deadlock, not a stall, which is why it gets its own test.
//!
//! The observable proof, without needing to expose the pipeline's slot bookkeeping: with two batches
//! submitted and NEITHER taken, CanSubmit() must become true -- and it only can if the pipeline gave a slot
//! back. The bounded poll loop is because "H2D queued" is not "H2D finished"; it fails the check rather
//! than hanging if the slot is never released.
static void TestInputSlotIsReturnedBeforeResultsAreRead() {
	GpuMemoryPool::Instance().ResetCursorForTesting();
	auto plan = MakeProjectionPlan();
	GpuStreamPipeline pipeline(*plan, {"a", "b"}, {GpuValueType::FLOAT64, GpuValueType::FLOAT64}, kMaxRows);

	const uint64_t rows = 512;
	std::vector<double> a(rows, 1.5), b(rows, 2.5);

	auto producer_slot = GpuMemoryPool::Instance().AcquireNext();
	CHECK(producer_slot.valid());
	CHECK(pipeline.CanSubmit());
	pipeline.Submit(MakeBatch(std::move(producer_slot), a, b));
	producer_slot = GpuMemoryPool::Instance().AcquireNext();
	CHECK(pipeline.CanSubmit());
	pipeline.Submit(MakeBatch(std::move(producer_slot), a, b));
	producer_slot = GpuMemoryPool::Instance().AcquireNext();
	CHECK(pipeline.in_flight() == 2);

	bool released_before_take = false;
	for (int i = 0; i < 1000000 && !released_before_take; i++) {
		released_before_take = pipeline.CanSubmit(); // polls internally
	}
	CHECK(released_before_take);
	CHECK(pipeline.in_flight() == 2); // still nothing taken -- that is the point

	// And the slot it freed is genuinely the next one in the pool's cyclic rotation: this acquire would
	// block forever if the pipeline had held on to it.
	pipeline.Submit(MakeBatch(std::move(producer_slot), a, b));
	producer_slot = GpuMemoryPool::Instance().AcquireNext();
	CHECK(producer_slot.valid());
	CHECK(pipeline.in_flight() == 3);

	GpuStreamResult result;
	while (pipeline.in_flight() > 0) {
		CHECK(pipeline.TakeBlocking(result));
		result.Reset();
	}
	CHECK(pipeline.in_flight() == 0);
}

static void TestSubmitRejectsBadBatches() {
	GpuMemoryPool::Instance().ResetCursorForTesting();
	auto plan = MakeProjectionPlan();
	GpuStreamPipeline pipeline(*plan, {"a", "b"}, {GpuValueType::FLOAT64, GpuValueType::FLOAT64}, kMaxRows);

	std::vector<double> a(16, 1.0), b(16, 2.0);
	// More rows than any stage buffer was allocated for: refused, rather than writing past the end of a
	// device buffer sized once at construction.
	auto oversized = MakeBatch(GpuMemoryPool::Instance().AcquireNext(), a, b);
	oversized.row_count = kMaxRows + 1;
	CHECK_THROWS(pipeline.Submit(std::move(oversized)));
}

int main() {
	RUN_TEST(TestSupportsAcceptsOnlyProjectionOverScan);
	RUN_TEST(TestConstructorRejectsMismatchedSchema);
	RUN_TEST(TestStreamsManyBatchesCorrectlyAndOverlaps);
	RUN_TEST(TestChainedProjectionLevels);
	RUN_TEST(TestNullPropagation);
	RUN_TEST(TestInputSlotIsReturnedBeforeResultsAreRead);
	RUN_TEST(TestSubmitRejectsBadBatches);
	TEST_MAIN_EPILOGUE();
}
