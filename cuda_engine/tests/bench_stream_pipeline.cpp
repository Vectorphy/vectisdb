// bench_stream_pipeline.cpp
//
// Answers ONE question that no unit test can: do GpuStreamPipeline's three CUDA streams actually overlap
// on the device, on this machine? It is the reproducible form of the acceptance check for the operator --
// run it under `nsys` (gpu_shell/bench_stream_pipeline.sh does exactly that and summarises the result).
//
// WHY IT IS A BENCHMARK AND NOT A TEST. Overlap is a scheduling property of the driver and the card, not
// a property of this code that can be asserted. On the GTX 1650 this was developed against, an assertion
// on it would be measuring the Windows/WDDM command-submission path as much as anything here.
//
// WHY IT DOES NO HOST WORK PER BATCH. Overlap can only happen when the device is the bottleneck. Fed by a
// real DuckDB scan, it is not: PhysicalGpuStreamingProjection measures the host scan at roughly 11 ms per
// million rows while the whole GPU round trip for those rows is around 6 ms, so the pipeline is always
// waiting on the producer and never queues deep enough for two batches to be on the device at once. That
// is a true and important fact about the engine -- the host scan is the bottleneck, exactly as
// docs/SESSION_30_BENCHMARK_RESULTS.md found -- but it means an end-to-end query cannot answer whether the
// pipeline overlaps. So this bench fills each ring slot ONCE and then resubmits the bytes already sitting
// in it, which makes the producer free and lets the device be the constraint.

#include "gpu_memory_pool.hpp"
#include "gpu_stream_pipeline.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace vector_gpu;

namespace {

constexpr uint64_t kRowsPerBatch = 1048576;
constexpr int kBatches = 60;

std::shared_ptr<GpuPlanNode> MakePlan() {
	auto scan = std::make_shared<GpuPlanNode>();
	scan->op_type = GpuOpType::SCAN;
	scan->table = {"memory", "main", "t"};
	scan->column_names = {"a", "b"};

	auto projection = std::make_shared<GpuPlanNode>();
	projection->op_type = GpuOpType::PROJECTION;
	projection->children.push_back(scan);
	GpuExpr expr;
	// The transcendental-heavy shape docs/SESSION_32_CACHE_BENCHMARK.md identified as the one that beats
	// the CPU -- i.e. the shape worth pipelining. A trivial `a + b` kernel finishes in microseconds and
	// would leave the copies with nothing to overlap.
	expr.generated_cuda_source = "((double*)out)[idx] = sin(((const double*)inputs[0])[row]) + "
	                             "cos(((const double*)inputs[1])[row]) + "
	                             "sqrt(fabs(((const double*)inputs[0])[row])) + "
	                             "log(fabs(((const double*)inputs[0])[row])+1.0) + "
	                             "exp(((const double*)inputs[1])[row]/100.0);";
	expr.input_columns = {"a", "b"};
	expr.output_type = GpuValueType::FLOAT64;
	projection->expressions.push_back(expr);
	return projection;
}

} // namespace

int main() {
	auto plan = MakePlan();
	GpuStreamPipeline pipeline(*plan, {"a", "b"}, {GpuValueType::FLOAT64, GpuValueType::FLOAT64}, kRowsPerBatch);

	std::vector<double> values(kRowsPerBatch);
	for (uint64_t i = 0; i < kRowsPerBatch; i++) {
		values[i] = static_cast<double>(i) * 0.5;
	}

	// Slots whose contents have already been written once. After the first rotation every slot is in here,
	// and from then on a "batch" costs nothing on the host but the pointer bookkeeping below.
	std::vector<const void *> filled;
	auto producer = GpuMemoryPool::Instance().AcquireNext();
	int submitted = 0;
	size_t deepest = 0;
	GpuStreamResult result;

	auto start = std::chrono::steady_clock::now();
	while (submitted < kBatches || pipeline.in_flight() > 0) {
		if (submitted < kBatches && pipeline.CanSubmit()) {
			auto *base = static_cast<uint8_t *>(producer.data());
			bool already_filled = false;
			for (auto *seen : filled) {
				already_filled = already_filled || seen == producer.data();
			}
			if (!already_filled) {
				std::memcpy(base, values.data(), kRowsPerBatch * sizeof(double));
				std::memcpy(base + kRowsPerBatch * sizeof(double), values.data(), kRowsPerBatch * sizeof(double));
				filled.push_back(producer.data());
			}

			GpuStreamBatch batch;
			batch.row_count = kRowsPerBatch;
			GpuColumn a;
			a.name = "a";
			a.type = GpuValueType::FLOAT64;
			a.data = base;
			a.row_count = kRowsPerBatch;
			GpuColumn b;
			b.name = "b";
			b.type = GpuValueType::FLOAT64;
			b.data = base + kRowsPerBatch * sizeof(double);
			b.row_count = kRowsPerBatch;
			batch.columns.push_back(std::move(a));
			batch.columns.push_back(std::move(b));
			// Hand over the slot, then immediately take the next in the rotation -- the producer discipline
			// GpuBatchAccumulator::TakeReadyBatch follows, and the one GpuMemoryPool's cyclic AcquireNext
			// requires of anything that draws from it.
			batch.slot = std::move(producer);
			pipeline.Submit(std::move(batch));
			producer = GpuMemoryPool::Instance().AcquireNext();
			submitted++;
			deepest = pipeline.in_flight() > deepest ? pipeline.in_flight() : deepest;
			continue;
		}
		if (!pipeline.TryTake(result) && !pipeline.TakeBlocking(result)) {
			break;
		}
		result.Reset();
	}
	auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start)
	                      .count();

	std::printf("batches=%d rows/batch=%llu deepest_in_flight=%zu wall=%lld ms\n", submitted,
	            (unsigned long long)kRowsPerBatch, deepest, (long long)elapsed_ms);
	std::printf("run under nsys to see the overlap: gpu_shell/bench_stream_pipeline.sh\n");
	return submitted == kBatches ? 0 : 1;
}
