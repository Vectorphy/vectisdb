// bench_batch_accumulator.cpp — performance benchmark for GpuBatchAccumulator / GpuMemoryPool.
//
// Measures accumulator throughput (rows/s and MB/s) across three kMaxStreamBatchRows candidates
// and five execution modes, all bounded by a strict per-scenario wall-time budget so the total
// run stays within 10 seconds on an 8-core CPU.
//
// Batch sizes under test (mirrors the three kMaxStreamBatchRows values the routing layer controls):
//   SMALL  131072  rows  (~1.05 MB at 8 B/row with one DOUBLE column, fits one 64 MiB slot)
//   PREV   262144  rows  (~2.10 MB at 8 B/row, fits one 64 MiB slot)
//   SCALE  350224384 rows (target 2.80 GB VRAM scale -- too large for one 64 MiB slot, so the
//                         benchmark cycles slots to accumulate the same total row count in many
//                         max-fitting-slot-sized batches and reports aggregate throughput.)
//
// Execution modes (simulate the five GPU pipeline configurations without requiring a live kernel):
//   GPU_ONLY        Pure accumulate+take with a memory "consume" touch (no second thread)
//   PIPELINE        Producer thread writes into slots; consumer thread takes them concurrently
//   PIPELINE_CACHED Pipeline, but consumer warm-touches the slot (no re-zero between cycles)
//   NO_PIPE_CACHED  Sequential: accumulate+take, warm-touch slot between cycles
//   NO_PIPE_NO_CACHE Sequential: accumulate+take, memset slot to 0 between cycles (cold)
//
// Column set: one DOUBLE column.

#include "gpu_batch_accumulator.hpp"

#include "duckdb/common/vector/flat_vector.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace duckdb;
using namespace vector_gpu;
using SteadyClock = std::chrono::steady_clock;
using MsDuration  = std::chrono::duration<double, std::milli>;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Budget per (batch_size, mode) cell in the 3x5 matrix.
static constexpr double kBudgetMs = 600.0;

static constexpr idx_t kSmall       = 131072ULL;
static constexpr idx_t kPrev        = 262144ULL;
static constexpr idx_t kScaleTarget = 350224384ULL;

// API-facing std::vectors (GpuBatchAccumulator takes std::vector).
static const std::vector<std::string>      kStdNames = {"v"};
static const std::vector<duckdb::LogicalType> kStdTypes = {LogicalType::DOUBLE};

// DuckDB-flavoured vector for DataChunk::Initialize.
static const duckdb::vector<duckdb::LogicalType> kDdbTypes = {LogicalType::DOUBLE};

static constexpr idx_t kChunkRows = STANDARD_VECTOR_SIZE; // 2048

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

unique_ptr<DataChunk> MakeChunk(idx_t count) {
	auto chunk = make_uniq<DataChunk>();
	chunk->Initialize(Allocator::DefaultAllocator(), kDdbTypes);
	chunk->SetCardinality(count);
	auto *data = FlatVector::GetDataMutable<double>(chunk->data[0]);
	for (idx_t i = 0; i < count; i++) {
		data[i] = static_cast<double>(i);
	}
	return chunk;
}

// Touch every cache line in the slot (warm simulate: no zeroing).
void WarmTouch(const GpuPinnedBatch &batch) {
	volatile const uint8_t *p = static_cast<const uint8_t *>(batch.slot.data());
	size_t cap = batch.slot.capacity();
	for (size_t off = 0; off < cap; off += 64) {
		(void)p[off];
	}
}

// Zero the entire slot (cold simulate: evict from cache).
void ColdZero(const GpuPinnedBatch &batch) {
	std::memset(batch.slot.data(), 0, batch.slot.capacity());
}

// Clamp target_rows to whatever actually fits in one GpuMemoryPool slot.
idx_t SlotBatchRows(idx_t target_rows) {
	idx_t fits = GpuBatchAccumulator::RowsThatFitInOneSlot(kStdTypes);
	return (target_rows <= fits) ? target_rows : fits;
}

} // namespace

// ---------------------------------------------------------------------------
// Result record
// ---------------------------------------------------------------------------

struct BenchResult {
	const char *batch_label;
	const char *mode_label;
	idx_t       batch_rows;
	idx_t       target_rows;
	double      elapsed_ms;
	uint64_t    total_rows;
	uint64_t    total_batches;
	double      rows_per_sec;
	double      mb_per_sec;
};

// ---------------------------------------------------------------------------
// Mode implementations
// ---------------------------------------------------------------------------

// GPU_ONLY: single thread, accumulate+take+warm-touch, no pipelining.
BenchResult RunGpuOnly(const char *label, idx_t target_rows) {
	GpuMemoryPool::Instance().ResetCursorForTesting();
	idx_t batch_rows = SlotBatchRows(target_rows);
	GpuBatchAccumulator acc(kStdNames, kStdTypes, batch_rows);
	auto chunk = MakeChunk(kChunkRows);
	idx_t chunks_per_batch = batch_rows / kChunkRows;

	uint64_t total_rows = 0, total_batches = 0;
	auto t0 = SteadyClock::now();
	double elapsed_ms = 0.0;

	while (elapsed_ms < kBudgetMs) {
		for (idx_t c = 0; c < chunks_per_batch; c++) {
			acc.Append(*chunk, nullptr, chunk->size());
		}
		auto batch = acc.TakeReadyBatch();
		WarmTouch(batch);
		total_rows += batch.row_count;
		total_batches++;
		elapsed_ms = MsDuration(SteadyClock::now() - t0).count();
	}

	double rps = (elapsed_ms > 0) ? (double)total_rows / (elapsed_ms / 1000.0) : 0;
	return {label, "GPU_ONLY", batch_rows, target_rows, elapsed_ms,
	        total_rows, total_batches, rps, rps * 8.0 / (1024.0 * 1024.0)};
}

// PIPELINE / PIPELINE_CACHED: producer+consumer threads.
BenchResult RunPipeline(const char *label, idx_t target_rows, bool warm_cache) {
	GpuMemoryPool::Instance().ResetCursorForTesting();
	idx_t batch_rows = SlotBatchRows(target_rows);
	GpuBatchAccumulator acc(kStdNames, kStdTypes, batch_rows);
	auto chunk = MakeChunk(kChunkRows);
	idx_t chunks_per_batch = batch_rows / kChunkRows;

	std::mutex              mtx;
	std::condition_variable cv_ready, cv_consumed;
	std::vector<GpuPinnedBatch> ready_queue;
	bool producer_done = false;

	uint64_t total_rows = 0, total_batches = 0;
	auto t0 = SteadyClock::now();
	double elapsed_ms = 0.0;

	std::thread consumer([&] {
		while (true) {
			GpuPinnedBatch batch;
			{
				std::unique_lock<std::mutex> lock(mtx);
				cv_ready.wait(lock, [&] { return !ready_queue.empty() || producer_done; });
				if (ready_queue.empty()) {
					break;
				}
				batch = std::move(ready_queue.front());
				ready_queue.erase(ready_queue.begin());
			}
			cv_consumed.notify_one();
			if (warm_cache) {
				WarmTouch(batch);
			} else {
				ColdZero(batch);
			}
			// slot returns to ring on batch destruction.
		}
	});

	while (elapsed_ms < kBudgetMs) {
		for (idx_t c = 0; c < chunks_per_batch; c++) {
			acc.Append(*chunk, nullptr, chunk->size());
		}
		auto batch = acc.TakeReadyBatch();
		total_rows += batch.row_count;
		total_batches++;
		{
			std::unique_lock<std::mutex> lock(mtx);
			// Mirror GpuStreamPipeline's kDepth=3 in-flight limit.
			cv_consumed.wait(lock, [&] { return ready_queue.size() < 3; });
			ready_queue.push_back(std::move(batch));
		}
		cv_ready.notify_one();
		elapsed_ms = MsDuration(SteadyClock::now() - t0).count();
	}
	{
		std::unique_lock<std::mutex> lock(mtx);
		producer_done = true;
	}
	cv_ready.notify_one();
	consumer.join();

	double rps = (elapsed_ms > 0) ? (double)total_rows / (elapsed_ms / 1000.0) : 0;
	const char *mode = warm_cache ? "PIPELINE_CACHED" : "PIPELINE";
	return {label, mode, batch_rows, target_rows, elapsed_ms,
	        total_rows, total_batches, rps, rps * 8.0 / (1024.0 * 1024.0)};
}

// NO_PIPE_CACHED / NO_PIPE_NO_CACHE: single-threaded sequential.
BenchResult RunNoPipeline(const char *label, idx_t target_rows, bool warm_cache) {
	GpuMemoryPool::Instance().ResetCursorForTesting();
	idx_t batch_rows = SlotBatchRows(target_rows);
	GpuBatchAccumulator acc(kStdNames, kStdTypes, batch_rows);
	auto chunk = MakeChunk(kChunkRows);
	idx_t chunks_per_batch = batch_rows / kChunkRows;

	uint64_t total_rows = 0, total_batches = 0;
	auto t0 = SteadyClock::now();
	double elapsed_ms = 0.0;

	while (elapsed_ms < kBudgetMs) {
		for (idx_t c = 0; c < chunks_per_batch; c++) {
			acc.Append(*chunk, nullptr, chunk->size());
		}
		auto batch = acc.TakeReadyBatch();
		total_rows += batch.row_count;
		total_batches++;
		if (warm_cache) {
			WarmTouch(batch);
		} else {
			ColdZero(batch);
		}
		elapsed_ms = MsDuration(SteadyClock::now() - t0).count();
	}

	double rps = (elapsed_ms > 0) ? (double)total_rows / (elapsed_ms / 1000.0) : 0;
	const char *mode = warm_cache ? "NO_PIPE_CACHED" : "NO_PIPE_NO_CACHE";
	return {label, mode, batch_rows, target_rows, elapsed_ms,
	        total_rows, total_batches, rps, rps * 8.0 / (1024.0 * 1024.0)};
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

static void PrintResult(const BenchResult &r) {
	std::printf("  %-22s  %-18s  batch=%9llu  batches=%6llu  rows/s=%.3e  MB/s=%7.1f  t=%.0f ms\n",
	            r.batch_label, r.mode_label,
	            (unsigned long long)r.batch_rows,
	            (unsigned long long)r.total_batches,
	            r.rows_per_sec, r.mb_per_sec, r.elapsed_ms);
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int main() {
	std::printf("=== GpuBatchAccumulator Performance Benchmark ===\n");
	std::printf("    Budget/cell: %.0f ms  |  Column: 1xDOUBLE  |  Chunk: %llu rows\n\n",
	            kBudgetMs, (unsigned long long)kChunkRows);

	idx_t fits = GpuBatchAccumulator::RowsThatFitInOneSlot(kStdTypes);
	std::printf("    RowsThatFitInOneSlot (1xDOUBLE, 64 MiB slot): %llu rows\n", (unsigned long long)fits);
	std::printf("    SCALE target: %llu rows -> slot-limited to %llu rows/batch\n\n",
	            (unsigned long long)kScaleTarget, (unsigned long long)SlotBatchRows(kScaleTarget));

	if (kSmall > fits) {
		std::fprintf(stderr, "ABORT: SMALL=%llu does not fit in one slot (%llu rows max)\n",
		             (unsigned long long)kSmall, (unsigned long long)fits);
		return 1;
	}
	if (kPrev > fits) {
		std::fprintf(stderr, "ABORT: PREV=%llu does not fit in one slot (%llu rows max)\n",
		             (unsigned long long)kPrev, (unsigned long long)fits);
		return 1;
	}

	struct BatchSpec { const char *label; idx_t target; };
	static const BatchSpec kSpecs[] = {
	    {"SMALL(131072)",    kSmall},
	    {"PREV(262144)",     kPrev},
	    {"SCALE(350224384)", kScaleTarget},
	};

	std::vector<BenchResult> results;
	results.reserve(15);

	auto wall_start = SteadyClock::now();

	for (auto &spec : kSpecs) {
		std::printf("--- %s ---\n", spec.label);
		auto r0 = RunGpuOnly(spec.label, spec.target);        PrintResult(r0); results.push_back(r0);
		auto r1 = RunPipeline(spec.label, spec.target, false); PrintResult(r1); results.push_back(r1);
		auto r2 = RunPipeline(spec.label, spec.target, true);  PrintResult(r2); results.push_back(r2);
		auto r3 = RunNoPipeline(spec.label, spec.target, true); PrintResult(r3); results.push_back(r3);
		auto r4 = RunNoPipeline(spec.label, spec.target, false); PrintResult(r4); results.push_back(r4);
		std::printf("\n");
	}

	double wall_ms = MsDuration(SteadyClock::now() - wall_start).count();
	std::printf("=== Total wall time: %.0f ms (budget: 10000 ms) ===\n\n", wall_ms);

	// CSV output for further analysis.
	std::printf("batch_label,mode,batch_rows,target_rows,batches,rows_per_sec,MB_per_sec,elapsed_ms\n");
	for (auto &r : results) {
		std::printf("%s,%s,%llu,%llu,%llu,%.3e,%.1f,%.0f\n",
		            r.batch_label, r.mode_label,
		            (unsigned long long)r.batch_rows,
		            (unsigned long long)r.target_rows,
		            (unsigned long long)r.total_batches,
		            r.rows_per_sec, r.mb_per_sec, r.elapsed_ms);
	}

	return 0;
}
