// test_batch_accumulator.cpp -- proves GpuBatchAccumulator/GpuMemoryPool against REAL DuckDB DataChunks
// and REAL pinned host memory (not hand-built GpuColumns the way cuda_engine's own tests use, and not
// SQL strings the way test_host_api.cpp uses -- this constructs DataChunks directly, the same technique
// extension/tests/test_vector_converter.cpp uses for vector_converter.cpp, since that is what proves the
// DataChunk-flattening logic rather than assuming ToUnifiedFormat behaves as documented).
//
// Needs a real GPU: GpuMemoryPool::EnsureInitialized pins real device-visible host memory.

#include "gpu_batch_accumulator.hpp"

#include "duckdb/common/vector/constant_vector.hpp"
#include "duckdb/common/vector/flat_vector.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace duckdb;
using namespace vector_gpu;

namespace {

int g_checks = 0;
int g_failures = 0;

void Check(bool condition, const std::string &description) {
	g_checks++;
	std::printf("  %s  %s\n", condition ? "PASS" : "FAIL", description.c_str());
	std::fflush(stdout);
	if (!condition) {
		g_failures++;
	}
}

template <class Fn>
bool Throws(Fn &&fn) {
	try {
		fn();
	} catch (const GpuBatchAccumulatorError &) {
		return true;
	}
	return false;
}

unique_ptr<DataChunk> MakeChunk(vector<LogicalType> types, idx_t count) {
	auto chunk = make_uniq<DataChunk>();
	chunk->Initialize(Allocator::DefaultAllocator(), types);
	chunk->SetCardinality(count);
	return chunk;
}

//! True iff every byte `ptr[0..len)` falls within `[slot_data, slot_data + slot_capacity)` -- the
//! structural proof that a GpuColumn's data/validity pointer aliases INTO the pinned slot rather than
//! into some separate (heap) allocation.
bool WithinSlot(const void *ptr, size_t len, const void *slot_data, size_t slot_capacity) {
	auto *p = static_cast<const uint8_t *>(ptr);
	auto *base = static_cast<const uint8_t *>(slot_data);
	return p >= base && (p + len) <= (base + slot_capacity);
}

} // namespace

static void TestAppendReturnsReadyExactlyAtThreshold() {
	GpuBatchAccumulator acc({"v"}, {LogicalType::INTEGER});
	const idx_t per_chunk = 2048;
	bool ready = false;
	idx_t chunks_appended = 0;
	while (!ready) {
		auto chunk = MakeChunk({LogicalType::INTEGER}, per_chunk);
		for (idx_t i = 0; i < per_chunk; i++) {
			FlatVector::GetDataMutable<int32_t>(chunk->data[0])[i] = static_cast<int32_t>(i);
		}
		ready = acc.Append(*chunk, nullptr, chunk->size());
		chunks_appended++;
		Check(acc.row_count() == chunks_appended * per_chunk, "row_count tracks rows appended so far");
	}
	Check(chunks_appended * per_chunk == GpuBatchAccumulator::kBatchReadyRows,
	      "ready fired at exactly kBatchReadyRows, i.e. after exactly 16 chunks");
	Check(acc.HasReadyBatch(), "HasReadyBatch reflects the true return from Append");
}

static void TestValuesAndConstantVectorSurviveAcrossChunks() {
	// Constant-vector broadcast is the exact bug vector_converter.hpp's header documents:
	// Vector::Flatten() would under-read a constant vector's internal buffer Size() (1), not the
	// chunk's real row count. ToUnifiedFormat is what this accumulator relies on instead.
	GpuMemoryPool::Instance().ResetCursorForTesting();
	GpuBatchAccumulator acc({"flat", "k"}, {LogicalType::DOUBLE, LogicalType::INTEGER});

	std::vector<double> expected_flat;
	std::vector<int32_t> expected_k;
	for (idx_t round = 0; round < GpuBatchAccumulator::kBatchReadyRows / 2048; round++) {
		auto chunk = MakeChunk({LogicalType::DOUBLE, LogicalType::INTEGER}, 2048);
		for (idx_t i = 0; i < 2048; i++) {
			double v = static_cast<double>(round) * 1000.0 + static_cast<double>(i);
			FlatVector::GetDataMutable<double>(chunk->data[0])[i] = v;
			expected_flat.push_back(v);
		}
		// Column 1 is a genuine CONSTANT vector for this whole chunk, not a flat one written per-row.
		chunk->data[1].SetVectorType(VectorType::CONSTANT_VECTOR);
		*ConstantVector::GetData<int32_t>(chunk->data[1]) = static_cast<int32_t>(round);
		for (idx_t i = 0; i < 2048; i++) {
			expected_k.push_back(static_cast<int32_t>(round));
		}
		acc.Append(*chunk, nullptr, chunk->size());
	}
	Check(acc.HasReadyBatch(), "batch reached the ready threshold");
	auto batch = acc.TakeReadyBatch();
	Check(batch.row_count == GpuBatchAccumulator::kBatchReadyRows, "taken batch reports the full row count");
	Check(batch.columns.size() == 2, "one GpuColumn per accumulated column");

	auto *flat_data = static_cast<const double *>(batch.columns[0].data);
	bool flat_ok = true;
	for (idx_t i = 0; i < batch.row_count; i++) {
		if (flat_data[i] != expected_flat[i]) {
			flat_ok = false;
			break;
		}
	}
	Check(flat_ok, "FLAT column values round-trip exactly across every chunk");

	auto *k_data = static_cast<const int32_t *>(batch.columns[1].data);
	bool k_ok = true;
	for (idx_t i = 0; i < batch.row_count; i++) {
		if (k_data[i] != expected_k[i]) {
			k_ok = false;
			break;
		}
	}
	Check(k_ok, "CONSTANT-vector column broadcasts its single value to every row of its chunk, not just row 0");
}

static void TestNoNullsMeansNullValidityPointer() {
	GpuBatchAccumulator acc({"v"}, {LogicalType::DOUBLE});
	for (idx_t round = 0; round < GpuBatchAccumulator::kBatchReadyRows / 2048; round++) {
		auto chunk = MakeChunk({LogicalType::DOUBLE}, 2048);
		for (idx_t i = 0; i < 2048; i++) {
			FlatVector::GetDataMutable<double>(chunk->data[0])[i] = 1.0;
		}
		acc.Append(*chunk, nullptr, chunk->size());
	}
	auto batch = acc.TakeReadyBatch();
	Check(batch.columns[0].validity == nullptr,
	      "a column that never saw a NULL leaves GpuColumn::validity nullptr (matches its own documented "
	      "convention: nullptr means every row is valid)");
}

static void TestNullsPackCorrectlyAcrossChunkBoundary() {
	// Null rows chosen to straddle a chunk boundary (2047, 2048) and land on a non-byte-aligned bit
	// position (100) within a chunk -- exactly the cases the packed-bitmap format could get wrong.
	GpuBatchAccumulator acc({"v"}, {LogicalType::INTEGER});
	std::vector<bool> expected_null(GpuBatchAccumulator::kBatchReadyRows, false);
	for (idx_t round = 0; round < GpuBatchAccumulator::kBatchReadyRows / 2048; round++) {
		auto chunk = MakeChunk({LogicalType::INTEGER}, 2048);
		for (idx_t i = 0; i < 2048; i++) {
			FlatVector::GetDataMutable<int32_t>(chunk->data[0])[i] = static_cast<int32_t>(i);
		}
		if (round == 0) {
			FlatVector::SetNull(chunk->data[0], 100, true);
			FlatVector::SetNull(chunk->data[0], 2047, true); // last row of chunk 0
			expected_null[100] = true;
			expected_null[2047] = true;
		}
		if (round == 1) {
			FlatVector::SetNull(chunk->data[0], 0, true); // first row of chunk 1 -> global row 2048
			expected_null[2048] = true;
		}
		acc.Append(*chunk, nullptr, chunk->size());
	}
	auto batch = acc.TakeReadyBatch();
	Check(batch.columns[0].validity != nullptr, "a column that saw a NULL gets a real validity pointer");
	bool packing_ok = true;
	for (idx_t r = 0; r < batch.row_count; r++) {
		bool bit_valid = (batch.columns[0].validity[r / 8] & (uint8_t(1) << (r % 8))) != 0;
		if (bit_valid == expected_null[r]) { // bit set means VALID, so this should never match "is null"
			packing_ok = false;
			break;
		}
	}
	Check(packing_ok, "packed validity bitmap marks exactly the NULL rows, correct across a chunk boundary "
	                  "and at a non-byte-aligned bit position");
}

static void TestEveryPointerAliasesIntoTheSlotNotAHeapBuffer() {
	// The acceptance bar this whole class exists for: no malloc/std::vector allocation backs data that
	// crosses the PCIe bus. Every GpuColumn's data/validity pointer must fall inside the returned slot's
	// own [data(), data()+capacity()) range.
	GpuBatchAccumulator acc({"a", "b", "c"},
	                        {LogicalType::INTEGER, LogicalType::DOUBLE, LogicalType::BOOLEAN});
	for (idx_t round = 0; round < GpuBatchAccumulator::kBatchReadyRows / 2048; round++) {
		auto chunk = MakeChunk({LogicalType::INTEGER, LogicalType::DOUBLE, LogicalType::BOOLEAN}, 2048);
		for (idx_t i = 0; i < 2048; i++) {
			FlatVector::GetDataMutable<int32_t>(chunk->data[0])[i] = static_cast<int32_t>(i);
			FlatVector::GetDataMutable<double>(chunk->data[1])[i] = static_cast<double>(i);
			FlatVector::GetDataMutable<bool>(chunk->data[2])[i] = (i % 2) == 0;
		}
		FlatVector::SetNull(chunk->data[1], 5, true); // force a real validity region for column b too
		acc.Append(*chunk, nullptr, chunk->size());
	}
	auto batch = acc.TakeReadyBatch();
	auto *slot_data = batch.slot.data();
	auto capacity = batch.slot.capacity();
	Check(WithinSlot(batch.columns[0].data, batch.row_count * sizeof(int32_t), slot_data, capacity),
	      "column a's dense data aliases into the pinned slot");
	Check(WithinSlot(batch.columns[1].data, batch.row_count * sizeof(double), slot_data, capacity),
	      "column b's dense data aliases into the pinned slot");
	Check(WithinSlot(batch.columns[2].data, batch.row_count * sizeof(bool), slot_data, capacity),
	      "column c's dense data aliases into the pinned slot");
	Check(batch.columns[1].validity != nullptr &&
	          WithinSlot(batch.columns[1].validity, (batch.row_count + 7) / 8, slot_data, capacity),
	      "column b's packed validity bitmap also aliases into the pinned slot");
}

static void TestTakeReadyBatchCyclesToADifferentSlot() {
	GpuMemoryPool::Instance().ResetCursorForTesting();
	GpuBatchAccumulator acc({"v"}, {LogicalType::INTEGER});
	std::vector<void *> slot_pointers;
	for (int b = 0; b < 3; b++) {
		for (idx_t round = 0; round < GpuBatchAccumulator::kBatchReadyRows / 2048; round++) {
			auto chunk = MakeChunk({LogicalType::INTEGER}, 2048);
			acc.Append(*chunk, nullptr, chunk->size());
		}
		auto batch = acc.TakeReadyBatch();
		slot_pointers.push_back(batch.slot.data());
		// `batch.slot` goes out of scope at the end of this iteration, releasing it back to the ring --
		// otherwise the 4th slot request two iterations from now would block forever (only 3 exist).
	}
	Check(slot_pointers[0] != slot_pointers[1] && slot_pointers[1] != slot_pointers[2] &&
	          slot_pointers[0] != slot_pointers[2],
	      "three consecutive TakeReadyBatch calls cycle through three DISTINCT physical slots");
}

static void TestPartialBatchCanBeTakenEarly() {
	GpuBatchAccumulator acc({"v"}, {LogicalType::INTEGER});
	auto chunk = MakeChunk({LogicalType::INTEGER}, 2048);
	bool ready = acc.Append(*chunk, nullptr, chunk->size());
	Check(!ready, "one chunk alone does not reach the 32768-row threshold");
	Check(acc.row_count() == 2048, "row_count reflects the partial batch before it is taken");
	auto batch = acc.TakeReadyBatch(); // simulates flushing the final partial batch when a scan ends
	Check(batch.row_count == 2048, "TakeReadyBatch honors a partial row count when called before threshold");
	Check(acc.row_count() == 0, "the accumulator resets to an empty batch after TakeReadyBatch");
	Check(!acc.HasReadyBatch(), "HasReadyBatch is false immediately after taking a batch");
}

static void TestAppendAfterReadyThrows() {
	GpuBatchAccumulator acc({"v"}, {LogicalType::INTEGER});
	bool ready = false;
	for (idx_t round = 0; round < GpuBatchAccumulator::kBatchReadyRows / 2048 && !ready; round++) {
		auto chunk = MakeChunk({LogicalType::INTEGER}, 2048);
		ready = acc.Append(*chunk, nullptr, chunk->size());
	}
	Check(ready, "setup: batch reached the threshold");
	auto extra = MakeChunk({LogicalType::INTEGER}, 2048);
	Check(Throws([&] { acc.Append(*extra, nullptr, extra->size()); }),
	      "Append refuses to run again before TakeReadyBatch collects the ready batch");
}

static void TestMismatchedColumnCountThrows() {
	GpuBatchAccumulator acc({"a", "b"}, {LogicalType::INTEGER, LogicalType::DOUBLE});
	auto chunk = MakeChunk({LogicalType::INTEGER}, 100); // only one column, accumulator expects two
	Check(Throws([&] { acc.Append(*chunk, nullptr, chunk->size()); }), "Append rejects a chunk whose column count doesn't match");
}

static void TestUnsupportedTypeRejectedAtConstruction() {
	Check(Throws([] { GpuBatchAccumulator acc({"s"}, {LogicalType::VARCHAR}); }),
	      "an unsupported column type is refused at construction, before any scanning could start");
}

static void TestOversizedColumnSetRejectedAtConstruction() {
	// 1500 DOUBLE columns * kBatchReadyRows rows exceeds one 350 MiB slot -- must fail cleanly at
	// construction (before ever touching GpuMemoryPool) rather than silently overrunning a slot later.
	std::vector<std::string> names;
	std::vector<LogicalType> types;
	for (int i = 0; i < 1500; i++) {
		names.push_back("c" + std::to_string(i));
		types.push_back(LogicalType::DOUBLE);
	}
	Check(Throws([&] { GpuBatchAccumulator acc(names, types); }),
	      "a column set whose per-batch footprint exceeds one GpuMemoryPool slot is refused at construction");
}

static void TestConcurrentAppends() {
	GpuBatchAccumulator acc({"id"}, {LogicalType::INTEGER});
	constexpr int kNumThreads = 4;
	constexpr idx_t kChunksPerThread = 4;
	constexpr idx_t kChunkSize = 2048;
	// 4 threads * 4 chunks * 2048 rows = 32,768 rows = kBatchReadyRows
	std::vector<std::thread> threads;
	for (int t = 0; t < kNumThreads; t++) {
		threads.emplace_back([&acc, t, kChunksPerThread, kChunkSize]() {
			for (idx_t c = 0; c < kChunksPerThread; c++) {
				auto chunk = MakeChunk({LogicalType::INTEGER}, kChunkSize);
				auto *data = FlatVector::GetDataMutable<int32_t>(chunk->data[0]);
				idx_t base = (t * kChunksPerThread + c) * kChunkSize;
				for (idx_t i = 0; i < kChunkSize; i++) {
					data[i] = static_cast<int32_t>(base + i);
				}
				acc.Append(*chunk, nullptr, kChunkSize);
			}
		});
	}
	for (auto &th : threads) {
		th.join();
	}

	Check(acc.HasReadyBatch(), "accumulator reached ready state with concurrent appends");
	auto batch = acc.TakeReadyBatch();
	Check(batch.row_count == GpuBatchAccumulator::kBatchReadyRows, "batch row count is exactly kBatchReadyRows");

	const auto *data = reinterpret_cast<const int32_t *>(batch.columns[0].data);
	std::vector<bool> seen(GpuBatchAccumulator::kBatchReadyRows, false);
	bool all_seen_once = true;
	for (idx_t i = 0; i < batch.row_count; i++) {
		int32_t v = data[i];
		if (v < 0 || static_cast<idx_t>(v) >= GpuBatchAccumulator::kBatchReadyRows || seen[v]) {
			all_seen_once = false;
			break;
		}
		seen[v] = true;
	}
	Check(all_seen_once, "all 32,768 integers from 4 concurrent threads were accumulated without loss or corruption");
}

int main() {
	std::setvbuf(stdout, nullptr, _IONBF, 0);
	std::printf("== GpuBatchAccumulator / GpuMemoryPool, against real DataChunks and real pinned memory ==\n");
	try {
		TestAppendReturnsReadyExactlyAtThreshold();
		TestValuesAndConstantVectorSurviveAcrossChunks();
		TestNoNullsMeansNullValidityPointer();
		TestNullsPackCorrectlyAcrossChunkBoundary();
		TestEveryPointerAliasesIntoTheSlotNotAHeapBuffer();
		TestTakeReadyBatchCyclesToADifferentSlot();
		TestPartialBatchCanBeTakenEarly();
		TestAppendAfterReadyThrows();
		TestMismatchedColumnCountThrows();
		TestUnsupportedTypeRejectedAtConstruction();
		TestOversizedColumnSetRejectedAtConstruction();
		TestConcurrentAppends();
	} catch (const std::exception &e) {
		std::fprintf(stderr, "FATAL EXCEPTION in main: %s\n", e.what());
		return 1;
	} catch (...) {
		std::fprintf(stderr, "UNKNOWN FATAL EXCEPTION in main\n");
		return 1;
	}
	std::printf("\n%d/%d checks passed\n", g_checks - g_failures, g_checks);
	return g_failures == 0 ? 0 : 1;
}

