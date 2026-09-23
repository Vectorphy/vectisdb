// Real unit tests for extension/src/vector_converter.cpp -- constructs actual DuckDB DataChunks/Vectors
// directly (no live connection/query needed) and round-trips them through
// ConvertDataChunkToGpuColumns/ConvertGpuColumnToVector, checking values and null flags survive exactly.
//
// Build (same duckdb_static.lib linkage as test_translator_integration.cpp, but this one needs no
// cuda_engine/CUDA libs at all -- vector_converter.cpp has no CUDA dependency):
//   cl /nologo /std:c++17 /EHsc /MT /I ../include /I ../../cuda_engine/include
//      /I <duckdb>/src/include /I <duckdb>/third_party/fmt/include
//      ../src/vector_converter.cpp test_vector_converter.cpp
//      /Fe:test_vector_converter.exe
//      /link <duckdb>/build_core/src/duckdb_static.lib
//            <duckdb>/build_core/extension/dummy_static_extension_loader.lib
//            <duckdb>/build_core/extension/duckdb_generated_extension_loader.lib
//            <duckdb>/build_core/extension/core_functions/core_functions_extension.lib
//            <duckdb>/build_core/extension/parquet/parquet_extension.lib
//            Rstrtmgr.lib

#include "vector_converter.hpp"

#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/vector/constant_vector.hpp"

#include <cstring>
#include <iostream>
#include <string>

using namespace duckdb;
using namespace vector_gpu;

namespace {

int g_checks = 0;
int g_failures = 0;

void Check(bool condition, const std::string &description) {
	g_checks++;
	if (!condition) {
		g_failures++;
		std::cerr << "[FAIL] " << description << "\n";
	} else {
		std::cerr << "[PASS] " << description << "\n";
	}
}

template <class Fn>
bool Throws(Fn &&fn) {
	try {
		fn();
	} catch (const GpuUnsupportedVector &) {
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

//! Builds a synthetic INT32 GpuColumn directly (no DataChunk/scan involved) -- used to test the
//! result-packing path (ConvertGpuColumnRangeToVector/PackGpuColumnsIntoCollection) in isolation, the way
//! a (currently hypothetical, since GpuEngine::ExecutePlan's op dispatch is still a stub) successful
//! GpuExecutionResult would arrive.
GpuColumn MakeSyntheticInt32Column(const std::string &name, const std::vector<int32_t> &values,
                                   const std::vector<bool> &nulls, std::vector<std::vector<uint8_t>> &owned) {
	GpuColumn column;
	column.name = name;
	column.type = GpuValueType::INT32;
	column.row_count = values.size();

	std::vector<uint8_t> data_bytes(values.size() * sizeof(int32_t));
	std::memcpy(data_bytes.data(), values.data(), data_bytes.size());
	owned.push_back(std::move(data_bytes));
	column.data = owned.back().data();

	if (!nulls.empty()) {
		std::vector<uint8_t> packed((values.size() + 7) / 8, 0xFF);
		for (size_t i = 0; i < nulls.size(); i++) {
			if (nulls[i]) {
				packed[i / 8] &= ~(uint8_t(1) << (i % 8));
			}
		}
		owned.push_back(std::move(packed));
		column.validity = owned.back().data();
	}
	return column;
}

} // namespace

static void TestRoundTripIntegerNoNulls() {
	auto chunk = MakeChunk({LogicalType::INTEGER}, 5);
	auto *data = FlatVector::GetDataMutable<int32_t>(chunk->data[0]);
	for (int i = 0; i < 5; i++) {
		data[i] = i * 10;
	}

	std::vector<std::vector<uint8_t>> owned;
	auto columns = ConvertDataChunkToGpuColumns(*chunk, {"a"}, owned);
	Check(columns.size() == 1, "one column produced");
	Check(columns[0].type == GpuValueType::INT32, "INTEGER maps to INT32");
	Check(columns[0].row_count == 5, "row count preserved");
	Check(columns[0].validity == nullptr, "no-nulls column has a null validity pointer");

	auto out_chunk = MakeChunk({LogicalType::INTEGER}, 5);
	ConvertGpuColumnToVector(columns[0], 5, out_chunk->data[0]);
	auto *out_data = FlatVector::GetData<int32_t>(out_chunk->data[0]);
	bool all_match = true;
	for (int i = 0; i < 5; i++) {
		if (out_data[i] != i * 10) {
			all_match = false;
		}
	}
	Check(all_match, "INTEGER values round-trip exactly");
}

static void TestRoundTripWithNulls() {
	auto chunk = MakeChunk({LogicalType::DOUBLE}, 6);
	auto *data = FlatVector::GetDataMutable<double>(chunk->data[0]);
	for (int i = 0; i < 6; i++) {
		data[i] = i * 1.5;
	}
	FlatVector::SetNull(chunk->data[0], 1, true);
	FlatVector::SetNull(chunk->data[0], 4, true);

	std::vector<std::vector<uint8_t>> owned;
	auto columns = ConvertDataChunkToGpuColumns(*chunk, {"d"}, owned);
	Check(columns[0].validity != nullptr, "column with nulls has a non-null validity pointer");

	auto out_chunk = MakeChunk({LogicalType::DOUBLE}, 6);
	ConvertGpuColumnToVector(columns[0], 6, out_chunk->data[0]);
	auto &out_validity = FlatVector::Validity(out_chunk->data[0]);
	Check(!out_validity.RowIsValid(1), "row 1 is null after round-trip");
	Check(!out_validity.RowIsValid(4), "row 4 is null after round-trip");
	Check(out_validity.RowIsValid(0) && out_validity.RowIsValid(2) && out_validity.RowIsValid(3) &&
	          out_validity.RowIsValid(5),
	      "non-null rows stayed valid after round-trip");
	auto *out_data = FlatVector::GetData<double>(out_chunk->data[0]);
	Check(out_data[0] == 0.0 && out_data[2] == 3.0 && out_data[5] == 7.5,
	      "non-null DOUBLE values round-trip exactly");
}

static void TestAllNullColumn() {
	auto chunk = MakeChunk({LogicalType::BIGINT}, 4);
	for (idx_t i = 0; i < 4; i++) {
		FlatVector::SetNull(chunk->data[0], i, true);
	}
	std::vector<std::vector<uint8_t>> owned;
	auto columns = ConvertDataChunkToGpuColumns(*chunk, {"x"}, owned);
	Check(columns[0].validity != nullptr, "all-null column still has a validity pointer");

	auto out_chunk = MakeChunk({LogicalType::BIGINT}, 4);
	ConvertGpuColumnToVector(columns[0], 4, out_chunk->data[0]);
	auto &out_validity = FlatVector::Validity(out_chunk->data[0]);
	bool all_null = true;
	for (idx_t i = 0; i < 4; i++) {
		if (out_validity.RowIsValid(i)) {
			all_null = false;
		}
	}
	Check(all_null, "all-null column round-trips as all-null");
}

static void TestEmptyChunk() {
	auto chunk = MakeChunk({LogicalType::FLOAT}, 0);
	std::vector<std::vector<uint8_t>> owned;
	auto columns = ConvertDataChunkToGpuColumns(*chunk, {"f"}, owned);
	Check(columns.size() == 1, "empty chunk still produces one (empty) column");
	Check(columns[0].row_count == 0, "empty column has row_count 0");

	auto out_chunk = MakeChunk({LogicalType::FLOAT}, 0);
	// Must not throw / crash on a zero-row round trip.
	bool threw = false;
	try {
		ConvertGpuColumnToVector(columns[0], 0, out_chunk->data[0]);
	} catch (...) {
		threw = true;
	}
	Check(!threw, "zero-row round trip doesn't throw");
}

static void TestAllSupportedTypesInOneChunk() {
	auto chunk = MakeChunk({LogicalType::INTEGER, LogicalType::BIGINT, LogicalType::FLOAT, LogicalType::DOUBLE,
	                       LogicalType::BOOLEAN},
	                      3);
	FlatVector::GetDataMutable<int32_t>(chunk->data[0])[0] = 1;
	FlatVector::GetDataMutable<int32_t>(chunk->data[0])[1] = 2;
	FlatVector::GetDataMutable<int32_t>(chunk->data[0])[2] = 3;
	FlatVector::GetDataMutable<int64_t>(chunk->data[1])[0] = 100000000000LL;
	FlatVector::GetDataMutable<float>(chunk->data[2])[0] = 1.5f;
	FlatVector::GetDataMutable<double>(chunk->data[3])[0] = 2.5;
	FlatVector::GetDataMutable<bool>(chunk->data[4])[0] = true;
	FlatVector::GetDataMutable<bool>(chunk->data[4])[1] = false;

	std::vector<std::vector<uint8_t>> owned;
	auto columns = ConvertDataChunkToGpuColumns(*chunk, {"i", "bi", "f", "d", "b"}, owned);
	Check(columns.size() == 5, "all 5 columns converted");
	Check(columns[0].type == GpuValueType::INT32 && columns[1].type == GpuValueType::INT64 &&
	          columns[2].type == GpuValueType::FLOAT32 && columns[3].type == GpuValueType::FLOAT64 &&
	          columns[4].type == GpuValueType::BOOLEAN,
	      "all 5 types mapped correctly");

	auto out_chunk = MakeChunk(
	    {LogicalType::INTEGER, LogicalType::BIGINT, LogicalType::FLOAT, LogicalType::DOUBLE, LogicalType::BOOLEAN}, 3);
	for (idx_t i = 0; i < 5; i++) {
		ConvertGpuColumnToVector(columns[i], 3, out_chunk->data[i]);
	}
	Check(FlatVector::GetData<int64_t>(out_chunk->data[1])[0] == 100000000000LL, "BIGINT round-trips exactly");
	Check(FlatVector::GetData<float>(out_chunk->data[2])[0] == 1.5f, "FLOAT round-trips exactly");
	Check(FlatVector::GetData<bool>(out_chunk->data[4])[0] == true &&
	          FlatVector::GetData<bool>(out_chunk->data[4])[1] == false,
	      "BOOLEAN round-trips exactly");
}

static void TestConstantVectorIsFlattenedFirst() {
	// A constant vector (e.g. produced by a scalar sub-expression or a literal broadcast across a chunk)
	// must be transparently expanded to `count` copies -- the GPU side has no concept of DuckDB's
	// constant-vector encoding.
	auto chunk = MakeChunk({LogicalType::INTEGER}, 4);
	chunk->data[0].SetVectorType(VectorType::CONSTANT_VECTOR);
	*ConstantVector::GetData<int32_t>(chunk->data[0]) = 42;

	std::vector<std::vector<uint8_t>> owned;
	auto columns = ConvertDataChunkToGpuColumns(*chunk, {"c"}, owned);
	Check(columns[0].row_count == 4, "constant vector expands to full row count");

	auto *expanded = static_cast<const int32_t *>(columns[0].data);
	bool all_42 = expanded[0] == 42 && expanded[1] == 42 && expanded[2] == 42 && expanded[3] == 42;
	Check(all_42, "constant vector's single value is correctly broadcast to all rows");
}

static void TestUnsupportedTypeThrows() {
	auto chunk = MakeChunk({LogicalType::VARCHAR}, 2);
	std::vector<std::vector<uint8_t>> owned;
	Check(Throws([&] { ConvertDataChunkToGpuColumns(*chunk, {"s"}, owned); }),
	      "VARCHAR column throws GpuUnsupportedVector");
}

static void TestNamesSizeMismatchThrows() {
	auto chunk = MakeChunk({LogicalType::INTEGER, LogicalType::INTEGER}, 2);
	std::vector<std::vector<uint8_t>> owned;
	Check(Throws([&] { ConvertDataChunkToGpuColumns(*chunk, {"only_one"}, owned); }),
	      "mismatched names.size() vs. ColumnCount() throws");
}

static void TestWriteBackTypeMismatchThrows() {
	auto chunk = MakeChunk({LogicalType::INTEGER}, 2);
	FlatVector::GetDataMutable<int32_t>(chunk->data[0])[0] = 1;
	FlatVector::GetDataMutable<int32_t>(chunk->data[0])[1] = 2;
	std::vector<std::vector<uint8_t>> owned;
	auto columns = ConvertDataChunkToGpuColumns(*chunk, {"i"}, owned);

	auto out_chunk = MakeChunk({LogicalType::DOUBLE}, 2); // wrong type on purpose
	Check(Throws([&] { ConvertGpuColumnToVector(columns[0], 2, out_chunk->data[0]); }),
	      "writing an INT32 column into a DOUBLE target vector throws");
}

static void TestWriteBackRowCountMismatchThrows() {
	auto chunk = MakeChunk({LogicalType::INTEGER}, 3);
	std::vector<std::vector<uint8_t>> owned;
	auto columns = ConvertDataChunkToGpuColumns(*chunk, {"i"}, owned);

	auto out_chunk = MakeChunk({LogicalType::INTEGER}, 5); // wrong count on purpose
	Check(Throws([&] { ConvertGpuColumnToVector(columns[0], 5, out_chunk->data[0]); }),
	      "row_count mismatch between GpuColumn and requested count throws");
}

static void TestAccumulatorMultiChunkWithMidByteBoundaryNulls() {
	// The whole reason GpuColumnAccumulator exists: appending a 5-row chunk then a 3-row chunk means the
	// second chunk starts at global row 5 -- NOT a byte boundary in the final packed bitmap (byte 0 covers
	// rows 0-7). If null-bit packing were naively done per chunk and concatenated, a null in the second
	// chunk would land in the wrong bit position. Accumulating unpacked and packing once at the end (the
	// actual design) avoids this by construction; this test would have caught it if it hadn't.
	auto accumulators = MakeGpuColumnAccumulators({"a"}, {LogicalType::INTEGER});

	auto chunk1 = MakeChunk({LogicalType::INTEGER}, 5);
	for (int i = 0; i < 5; i++) {
		FlatVector::GetDataMutable<int32_t>(chunk1->data[0])[i] = i;
	}
	FlatVector::SetNull(chunk1->data[0], 4, true); // global row 4 is null
	AppendChunkToAccumulators(*chunk1, accumulators);

	auto chunk2 = MakeChunk({LogicalType::INTEGER}, 3);
	for (int i = 0; i < 3; i++) {
		FlatVector::GetDataMutable<int32_t>(chunk2->data[0])[i] = 100 + i;
	}
	FlatVector::SetNull(chunk2->data[0], 1, true); // global row 5+1=6 is null
	AppendChunkToAccumulators(*chunk2, accumulators);

	std::vector<std::vector<uint8_t>> owned;
	auto columns = FinalizeGpuColumnAccumulators(accumulators, owned);
	Check(columns[0].row_count == 8, "accumulated row count is 5+3=8 across two chunks");

	auto *data = static_cast<const int32_t *>(columns[0].data);
	bool values_correct = data[0] == 0 && data[3] == 3 && data[5] == 100 && data[7] == 102;
	Check(values_correct, "values from both chunks land at the correct global row offsets");

	auto BitIsSet = [](const uint8_t *bitmap, idx_t row) {
		return (bitmap[row / 8] & (uint8_t(1) << (row % 8))) != 0;
	};
	Check(columns[0].validity != nullptr, "cross-chunk accumulation with nulls produces a validity pointer");
	bool nulls_correct = !BitIsSet(columns[0].validity, 4) && !BitIsSet(columns[0].validity, 6) &&
	                     BitIsSet(columns[0].validity, 0) && BitIsSet(columns[0].validity, 5) &&
	                     BitIsSet(columns[0].validity, 7);
	Check(nulls_correct, "null at global row 4 (chunk 1) and row 6 (chunk 2, non-byte-aligned start) both "
	                     "land at the correct bit after packing");
}

static void TestAccumulatorEmptyChunkIsNoop() {
	auto accumulators = MakeGpuColumnAccumulators({"a"}, {LogicalType::INTEGER});
	auto chunk = MakeChunk({LogicalType::INTEGER}, 0);
	AppendChunkToAccumulators(*chunk, accumulators); // must not throw or corrupt state
	auto real_chunk = MakeChunk({LogicalType::INTEGER}, 2);
	FlatVector::GetDataMutable<int32_t>(real_chunk->data[0])[0] = 7;
	FlatVector::GetDataMutable<int32_t>(real_chunk->data[0])[1] = 8;
	AppendChunkToAccumulators(*real_chunk, accumulators);

	std::vector<std::vector<uint8_t>> owned;
	auto columns = FinalizeGpuColumnAccumulators(accumulators, owned);
	Check(columns[0].row_count == 2, "an empty chunk appended before real data contributes 0 rows, not a crash");
	auto *data = static_cast<const int32_t *>(columns[0].data);
	Check(data[0] == 7 && data[1] == 8, "real data after an empty-chunk append is still correct");
}

static void TestAccumulatorNoNullsProducesNullValidityPointer() {
	auto accumulators = MakeGpuColumnAccumulators({"a"}, {LogicalType::INTEGER});
	auto chunk = MakeChunk({LogicalType::INTEGER}, 4);
	for (int i = 0; i < 4; i++) {
		FlatVector::GetDataMutable<int32_t>(chunk->data[0])[i] = i;
	}
	AppendChunkToAccumulators(*chunk, accumulators);
	std::vector<std::vector<uint8_t>> owned;
	auto columns = FinalizeGpuColumnAccumulators(accumulators, owned);
	Check(columns[0].validity == nullptr, "no nulls across any appended chunk -> validity pointer stays null");
}

// ============================================================================
// ConvertGpuColumnRangeToVector -- read-side slicing, the counterpart used by PackGpuColumnsIntoCollection
// ============================================================================

static void TestRangeToVectorMiddleSlice() {
	std::vector<std::vector<uint8_t>> owned;
	auto column = MakeSyntheticInt32Column("v", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {}, owned);

	auto out_chunk = MakeChunk({LogicalType::INTEGER}, 4);
	ConvertGpuColumnRangeToVector(column, 3, 4, out_chunk->data[0]); // rows [3,7)
	auto *data = FlatVector::GetData<int32_t>(out_chunk->data[0]);
	Check(data[0] == 3 && data[1] == 4 && data[2] == 5 && data[3] == 6, "middle slice [3,7) reads the correct values");
}

static void TestRangeToVectorExactEnd() {
	std::vector<std::vector<uint8_t>> owned;
	auto column = MakeSyntheticInt32Column("v", {10, 20, 30}, {}, owned);
	auto out_chunk = MakeChunk({LogicalType::INTEGER}, 1);
	ConvertGpuColumnRangeToVector(column, 2, 1, out_chunk->data[0]); // last row exactly
	Check(FlatVector::GetData<int32_t>(out_chunk->data[0])[0] == 30, "range ending exactly at row_count reads correctly");
}

static void TestRangeToVectorOutOfBoundsThrows() {
	std::vector<std::vector<uint8_t>> owned;
	auto column = MakeSyntheticInt32Column("v", {1, 2, 3}, {}, owned);
	auto out_chunk = MakeChunk({LogicalType::INTEGER}, 2);
	Check(Throws([&] { ConvertGpuColumnRangeToVector(column, 2, 2, out_chunk->data[0]); }), // [2,4) exceeds row_count=3
	      "range exceeding column row_count throws");
}

static void TestRangeToVectorNonByteAlignedOffsetWithNulls() {
	// Offset 5 is not a validity-byte boundary (byte 0 covers rows 0-7) -- this is exactly the case that
	// would silently read the wrong null bits if BitIsSet(bitmap, i) were used instead of
	// BitIsSet(bitmap, offset + i).
	std::vector<bool> nulls(12, false);
	nulls[5] = true;
	nulls[9] = true;
	std::vector<std::vector<uint8_t>> owned;
	auto column = MakeSyntheticInt32Column("v", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, nulls, owned);

	auto out_chunk = MakeChunk({LogicalType::INTEGER}, 5);
	ConvertGpuColumnRangeToVector(column, 5, 5, out_chunk->data[0]); // rows [5,10) -> local indices 0..4
	auto &validity = FlatVector::Validity(out_chunk->data[0]);
	Check(!validity.RowIsValid(0), "global row 5 (local index 0 of the slice) is null");
	Check(!validity.RowIsValid(4), "global row 9 (local index 4 of the slice) is null");
	Check(validity.RowIsValid(1) && validity.RowIsValid(2) && validity.RowIsValid(3),
	      "global rows 6,7,8 (local indices 1-3) are valid");
}

// ============================================================================
// PackGpuColumnsIntoCollection -- the result path GpuEngine::ExecutePlan's (future) success case will use
// ============================================================================

static void TestPackIntoCollectionMultiChunk() {
	// 5000 rows: spans more than two STANDARD_VECTOR_SIZE (2048) output chunks, so PackGpuColumnsIntoCollection's
	// own chunking loop is genuinely exercised, not just a single-batch case.
	std::vector<int32_t> a_values(5000), b_values(5000);
	for (int i = 0; i < 5000; i++) {
		a_values[i] = i;
		b_values[i] = i % 7;
	}
	std::vector<std::vector<uint8_t>> owned;
	auto col_a = MakeSyntheticInt32Column("a", a_values, {}, owned);
	auto col_b = MakeSyntheticInt32Column("b", b_values, {}, owned);

	ColumnDataCollection collection(Allocator::DefaultAllocator(),
	                                vector<LogicalType> {LogicalType::INTEGER, LogicalType::INTEGER});
	PackGpuColumnsIntoCollection({col_a, col_b}, {LogicalType::INTEGER, LogicalType::INTEGER}, collection);

	Check(collection.Count() == 5000, "collection contains all 5000 packed rows");

	ColumnDataScanState scan_state;
	collection.InitializeScan(scan_state);
	DataChunk out_chunk;
	out_chunk.Initialize(Allocator::DefaultAllocator(), {LogicalType::INTEGER, LogicalType::INTEGER});

	idx_t seen = 0;
	bool all_correct = true;
	while (collection.Scan(scan_state, out_chunk)) {
		auto *a_data = FlatVector::GetData<int32_t>(out_chunk.data[0]);
		auto *b_data = FlatVector::GetData<int32_t>(out_chunk.data[1]);
		for (idx_t i = 0; i < out_chunk.size(); i++) {
			auto global_row = seen + i;
			if (a_data[i] != static_cast<int32_t>(global_row) || b_data[i] != static_cast<int32_t>(global_row % 7)) {
				all_correct = false;
			}
		}
		seen += out_chunk.size();
	}
	Check(seen == 5000, "scanning the packed collection back out yields all 5000 rows");
	Check(all_correct, "every row scanned back out of the collection matches the original GpuColumn values");
}

static void TestPackIntoCollectionWithNulls() {
	std::vector<int32_t> values(10);
	std::vector<bool> nulls(10, false);
	for (int i = 0; i < 10; i++) {
		values[i] = i;
	}
	nulls[0] = true;
	nulls[9] = true;
	std::vector<std::vector<uint8_t>> owned;
	auto column = MakeSyntheticInt32Column("n", values, nulls, owned);

	ColumnDataCollection collection(Allocator::DefaultAllocator(), vector<LogicalType> {LogicalType::INTEGER});
	PackGpuColumnsIntoCollection({column}, {LogicalType::INTEGER}, collection);

	ColumnDataScanState scan_state;
	collection.InitializeScan(scan_state);
	DataChunk out_chunk;
	out_chunk.Initialize(Allocator::DefaultAllocator(), {LogicalType::INTEGER});
	Check(collection.Scan(scan_state, out_chunk), "single-chunk collection scan returns data");
	auto &validity = FlatVector::Validity(out_chunk.data[0]);
	Check(!validity.RowIsValid(0) && !validity.RowIsValid(9), "nulls at rows 0 and 9 survive packing into the collection");
	Check(validity.RowIsValid(1) && validity.RowIsValid(8), "non-null rows stay valid after packing");
}

static void TestPackIntoCollectionEmptyIsNoop() {
	ColumnDataCollection collection(Allocator::DefaultAllocator(), vector<LogicalType> {LogicalType::INTEGER});
	std::vector<GpuColumn> empty_columns; // an empty result (0 output columns) -- degenerate but valid
	PackGpuColumnsIntoCollection(empty_columns, {}, collection);
	Check(collection.Count() == 0, "packing zero columns leaves the collection empty, doesn't crash");
}

static void TestPackIntoCollectionMismatchedRowCountThrows() {
	std::vector<std::vector<uint8_t>> owned;
	auto col_a = MakeSyntheticInt32Column("a", {1, 2, 3}, {}, owned);
	auto col_b = MakeSyntheticInt32Column("b", {1, 2}, {}, owned); // deliberately shorter
	ColumnDataCollection collection(Allocator::DefaultAllocator(),
	                                vector<LogicalType> {LogicalType::INTEGER, LogicalType::INTEGER});
	Check(Throws([&] {
		      PackGpuColumnsIntoCollection({col_a, col_b}, {LogicalType::INTEGER, LogicalType::INTEGER}, collection);
	      }),
	      "columns disagreeing on row_count throws rather than silently truncating");
}

static void TestPackIntoCollectionSizeMismatchThrows() {
	std::vector<std::vector<uint8_t>> owned;
	auto col_a = MakeSyntheticInt32Column("a", {1, 2, 3}, {}, owned);
	ColumnDataCollection collection(Allocator::DefaultAllocator(), vector<LogicalType> {LogicalType::INTEGER});
	Check(Throws([&] { PackGpuColumnsIntoCollection({col_a}, {LogicalType::INTEGER, LogicalType::INTEGER}, collection); }),
	      "columns.size() != types.size() throws");
}

int main() {
	TestRoundTripIntegerNoNulls();
	TestRoundTripWithNulls();
	TestAllNullColumn();
	TestEmptyChunk();
	TestAllSupportedTypesInOneChunk();
	TestConstantVectorIsFlattenedFirst();
	TestUnsupportedTypeThrows();
	TestNamesSizeMismatchThrows();
	TestWriteBackTypeMismatchThrows();
	TestWriteBackRowCountMismatchThrows();
	TestAccumulatorMultiChunkWithMidByteBoundaryNulls();
	TestAccumulatorEmptyChunkIsNoop();
	TestAccumulatorNoNullsProducesNullValidityPointer();
	TestRangeToVectorMiddleSlice();
	TestRangeToVectorExactEnd();
	TestRangeToVectorOutOfBoundsThrows();
	TestRangeToVectorNonByteAlignedOffsetWithNulls();
	TestPackIntoCollectionMultiChunk();
	TestPackIntoCollectionWithNulls();
	TestPackIntoCollectionEmptyIsNoop();
	TestPackIntoCollectionMismatchedRowCountThrows();
	TestPackIntoCollectionSizeMismatchThrows();

	std::cout << "\n" << (g_checks - g_failures) << "/" << g_checks << " checks passed\n";
	return g_failures == 0 ? 0 : 1;
}
