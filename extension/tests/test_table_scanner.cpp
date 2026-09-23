// Real, direct test of extension/src/table_scanner.cpp: opens a real in-memory DuckDB, creates real
// tables via SQL, and calls ScanTableToGpuColumns directly (bypassing the optimizer entirely) to verify
// the scan+accumulate+convert pipeline produces exactly correct data -- including across chunk
// boundaries (STANDARD_VECTOR_SIZE = 2048 rows/chunk), which is the part that's easy to get subtly wrong
// (see vector_converter.hpp's GpuColumnAccumulator comments on why null-bit packing is done once at the
// end rather than per chunk).
//
// Build: same linkage as test_translator_integration.cpp (duckdb_static + real cuda_engine + CUDA libs --
// table_scanner.cpp itself has no CUDA dependency, but vector_converter.hpp/gpu_engine.hpp are shared
// with cuda_engine's build, so linking the same way avoids a second one-off configuration).

#include "duckdb.hpp"
#include "table_scanner.hpp"

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
	} catch (const std::exception &) {
		return true;
	}
	return false;
}

} // namespace

int main() {
	DuckDB db(nullptr);
	Connection con(db);

	// 5000 rows: comfortably spans more than two 2048-row chunks (5000 = 2048 + 2048 + 904), so the
	// accumulation loop's chunk-boundary handling is genuinely exercised, not just a single-chunk case.
	auto create_result = con.Query("CREATE TABLE t AS SELECT range::INTEGER AS a, (range % 7)::INTEGER AS b, "
	                               "range::DOUBLE AS d FROM range(5000)");
	Check(!create_result->HasError(), "create 5000-row table succeeds: " +
	                                      (create_result->HasError() ? create_result->GetError() : "ok"));

	std::vector<GpuColumn> columns;
	std::vector<std::vector<uint8_t>> owned_buffers;
	con.context->RunFunctionInTransaction(
	    [&] { columns = ScanTableToGpuColumns(*con.context, "t", {"a", "b", "d"}, owned_buffers); });

	Check(columns.size() == 3, "three columns returned");
	Check(columns[0].row_count == 5000 && columns[1].row_count == 5000 && columns[2].row_count == 5000,
	      "all columns report the full 5000-row count (spans multiple chunks)");
	Check(columns[0].type == GpuValueType::INT32 && columns[1].type == GpuValueType::INT32 &&
	          columns[2].type == GpuValueType::FLOAT64,
	      "column types resolved correctly from the table's real schema");
	Check(columns[0].validity == nullptr, "no-nulls column ('a') has a null validity pointer");

	auto *a_data = static_cast<const int32_t *>(columns[0].data);
	auto *b_data = static_cast<const int32_t *>(columns[1].data);
	auto *d_data = static_cast<const double *>(columns[2].data);

	// Spot-check values spanning chunk boundaries: row 2047 (last row of chunk 1), row 2048 (first row of
	// chunk 2), row 4095/4096 (boundary between chunk 2 and chunk 3), and the very last row (904th row of
	// the final, partial chunk).
	bool boundary_values_correct = true;
	for (idx_t row : {idx_t(0), idx_t(2047), idx_t(2048), idx_t(4095), idx_t(4096), idx_t(4999)}) {
		if (a_data[row] != static_cast<int32_t>(row) || b_data[row] != static_cast<int32_t>(row % 7) ||
		    d_data[row] != static_cast<double>(row)) {
			boundary_values_correct = false;
		}
	}
	Check(boundary_values_correct, "values at chunk-boundary rows (0, 2047, 2048, 4095, 4096, 4999) are all correct");

	// Every single row, not just boundary spot checks -- catches any off-by-one in the accumulation loop
	// that a sparse spot check might miss.
	bool all_rows_correct = true;
	for (idx_t row = 0; row < 5000; row++) {
		if (a_data[row] != static_cast<int32_t>(row) || b_data[row] != static_cast<int32_t>(row % 7)) {
			all_rows_correct = false;
			break;
		}
	}
	Check(all_rows_correct, "all 5000 rows of columns 'a' and 'b' are correct, not just boundary samples");

	// --- Column order is preserved exactly as requested, even when it doesn't match the table's physical
	// column order (b, a instead of a, b) -----------------------------------------------------------
	{
		std::vector<GpuColumn> reordered;
		std::vector<std::vector<uint8_t>> reordered_buffers;
		con.context->RunFunctionInTransaction(
		    [&] { reordered = ScanTableToGpuColumns(*con.context, "t", {"b", "a"}, reordered_buffers); });
		Check(reordered.size() == 2, "reordered scan returns exactly 2 columns");
		auto *first = static_cast<const int32_t *>(reordered[0].data);  // should be 'b'
		auto *second = static_cast<const int32_t *>(reordered[1].data); // should be 'a'
		Check(first[10] == 10 % 7 && second[10] == 10, "requested column order ('b' then 'a') is honored exactly");
	}

	// --- A table small enough to fit in a single chunk still works (the other tests all exercise the
	// multi-chunk path; this confirms the single-chunk path wasn't broken along the way) -------------
	{
		con.Query("CREATE TABLE small_t AS SELECT range::INTEGER AS x FROM range(3)");
		std::vector<GpuColumn> small_columns;
		std::vector<std::vector<uint8_t>> small_buffers;
		con.context->RunFunctionInTransaction(
		    [&] { small_columns = ScanTableToGpuColumns(*con.context, "small_t", {"x"}, small_buffers); });
		Check(small_columns.size() == 1 && small_columns[0].row_count == 3, "single-chunk (3-row) table scans correctly");
		auto *x_data = static_cast<const int32_t *>(small_columns[0].data);
		Check(x_data[0] == 0 && x_data[1] == 1 && x_data[2] == 2, "single-chunk table values are correct");
	}

	// --- An empty table (0 rows) must not crash and must report row_count == 0 ----------------------
	{
		con.Query("CREATE TABLE empty_t (x INTEGER)");
		std::vector<GpuColumn> empty_columns;
		std::vector<std::vector<uint8_t>> empty_buffers;
		bool threw = false;
		try {
			con.context->RunFunctionInTransaction(
			    [&] { empty_columns = ScanTableToGpuColumns(*con.context, "empty_t", {"x"}, empty_buffers); });
		} catch (...) {
			threw = true;
		}
		Check(!threw, "scanning an empty (0-row) table doesn't throw");
		if (!threw) {
			Check(empty_columns.size() == 1 && empty_columns[0].row_count == 0,
			      "empty table reports row_count 0, not a crash or garbage value");
		}
	}

	// --- Nulls survive a real multi-chunk scan correctly --------------------------------------------
	{
		con.Query("CREATE TABLE nullable_t AS SELECT range::INTEGER AS a, "
		         "CASE WHEN range % 1000 = 0 THEN NULL ELSE range END AS n FROM range(3000)");
		std::vector<GpuColumn> null_columns;
		std::vector<std::vector<uint8_t>> null_buffers;
		con.context->RunFunctionInTransaction(
		    [&] { null_columns = ScanTableToGpuColumns(*con.context, "nullable_t", {"a", "n"}, null_buffers); });
		Check(null_columns[1].validity != nullptr, "column with nulls has a non-null validity pointer after a real scan");
		auto BitIsSet = [](const uint8_t *bitmap, idx_t row) {
			return (bitmap[row / 8] & (uint8_t(1) << (row % 8))) != 0;
		};
		// Nulls land at rows 0, 1000, 2000 -- one in each of the three 1000-row-aligned positions, so this
		// also confirms nulls are handled correctly regardless of which chunk they fall in (0 is chunk 1,
		// 1000 is chunk 1, 2000 is chunk 1... range(3000) with 2048/chunk puts 1000 in chunk 1 and 2000 in
		// chunk 1 too; use exact positions instead of assuming which chunk, the point is chunk-spanning
		// accumulation, already covered by the boundary test above -- this test is specifically about
		// nulls surviving at all after going through the unpacked-then-repacked accumulator).
		bool nulls_correct = !BitIsSet(null_columns[1].validity, 0) && !BitIsSet(null_columns[1].validity, 1000) &&
		                     !BitIsSet(null_columns[1].validity, 2000) && BitIsSet(null_columns[1].validity, 1) &&
		                     BitIsSet(null_columns[1].validity, 2999);
		Check(nulls_correct, "nulls at rows 0, 1000, 2000 are null; rows 1 and 2999 are valid, after a real scan");
	}

	// --- Error paths ---------------------------------------------------------------------------------
	{
		std::vector<GpuColumn> unused;
		std::vector<std::vector<uint8_t>> unused_buffers;
		Check(Throws([&] {
			      con.context->RunFunctionInTransaction(
			          [&] { unused = ScanTableToGpuColumns(*con.context, "no_such_table", {"x"}, unused_buffers); });
		      }),
		      "scanning a nonexistent table throws");
		Check(Throws([&] {
			      con.context->RunFunctionInTransaction(
			          [&] { unused = ScanTableToGpuColumns(*con.context, "t", {"no_such_column"}, unused_buffers); });
		      }),
		      "scanning a nonexistent column throws");

		con.Query("CREATE TABLE string_t (s VARCHAR)");
		Check(Throws([&] {
			      con.context->RunFunctionInTransaction(
			          [&] { unused = ScanTableToGpuColumns(*con.context, "string_t", {"s"}, unused_buffers); });
		      }),
		      "scanning an unsupported column type (VARCHAR) throws");
	}

	std::cout << "\n" << (g_checks - g_failures) << "/" << g_checks << " checks passed\n";
	return g_failures == 0 ? 0 : 1;
}
