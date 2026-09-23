#pragma once

#include "duckdb.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "gpu_engine.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace vector_gpu {

//! Thrown for a vector/type shape this converter doesn't (yet) handle -- e.g. a type outside
//! TableChecker's whitelist. Mirrors GpuUnsupportedExpression's role: a normal, expected signal for
//! "not supported", not a bug report. Callers should treat it as a fallback trigger, same as everywhere
//! else in this extension.
struct GpuUnsupportedVector : std::runtime_error {
	explicit GpuUnsupportedVector(const std::string &msg) : std::runtime_error(msg) {
	}
};

//! Converts a DuckDB Vector (`count` logical rows) into a GpuColumn. Supports the same numeric/boolean
//! type whitelist TableChecker and expression_translator already enforce (INTEGER, BIGINT, FLOAT,
//! DOUBLE, BOOLEAN) -- throws GpuUnsupportedVector for anything else.
//!
//! Reads via Vector::ToUnifiedFormat rather than requiring a pre-flattened vector -- deliberately NOT
//! "call Flatten() first": DataChunk::Flatten()/Vector::Flatten() were tried first and found to silently
//! under-read a constant vector (they materialize based on the vector buffer's own internal Size(),
//! which is 1 for a constant vector, not the chunk's actual row count -- see
//! docs/EXECUTION_TRACKER.md). ToUnifiedFormat's selection vector correctly broadcasts a constant
//! vector's single value to every requested row instead.
//!
//! Copies data into a freshly allocated buffer appended to `owned_buffers` -- the caller controls that
//! buffer's lifetime (same contract as GpuExecutionResult::owned_buffers on the result path). This is a
//! real copy, not a zero-copy pointer handoff (docs/ARCHITECTURE.md's eventual design uses
//! cudaHostRegister-pinned DuckDB-owned memory instead) -- correctness first; pinning is a follow-up
//! optimization once this path is proven end-to-end, tracked in docs/KNOWN_ISSUES.md.
GpuColumn ConvertVectorToGpuColumn(duckdb::Vector &vec, duckdb::idx_t count, std::string name,
                                   std::vector<std::vector<uint8_t>> &owned_buffers);

//! Converts every column of `chunk` into GpuColumns (see ConvertVectorToGpuColumn -- handles constant/
//! dictionary-encoded columns transparently, no need to flatten `chunk` first). `names` must have
//! exactly `chunk.ColumnCount()` entries, in column order.
std::vector<GpuColumn> ConvertDataChunkToGpuColumns(duckdb::DataChunk &chunk, const std::vector<std::string> &names,
                                                    std::vector<std::vector<uint8_t>> &owned_buffers);

//! Converts a GpuColumn's buffer back into an existing DuckDB Vector. `target` must already be a FLAT
//! vector with capacity for `count` rows (e.g. a column of a DataChunk sized via Initialize()) and its
//! LogicalType must match `column.type` (checked, throws GpuUnsupportedVector on mismatch rather than
//! silently reinterpreting bytes as the wrong type).
void ConvertGpuColumnToVector(const GpuColumn &column, duckdb::idx_t count, duckdb::Vector &target);

//! Converts `count` rows starting at logical row `offset` within `column` into `target` (a flat vector
//! sized for at least `count` rows). The read-side counterpart of the accumulator's write-side slicing:
//! lets a single, arbitrarily-large GpuColumn (e.g. GpuEngine::ExecutePlan's whole-result output) be
//! packed into a ColumnDataCollection STANDARD_VECTOR_SIZE rows at a time (see
//! PackGpuColumnsIntoCollection) without materializing a full-size intermediate DataChunk per column.
//! Correctly handles `offset` values that don't fall on a validity-bitmap byte boundary. Throws
//! GpuUnsupportedVector if `target`'s type doesn't match `column.type`, or if `offset + count` exceeds
//! `column.row_count`.
void ConvertGpuColumnRangeToVector(const GpuColumn &column, duckdb::idx_t offset, duckdb::idx_t count,
                                  duckdb::Vector &target);

//! Packs a full set of GpuColumns (e.g. a successful GpuExecutionResult::columns, potentially far larger
//! than one DataChunk) into `collection`, chunking internally at STANDARD_VECTOR_SIZE. All columns must
//! report the same `row_count`, and `types` must match `columns` 1:1 in order (GpuEngine::ExecutePlan is
//! expected to return result columns in the physical operator's own output order -- the same convention
//! ScanTableToGpuColumns's callers already rely on for input columns). Throws GpuUnsupportedVector if
//! `columns.size()` != `types.size()`, or if the columns disagree on row_count.
void PackGpuColumnsIntoCollection(const std::vector<GpuColumn> &columns, const std::vector<duckdb::LogicalType> &types,
                                  duckdb::ColumnDataCollection &collection);

//! Growing per-column buffer for accumulating MANY DataChunks (e.g. every chunk of a table scan) into one
//! final GpuColumn. Needed because a table scan produces data STANDARD_VECTOR_SIZE (2048) rows at a time,
//! but GpuEngine::ExecutePlan wants one contiguous buffer per column covering the whole scanned table.
//!
//! Null tracking is deliberately unpacked (one byte per row) during accumulation rather than reusing
//! GpuColumn's packed bitmap format directly -- packed bits from one chunk almost never end on a byte
//! boundary (chunk sizes are 2048 except the last), so naively concatenating packed-bitmap bytes across
//! chunks would misalign every bit after the first chunk. Packing happens once, correctly, in
//! FinalizeGpuColumnAccumulators.
struct GpuColumnAccumulator {
	std::string name;
	GpuValueType type;
	std::vector<uint8_t> data;       // grows by one chunk's worth of typed bytes per AppendChunkToAccumulators call
	std::vector<uint8_t> null_flags; // one byte per row (1 = null, 0 = valid), same length as logical row count so far
	uint64_t row_count = 0;
};

//! Creates one accumulator per column, ready for repeated AppendChunkToAccumulators calls. `names` and
//! `types` must be the same length, in column order. Throws GpuUnsupportedVector immediately for any
//! unsupported type, rather than discovering it only after scanning has already started.
std::vector<GpuColumnAccumulator> MakeGpuColumnAccumulators(const std::vector<std::string> &names,
                                                            const std::vector<duckdb::LogicalType> &types);

//! Appends one DataChunk's worth of data to each accumulator (`accumulators.size()` must equal
//! `chunk.ColumnCount()`, in the same column order used to create them). Safe to call with an empty
//! chunk (`chunk.size() == 0`) -- a no-op, matching a table scan's own "no more data" signal.
void AppendChunkToAccumulators(duckdb::DataChunk &chunk, std::vector<GpuColumnAccumulator> &accumulators);

//! Finalizes accumulators into GpuColumns: packs each accumulator's unpacked null_flags into the bitmap
//! format documented on GpuColumn::validity (nullptr if no nulls were ever seen), and appends the final
//! buffers to `owned_buffers` for the caller to own. Consumes (moves out of) `accumulators`.
std::vector<GpuColumn> FinalizeGpuColumnAccumulators(std::vector<GpuColumnAccumulator> &accumulators,
                                                     std::vector<std::vector<uint8_t>> &owned_buffers);

} // namespace vector_gpu
