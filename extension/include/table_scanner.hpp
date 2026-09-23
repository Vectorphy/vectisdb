#pragma once

#include "duckdb.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "gpu_engine.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace vector_gpu {

//! Thrown when a table/column named in a GpuPlanNode can't actually be scanned (e.g. it was dropped
//! between planning and execution, or a column type isn't in vector_converter's supported whitelist).
//! Unlike GpuUnsupportedExpression/GpuUnsupportedVector, this is NOT a normal fallback signal -- by the
//! time PhysicalGpuExecute runs, the optimizer has already committed to the GPU plan and there is no
//! surviving CPU-fallback plan to run instead. Callers should let this propagate as a genuine query
//! error (wrapped in a non-fatal DuckDB exception type -- see docs/KNOWN_ISSUES.md's standing rule about
//! never using InternalException for an expected-ish failure).
struct GpuTableScanError : std::runtime_error {
	explicit GpuTableScanError(const std::string &msg) : std::runtime_error(msg) {
	}
};

//! Performs a real, low-level DataTable scan of `table` (resolved by its FULL catalog.schema.name
//! identity, NOT through the session's current search path) reading exactly `column_names`, in that
//! order, and returns the whole table's data as GpuColumns (accumulated across every 2048-row chunk the
//! scan produces via GpuColumnAccumulator -- see vector_converter.hpp).
//!
//! Must be called from a context where a transaction is already active (true for any call made from
//! within normal query execution, e.g. from PhysicalGpuExecute -- ClientContext::BeginQueryInternal
//! starts one before any physical operator runs). Throws GpuTableScanError if the table doesn't exist,
//! a requested column doesn't exist, or a column's type isn't supported by vector_converter.
//!
//! Used only by the WHOLE-INPUT path (GROUP_BY_AGGREGATE/HASH_JOIN/CROSS_PRODUCT, which need every row at
//! once -- see IsRowIndependent). Row-independent plans use StreamingTableScanner below instead, which
//! hands back rows as they become available rather than only after the whole table is scanned.
std::vector<GpuColumn> ScanTableToGpuColumns(duckdb::ClientContext &context, const GpuTableRef &table,
                                             const std::vector<std::string> &column_names,
                                             std::vector<std::vector<uint8_t>> &owned_buffers);

//! One independently-owned batch of scanned rows, ready to execute as a GPU chunk without any further
//! slicing. Unlike ScanTableToGpuColumns's single accumulate-the-whole-table result, a streamed scan
//! produces a SEQUENCE of these -- each owns its backing buffers outright, so handing one to a consumer
//! (to start GPU execution) while the NEXT is still being scanned is safe: producing batch N+1 can never
//! move or invalidate batch N's memory the way growing one shared accumulator buffer could.
struct ScannedRowBatch {
	std::vector<GpuColumn> columns;
	std::vector<std::vector<uint8_t>> owned_buffers; // backs `columns`; must outlive it
	duckdb::idx_t row_count = 0;
	//! Wall time this specific batch's NextBatch() call spent scanning, microseconds. Unlike the old
	//! whole-table-scan-then-chunk design (which attributed ALL scan cost to chunk 0 because scanning
	//! genuinely happened once, up front), each batch here does its own real scanning, so each batch
	//! reports its own honest cost rather than lumping it onto the first chunk.
	uint64_t scan_us = 0;
	//! True once the table has been fully scanned -- THIS batch may still hold rows (the last real batch
	//! and the exhaustion signal arrive together, not as a separate empty batch) unless row_count == 0,
	//! which happens only when NextBatch is called again after a previous batch already set this true.
	bool exhausted = false;
};

//! Drives a resumable, ramping-batch-size scan of one table, entirely on the CALLING thread.
//!
//! Deliberately single-threaded: DuckDB's TableScanState/DataTable::Scan are driven, elsewhere in this
//! codebase and in DuckDB core itself, only from the thread that owns the pipeline's transaction context
//! -- nothing in this class (or its caller) attempts to scan from a second thread, since that contract is
//! not documented as safe and getting it wrong would be silent corruption, not a clean crash. The overlap
//! a streaming pipeline wants instead comes from where EXECUTION is dispatched: PhysicalGpuExecute calls
//! NextBatch() (blocking, here, on the calling thread), then hands the returned batch to its EXISTING
//! std::async-based chunk execution (see physical_gpu_execute.cpp's StartPrefetch), then immediately calls
//! NextBatch() again for the NEXT batch -- so scanning batch N+1 genuinely overlaps executing batch N,
//! using only the already-proven-safe threading pattern this engine already had for GPU execution, not a
//! new one for DuckDB storage access. See docs/STREAMING_INGEST_DESIGN.md.
class StreamingTableScanner {
public:
	//! Throws GpuTableScanError immediately (before any scanning) if the table or a requested column
	//! doesn't exist, or a column's type isn't supported -- same checks ScanTableToGpuColumns makes,
	//! at the same point (construction), so a caller doesn't discover the failure only after wasting a
	//! partial scan.
	StreamingTableScanner(duckdb::ClientContext &context, const GpuTableRef &table,
	                      const std::vector<std::string> &column_names);

	//! Scans forward until at least `target_rows` NEW rows have been accumulated into this batch, or the
	//! table is exhausted, whichever comes first. `target_rows` == 0 is treated as 1 (scan at least one
	//! more DuckDB chunk). Calling this again after exhausted() returns a 0-row, exhausted batch --
	//! callers should check exhausted() rather than rely on row_count == 0 to mean "done", since a
	//! genuinely empty table produces exactly that on the FIRST call too.
	ScannedRowBatch NextBatch(duckdb::idx_t target_rows);

	bool exhausted() const {
		return exhausted_;
	}

private:
	duckdb::ClientContext &context_;
	std::vector<std::string> column_names_;
	duckdb::vector<duckdb::LogicalType> column_types_;
	duckdb::optional_ptr<duckdb::TableCatalogEntry> table_entry_;
	duckdb::DuckTransaction *transaction_ = nullptr;
	duckdb::TableScanState scan_state_;
	duckdb::DataChunk chunk_;
	bool exhausted_ = false;
};

} // namespace vector_gpu
