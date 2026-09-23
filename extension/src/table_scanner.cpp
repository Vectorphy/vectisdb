#include "table_scanner.hpp"
#include "vector_converter.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/parser/qualified_name.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/transaction/duck_transaction.hpp"

#include <chrono>
#include <thread>

namespace vector_gpu {

using namespace duckdb;

//! Resolves a SCAN node's table by its FULL catalog.schema.name identity.
//!
//! Emphatically not an unqualified lookup through the search path, which is what this used to do: the same
//! plan would then read whichever `t` the session's CURRENT search path pointed at, so `SELECT ... FROM
//! s.t` silently returned `main.t`'s rows, and a later `USE s` changed what an already-planned query read.
//! The translator has the bound TableCatalogEntry and records all three components (see GpuTableRef), so
//! there is no reason to re-resolve by bare name and hope.
static optional_ptr<TableCatalogEntry> ResolveTable(ClientContext &context, const GpuTableRef &table,
                                                    const char *caller) {
	try {
		QualifiedName qualified(Identifier(table.catalog), Identifier(table.schema), Identifier(table.table));
		return &Catalog::GetEntry<TableCatalogEntry>(context, qualified);
	} catch (const std::exception &e) {
		throw GpuTableScanError(std::string(caller) + ": table '" + table.ToString() + "' not found: " + e.what());
	}
}

std::vector<GpuColumn> ScanTableToGpuColumns(ClientContext &context, const GpuTableRef &table,
                                             const std::vector<std::string> &column_names,
                                             std::vector<std::vector<uint8_t>> &owned_buffers) {
	auto table_entry = ResolveTable(context, table, "ScanTableToGpuColumns");

	// Resolve each requested column name to (a) the StorageIndex the scan API needs and (b) its
	// LogicalType, in the exact order requested -- this order is what GpuPlanNode's translation already
	// assumes when it built `column_names` (see gpu_offload_extension.cpp's TranslateToGpuPlan), so it
	// must be preserved exactly, not re-sorted or deduplicated.
	//
	// Note: `vector<T>` here is duckdb::vector (DuckDB's own vector wrapper, brought in via
	// `using namespace duckdb`), not std::vector -- required because DataTable::InitializeScan and
	// DataChunk::Initialize take duckdb::vector specifically, and the two are distinct template
	// instantiations that don't implicitly convert.
	vector<StorageIndex> storage_ids;
	vector<LogicalType> column_types;
	storage_ids.reserve(column_names.size());
	column_types.reserve(column_names.size());
	for (auto &name : column_names) {
		LogicalIndex logical_idx = [&]() -> LogicalIndex {
			try {
				Identifier ident(name);
				return table_entry->GetColumnIndex(ident);
			} catch (const std::exception &e) {
				throw GpuTableScanError("ScanTableToGpuColumns: column '" + name + "' not found on table '" +
				                       table.ToString() + "': " + e.what());
			}
		}();
		storage_ids.push_back(table_entry->GetStorageIndex(ColumnIndex(logical_idx.index)));
		column_types.push_back(table_entry->GetColumn(logical_idx).Type());
	}

	// Throws immediately (before any scanning happens) if a column's type isn't supported -- cheaper to
	// fail here than partway through a possibly-large scan. vector_converter.hpp's API takes std::vector
	// (it has no DuckDB dependency by design), hence the explicit conversion from duckdb::vector here.
	auto accumulators =
	    MakeGpuColumnAccumulators(column_names, std::vector<LogicalType>(column_types.begin(), column_types.end()));

	auto &storage = table_entry->GetStorage();
	auto &transaction = DuckTransaction::Get(context, table_entry->catalog);

	idx_t total_rows = storage.GetTotalRows();
	idx_t num_threads = TaskScheduler::GetScheduler(context).NumberOfThreads();

	if (num_threads <= 1 || total_rows < 65536) {
		TableScanState scan_state;
		storage.InitializeScan(context, transaction, scan_state, storage_ids);

		DataChunk chunk;
		chunk.Initialize(context, column_types);
		while (true) {
			chunk.Reset();
			storage.Scan(transaction, chunk, scan_state);
			if (chunk.size() == 0) {
				break;
			}
			AppendChunkToAccumulators(chunk, accumulators);
		}
		return FinalizeGpuColumnAccumulators(accumulators, owned_buffers);
	}

	vector<ColumnIndex> column_indexes;
	column_indexes.reserve(storage_ids.size());
	for (auto &sid : storage_ids) {
		column_indexes.emplace_back(ColumnIndex(sid.GetPrimaryIndex()));
	}

	ParallelTableScanState parallel_state;
	storage.InitializeParallelScan(context, parallel_state, column_indexes);

	idx_t actual_workers = std::min<idx_t>(num_threads, 8);
	std::vector<std::vector<GpuColumnAccumulator>> thread_accumulators(actual_workers);
	for (idx_t t = 0; t < actual_workers; t++) {
		thread_accumulators[t] = MakeGpuColumnAccumulators(
		    column_names, std::vector<LogicalType>(column_types.begin(), column_types.end()));
	}

	std::vector<std::thread> workers;
	workers.reserve(actual_workers);
	for (idx_t t = 0; t < actual_workers; t++) {
		workers.emplace_back([&, t]() {
			TableScanState local_scan_state;
			local_scan_state.Initialize(storage_ids, &context);
			DataChunk local_chunk;
			local_chunk.Initialize(context, column_types);

			while (storage.NextParallelScan(context, parallel_state, local_scan_state) > 0) {
				while (true) {
					local_chunk.Reset();
					storage.Scan(transaction, local_chunk, local_scan_state);
					if (local_chunk.size() == 0) {
						break;
					}
					AppendChunkToAccumulators(local_chunk, thread_accumulators[t]);
				}
			}
		});
	}

	for (auto &w : workers) {
		w.join();
	}

	for (size_t c = 0; c < accumulators.size(); c++) {
		idx_t col_total_rows = 0;
		size_t col_total_bytes = 0;
		for (idx_t t = 0; t < actual_workers; t++) {
			col_total_rows += thread_accumulators[t][c].row_count;
			col_total_bytes += thread_accumulators[t][c].data.size();
		}
		accumulators[c].row_count = col_total_rows;
		accumulators[c].data.resize(col_total_bytes);
		accumulators[c].null_flags.resize(col_total_rows);

		size_t data_offset = 0;
		idx_t row_offset = 0;
		for (idx_t t = 0; t < actual_workers; t++) {
			auto &t_acc = thread_accumulators[t][c];
			if (!t_acc.data.empty()) {
				std::memcpy(accumulators[c].data.data() + data_offset, t_acc.data.data(), t_acc.data.size());
				data_offset += t_acc.data.size();
			}
			if (!t_acc.null_flags.empty()) {
				std::memcpy(accumulators[c].null_flags.data() + row_offset, t_acc.null_flags.data(),
				            t_acc.null_flags.size());
				row_offset += t_acc.null_flags.size();
			}
		}
	}

	return FinalizeGpuColumnAccumulators(accumulators, owned_buffers);
}

StreamingTableScanner::StreamingTableScanner(ClientContext &context, const GpuTableRef &table,
                                             const std::vector<std::string> &column_names)
    : context_(context), column_names_(column_names) {
	// Same resolution ScanTableToGpuColumns performs, and at the same point (construction, before any
	// scanning) -- see that function's comments for why each step can fail and what it means.
	table_entry_ = ResolveTable(context_, table, "StreamingTableScanner");

	vector<StorageIndex> storage_ids;
	storage_ids.reserve(column_names_.size());
	column_types_.reserve(column_names_.size());
	for (auto &name : column_names_) {
		LogicalIndex logical_idx = [&]() -> LogicalIndex {
			try {
				Identifier ident(name);
				return table_entry_->GetColumnIndex(ident);
			} catch (const std::exception &e) {
				throw GpuTableScanError("StreamingTableScanner: column '" + name + "' not found on table '" +
				                       table.ToString() + "': " + e.what());
			}
		}();
		storage_ids.push_back(table_entry_->GetStorageIndex(ColumnIndex(logical_idx.index)));
		column_types_.push_back(table_entry_->GetColumn(logical_idx).Type());
	}

	transaction_ = &DuckTransaction::Get(context_, table_entry_->catalog);
	table_entry_->GetStorage().InitializeScan(context_, *transaction_, scan_state_, storage_ids);
	chunk_.Initialize(context_, column_types_);
}

ScannedRowBatch StreamingTableScanner::NextBatch(idx_t target_rows) {
	ScannedRowBatch batch;
	if (exhausted_) {
		batch.exhausted = true;
		return batch;
	}
	if (target_rows == 0) {
		target_rows = 1;
	}
	auto scan_start = std::chrono::steady_clock::now();
	auto accumulators = MakeGpuColumnAccumulators(
	    column_names_, std::vector<LogicalType>(column_types_.begin(), column_types_.end()));
	idx_t scanned = 0;
	while (scanned < target_rows) {
		chunk_.Reset();
		table_entry_->GetStorage().Scan(*transaction_, chunk_, scan_state_);
		if (chunk_.size() == 0) {
			exhausted_ = true;
			break;
		}
		AppendChunkToAccumulators(chunk_, accumulators);
		scanned += chunk_.size();
	}
	batch.columns = FinalizeGpuColumnAccumulators(accumulators, batch.owned_buffers);
	batch.row_count = batch.columns.empty() ? idx_t(0) : idx_t(batch.columns.front().row_count);
	batch.exhausted = exhausted_;
	batch.scan_us = static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - scan_start).count());
	return batch;
}

} // namespace vector_gpu
