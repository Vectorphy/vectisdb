#pragma once

#include "duckdb.hpp"
#include "gpu_engine.hpp"       // GpuColumn, GpuValueType
#include "gpu_memory_pool.hpp"  // GpuMemoryPool, GpuMemoryPoolSlot

#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

// GpuBatchAccumulator flattens a sequence of DuckDB DataChunks DIRECTLY into a GpuMemoryPool ring slot's
// pinned memory, so no unpinned heap buffer ever sits between a DataChunk and the page-locked staging
// area a real H2D transfer would read from.
//
// This is a deliberately different shape from the existing GpuColumnAccumulator (vector_converter.hpp),
// not a duplicate of it. That accumulator grows a plain std::vector<uint8_t> per column -- correct, and
// exactly what ScanTableToGpuColumns/StreamingTableScanner need for the WHOLE-TABLE and ramping-batch
// paths already wired into routing -- but the buffer it produces is still pageable: gpu_executor.cu's
// ExecuteScan memcpy's it into a PinnedBufferPool slot immediately before the async H2D copy, which is
// the double-copy vector_converter.hpp's own header comment flags as a follow-up ("a real copy, not a
// zero-copy pointer handoff... pinning is a follow-up optimization"). This class IS that follow-up, for a
// new fixed-size streaming shape: instead of one growing buffer per column, the SLOT's memory is
// subdivided into a small, fixed number of DENSE, VALIDITY and PACKED-VALIDITY regions per column, sized
// once (at construction) for exactly the chosen batch size, and every Append() writes straight into them.

namespace vector_gpu {

//! Thrown for a column type outside the supported numeric/boolean whitelist, or when a column set's
//! per-batch footprint would not fit inside one GpuMemoryPool slot. Same "normal, expected fallback
//! signal" role as GpuUnsupportedVector in vector_converter.hpp -- not a bug report.
struct GpuBatchAccumulatorError : std::runtime_error {
	explicit GpuBatchAccumulatorError(const std::string &msg) : std::runtime_error(msg) {
	}
};

//! One accumulated batch, ready for an H2D transfer. `slot` (moved out of the accumulator that produced
//! this batch) owns the pinned memory every GpuColumn in `columns` points into -- `slot` must outlive
//! whatever issues the transfer, and once it is destroyed every pointer in `columns` is dangling.
//! `columns[i].data`/`validity` ALIAS INTO `slot`'s buffer; nothing here is copied or freshly allocated.
struct GpuPinnedBatch {
	GpuMemoryPoolSlot slot;
	std::vector<GpuColumn> columns;
	duckdb::idx_t row_count = 0;
};

//! Accumulates DataChunks into one GpuMemoryPool ring slot at a time, cycling to the next slot each time
//! a batch is taken.
//!
//! SINGLE-THREADED BY DESIGN, matching StreamingTableScanner's own documented contract: Append() must be
//! called only from the thread driving the DuckDB scan (DataTable::Scan is not documented safe to drive
//! from a second thread). The overlap this ring buffer exists for comes from handing a FINISHED
//! GpuPinnedBatch to a different thread for its H2D transfer via TakeReadyBatch -- ownership of the slot
//! moves at that point, so there is no concurrent access to the same memory to synchronize, only the
//! GpuMemoryPool cyclic handoff itself (already internally synchronized).
class GpuBatchAccumulator {
public:
	//! Default batch size, used when a caller does not choose one. Divides evenly by DuckDB's
	//! STANDARD_VECTOR_SIZE (2048 * 16 = 32768) so a batch boundary always lands exactly on a DataChunk
	//! boundary -- Append() never has to split one chunk's rows across two ring slots. Every batch size,
	//! default or explicit, must keep that property; the constructor enforces it.
	static constexpr duckdb::idx_t kBatchReadyRows = 32768;

	//! The largest batch size the given column set fits inside ONE GpuMemoryPool slot, rounded down to a
	//! whole number of DataChunks (and therefore already legal to pass as `batch_rows`). Returns 0 for an
	//! empty column list.
	//!
	//! Exists because the right batch size is a property of the CONSUMER, not of this class. A consumer
	//! that hands each batch to a blocking execute call wants them small, for time-to-first-row. A
	//! consumer that overlaps batches on separate CUDA streams (GpuStreamPipeline) wants them large: at
	//! 32,768 rows a batch's whole H2D is ~30 microseconds, well under the ~100-microsecond command-
	//! submission latency Windows/WDDM imposes per flush, so successive batches cannot overlap no matter
	//! how they are scheduled -- measured on an `nsys` capture, where every batch's upload, kernel and
	//! download ran strictly one after another with 100+ microsecond gaps between them.
	static duckdb::idx_t RowsThatFitInOneSlot(const std::vector<duckdb::LogicalType> &types);

	//! `names`/`types` describe every column an appended DataChunk will have, in order (same contract as
	//! MakeGpuColumnAccumulators in vector_converter.hpp). Throws GpuBatchAccumulatorError immediately --
	//! before any scanning starts -- if a type is outside the supported whitelist, or if this column
	//! set's per-batch footprint (`batch_rows` rows of dense data, plus validity scratch, per column)
	//! would not fit inside one GpuMemoryPool::kSlotBytes slot. Acquires this accumulator's first slot.
	//! `batch_rows` must be a non-zero multiple of STANDARD_VECTOR_SIZE (see kBatchReadyRows) -- throws
	//! GpuBatchAccumulatorError otherwise, and likewise if this column set at this batch size does not fit
	//! one slot. RowsThatFitInOneSlot above computes the largest legal value for a given column set.
	GpuBatchAccumulator(const std::vector<std::string> &names, const std::vector<duckdb::LogicalType> &types,
	                    duckdb::idx_t batch_rows = kBatchReadyRows);

	GpuBatchAccumulator(const GpuBatchAccumulator &) = delete;
	GpuBatchAccumulator &operator=(const GpuBatchAccumulator &) = delete;
	//! No custom destructor needed or written: `slot_` is a GpuMemoryPoolSlot member, so its own
	//! destructor returns the checked-out ring slot to GpuMemoryPool automatically, whether this object
	//! is destroyed normally or via an exception unwinding through it. This IS "cleans up safely on query
	//! destruction" -- there is no separate cleanup path to get wrong.

	//! Appends one DataChunk's rows into the active slot, flattening every column (constant/dictionary
	//! encodings included) directly into pinned memory via Vector::ToUnifiedFormat -- the same technique
	//! vector_converter.cpp's AppendTyped uses, and for the same documented reason: Vector::Flatten()
	//! reads a constant vector's internal buffer Size() (1), not the chunk's actual row count, and
	//! silently under-reads past the single real value.
	//!
	//! Returns true once the active batch has reached batch_rows() -- the caller must call
	//! TakeReadyBatch() before appending again; Append() throws rather than overrun a slot that has
	//! already reached the threshold. A chunk with chunk.size() == 0 is a no-op returning false, matching
	//! AppendChunkToAccumulators' contract for a scan's own "no more data" signal.
	//! Throws GpuBatchAccumulatorError if chunk.ColumnCount() disagrees with the constructor's column
	//! count, or (defensively -- not reachable for an ordinary table scan, whose chunks are all
	//! STANDARD_VECTOR_SIZE except the last, and kBatchReadyRows is a multiple of that) if this chunk's
	//! rows would overrun the active batch.
	bool Append(duckdb::DataChunk &chunk, const duckdb::SelectionVector *sel, duckdb::idx_t count);

	//! True once Append() has returned true and the ready batch has not yet been taken via
	//! TakeReadyBatch(). Append() refuses to be called again while this is true.
	bool HasReadyBatch() const {
		std::lock_guard<std::mutex> guard(mutex_);
		return ready_;
	}

	//! Packs each column's validity into the Arrow-bitmap format GpuColumn::validity documents -- still
	//! entirely within this slot's pinned memory, no heap allocation -- and hands back the batch,
	//! acquiring the NEXT ring slot for whatever is appended after this call.
	//!
	//! Valid to call even when row_count() < batch_rows(): HasReadyBatch() reports only the
	//! *threshold*, not whether any data is available at all, so a caller flushing a final partial batch
	//! at the end of a scan (mirroring ScannedRowBatch::exhausted's "this may be the last one, still
	//! process it" contract) should check row_count() > 0 instead. Calling this with row_count() == 0
	//! still cycles to a fresh slot and returns an empty (zero-column-data, zero-row) batch -- harmless,
	//! but pointless; callers should guard on row_count() first.
	GpuPinnedBatch TakeReadyBatch();

	//! Rows accumulated into the batch that has not yet been taken.
	duckdb::idx_t row_count() const {
		std::lock_guard<std::mutex> guard(mutex_);
		return row_count_;
	}

	//! Rows this accumulator offers a batch at, as chosen at construction.
	duckdb::idx_t batch_rows() const {
		return batch_rows_;
	}

	//! The GPU type each column is written to pinned memory as, in column order.
	//!
	//! Exposed rather than left for the caller to re-derive: a consumer of the batches (GpuStreamPipeline)
	//! has to be told what the bytes in the slot mean, and if its answer disagreed with the mapping used
	//! HERE the bytes would simply be reinterpreted -- an INT32 column read as FLOAT32 produces numbers,
	//! not an error. One source of truth removes that failure mode entirely.
	std::vector<GpuValueType> column_types() const {
		std::vector<GpuValueType> types;
		types.reserve(layout_.size());
		for (auto &column : layout_) {
			types.push_back(column.type);
		}
		return types;
	}

private:
	//! Where one column's three pinned regions live within the active slot, computed once at
	//! construction and reused (offsets never change; only the bytes at them do).
	struct ColumnLayout {
		std::string name;
		GpuValueType type;
		size_t value_size;       // bytes/row for `type` (e.g. 8 for FLOAT64)
		size_t data_offset;      // dense data: batch_rows_ * value_size bytes
		size_t null_flag_offset; // UNPACKED null scratch, 1 byte/row: written every Append, packed at Take
		size_t packed_offset;    // PACKED Arrow-bitmap form, written only at TakeReadyBatch
		bool any_null = false;   // set the first time Append sees a NULL in the row range since the last Take
	};

	void AcquireFreshSlot();
	template <typename T>
	void AppendTypedColumn(duckdb::Vector &vec, duckdb::idx_t count, ColumnLayout &layout, const duckdb::SelectionVector *sel);

	std::vector<ColumnLayout> layout_;
	duckdb::idx_t batch_rows_;
	GpuMemoryPoolSlot slot_;
	duckdb::idx_t row_count_ = 0;
	bool ready_ = false;
	mutable std::mutex mutex_;
};

} // namespace vector_gpu
