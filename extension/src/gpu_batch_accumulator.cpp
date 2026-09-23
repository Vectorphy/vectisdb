#include "gpu_batch_accumulator.hpp"

#include "duckdb/common/vector/unified_vector_format.hpp"

#include <cstring>

namespace vector_gpu {

using namespace duckdb;

namespace {
//! Mirrors expression_translator.cpp's/table_checker.cpp's/vector_converter.cpp's MapLogicalType exactly.
//! Deliberately NOT shared via a common header -- vector_converter.cpp's own comment on its copy states
//! the convention this follows: each file's version stays intentionally-duplicated, self-contained logic
//! close to its own use.
GpuValueType MapLogicalType(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::SMALLINT:
		return GpuValueType::INT16;
	case LogicalTypeId::INTEGER:
		return GpuValueType::INT32;
	case LogicalTypeId::BIGINT:
		return GpuValueType::INT64;
	case LogicalTypeId::HUGEINT:
		return GpuValueType::HUGEINT;
	case LogicalTypeId::FLOAT:
		return GpuValueType::FLOAT32;
	case LogicalTypeId::DOUBLE:
		return GpuValueType::FLOAT64;
	case LogicalTypeId::BOOLEAN:
		return GpuValueType::BOOLEAN;
	default:
		throw GpuBatchAccumulatorError("unsupported type for GPU batch accumulation: " + type.ToString());
	}
}

size_t GpuValueTypeSize(GpuValueType type) {
	switch (type) {
	case GpuValueType::INT16:
		return sizeof(int16_t);
	case GpuValueType::INT32:
		return sizeof(int32_t);
	case GpuValueType::INT64:
		return sizeof(int64_t);
	case GpuValueType::HUGEINT:
		return sizeof(GpuHugeInt);
	case GpuValueType::FLOAT32:
		return sizeof(float);
	case GpuValueType::FLOAT64:
		return sizeof(double);
	case GpuValueType::BOOLEAN:
		return sizeof(bool);
	case GpuValueType::DICTIONARY_STRING:
		throw GpuBatchAccumulatorError("dictionary-encoded strings are not supported by GpuBatchAccumulator");
	}
	throw GpuBatchAccumulatorError("unknown GpuValueType");
}
} // namespace

GpuBatchAccumulator::GpuBatchAccumulator(const std::vector<std::string> &names,
                                         const std::vector<LogicalType> &types, idx_t batch_rows)
    : batch_rows_(batch_rows) {
	if (names.size() != types.size()) {
		throw GpuBatchAccumulatorError("GpuBatchAccumulator: names.size() (" + std::to_string(names.size()) +
		                               ") does not match types.size() (" + std::to_string(types.size()) + ")");
	}
	if (batch_rows_ == 0 || batch_rows_ % STANDARD_VECTOR_SIZE != 0) {
		// A multiple of STANDARD_VECTOR_SIZE, always: that is what makes a batch boundary land exactly on
		// a DataChunk boundary, so Append() never has to split one chunk's rows across two ring slots.
		throw GpuBatchAccumulatorError("GpuBatchAccumulator: batch_rows must be a non-zero multiple of " +
		                               std::to_string(idx_t(STANDARD_VECTOR_SIZE)) + ", got " +
		                               std::to_string(batch_rows_));
	}

	layout_.reserve(names.size());
	size_t offset = 0;
	for (size_t i = 0; i < names.size(); i++) {
		ColumnLayout col;
		col.name = names[i];
		col.type = MapLogicalType(types[i]); // throws immediately for an unsupported type
		col.value_size = GpuValueTypeSize(col.type);

		col.data_offset = offset;
		offset += static_cast<size_t>(batch_rows_) * col.value_size;
		col.null_flag_offset = offset;
		offset += static_cast<size_t>(batch_rows_); // unpacked scratch: 1 byte/row
		col.packed_offset = offset;
		offset += (static_cast<size_t>(batch_rows_) + 7) / 8; // Arrow-bitmap packed form

		layout_.push_back(std::move(col));
	}
	if (offset > GpuMemoryPool::kSlotBytes) {
		throw GpuBatchAccumulatorError(
		    "GpuBatchAccumulator: " + std::to_string(names.size()) + " column(s) need " + std::to_string(offset) +
		    " bytes per " + std::to_string(batch_rows_) + "-row batch, which exceeds one GpuMemoryPool "
		    "slot (" + std::to_string(GpuMemoryPool::kSlotBytes) + " bytes) -- too many/too wide columns "
		    "for this ring buffer's slot size");
	}

	AcquireFreshSlot();
}

idx_t GpuBatchAccumulator::RowsThatFitInOneSlot(const std::vector<LogicalType> &types) {
	// Bytes one row of the whole column set occupies inside a slot: dense value + one unpacked null flag
	// + one packed bit. Mirrors the layout the constructor lays down, and must stay in step with it --
	// the constructor is still the authority, and still throws if a caller asks for more than fits.
	size_t per_row = 0;
	for (auto &type : types) {
		per_row += GpuValueTypeSize(MapLogicalType(type)) + 1; // value + unpacked null flag
	}
	// The packed bitmaps add one bit per row per column; count them as a whole byte each so the estimate
	// is never optimistic.
	per_row += types.size();
	if (per_row == 0) {
		return 0;
	}
	auto rows = GpuMemoryPool::kSlotBytes / per_row;
	return static_cast<idx_t>(rows / STANDARD_VECTOR_SIZE) * STANDARD_VECTOR_SIZE; // whole DataChunks only
}

void GpuBatchAccumulator::AcquireFreshSlot() {
	slot_ = GpuMemoryPool::Instance().AcquireNext();
}

template <typename T>
void GpuBatchAccumulator::AppendTypedColumn(Vector &vec, idx_t count, ColumnLayout &layout, const duckdb::SelectionVector *sel) {
	UnifiedVectorFormat format;
	vec.ToUnifiedFormat(count, format);
	auto *typed_data = UnifiedVectorFormat::GetData<T>(format);

	auto *base = static_cast<uint8_t *>(slot_.data());
	auto *dest = reinterpret_cast<T *>(base + layout.data_offset) + row_count_;
	auto *null_flags = base + layout.null_flag_offset + row_count_;

	for (idx_t i = 0; i < count; i++) {
		idx_t row_idx = sel ? sel->get_index(i) : i;
		idx_t phys_idx = format.sel->get_index(row_idx);
		dest[i] = typed_data[phys_idx];
		// Every row in this call's range is written explicitly, valid or not: cudaHostAlloc'd memory
		// carries no zero-init guarantee, and this same physical region held a DIFFERENT batch's bytes
		// the last time this slot cycled around the ring, so nothing here may rely on residual state.
		if (format.validity.RowIsValid(phys_idx)) {
			null_flags[i] = 0;
		} else {
			null_flags[i] = 1;
			layout.any_null = true;
		}
	}
}

bool GpuBatchAccumulator::Append(DataChunk &chunk, const duckdb::SelectionVector *sel, duckdb::idx_t count) {
	std::lock_guard<std::mutex> guard(mutex_);
	if (chunk.ColumnCount() != layout_.size()) {
		throw GpuBatchAccumulatorError("GpuBatchAccumulator::Append: chunk has " +
		                               std::to_string(chunk.ColumnCount()) + " column(s), expected " +
		                               std::to_string(layout_.size()));
	}
	if (ready_) {
		throw GpuBatchAccumulatorError("GpuBatchAccumulator::Append: called again before TakeReadyBatch() "
		                               "collected the previous ready batch");
	}
	if (count == 0) {
		return false; // matches AppendChunkToAccumulators' "empty chunk is a no-op" contract
	}
	if (static_cast<uint64_t>(row_count_) + count > batch_rows_) {
		// Not reachable for an ordinary table scan (STANDARD_VECTOR_SIZE chunks divide evenly into
		// batch_rows_), but a caller handing in an oversized or oddly-sized chunk must fail cleanly
		// here rather than write past this column's reserved region into the next column's.
		throw GpuBatchAccumulatorError("GpuBatchAccumulator::Append: chunk would overrun the active batch (" +
		                               std::to_string(row_count_) + " + " + std::to_string(count) + " > " +
		                               std::to_string(batch_rows_) + " rows) -- call TakeReadyBatch() first");
	}

	for (idx_t i = 0; i < chunk.ColumnCount(); i++) {
		auto &col = layout_[i];
		switch (col.type) {
		case GpuValueType::INT16:
			AppendTypedColumn<int16_t>(chunk.data[i], count, col, sel);
			break;
		case GpuValueType::INT32:
			AppendTypedColumn<int32_t>(chunk.data[i], count, col, sel);
			break;
		case GpuValueType::INT64:
			AppendTypedColumn<int64_t>(chunk.data[i], count, col, sel);
			break;
		case GpuValueType::HUGEINT:
			AppendTypedColumn<hugeint_t>(chunk.data[i], count, col, sel);
			break;
		case GpuValueType::FLOAT32:
			AppendTypedColumn<float>(chunk.data[i], count, col, sel);
			break;
		case GpuValueType::FLOAT64:
			AppendTypedColumn<double>(chunk.data[i], count, col, sel);
			break;
		case GpuValueType::BOOLEAN:
			AppendTypedColumn<bool>(chunk.data[i], count, col, sel);
			break;
		case GpuValueType::DICTIONARY_STRING:
			throw GpuBatchAccumulatorError("dictionary-encoded strings are not supported by GpuBatchAccumulator");
		}
	}

	row_count_ += count;
	if (row_count_ >= batch_rows_) {
		ready_ = true;
	}
	return ready_;
}

GpuPinnedBatch GpuBatchAccumulator::TakeReadyBatch() {
	std::lock_guard<std::mutex> guard(mutex_);
	GpuPinnedBatch batch;
	batch.row_count = row_count_;
	batch.columns.reserve(layout_.size());

	auto *base = static_cast<uint8_t *>(slot_.data());
	for (auto &col : layout_) {
		GpuColumn column;
		column.name = col.name;
		column.type = col.type;
		column.row_count = row_count_;
		column.data = base + col.data_offset;
		if (col.any_null) {
			// Packs the unpacked (1 byte/row) scratch into the Arrow-bitmap format GpuColumn::validity
			// documents -- LSB-first, bit set = valid -- in place, still inside this slot's pinned
			// memory. Matches FinalizeGpuColumnAccumulators' pack step exactly, minus the heap buffer.
			auto *packed = base + col.packed_offset;
			auto packed_bytes = (static_cast<size_t>(row_count_) + 7) / 8;
			std::memset(packed, 0xFF, packed_bytes);
			auto *null_flags = base + col.null_flag_offset;
			for (idx_t r = 0; r < row_count_; r++) {
				if (null_flags[r]) {
					packed[r / 8] &= ~(uint8_t(1) << (r % 8));
				}
			}
			column.validity = packed;
		}
		batch.columns.push_back(std::move(column));
	}

	batch.slot = std::move(slot_); // this accumulator's slot_ is now empty; must reacquire before reuse
	AcquireFreshSlot();
	row_count_ = 0;
	ready_ = false;
	for (auto &col : layout_) {
		col.any_null = false;
	}
	return batch;
}

} // namespace vector_gpu
