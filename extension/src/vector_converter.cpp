#include "vector_converter.hpp"

#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/vector/unified_vector_format.hpp"
#include "duckdb/common/vector_size.hpp"

#include <cstring>

namespace vector_gpu {

using namespace duckdb;

namespace {
//! Mirrors expression_translator.cpp's (anonymous-namespace) MapLogicalType and table_checker.cpp's
//! IsSupportedType exactly -- all three must stay in sync, or a column could pass the Table Checker /
//! translate successfully but then fail (or silently misconvert) here. Deliberately not shared via a
//! common header: each file's version is intentionally-duplicated, self-contained logic close to its own
//! use, matching the pattern already established in table_checker.cpp's own comment about this.
GpuValueType MapLogicalType(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::DECIMAL:
		switch (type.InternalType()) {
		case PhysicalType::INT16:
			return GpuValueType::INT16;
		case PhysicalType::INT32:
			return GpuValueType::INT32;
		case PhysicalType::INT64:
			return GpuValueType::INT64;
		case PhysicalType::INT128:
			return GpuValueType::HUGEINT;
		default:
			throw GpuUnsupportedVector("unsupported decimal width");
		}
	case LogicalTypeId::HUGEINT:
		return GpuValueType::HUGEINT;
	case LogicalTypeId::SMALLINT:
		return GpuValueType::INT16;
	case LogicalTypeId::INTEGER:
		return GpuValueType::INT32;
	case LogicalTypeId::BIGINT:
		return GpuValueType::INT64;
	case LogicalTypeId::FLOAT:
		return GpuValueType::FLOAT32;
	case LogicalTypeId::DOUBLE:
		return GpuValueType::FLOAT64;
	case LogicalTypeId::BOOLEAN:
		return GpuValueType::BOOLEAN;
	default:
		throw GpuUnsupportedVector("unsupported type for GPU column conversion: " + type.ToString());
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
	case GpuValueType::FLOAT32:
		return sizeof(float);
	case GpuValueType::FLOAT64:
		return sizeof(double);
	case GpuValueType::BOOLEAN:
		return sizeof(bool);
	case GpuValueType::DICTIONARY_STRING:
		throw GpuUnsupportedVector("dictionary-encoded strings are not supported by vector_converter");
	}
	throw GpuUnsupportedVector("unknown GpuValueType");
}

bool BitIsSet(const uint8_t *bitmap, idx_t row) {
	return (bitmap[row / 8] & (uint8_t(1) << (row % 8))) != 0;
}

//! Copies `count` logical rows of `vec` into a freshly allocated, tightly-packed buffer, correctly
//! handling FLAT, CONSTANT, and DICTIONARY encodings uniformly via ToUnifiedFormat -- rather than
//! requiring the caller to pre-flatten. This matters for a reason found the hard way (see
//! docs/EXECUTION_TRACKER.md): DataChunk::Flatten()/Vector::Flatten() use the vector BUFFER's own
//! internal Size() to decide how many elements to materialize, which for a constant vector is 1, not the
//! chunk's actual row count -- so naively flattening first and then reading `count` elements silently
//! reads garbage past the single real value. ToUnifiedFormat's selection vector (a real all-zeros
//! selection for constant vectors, confirmed by reading ConstantVector::ZeroSelectionVector's actual
//! implementation) correctly maps every logical row to the right physical value regardless of encoding.
template <class T>
void CopyTyped(Vector &vec, idx_t count, uint8_t *dest_bytes, std::vector<uint8_t> *validity_out) {
	UnifiedVectorFormat format;
	vec.ToUnifiedFormat(format);
	auto *typed_data = UnifiedVectorFormat::GetData<T>(format);
	auto *dest = reinterpret_cast<T *>(dest_bytes);

	bool any_null = false;
	for (idx_t i = 0; i < count; i++) {
		auto phys_idx = format.sel->get_index(i);
		dest[i] = typed_data[phys_idx];
		if (!format.validity.RowIsValid(phys_idx)) {
			any_null = true;
		}
	}
	if (any_null && validity_out) {
		validity_out->assign((count + 7) / 8, 0xFF);
		for (idx_t i = 0; i < count; i++) {
			if (!format.validity.RowIsValid(format.sel->get_index(i))) {
				(*validity_out)[i / 8] &= ~(uint8_t(1) << (i % 8));
			}
		}
	}
}

//! Appends `count` rows of `vec` onto the end of `data` (growing it) and appends one byte per row (1 =
//! null, 0 = valid) onto the end of `null_flags` -- the unpacked format GpuColumnAccumulator uses so that
//! chunk boundaries never land mid-byte the way a packed bitmap's would.
template <class T>
void AppendTyped(Vector &vec, idx_t count, std::vector<uint8_t> &data, std::vector<uint8_t> &null_flags) {
	UnifiedVectorFormat format;
	vec.ToUnifiedFormat(format);
	auto *typed_data = UnifiedVectorFormat::GetData<T>(format);

	auto old_size = data.size();
	data.resize(old_size + count * sizeof(T));
	auto *dest = reinterpret_cast<T *>(data.data() + old_size);

	auto null_old_size = null_flags.size();
	null_flags.resize(null_old_size + count, 0);

	for (idx_t i = 0; i < count; i++) {
		auto phys_idx = format.sel->get_index(i);
		dest[i] = typed_data[phys_idx];
		if (!format.validity.RowIsValid(phys_idx)) {
			null_flags[null_old_size + i] = 1;
		}
	}
}
} // namespace

GpuColumn ConvertVectorToGpuColumn(Vector &vec, idx_t count, std::string name,
                                   std::vector<std::vector<uint8_t>> &owned_buffers) {
	GpuColumn column;
	column.name = std::move(name);
	column.type = MapLogicalType(vec.GetType());
	column.row_count = count;

	auto value_size = GpuValueTypeSize(column.type);
	std::vector<uint8_t> data_buffer(value_size * count);
	std::vector<uint8_t> validity_buffer; // stays empty unless a null is actually found

	if (count > 0) {
		switch (column.type) {
		case GpuValueType::INT16:
			CopyTyped<int16_t>(vec, count, data_buffer.data(), &validity_buffer);
			break;
		case GpuValueType::INT32:
			CopyTyped<int32_t>(vec, count, data_buffer.data(), &validity_buffer);
			break;
		case GpuValueType::INT64:
			CopyTyped<int64_t>(vec, count, data_buffer.data(), &validity_buffer);
			break;
		case GpuValueType::FLOAT32:
			CopyTyped<float>(vec, count, data_buffer.data(), &validity_buffer);
			break;
		case GpuValueType::FLOAT64:
			CopyTyped<double>(vec, count, data_buffer.data(), &validity_buffer);
			break;
		case GpuValueType::BOOLEAN:
			CopyTyped<bool>(vec, count, data_buffer.data(), &validity_buffer);
			break;
		case GpuValueType::DICTIONARY_STRING:
			throw GpuUnsupportedVector("dictionary-encoded strings are not supported by vector_converter");
		}
	}

	owned_buffers.push_back(std::move(data_buffer));
	column.data = owned_buffers.back().data();

	if (!validity_buffer.empty()) {
		owned_buffers.push_back(std::move(validity_buffer));
		column.validity = owned_buffers.back().data();
	}

	return column;
}

std::vector<GpuColumn> ConvertDataChunkToGpuColumns(DataChunk &chunk, const std::vector<std::string> &names,
                                                    std::vector<std::vector<uint8_t>> &owned_buffers) {
	if (names.size() != chunk.ColumnCount()) {
		throw GpuUnsupportedVector("ConvertDataChunkToGpuColumns: names.size() (" + std::to_string(names.size()) +
		                           ") does not match chunk.ColumnCount() (" +
		                           std::to_string(chunk.ColumnCount()) + ")");
	}

	std::vector<GpuColumn> columns;
	columns.reserve(chunk.ColumnCount());
	for (idx_t i = 0; i < chunk.ColumnCount(); i++) {
		columns.push_back(ConvertVectorToGpuColumn(chunk.data[i], chunk.size(), names[i], owned_buffers));
	}
	return columns;
}

void ConvertGpuColumnToVector(const GpuColumn &column, idx_t count, Vector &target) {
	if (target.GetVectorType() != VectorType::FLAT_VECTOR) {
		throw GpuUnsupportedVector("ConvertGpuColumnToVector requires an already-flat target vector for column '" +
		                           column.name + "'");
	}
	auto expected_type = MapLogicalType(target.GetType());
	if (expected_type != column.type) {
		throw GpuUnsupportedVector("ConvertGpuColumnToVector: type mismatch for column '" + column.name +
		                           "' (target vector's type doesn't match column.type)");
	}
	if (column.row_count != count) {
		throw GpuUnsupportedVector("ConvertGpuColumnToVector: row_count mismatch for column '" + column.name +
		                           "' (column has " + std::to_string(column.row_count) + ", expected " +
		                           std::to_string(count) + ")");
	}

	auto value_size = GpuValueTypeSize(column.type);
	if (count > 0) {
		if (!column.data) {
			throw GpuUnsupportedVector("ConvertGpuColumnToVector: column '" + column.name +
			                          "' has row_count > 0 but a null data pointer");
		}
		std::memcpy(FlatVector::GetDataMutable(target), column.data, value_size * count);
	}

	auto &target_validity = FlatVector::ValidityMutable(target);
	if (column.validity) {
		for (idx_t i = 0; i < count; i++) {
			if (!BitIsSet(column.validity, i)) {
				target_validity.SetInvalid(i);
			}
		}
	}
	// column.validity == nullptr means "no nulls" -- target_validity is left at its default (freshly
	// initialized flat vectors in DuckDB start all-valid), so there's nothing to do in that case.
}

void ConvertGpuColumnRangeToVector(const GpuColumn &column, idx_t offset, idx_t count, Vector &target) {
	if (target.GetVectorType() != VectorType::FLAT_VECTOR) {
		throw GpuUnsupportedVector("ConvertGpuColumnRangeToVector requires an already-flat target vector for "
		                          "column '" +
		                          column.name + "'");
	}
	auto expected_type = MapLogicalType(target.GetType());
	if (expected_type != column.type) {
		throw GpuUnsupportedVector("ConvertGpuColumnRangeToVector: type mismatch for column '" + column.name +
		                           "' (target vector's type doesn't match column.type)");
	}
	if (offset + count > column.row_count) {
		throw GpuUnsupportedVector("ConvertGpuColumnRangeToVector: range [" + std::to_string(offset) + ", " +
		                           std::to_string(offset + count) + ") exceeds column '" + column.name +
		                           "'s row_count (" + std::to_string(column.row_count) + ")");
	}

	auto value_size = GpuValueTypeSize(column.type);
	if (count > 0) {
		if (!column.data) {
			throw GpuUnsupportedVector("ConvertGpuColumnRangeToVector: column '" + column.name +
			                          "' has row_count > 0 but a null data pointer");
		}
		auto *src = static_cast<const uint8_t *>(column.data) + offset * value_size;
		std::memcpy(FlatVector::GetDataMutable(target), src, value_size * count);
	}

	auto &target_validity = FlatVector::ValidityMutable(target);
	if (column.validity) {
		for (idx_t i = 0; i < count; i++) {
			if (!BitIsSet(column.validity, offset + i)) {
				target_validity.SetInvalid(i);
			}
		}
	}
	// column.validity == nullptr means "no nulls anywhere in the whole column" -- nothing to do for this
	// range either, same reasoning as ConvertGpuColumnToVector.
}

void PackGpuColumnsIntoCollection(const std::vector<GpuColumn> &columns, const std::vector<LogicalType> &types,
                                 ColumnDataCollection &collection) {
	if (columns.size() != types.size()) {
		throw GpuUnsupportedVector("PackGpuColumnsIntoCollection: columns.size() (" + std::to_string(columns.size()) +
		                           ") does not match types.size() (" + std::to_string(types.size()) + ")");
	}
	if (columns.empty()) {
		return; // nothing to pack -- a zero-column result is a degenerate but valid case (e.g. a query
		        // whose SELECT list somehow yields no columns), not an error.
	}
	auto total_rows = columns[0].row_count;
	for (auto &column : columns) {
		if (column.row_count != total_rows) {
			throw GpuUnsupportedVector("PackGpuColumnsIntoCollection: column '" + column.name +
			                          "' has row_count " + std::to_string(column.row_count) +
			                          ", but column '" + columns[0].name + "' has " + std::to_string(total_rows) +
			                          " -- all result columns must agree on row count");
		}
	}

	DataChunk chunk;
	// duckdb::vector (not std::vector) is what DataChunk::Initialize requires -- the two are distinct
	// template instantiations that don't implicitly convert (same issue hit in table_scanner.cpp).
	chunk.Initialize(Allocator::DefaultAllocator(), vector<LogicalType>(types.begin(), types.end()));
	for (idx_t offset = 0; offset < total_rows; offset += STANDARD_VECTOR_SIZE) {
		auto batch_size = MinValue<idx_t>(STANDARD_VECTOR_SIZE, total_rows - offset);
		chunk.Reset();
		for (idx_t col = 0; col < columns.size(); col++) {
			ConvertGpuColumnRangeToVector(columns[col], offset, batch_size, chunk.data[col]);
		}
		chunk.SetCardinality(batch_size);
		collection.Append(chunk);
	}
}

std::vector<GpuColumnAccumulator> MakeGpuColumnAccumulators(const std::vector<std::string> &names,
                                                            const std::vector<LogicalType> &types) {
	if (names.size() != types.size()) {
		throw GpuUnsupportedVector("MakeGpuColumnAccumulators: names.size() (" + std::to_string(names.size()) +
		                           ") does not match types.size() (" + std::to_string(types.size()) + ")");
	}
	std::vector<GpuColumnAccumulator> accumulators;
	accumulators.reserve(names.size());
	for (size_t i = 0; i < names.size(); i++) {
		GpuColumnAccumulator acc;
		acc.name = names[i];
		acc.type = MapLogicalType(types[i]); // throws immediately for an unsupported type, before any scanning
		accumulators.push_back(std::move(acc));
	}
	return accumulators;
}

void AppendChunkToAccumulators(DataChunk &chunk, std::vector<GpuColumnAccumulator> &accumulators) {
	if (chunk.size() == 0) {
		return; // matches a table scan's own "no more data" signal -- nothing to append
	}
	if (accumulators.size() != chunk.ColumnCount()) {
		throw GpuUnsupportedVector("AppendChunkToAccumulators: accumulators.size() (" +
		                           std::to_string(accumulators.size()) + ") does not match chunk.ColumnCount() (" +
		                           std::to_string(chunk.ColumnCount()) + ")");
	}
	auto count = chunk.size();
	for (idx_t i = 0; i < chunk.ColumnCount(); i++) {
		auto &acc = accumulators[i];
		switch (acc.type) {
		case GpuValueType::INT16:
			AppendTyped<int16_t>(chunk.data[i], count, acc.data, acc.null_flags);
			break;
		case GpuValueType::INT32:
			AppendTyped<int32_t>(chunk.data[i], count, acc.data, acc.null_flags);
			break;
		case GpuValueType::INT64:
			AppendTyped<int64_t>(chunk.data[i], count, acc.data, acc.null_flags);
			break;
		case GpuValueType::HUGEINT:
			AppendTyped<hugeint_t>(chunk.data[i], count, acc.data, acc.null_flags);
			break;
		case GpuValueType::FLOAT32:
			AppendTyped<float>(chunk.data[i], count, acc.data, acc.null_flags);
			break;
		case GpuValueType::FLOAT64:
			AppendTyped<double>(chunk.data[i], count, acc.data, acc.null_flags);
			break;
		case GpuValueType::BOOLEAN:
			AppendTyped<bool>(chunk.data[i], count, acc.data, acc.null_flags);
			break;
		case GpuValueType::DICTIONARY_STRING:
			throw GpuUnsupportedVector("dictionary-encoded strings are not supported by vector_converter");
		}
		acc.row_count += count;
	}
}

std::vector<GpuColumn> FinalizeGpuColumnAccumulators(std::vector<GpuColumnAccumulator> &accumulators,
                                                     std::vector<std::vector<uint8_t>> &owned_buffers) {
	std::vector<GpuColumn> columns;
	columns.reserve(accumulators.size());
	for (auto &acc : accumulators) {
		GpuColumn column;
		column.name = std::move(acc.name);
		column.type = acc.type;
		column.row_count = acc.row_count;

		owned_buffers.push_back(std::move(acc.data));
		column.data = owned_buffers.back().data();

		bool any_null = false;
		for (auto flag : acc.null_flags) {
			if (flag) {
				any_null = true;
				break;
			}
		}
		if (any_null) {
			std::vector<uint8_t> packed((acc.row_count + 7) / 8, 0xFF);
			for (uint64_t i = 0; i < acc.row_count; i++) {
				if (acc.null_flags[i]) {
					packed[i / 8] &= ~(uint8_t(1) << (i % 8));
				}
			}
			owned_buffers.push_back(std::move(packed));
			column.validity = owned_buffers.back().data();
		}

		columns.push_back(std::move(column));
	}
	return columns;
}

} // namespace vector_gpu
