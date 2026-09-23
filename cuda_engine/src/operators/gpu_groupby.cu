// gpu_groupby.cu
//
// Real GPU GROUP BY ... AGGREGATE. This replaces the `ExecuteGroupByStub` that used to live here and
// return "not implemented — libcudf unavailable in this build environment".
//
// WHY THIS IS NOT libcudf. The original design (and docs/KNOWN_ISSUES.md) treated group-by as blocked on
// RAPIDS libcudf, which has no native Windows build. That was a design assumption, not a technical
// requirement: the grouping and reduction primitives needed here are plain Thrust, which is already part
// of the CUDA Toolkit and already what operators/gpu_filter.cu and gpu_executor.cu are built on. Doing it
// directly in Thrust removes the WSL2/BIOS dependency for this operator entirely. libcudf remains the
// better long-term answer for the harder cases (string keys, null-aware aggregation, spill) — it is not
// needed for the fixed-width, non-null subset the rest of this engine already supports.
//
// ALGORITHM (sort-based grouping, one pass per key column plus one pass per aggregate):
//   1. Widen every group-key column to uint64 so the sort/compare code is type-free. Keys are grouped by
//      raw bit pattern, so equality of widened values is exactly equality of the original values.
//   2. Sort a permutation of row indices by the key columns, least-significant key first with a STABLE
//      sort each pass (an LSD radix ordering) — after the last pass rows with equal keys are adjacent.
//   3. Mark the first row of each group (any key differing from the previous row), inclusive-scan those
//      flags into a per-row group id, and read the group count off the last element.
//   4. Gather each group's representative row to produce the output key columns; run one Thrust
//      reduce_by_key per aggregate to produce the output aggregate columns.
//
// DELIBERATE RESTRICTIONS (each is a clean refusal -> CPU fallback, never a wrong answer):
//   - Integer/boolean group keys only. FLOAT keys would group by bit pattern, under which -0.0 and +0.0
//     land in different groups and NaN never equals itself — both differ from DuckDB's grouping.
//   - MIN/MAX on integer/boolean only. DuckDB orders NaN above every other value; thrust::minimum/
//     maximum use operator<, which is false for NaN, so a NaN would silently change the answer.
//   - SUM on FLOAT/DOUBLE only, producing DOUBLE. DuckDB's SUM over integers returns HUGEINT, which this
//     engine has no type for (TableChecker rejects such a plan before it ever reaches here).
//   - Nullable input columns are already refused upstream (gpu_executor.cu), which is also why
//     COUNT(col) is computed identically to COUNT(*) here: with no nulls present the two agree by
//     definition.
//
// Floating-point SUM is order-dependent: a parallel reduction adds in a different order than DuckDB's
// CPU aggregate, so results can differ in the last ULP. DuckDB's own aggregate is likewise parallel and
// not order-stable, so this is inherent to both, not a divergence introduced here.

#include "gpu_executor_internal.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

namespace vector_gpu {

namespace {

void SyncOrThrow(const char *what) {
	auto status = cudaDeviceSynchronize();
	if (status != cudaSuccess) {
		throw std::runtime_error(std::string("gpu_groupby: ") + what + " failed: " + cudaGetErrorString(status));
	}
}

//! Widens a fixed-width key column to uint64. Integer/boolean only (see the file header): the widened
//! value is the key's raw bit pattern, zero-extended, so distinct keys stay distinct.
__global__ void WidenKeyKernel(const void *src, unsigned type_size, uint64_t *dst, uint64_t n) {
	uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
	for (uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
		switch (type_size) {
		case 1:
			dst[idx] = static_cast<const uint8_t *>(src)[idx];
			break;
		case 2:
			dst[idx] = static_cast<const uint16_t *>(src)[idx];
			break;
		case 4:
			dst[idx] = static_cast<const uint32_t *>(src)[idx];
			break;
		default:
			dst[idx] = static_cast<const uint64_t *>(src)[idx];
			break;
		}
	}
}

//! Sets head[idx] wherever this key column differs from the previous row's, in sorted order. Called once
//! per key column against the same `head` buffer, so a row starts a new group if ANY key changed.
__global__ void MarkKeyChangeKernel(const uint64_t *sorted_key, uint64_t *head, uint64_t n) {
	uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
	for (uint64_t idx = 1 + blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
		if (sorted_key[idx] != sorted_key[idx - 1]) {
			head[idx] = 1;
		}
	}
}

constexpr unsigned kBlockSize = 256;

unsigned GridFor(uint64_t n) {
	uint64_t grid = (n + kBlockSize - 1) / kBlockSize;
	return grid > 65535u ? 65535u : static_cast<unsigned>(grid); // grid-stride loops cover the remainder
}

template <typename T> struct LimitTraits {};
template <> struct LimitTraits<int16_t> { static __device__ int16_t Min() { return -32768; } static __device__ int16_t Max() { return 32767; } };
template <> struct LimitTraits<int32_t> { static __device__ int32_t Min() { return -2147483647 - 1; } static __device__ int32_t Max() { return 2147483647; } };
template <> struct LimitTraits<int64_t> { static __device__ int64_t Min() { return -9223372036854775807LL - 1; } static __device__ int64_t Max() { return 9223372036854775807LL; } };
template <> struct LimitTraits<uint8_t> { static __device__ uint8_t Min() { return 0; } static __device__ uint8_t Max() { return 255; } };

template <typename T>
struct NullSafeMin {
	__host__ __device__ T operator()(const thrust::tuple<T, bool>& t) const {
		return thrust::get<1>(t) ? thrust::get<0>(t) : LimitTraits<T>::Max();
	}
};

template <typename T>
struct NullSafeMax {
	__host__ __device__ T operator()(const thrust::tuple<T, bool>& t) const {
		return thrust::get<1>(t) ? thrust::get<0>(t) : LimitTraits<T>::Min();
	}
};

struct NullSafeCount {
	__host__ __device__ int64_t operator()(bool valid) const {
		return valid ? 1 : 0;
	}
};

struct IsNonZero {
	__host__ __device__ bool operator()(uint64_t value) const {
		return value != 0;
	}
};

template <typename T>
struct IntToDouble {
	__host__ __device__ double operator()(const thrust::tuple<T, bool>& t) const {
		return thrust::get<1>(t) ? static_cast<double>(thrust::get<0>(t)) : 0.0;
	}
	__host__ __device__ double operator()(T value) const {
		return static_cast<double>(value);
	}
};

struct DivideByCount {
	__host__ __device__ double operator()(double sum, int64_t count) const {
		return count == 0 ? 0.0 : sum / static_cast<double>(count);
	}
};

struct FloatToDouble {
	__host__ __device__ double operator()(const thrust::tuple<float, bool>& t) const {
		return thrust::get<1>(t) ? static_cast<double>(thrust::get<0>(t)) : 0.0;
	}
	__host__ __device__ double operator()(float value) const {
		return static_cast<double>(value);
	}
};

template <typename T, typename Iterator, typename Op>
void ReduceByGroup(const uint64_t *group_id, uint64_t rows, Iterator vals, void *out, Op op) {
	thrust::device_ptr<const uint64_t> keys(group_id);
	thrust::device_ptr<T> dst(static_cast<T *>(out));
	thrust::reduce_by_key(thrust::device, keys, keys + rows, vals, thrust::make_discard_iterator(), dst,
	                      thrust::equal_to<uint64_t>(), op);
}

//! MIN/MAX dispatch over the types this path accepts. Separate from ReduceByGroup so the min/max functor
//! is instantiated once per type rather than once per (type, direction) at every call site.
void ReduceMinMax(GpuValueType type, bool want_min, const uint64_t *group_id, uint64_t rows, const void *values,
                  const bool *valid, void *out) {
	auto do_reduce = [&](auto dummy) {
		using T = decltype(dummy);
		thrust::device_ptr<const T> val_ptr(static_cast<const T *>(values));
		if (valid) {
			thrust::device_ptr<const bool> valid_ptr(valid);
			auto zipped = thrust::make_zip_iterator(thrust::make_tuple(val_ptr, valid_ptr));
			if (want_min) {
				ReduceByGroup<T>(group_id, rows, thrust::make_transform_iterator(zipped, NullSafeMin<T>()), out, thrust::minimum<T>());
			} else {
				ReduceByGroup<T>(group_id, rows, thrust::make_transform_iterator(zipped, NullSafeMax<T>()), out, thrust::maximum<T>());
			}
		} else {
			if (want_min) {
				ReduceByGroup<T>(group_id, rows, val_ptr, out, thrust::minimum<T>());
			} else {
				ReduceByGroup<T>(group_id, rows, val_ptr, out, thrust::maximum<T>());
			}
		}
	};
	switch (type) {
	case GpuValueType::INT16:
		do_reduce(int16_t());
		return;
	case GpuValueType::INT32:
		do_reduce(int32_t());
		return;
	case GpuValueType::INT64:
		do_reduce(int64_t());
		return;
	case GpuValueType::BOOLEAN:
		do_reduce(uint8_t());
		return;
	default:
		// See the file header: float MIN/MAX would disagree with DuckDB's NaN ordering.
		throw std::runtime_error("gpu_groupby: MIN/MAX on this GPU path supports only integer/boolean columns");
	}
}

//! Validates an aggregate against what this file can actually execute, before any device work happens.
//! Mirrors the checks gpu_offload_extension.cpp makes at translation time — this is the engine-side
//! backstop for a plan built by hand (or by a future caller) rather than by the translator.
void ValidateAggregate(const GpuAggregate &aggregate, const std::vector<DeviceColumn> &child_columns, uint64_t rows) {
	if (aggregate.kind == GpuAggregateKind::COUNT_STAR) {
		if (aggregate.output_type != GpuValueType::INT64) {
			throw std::runtime_error("gpu_groupby: COUNT(*) must produce INT64");
		}
		return;
	}
	auto &column = FindColumn(child_columns, aggregate.input_column);
	if (column.rows != rows) {
		throw std::runtime_error("gpu_groupby: aggregate input column '" + aggregate.input_column +
		                         "' row count mismatch");
	}
	switch (aggregate.kind) {
	case GpuAggregateKind::COUNT:
		if (aggregate.output_type != GpuValueType::INT64) {
			throw std::runtime_error("gpu_groupby: COUNT must produce INT64");
		}
		return;
	case GpuAggregateKind::SUM:
		if (aggregate.output_type != GpuValueType::FLOAT64 ||
		    (column.type != GpuValueType::FLOAT32 && column.type != GpuValueType::FLOAT64)) {
			throw std::runtime_error("gpu_groupby: SUM on this GPU path supports only FLOAT/DOUBLE input "
			                         "producing DOUBLE");
		}
		return;
	case GpuAggregateKind::AVG:
		// DuckDB's AVG returns DOUBLE for every numeric input type, so the accumulation is in double
		// regardless of the column type.
		if (aggregate.output_type != GpuValueType::FLOAT64) {
			throw std::runtime_error("gpu_groupby: AVG must produce DOUBLE");
		}
		if (column.type != GpuValueType::INT16 && column.type != GpuValueType::INT32 && column.type != GpuValueType::INT64 &&
		    column.type != GpuValueType::FLOAT32 && column.type != GpuValueType::FLOAT64) {
			throw std::runtime_error("gpu_groupby: AVG supports numeric columns only");
		}
		return;
	case GpuAggregateKind::MIN:
	case GpuAggregateKind::MAX:
		if (aggregate.output_type != column.type) {
			throw std::runtime_error("gpu_groupby: MIN/MAX output type must match its input column type");
		}
		if (column.type != GpuValueType::INT16 && column.type != GpuValueType::INT32 && column.type != GpuValueType::INT64 &&
		    column.type != GpuValueType::BOOLEAN) {
			throw std::runtime_error("gpu_groupby: MIN/MAX on this GPU path supports only integer/boolean "
			                         "columns");
		}
		return;
	default:
		throw std::runtime_error("gpu_groupby: unknown aggregate kind");
	}
}

//! Builds the zero-row output shape (correct whenever the input has no rows AND the query groups by
//! something — with no rows there are simply no groups).
std::vector<DeviceColumn> EmptyGroupByResult(const GpuPlanNode &node, const std::vector<DeviceColumn> &child_columns,
                                             ExecContext &ctx) {
	std::vector<DeviceColumn> result;
	result.reserve(node.group_keys.size() + node.aggregates.size());
	for (auto &key_name : node.group_keys) {
		DeviceColumn column;
		column.name = key_name;
		column.type = FindColumn(child_columns, key_name).type;
		column.rows = 0;
		column.data = ctx.arena.Alloc(0);
		result.push_back(std::move(column));
	}
	for (size_t i = 0; i < node.aggregates.size(); i++) {
		DeviceColumn column;
		column.name = "agg_" + std::to_string(i);
		column.type = node.aggregates[i].output_type;
		column.rows = 0;
		column.data = ctx.arena.Alloc(0);
		result.push_back(std::move(column));
	}
	return result;
}

} // namespace

std::vector<DeviceColumn> ExecuteGroupByNode(const GpuPlanNode &node, const std::vector<DeviceColumn> &child_columns,
                                             ExecContext &ctx) {
	if (child_columns.empty()) {
		throw std::runtime_error("gpu_groupby: input operator produced no columns to group");
	}
	if (node.group_keys.empty() && node.aggregates.empty()) {
		throw std::runtime_error("gpu_groupby: node has neither group keys nor aggregates");
	}
	if (node.aggregates.size() != node.aggregate_exprs.size() && !node.aggregate_exprs.empty()) {
		throw std::runtime_error("gpu_groupby: aggregates and aggregate_exprs disagree in length");
	}
	auto rows = child_columns.front().rows;

	// Resolve and validate the group keys before touching the device.
	std::vector<const DeviceColumn *> key_columns;
	key_columns.reserve(node.group_keys.size());
	for (auto &key_name : node.group_keys) {
		auto &column = FindColumn(child_columns, key_name);
		if (column.rows != rows) {
			throw std::runtime_error("gpu_groupby: group key '" + key_name + "' row count mismatch");
		}
		switch (column.type) {
		case GpuValueType::INT16:
		case GpuValueType::INT32:
		case GpuValueType::INT64:
		case GpuValueType::BOOLEAN:
			break;
		default:
			// See the file header: grouping floats by bit pattern disagrees with DuckDB for -0.0/NaN.
			throw std::runtime_error("gpu_groupby: group key '" + key_name +
			                         "' is not an integer/boolean column, which this GPU path requires");
		}
		key_columns.push_back(&column);
	}
	for (auto &aggregate : node.aggregates) {
		ValidateAggregate(aggregate, child_columns, rows);
	}

	if (rows == 0) {
		if (node.group_keys.empty()) {
			// A global aggregate over an empty input is one row, and DuckDB returns NULL for
			// SUM/MIN/MAX there. This path carries no validity bitmap, so refuse rather than emit 0.
			throw std::runtime_error("gpu_groupby: global aggregate over an empty input needs NULL results, "
			                         "which this GPU path cannot represent");
		}
		return EmptyGroupByResult(node, child_columns, ctx);
	}

	const auto key_count = key_columns.size();
	auto *permutation = static_cast<uint64_t *>(ctx.arena.Alloc(static_cast<size_t>(rows) * sizeof(uint64_t)));
	thrust::device_ptr<uint64_t> permutation_ptr(permutation);
	thrust::sequence(thrust::device, permutation_ptr, permutation_ptr + rows);

	// 1. Widen each key column once; every later pass works on uint64 regardless of the source type.
	std::vector<uint64_t *> widened(key_count);
	for (size_t k = 0; k < key_count; k++) {
		widened[k] = static_cast<uint64_t *>(ctx.arena.Alloc(static_cast<size_t>(rows) * sizeof(uint64_t)));
		WidenKeyKernel<<<GridFor(rows), kBlockSize>>>(key_columns[k]->data,
		                                              static_cast<unsigned>(TypeSize(key_columns[k]->type)),
		                                              widened[k], rows);
	}
	SyncOrThrow("group-key widening kernel");

	// 2. LSD ordering: stable-sort by the last key first, so after the final pass the permutation orders
	//    rows lexicographically by the whole key tuple and equal keys are adjacent.
	auto *sort_key = static_cast<uint64_t *>(ctx.arena.Alloc(static_cast<size_t>(rows) * sizeof(uint64_t)));
	thrust::device_ptr<uint64_t> sort_key_ptr(sort_key);
	for (size_t pass = 0; pass < key_count; pass++) {
		auto k = key_count - 1 - pass;
		thrust::device_ptr<const uint64_t> source(widened[k]);
		thrust::gather(thrust::device, permutation_ptr, permutation_ptr + rows, source, sort_key_ptr);
		thrust::stable_sort_by_key(thrust::device, sort_key_ptr, sort_key_ptr + rows, permutation_ptr);
	}
	SyncOrThrow("group-key sort");

	// 3. Group boundaries -> per-row group id. With no key columns at all this leaves a single group,
	//    which is exactly the global-aggregate case.
	auto *head = static_cast<uint64_t *>(ctx.arena.Alloc(static_cast<size_t>(rows) * sizeof(uint64_t)));
	auto status = cudaMemset(head, 0, static_cast<size_t>(rows) * sizeof(uint64_t));
	if (status != cudaSuccess) {
		throw std::runtime_error(std::string("gpu_groupby: head-flag clear failed: ") + cudaGetErrorString(status));
	}
	const uint64_t one = 1;
	status = cudaMemcpy(head, &one, sizeof(uint64_t), cudaMemcpyHostToDevice); // row 0 always starts a group
	if (status != cudaSuccess) {
		throw std::runtime_error(std::string("gpu_groupby: head-flag init failed: ") + cudaGetErrorString(status));
	}
	for (size_t k = 0; k < key_count; k++) {
		thrust::device_ptr<const uint64_t> source(widened[k]);
		thrust::gather(thrust::device, permutation_ptr, permutation_ptr + rows, source, sort_key_ptr);
		if (rows > 1) {
			MarkKeyChangeKernel<<<GridFor(rows), kBlockSize>>>(sort_key, head, rows);
		}
	}
	SyncOrThrow("group boundary detection");

	// group_id is the 1-based running group number; reduce_by_key only needs equal-and-adjacent, so the
	// scan output is used directly rather than rebased to 0.
	auto *group_id = static_cast<uint64_t *>(ctx.arena.Alloc(static_cast<size_t>(rows) * sizeof(uint64_t)));
	thrust::device_ptr<uint64_t> group_id_ptr(group_id);
	thrust::device_ptr<const uint64_t> head_ptr(head);
	thrust::inclusive_scan(thrust::device, head_ptr, head_ptr + rows, group_id_ptr);
	SyncOrThrow("group id scan");

	uint64_t group_count = 0;
	status = cudaMemcpy(&group_count, group_id + (rows - 1), sizeof(uint64_t), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		throw std::runtime_error(std::string("gpu_groupby: group count readback failed: ") +
		                         cudaGetErrorString(status));
	}
	if (group_count == 0) {
		throw std::runtime_error("gpu_groupby: computed zero groups for a non-empty input");
	}

	// 4a. One representative row per group, in the same order reduce_by_key will emit its results.
	auto *group_rows = static_cast<uint64_t *>(ctx.arena.Alloc(static_cast<size_t>(group_count) * sizeof(uint64_t)));
	thrust::device_ptr<uint64_t> group_rows_ptr(group_rows);
	thrust::copy_if(thrust::device, permutation_ptr, permutation_ptr + rows, head_ptr, group_rows_ptr, IsNonZero());
	SyncOrThrow("group representative selection");

	std::vector<DeviceColumn> result;
	result.reserve(key_count + node.aggregates.size());
	for (size_t k = 0; k < key_count; k++) {
		// GatherColumn names the result after its source, which is the group key's own name.
		result.push_back(GatherColumn(*key_columns[k], group_rows, group_count, ctx));
	}

	// 4b. One reduce_by_key per aggregate, over values permuted into group-sorted order.
	for (size_t i = 0; i < node.aggregates.size(); i++) {
		auto &aggregate = node.aggregates[i];
		DeviceColumn output;
		output.name = "agg_" + std::to_string(i); // matches CollectBindingNames' aggregate-output naming
		output.type = aggregate.output_type;
		output.rows = group_count;
		output.data = ctx.arena.Alloc(static_cast<size_t>(group_count) * TypeSize(aggregate.output_type));

		if (aggregate.kind == GpuAggregateKind::COUNT_STAR) {
			thrust::reduce_by_key(thrust::device, group_id_ptr, group_id_ptr + rows,
			                      thrust::make_constant_iterator<int64_t>(1), thrust::make_discard_iterator(),
			                      thrust::device_ptr<int64_t>(static_cast<int64_t *>(output.data)),
			                      thrust::equal_to<uint64_t>(), thrust::plus<int64_t>());
			SyncOrThrow("COUNT_STAR reduction");
			result.push_back(std::move(output));
			continue;
		}

		auto &source_column = FindColumn(child_columns, aggregate.input_column);
		auto sorted_values = GatherColumn(source_column, permutation, rows, ctx);

		if (sorted_values.valid != nullptr) {
			output.valid = static_cast<bool *>(ctx.arena.Alloc(static_cast<size_t>(group_count) * sizeof(bool)));
			thrust::device_ptr<const bool> valid_in(sorted_values.valid);
			thrust::device_ptr<bool> valid_out(output.valid);
			thrust::reduce_by_key(thrust::device, group_id_ptr, group_id_ptr + rows, valid_in, 
			                      thrust::make_discard_iterator(), valid_out, 
			                      thrust::equal_to<uint64_t>(), thrust::logical_or<bool>());
			SyncOrThrow("aggregate output validity reduction");
		}

		if (aggregate.kind == GpuAggregateKind::COUNT) {
			if (sorted_values.valid != nullptr) {
				thrust::device_ptr<const bool> valid_in(sorted_values.valid);
				thrust::reduce_by_key(thrust::device, group_id_ptr, group_id_ptr + rows,
				                      thrust::make_transform_iterator(valid_in, NullSafeCount()),
				                      thrust::make_discard_iterator(),
				                      thrust::device_ptr<int64_t>(static_cast<int64_t *>(output.data)),
				                      thrust::equal_to<uint64_t>(), thrust::plus<int64_t>());
				// COUNT never produces NULLs, so clear the validity output
				output.valid = nullptr;
			} else {
				thrust::reduce_by_key(thrust::device, group_id_ptr, group_id_ptr + rows,
				                      thrust::make_constant_iterator<int64_t>(1), thrust::make_discard_iterator(),
				                      thrust::device_ptr<int64_t>(static_cast<int64_t *>(output.data)),
				                      thrust::equal_to<uint64_t>(), thrust::plus<int64_t>());
			}
			SyncOrThrow("COUNT reduction");
			result.push_back(std::move(output));
			continue;
		}

		switch (aggregate.kind) {
		case GpuAggregateKind::SUM: {
			if (source_column.type == GpuValueType::FLOAT32) {
				thrust::device_ptr<const float> values(static_cast<const float *>(sorted_values.data));
				if (sorted_values.valid != nullptr) {
					thrust::device_ptr<const bool> valid_in(sorted_values.valid);
					auto zipped = thrust::make_zip_iterator(thrust::make_tuple(values, valid_in));
					thrust::reduce_by_key(thrust::device, group_id_ptr, group_id_ptr + rows,
					                      thrust::make_transform_iterator(zipped, FloatToDouble()),
					                      thrust::make_discard_iterator(),
					                      thrust::device_ptr<double>(static_cast<double *>(output.data)),
					                      thrust::equal_to<uint64_t>(), thrust::plus<double>());
				} else {
					thrust::reduce_by_key(thrust::device, group_id_ptr, group_id_ptr + rows,
					                      thrust::make_transform_iterator(values, FloatToDouble()),
					                      thrust::make_discard_iterator(),
					                      thrust::device_ptr<double>(static_cast<double *>(output.data)),
					                      thrust::equal_to<uint64_t>(), thrust::plus<double>());
				}
			} else {
				thrust::device_ptr<const double> values(static_cast<const double *>(sorted_values.data));
				if (sorted_values.valid != nullptr) {
					thrust::device_ptr<const bool> valid_in(sorted_values.valid);
					auto zipped = thrust::make_zip_iterator(thrust::make_tuple(values, valid_in));
					ReduceByGroup<double>(group_id, rows, thrust::make_transform_iterator(zipped, IntToDouble<double>()), output.data, thrust::plus<double>());
				} else {
					ReduceByGroup<double>(group_id, rows, values, output.data, thrust::plus<double>());
				}
			}
			break;
		}
		case GpuAggregateKind::AVG: {
			auto *sums = static_cast<double *>(
			    ctx.arena.Alloc(static_cast<size_t>(group_count) * sizeof(double)));
			auto *counts = static_cast<int64_t *>(
			    ctx.arena.Alloc(static_cast<size_t>(group_count) * sizeof(int64_t)));
			thrust::device_ptr<double> sums_ptr(sums);
			thrust::device_ptr<int64_t> counts_ptr(counts);

			auto do_avg_sum = [&](auto dummy) {
				using T = decltype(dummy);
				thrust::device_ptr<const T> values(static_cast<const T *>(sorted_values.data));
				if (sorted_values.valid != nullptr) {
					thrust::device_ptr<const bool> valid_in(sorted_values.valid);
					auto zipped = thrust::make_zip_iterator(thrust::make_tuple(values, valid_in));
					if constexpr (std::is_same_v<T, float>) {
						thrust::reduce_by_key(thrust::device, group_id_ptr, group_id_ptr + rows,
						                      thrust::make_transform_iterator(zipped, FloatToDouble()),
						                      thrust::make_discard_iterator(), sums_ptr, thrust::equal_to<uint64_t>(),
						                      thrust::plus<double>());
					} else {
						thrust::reduce_by_key(thrust::device, group_id_ptr, group_id_ptr + rows,
						                      thrust::make_transform_iterator(zipped, IntToDouble<T>()),
						                      thrust::make_discard_iterator(), sums_ptr, thrust::equal_to<uint64_t>(),
						                      thrust::plus<double>());
					}
				} else {
					if constexpr (std::is_same_v<T, float>) {
						thrust::reduce_by_key(thrust::device, group_id_ptr, group_id_ptr + rows,
						                      thrust::make_transform_iterator(values, FloatToDouble()),
						                      thrust::make_discard_iterator(), sums_ptr, thrust::equal_to<uint64_t>(),
						                      thrust::plus<double>());
					} else if constexpr (std::is_same_v<T, double>) {
						ReduceByGroup<double>(group_id, rows, values, sums, thrust::plus<double>());
					} else {
						thrust::reduce_by_key(thrust::device, group_id_ptr, group_id_ptr + rows,
						                      thrust::make_transform_iterator(values, IntToDouble<T>()),
						                      thrust::make_discard_iterator(), sums_ptr, thrust::equal_to<uint64_t>(),
						                      thrust::plus<double>());
					}
				}
			};

			switch (source_column.type) {
			case GpuValueType::FLOAT64: do_avg_sum(double()); break;
			case GpuValueType::FLOAT32: do_avg_sum(float()); break;
			case GpuValueType::INT16: do_avg_sum(int16_t()); break;
			case GpuValueType::INT32: do_avg_sum(int32_t()); break;
			default: do_avg_sum(int64_t()); break;
			}
			
			if (sorted_values.valid != nullptr) {
				thrust::device_ptr<const bool> valid_in(sorted_values.valid);
				thrust::reduce_by_key(thrust::device, group_id_ptr, group_id_ptr + rows,
				                      thrust::make_transform_iterator(valid_in, NullSafeCount()), thrust::make_discard_iterator(),
				                      counts_ptr, thrust::equal_to<uint64_t>(), thrust::plus<int64_t>());
			} else {
				thrust::reduce_by_key(thrust::device, group_id_ptr, group_id_ptr + rows,
				                      thrust::make_constant_iterator<int64_t>(1), thrust::make_discard_iterator(),
				                      counts_ptr, thrust::equal_to<uint64_t>(), thrust::plus<int64_t>());
			}

			thrust::transform(thrust::device, sums_ptr, sums_ptr + group_count, counts_ptr,
			                  thrust::device_ptr<double>(static_cast<double *>(output.data)), DivideByCount());
			break;
		}
		case GpuAggregateKind::MIN:
		case GpuAggregateKind::MAX:
			ReduceMinMax(source_column.type, aggregate.kind == GpuAggregateKind::MIN, group_id, rows,
			             sorted_values.data, sorted_values.valid, output.data);
			break;
		default:
			throw std::runtime_error("gpu_groupby: unknown aggregate kind");
		}
		SyncOrThrow("aggregate reduction");
		result.push_back(std::move(output));
	}

	return result;
}

} // namespace vector_gpu
