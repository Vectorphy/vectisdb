// gpu_hash_join.cu
//
// Real GPU INNER equi-join. This replaces the `ExecuteHashJoinStub` that used to live here and return
// "not implemented — libcudf unavailable in this build environment".
//
// WHY THIS IS NOT libcudf, and why it is not literally a hash join. The libcudf dependency was a design
// assumption, not a requirement — see gpu_groupby.cu's header for the full reasoning. The physical
// strategy here is SORT-MERGE, not hashing: `GpuOpType::HASH_JOIN` names the DuckDB *logical* operator
// being lowered, not the algorithm. Sort-merge is a deliberate choice over a hand-rolled GPU hash table,
// which would need atomics, collision handling and load-factor tuning to be correct under contention;
// the sort/binary-search primitives used below are Thrust's, already exercised by gpu_filter.cu and
// gpu_groupby.cu, and give an exact answer with far less that can go subtly wrong.
//
// ALGORITHM (one sort of the right side, then a vectorized binary search per left row):
//   1. Widen both join keys to uint64 so equality is a single type-free comparison. As in gpu_groupby.cu
//      the widened value is the key's raw bit pattern, so equal keys stay equal and distinct keys stay
//      distinct.
//   2. Sort a permutation of the right side's rows by its key.
//   3. For every left row, lower_bound/upper_bound that key in the sorted right keys. The gap between
//      them is exactly how many right rows that left row matches (0 for no match, N for a many-match).
//   4. Exclusive-scan those per-row match counts into output offsets; the total is the join's row count.
//   5. Expand: one output row per (left row, matching right row) pair, as two index vectors.
//   6. Gather each requested output column through the left or right index vector.
//
// DELIBERATE RESTRICTIONS (each a clean refusal -> CPU fallback, never a wrong answer):
//   - INNER only. Enforced at translation time already; outer/semi/anti/mark joins have different output
//     shapes and null semantics this operator does not model.
//   - Exactly ONE equi-join key column. A composite key needs the key tuple collapsed to a single
//     comparable value (the dense-id trick gpu_groupby.cu uses); not attempted here.
//   - Both key columns must be the SAME integer/boolean type. Float keys would match by bit pattern,
//     under which -0.0 != +0.0 and NaN != NaN, disagreeing with DuckDB. Mixed widths are refused
//     because widening is bit-pattern-based: INT32 -1 and INT64 -1 widen to different uint64 values, so
//     a mixed-width join would silently drop negative matches.
//   - Output column names must be unambiguous across the two sides (no name appearing on both). Columns
//     are selected by name, so a collision could otherwise silently take the wrong side's data. Enforced
//     by ResolveSideColumn, shared with operators/gpu_cross_join.cu.
//   - Nullable input columns are refused upstream by gpu_executor.cu, so no null-vs-null match question
//     arises here (SQL equi-joins never match NULL to NULL anyway).
//
// Skew note: step 5 gives each left row's matches to a single thread, so one left row matching very many
// right rows serializes that row's expansion. Correct, but a load-balanced expansion would be faster on
// heavily skewed data.

#include "gpu_executor_internal.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace vector_gpu {

namespace {

void SyncOrThrow(const char *what) {
	auto status = cudaDeviceSynchronize();
	if (status != cudaSuccess) {
		throw std::runtime_error(std::string("gpu_hash_join: ") + what + " failed: " + cudaGetErrorString(status));
	}
}

//! Same widening contract as gpu_groupby.cu: zero-extend the key's raw bit pattern into uint64.
__global__ void WidenJoinKeyKernel(const void *src, unsigned type_size, uint64_t *dst, uint64_t n) {
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

//! Writes one (left row, right row) index pair per match. `lower[i]` is where left row i's matches begin
//! in the sorted right side, `counts[i]` how many there are, `offsets[i]` where they go in the output.
__global__ void ExpandMatchesKernel(const uint64_t *lower, const uint64_t *counts, const uint64_t *offsets,
                                    const uint64_t *right_permutation, uint64_t *out_left, uint64_t *out_right,
                                    uint64_t left_rows) {
	uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
	for (uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < left_rows; idx += stride) {
		uint64_t base = offsets[idx];
		uint64_t begin = lower[idx];
		uint64_t count = counts[idx];
		for (uint64_t j = 0; j < count; j++) {
			out_left[base + j] = idx;
			out_right[base + j] = right_permutation[begin + j];
		}
	}
}

constexpr unsigned kBlockSize = 256;

unsigned GridFor(uint64_t n) {
	uint64_t grid = (n + kBlockSize - 1) / kBlockSize;
	return grid > 65535u ? 65535u : static_cast<unsigned>(grid); // grid-stride loops cover the remainder
}

//! Difference of two device arrays, elementwise: counts[i] = upper[i] - lower[i].
__global__ void MatchCountKernel(const uint64_t *lower, const uint64_t *upper, uint64_t *counts, uint64_t n) {
	uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
	for (uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
		counts[idx] = upper[idx] - lower[idx];
	}
}

void ValidateJoinKeyType(const DeviceColumn &column, const char *side) {
	switch (column.type) {
	case GpuValueType::INT16:
	case GpuValueType::INT32:
	case GpuValueType::INT64:
	case GpuValueType::BOOLEAN:
		return;
	default:
		throw std::runtime_error(std::string("gpu_hash_join: ") + side +
		                         " join key is not an integer/boolean column, which this GPU path requires");
	}
}

} // namespace

std::vector<DeviceColumn> ExecuteJoinNode(const GpuPlanNode &node, const std::vector<DeviceColumn> &left_columns,
                                          const std::vector<DeviceColumn> &right_columns, ExecContext &ctx) {
	if (node.build_keys.size() != 1 || node.probe_keys.size() != 1) {
		throw std::runtime_error("gpu_hash_join: this GPU path joins on exactly one equality key column");
	}
	if (node.output_columns.empty()) {
		throw std::runtime_error("gpu_hash_join: node has no output columns");
	}

	// build_keys names a children[0] column, probe_keys a children[1] column -- each is looked up only on
	// its own side, so a name shared by both sides is never ambiguous *here* (only in the output list).
	auto &left_key = FindColumn(left_columns, node.build_keys[0]);
	auto &right_key = FindColumn(right_columns, node.probe_keys[0]);
	ValidateJoinKeyType(left_key, "left");
	ValidateJoinKeyType(right_key, "right");
	if (left_key.type != right_key.type) {
		// Widening is bit-pattern based, so INT32 -1 and INT64 -1 do not compare equal.
		throw std::runtime_error("gpu_hash_join: join key columns must have the same type on both sides");
	}

	auto left_rows = left_key.rows;
	auto right_rows = right_key.rows;

	// Resolve the output shape up front so the empty-result path returns correctly-typed columns.
	std::vector<const DeviceColumn *> output_sources;
	std::vector<bool> output_from_left;
	output_sources.reserve(node.output_columns.size());
	output_from_left.reserve(node.output_columns.size());
	for (auto &name : node.output_columns) {
		bool from_left = false;
		auto &column = ResolveSideColumn(name, left_columns, right_columns, from_left);
		output_sources.push_back(&column);
		output_from_left.push_back(from_left);
	}

	auto empty_result = [&]() {
		std::vector<DeviceColumn> result;
		result.reserve(output_sources.size());
		for (size_t i = 0; i < output_sources.size(); i++) {
			DeviceColumn column;
			column.name = node.output_columns[i];
			column.type = output_sources[i]->type;
			column.rows = 0;
			column.data = ctx.arena.Alloc(0);
			result.push_back(std::move(column));
		}
		return result;
	};

	if (left_rows == 0 || right_rows == 0) {
		return empty_result(); // an INNER join with an empty side matches nothing
	}

	// 1. Widen both keys.
	auto *left_widened = static_cast<uint64_t *>(ctx.arena.Alloc(static_cast<size_t>(left_rows) * sizeof(uint64_t)));
	auto *right_widened =
	    static_cast<uint64_t *>(ctx.arena.Alloc(static_cast<size_t>(right_rows) * sizeof(uint64_t)));
	auto key_size = static_cast<unsigned>(TypeSize(left_key.type));
	WidenJoinKeyKernel<<<GridFor(left_rows), kBlockSize>>>(left_key.data, key_size, left_widened, left_rows);
	WidenJoinKeyKernel<<<GridFor(right_rows), kBlockSize>>>(right_key.data, key_size, right_widened, right_rows);
	SyncOrThrow("join-key widening kernel");

	// 2. Sort the right side by key, keeping the permutation so matches map back to original rows.
	auto *right_permutation =
	    static_cast<uint64_t *>(ctx.arena.Alloc(static_cast<size_t>(right_rows) * sizeof(uint64_t)));
	thrust::device_ptr<uint64_t> right_permutation_ptr(right_permutation);
	thrust::device_ptr<uint64_t> right_widened_ptr(right_widened);
	thrust::sequence(thrust::device, right_permutation_ptr, right_permutation_ptr + right_rows);
	thrust::sort_by_key(thrust::device, right_widened_ptr, right_widened_ptr + right_rows, right_permutation_ptr);
	SyncOrThrow("right-side join key sort");

	// 3. Match range per left row. lower_bound/upper_bound are vectorized: one search per left key.
	auto *lower = static_cast<uint64_t *>(ctx.arena.Alloc(static_cast<size_t>(left_rows) * sizeof(uint64_t)));
	auto *upper = static_cast<uint64_t *>(ctx.arena.Alloc(static_cast<size_t>(left_rows) * sizeof(uint64_t)));
	auto *counts = static_cast<uint64_t *>(ctx.arena.Alloc(static_cast<size_t>(left_rows) * sizeof(uint64_t)));
	thrust::device_ptr<const uint64_t> left_widened_ptr(left_widened);
	thrust::lower_bound(thrust::device, right_widened_ptr, right_widened_ptr + right_rows, left_widened_ptr,
	                    left_widened_ptr + left_rows, thrust::device_ptr<uint64_t>(lower));
	thrust::upper_bound(thrust::device, right_widened_ptr, right_widened_ptr + right_rows, left_widened_ptr,
	                    left_widened_ptr + left_rows, thrust::device_ptr<uint64_t>(upper));
	MatchCountKernel<<<GridFor(left_rows), kBlockSize>>>(lower, upper, counts, left_rows);
	SyncOrThrow("join match-range search");

	// 4. Output offsets + total row count.
	auto *offsets = static_cast<uint64_t *>(ctx.arena.Alloc(static_cast<size_t>(left_rows) * sizeof(uint64_t)));
	thrust::device_ptr<const uint64_t> counts_ptr(counts);
	thrust::device_ptr<uint64_t> offsets_ptr(offsets);
	thrust::exclusive_scan(thrust::device, counts_ptr, counts_ptr + left_rows, offsets_ptr);
	SyncOrThrow("join offset scan");

	uint64_t last_offset = 0;
	uint64_t last_count = 0;
	auto status = cudaMemcpy(&last_offset, offsets + (left_rows - 1), sizeof(uint64_t), cudaMemcpyDeviceToHost);
	if (status == cudaSuccess) {
		status = cudaMemcpy(&last_count, counts + (left_rows - 1), sizeof(uint64_t), cudaMemcpyDeviceToHost);
	}
	if (status != cudaSuccess) {
		throw std::runtime_error(std::string("gpu_hash_join: output size readback failed: ") +
		                         cudaGetErrorString(status));
	}
	uint64_t total_rows = last_offset + last_count;
	if (total_rows == 0) {
		return empty_result();
	}
	// A many-to-many join can multiply row counts far past anything that fits; refuse before asking the
	// allocator for an absurd size, so the query falls back to CPU instead of thrashing the device.
	constexpr uint64_t kMaxJoinOutputRows = 1ull << 32;
	if (total_rows > kMaxJoinOutputRows) {
		throw std::runtime_error("gpu_hash_join: join result of " + std::to_string(total_rows) +
		                         " rows is too large for GPU execution");
	}

	// 5. Expand into one index pair per matching row pair.
	auto *out_left = static_cast<uint64_t *>(ctx.arena.Alloc(static_cast<size_t>(total_rows) * sizeof(uint64_t)));
	auto *out_right = static_cast<uint64_t *>(ctx.arena.Alloc(static_cast<size_t>(total_rows) * sizeof(uint64_t)));
	ExpandMatchesKernel<<<GridFor(left_rows), kBlockSize>>>(lower, counts, offsets, right_permutation, out_left,
	                                                        out_right, left_rows);
	SyncOrThrow("join match expansion");

	// 6. Gather every requested output column through its side's index vector.
	std::vector<DeviceColumn> result;
	result.reserve(output_sources.size());
	for (size_t i = 0; i < output_sources.size(); i++) {
		auto gathered =
		    GatherColumn(*output_sources[i], output_from_left[i] ? out_left : out_right, total_rows, ctx);
		gathered.name = node.output_columns[i];
		result.push_back(std::move(gathered));
	}
	return result;
}

} // namespace vector_gpu
