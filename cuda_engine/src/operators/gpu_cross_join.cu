// gpu_cross_join.cu
//
// CROSS_PRODUCT: every left row paired with every right row. This is the one join shape that needs no
// matching machinery at all -- no sort, no binary search, no hash table (contrast operators/
// gpu_hash_join.cu, which needs all of that to find which rows pair up). Output row r is unconditionally
// (left row r / right_rows, right row r % right_rows), so each of the n*m output slots is computed
// independently by one thread from an index alone.
//
// WHY IT IS WORTH OFFLOADING AT ALL, when the same argument sank a bare SCAN (see
// PerformsGpuComputation in extension/src/gpu_offload_extension.cpp): a cross product turns a small
// input into a large one. A few MB uploaded can drive billions of pairwise evaluations in whatever
// PROJECTION / FILTER / GROUP_BY sits above it, and those never touch the host. That is the
// arithmetic-bound regime a discrete GPU actually wins. A cross product whose result is copied straight
// BACK to the host wins nothing and is refused at routing time for exactly the bare-SCAN reason.
//
// ROW ORDER is left-major, matching DuckDB's own PhysicalCrossProduct (verified against
// build_core/duckdb.exe: `SELECT x, y FROM sa, sb` emits every y for x=0, then every y for x=1, ...).
// SQL does not promise an order without ORDER BY, but agreeing with the CPU path keeps a differential
// diff of raw rows meaningful instead of forcing every test through an aggregate.
//
// NO INDEX VECTORS. The obvious implementation builds two n*m index arrays and reuses GatherColumn, the
// way gpu_hash_join.cu does. That costs 16 extra bytes per output row of device memory -- more than the
// output columns themselves in the common one- or two-column case -- for indices that are a division and
// a modulo away. Computing them in the copy kernel keeps peak VRAM equal to the output, which is what the
// routing-time size guard is sized against.
//
// DELIBERATE RESTRICTIONS (each a clean refusal -> CPU fallback, never a wrong answer):
//   - Output column names must be unambiguous across the two sides, enforced by the shared
//     ResolveSideColumn. Same reason as the join: columns are selected by name.
//   - n*m is capped (kMaxCrossProductOutputRows). A cross product is the fastest way in this engine to
//     ask for more memory than exists, and DeviceArena has no spill path.
//
// NOTE ON THE CAP: this engine-side check is a backstop, not the primary guard. Reaching it FAILS the
// query rather than falling back, so extension/src/table_checker.cpp refuses an oversized cross product
// at routing time, where a refusal is silent. See docs/CONTEXT.md gotcha 2.

#include "gpu_executor_internal.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace vector_gpu {

namespace {

constexpr unsigned kBlockSize = 256;

unsigned GridFor(uint64_t n) {
	uint64_t grid = (n + kBlockSize - 1) / kBlockSize;
	return grid > 65535u ? 65535u : static_cast<unsigned>(grid); // grid-stride loops cover the remainder
}

//! Upper bound on n*m, enforced before any allocation. 2^32 output rows is already 32 GB for a single
//! DOUBLE column -- far past any device this runs on -- so anything above it is certainly a mistake
//! rather than a workload. The real limit is VRAM, applied at routing time.
constexpr uint64_t kMaxCrossProductOutputRows = 1ull << 32;

//! One output column of the pairwise expansion. `divisor` is the right side's row count: a left column
//! reads base row idx / divisor (consecutive threads hit the same element -- a broadcast), a right column
//! reads idx % divisor (consecutive threads hit consecutive elements -- coalesced).
template <typename T>
__global__ void ExpandCrossProductKernel(const T *src, T *dst, uint64_t divisor, bool from_left, uint64_t n) {
	uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
	for (uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
		dst[idx] = src[from_left ? idx / divisor : idx % divisor];
	}
}

//! Dispatches the expansion on the column's byte width. Byte width rather than GpuValueType because the
//! kernel only copies -- it never interprets the value -- which is the same trick GatherColumn uses.
void LaunchExpansion(const DeviceColumn &source, DeviceColumn &destination, uint64_t divisor, bool from_left,
                     uint64_t rows) {
	switch (TypeSize(source.type)) {
	case 1:
		ExpandCrossProductKernel<uint8_t><<<GridFor(rows), kBlockSize>>>(
		    static_cast<const uint8_t *>(source.data), static_cast<uint8_t *>(destination.data), divisor, from_left,
		    rows);
		break;
	case 2:
		ExpandCrossProductKernel<uint16_t><<<GridFor(rows), kBlockSize>>>(
		    static_cast<const uint16_t *>(source.data), static_cast<uint16_t *>(destination.data), divisor,
		    from_left, rows);
		break;
	case 4:
		ExpandCrossProductKernel<uint32_t><<<GridFor(rows), kBlockSize>>>(
		    static_cast<const uint32_t *>(source.data), static_cast<uint32_t *>(destination.data), divisor,
		    from_left, rows);
		break;
	case 8:
		ExpandCrossProductKernel<uint64_t><<<GridFor(rows), kBlockSize>>>(
		    static_cast<const uint64_t *>(source.data), static_cast<uint64_t *>(destination.data), divisor,
		    from_left, rows);
		break;
	case 16:
		ExpandCrossProductKernel<uint4><<<GridFor(rows), kBlockSize>>>(
		    static_cast<const uint4 *>(source.data), static_cast<uint4 *>(destination.data), divisor, from_left,
		    rows);
		break;
	default:
		throw std::runtime_error("gpu_cross_join: LaunchExpansion: unsupported column byte width " +
		                         std::to_string(TypeSize(source.type)));
	}
}

} // namespace

std::vector<DeviceColumn> ExecuteCrossProductNode(const GpuPlanNode &node,
                                                  const std::vector<DeviceColumn> &left_columns,
                                                  const std::vector<DeviceColumn> &right_columns, ExecContext &ctx) {
	if (node.output_columns.empty()) {
		throw std::runtime_error("gpu_cross_join: node has no output columns");
	}
	if (left_columns.empty() || right_columns.empty()) {
		throw std::runtime_error("gpu_cross_join: both inputs must produce at least one column");
	}

	// Resolve the output shape up front so the empty-result path still returns correctly-typed columns.
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

	auto left_rows = left_columns.front().rows;
	auto right_rows = right_columns.front().rows;
	// The kernel indexes a source column by a row number derived from the OTHER side's count, so a
	// relation whose columns disagree on length would read out of bounds rather than fail.
	for (auto &column : left_columns) {
		if (column.rows != left_rows) {
			throw std::runtime_error("gpu_cross_join: left input column '" + column.name + "' row count mismatch");
		}
	}
	for (auto &column : right_columns) {
		if (column.rows != right_rows) {
			throw std::runtime_error("gpu_cross_join: right input column '" + column.name + "' row count mismatch");
		}
	}

	std::vector<DeviceColumn> result;
	result.reserve(output_sources.size());

	if (left_rows == 0 || right_rows == 0) {
		for (size_t i = 0; i < output_sources.size(); i++) {
			DeviceColumn column;
			column.name = node.output_columns[i];
			column.type = output_sources[i]->type;
			column.rows = 0;
			column.data = ctx.arena.Alloc(0);
			result.push_back(std::move(column));
		}
		return result;
	}

	// Overflow-safe n*m. The division test rather than a multiply-and-compare: the product is exactly
	// what would wrap, so it must never be formed before being checked.
	if (left_rows > kMaxCrossProductOutputRows / right_rows) {
		throw std::runtime_error("gpu_cross_join: cross product of " + std::to_string(left_rows) + " x " +
		                         std::to_string(right_rows) + " rows is too large for GPU execution");
	}
	uint64_t total_rows = left_rows * right_rows;
	if (total_rows > kMaxCrossProductOutputRows) {
		throw std::runtime_error("gpu_cross_join: cross product result of " + std::to_string(total_rows) +
		                         " rows is too large for GPU execution");
	}

	for (size_t i = 0; i < output_sources.size(); i++) {
		auto &source = *output_sources[i];
		DeviceColumn column;
		column.name = node.output_columns[i];
		column.type = source.type;
		column.rows = total_rows;
		column.data = ctx.arena.Alloc(static_cast<size_t>(total_rows) * TypeSize(source.type));
		LaunchExpansion(source, column, right_rows, output_from_left[i], total_rows);
		result.push_back(std::move(column));
	}

	auto status = cudaDeviceSynchronize();
	if (status != cudaSuccess) {
		throw std::runtime_error(std::string("gpu_cross_join: pairwise expansion kernel failed: ") +
		                         cudaGetErrorString(status));
	}
	return result;
}

} // namespace vector_gpu
