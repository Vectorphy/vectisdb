// gpu_executor.cu
//
// The real recursive GPU plan executor. This is what replaced the per-operator stubs that used to make
// GpuEngine::ExecutePlan always return success=false ("kernel dispatch not wired up"): a GpuPlanNode tree
// is now genuinely executed on the device — host data is uploaded, per-row work runs in NVRTC-compiled
// fused kernels and Thrust primitives, and only the final result is copied back.
//
// Supported today, fully on GPU:
//   SCAN        — upload the scanned columns to device memory
//   PROJECTION  — one fused NVRTC kernel per output expression (arithmetic/comparison over columns)
//   FILTER      — one fused NVRTC kernel per predicate producing a bool mask, masks AND-combined,
//                 then a single index-compaction + gather over every carried column
//   GROUP_BY_AGGREGATE — sort-based grouping + Thrust reduce_by_key, in operators/gpu_groupby.cu
//   HASH_JOIN   — INNER equi-join by sort + vectorized binary search, in operators/gpu_hash_join.cu
//   CROSS_PRODUCT — O(n*m) pairwise expansion, one kernel per output column, in
//                 operators/gpu_cross_join.cu
//
// Every operator refuses shapes it cannot execute exactly (nullable columns, float group/join keys,
// composite join keys, ...) by throwing, which ExecuteGpuPlan turns into success=false and the caller
// turns into a CPU fallback. The optimizer additionally declines to offload most of those plans in the
// first place. See each operator's file header for the specific restrictions and why they exist.
//
// Null handling: for a row-independent plan (SCAN/FILTER/PROJECTION only -- see IsRowIndependent), a
// nullable input column is unpacked into a dense device validity array and threaded through the fused
// expression kernels, which now compute out_valid[idx] as the AND of every referenced input's validity at
// that row -- ordinary SQL NULL propagation. GROUP_BY_AGGREGATE/HASH_JOIN/CROSS_PRODUCT are unchanged:
// their Thrust-based kernels still have no null semantics, so a nullable column reaching one of them is
// still rejected up front (clean failure -> CPU fallback), exactly as before this prototype.

#include "code_generator.hpp"
#include "gpu_column_cache.hpp"
#include "gpu_engine.hpp"
#include "gpu_executor_internal.hpp"
#include "gpu_logger.hpp"
#include "kernel_cache.hpp"
#include "pinned_buffer_pool.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>

namespace vector_gpu {

// Defined at namespace scope (not in the anonymous namespace below) because operators/gpu_groupby.cu
// links against them through gpu_executor_internal.hpp.

size_t TypeSize(GpuValueType type) {
	switch (type) {
	case GpuValueType::INT16:
		return 2;
	case GpuValueType::INT32:
		return 4;
	case GpuValueType::INT64:
		return 8;
	case GpuValueType::HUGEINT:
		return 16;
	case GpuValueType::FLOAT32:
		return 4;
	case GpuValueType::FLOAT64:
		return 8;
	case GpuValueType::BOOLEAN:
		return 1; // matches CudaTypeName(BOOLEAN) == "bool", which is 1 byte in CUDA C++
	case GpuValueType::DICTIONARY_STRING:
		throw std::runtime_error("gpu_executor: dictionary-encoded string columns cannot be executed on GPU");
	}
	throw std::runtime_error("gpu_executor: unknown GpuValueType");
}

DeviceArena::~DeviceArena() {
	for (auto *ptr : buffers_) {
		cudaFree(ptr);
	}
}

void *DeviceArena::Alloc(size_t bytes) {
	if (bytes == 0) {
		bytes = 1; // a zero-row intermediate still needs a valid, distinct pointer
	}
	void *ptr = nullptr;
	auto status = cudaMalloc(&ptr, bytes);
	if (status != cudaSuccess) {
		throw std::runtime_error(std::string("gpu_executor: cudaMalloc(") + std::to_string(bytes) +
		                         ") failed: " + cudaGetErrorString(status));
	}
	buffers_.push_back(ptr);
	VGPU_LOG(LogLevel::DEBUG, "alloc", "\"src\":\"device_arena\",\"bytes\":" + std::to_string(bytes));
	return ptr;
}

void DeviceArena::AdoptExternal(void *ptr) {
	if (ptr != nullptr) {
		buffers_.push_back(ptr);
	}
}

const DeviceColumn &FindColumn(const std::vector<DeviceColumn> &columns, const std::string &name) {
	for (auto &column : columns) {
		if (column.name == name) {
			return column;
		}
	}
	throw std::runtime_error("gpu_executor: expression references column '" + name +
	                         "' which is not produced by its input operator");
}

const DeviceColumn &ResolveSideColumn(const std::string &name, const std::vector<DeviceColumn> &left_columns,
                                      const std::vector<DeviceColumn> &right_columns, bool &from_left) {
	const DeviceColumn *found = nullptr;
	for (auto &column : left_columns) {
		if (column.name == name) {
			found = &column;
			from_left = true;
			break;
		}
	}
	for (auto &column : right_columns) {
		if (column.name == name) {
			if (found != nullptr) {
				throw std::runtime_error("gpu_executor: output column '" + name +
				                         "' is produced by both inputs, so selecting it by name is ambiguous");
			}
			found = &column;
			from_left = false;
			break;
		}
	}
	if (found == nullptr) {
		throw std::runtime_error("gpu_executor: output column '" + name + "' is produced by neither input");
	}
	return *found;
}

//! Gathers `column` at the given device indices into a new device column of `count` rows.
DeviceColumn GatherColumn(const DeviceColumn &column, const uint64_t *indices, uint64_t count, ExecContext &ctx) {
	DeviceColumn result;
	result.name = column.name;
	result.type = column.type;
	result.rows = count;
	result.data = ctx.arena.Alloc(static_cast<size_t>(count) * TypeSize(column.type));
	if (count == 0) {
		return result;
	}

	thrust::device_ptr<const uint64_t> index_ptr(indices);
	switch (TypeSize(column.type)) {
	case 1: {
		thrust::device_ptr<const uint8_t> src(static_cast<const uint8_t *>(column.data));
		thrust::device_ptr<uint8_t> dst(static_cast<uint8_t *>(result.data));
		thrust::gather(thrust::device, index_ptr, index_ptr + count, src, dst);
		break;
	}
	case 2: {
		thrust::device_ptr<const uint16_t> src(static_cast<const uint16_t *>(column.data));
		thrust::device_ptr<uint16_t> dst(static_cast<uint16_t *>(result.data));
		thrust::gather(thrust::device, index_ptr, index_ptr + count, src, dst);
		break;
	}
	case 4: {
		thrust::device_ptr<const uint32_t> src(static_cast<const uint32_t *>(column.data));
		thrust::device_ptr<uint32_t> dst(static_cast<uint32_t *>(result.data));
		thrust::gather(thrust::device, index_ptr, index_ptr + count, src, dst);
		break;
	}
	case 8: {
		thrust::device_ptr<const uint64_t> src(static_cast<const uint64_t *>(column.data));
		thrust::device_ptr<uint64_t> dst(static_cast<uint64_t *>(result.data));
		thrust::gather(thrust::device, index_ptr, index_ptr + count, src, dst);
		break;
	}
	case 16: {
		// uint4 is a built-in CUDA 16-byte vector type, convenient for 128-bit transfers
		thrust::device_ptr<const uint4> src(static_cast<const uint4 *>(column.data));
		thrust::device_ptr<uint4> dst(static_cast<uint4 *>(result.data));
		thrust::gather(thrust::device, index_ptr, index_ptr + count, src, dst);
		break;
	}
	default: {
		throw std::runtime_error("gpu_executor: GatherColumn encountered unknown TypeSize");
	}
	}
	if (column.valid != nullptr) {
		result.valid = static_cast<bool *>(ctx.arena.Alloc(static_cast<size_t>(count)));
		thrust::device_ptr<const bool> src_valid(column.valid);
		thrust::device_ptr<bool> dst_valid(result.valid);
		thrust::gather(thrust::device, index_ptr, index_ptr + count, src_valid, dst_valid);
	}
	auto status = cudaDeviceSynchronize();
	if (status != cudaSuccess) {
		throw std::runtime_error(std::string("gpu_executor: gather failed: ") + cudaGetErrorString(status));
	}
	return result;
}


DeviceColumn MaterializeColumn(const DeviceColumn &column, ExecContext &ctx) {
	if (column.selection == nullptr) {
		return column;
	}
	DeviceColumn dense;
	dense.name = column.name;
	dense.type = column.type;
	dense.rows = column.rows;
	dense.data = ctx.arena.Alloc(static_cast<size_t>(column.rows) * TypeSize(column.type));
	if (column.rows == 0) {
		return dense;
	}
	thrust::device_ptr<const uint32_t> index_ptr(column.selection);
	switch (TypeSize(column.type)) {
	case 1: {
		thrust::device_ptr<const uint8_t> src(static_cast<const uint8_t *>(column.data));
		thrust::gather(thrust::device, index_ptr, index_ptr + column.rows, src,
		               thrust::device_ptr<uint8_t>(static_cast<uint8_t *>(dense.data)));
		break;
	}
	case 4: {
		thrust::device_ptr<const uint32_t> src(static_cast<const uint32_t *>(column.data));
		thrust::gather(thrust::device, index_ptr, index_ptr + column.rows, src,
		               thrust::device_ptr<uint32_t>(static_cast<uint32_t *>(dense.data)));
		break;
	}
	default: {
		thrust::device_ptr<const uint64_t> src(static_cast<const uint64_t *>(column.data));
		thrust::gather(thrust::device, index_ptr, index_ptr + column.rows, src,
		               thrust::device_ptr<uint64_t>(static_cast<uint64_t *>(dense.data)));
		break;
	}
	}
	if (column.valid != nullptr) {
		// Same selection, applied to the validity array: a materialized (post-filter) column must carry
		// the SAME rows' nullness as its data, or the two would go out of sync the moment a selection ever
		// reorders/drops rows -- which is exactly what a preceding FILTER does.
		dense.valid = static_cast<bool *>(ctx.arena.Alloc(static_cast<size_t>(column.rows) * sizeof(bool)));
		thrust::device_ptr<const bool> valid_src(column.valid);
		thrust::gather(thrust::device, index_ptr, index_ptr + column.rows, valid_src,
		               thrust::device_ptr<bool>(dense.valid));
	}
	auto status = cudaDeviceSynchronize();
	if (status != cudaSuccess) {
		throw std::runtime_error(std::string("gpu_executor: materialize failed: ") + cudaGetErrorString(status));
	}
	return dense;
}

std::vector<DeviceColumn> MaterializeAll(const std::vector<DeviceColumn> &columns, ExecContext &ctx) {
	std::vector<DeviceColumn> dense;
	dense.reserve(columns.size());
	for (auto &column : columns) {
		dense.push_back(MaterializeColumn(column, ctx));
	}
	return dense;
}

namespace {


//! Expands a packed Arrow-style validity bitmap (GpuColumn::validity's format: 1 bit/row, LSB-first,
//! bit=1 valid) into a dense bool array, one byte per row. Kernels read/write validity as plain bools
//! (see gpu_executor_internal.hpp's DeviceColumn::valid) so no thread ever has to share a byte with
//! another thread's write -- packing only happens once more, host-side, on the way back out (see
//! ExecuteGpuPlan's final D2H).
__global__ void UnpackValidityKernel(const uint8_t *packed, bool *out, uint64_t n) {
	for (uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
		out[idx] = (packed[idx / 8] & (uint8_t(1) << (idx % 8))) != 0;
	}
}

//! Compiles `expr` and launches it over `rows`, returning a freshly allocated device column holding the
//! per-row result. The generated source reads inputs[i] in the order of expr.input_columns.
DeviceColumn EvaluateExpression(const GpuExpr &expr, const std::vector<DeviceColumn> &child_columns, uint64_t rows,
                                ExecContext &ctx, std::string result_name) {
	std::vector<const void *> kernel_inputs;
	std::vector<const bool *> kernel_input_valid;
	kernel_inputs.reserve(expr.input_columns.size());
	kernel_input_valid.reserve(expr.input_columns.size());
	bool any_nullable = false;
	for (auto &column_name : expr.input_columns) {
		auto &column = FindColumn(child_columns, column_name);
		if (column.rows != rows) {
			throw std::runtime_error("gpu_executor: input column '" + column_name + "' row count mismatch");
		}
		kernel_inputs.push_back(column.data);
		kernel_input_valid.push_back(column.valid);
		any_nullable |= (column.valid != nullptr);
	}

	DeviceColumn result;
	result.name = std::move(result_name);
	result.type = expr.output_type;
	result.rows = rows;
	result.data = ctx.arena.Alloc(static_cast<size_t>(rows) * TypeSize(expr.output_type));
	if (any_nullable && rows > 0) {
		// Ordinary SQL NULL propagation for a scalar expression: this output row is NULL iff any
		// referenced input was NULL at that row. See code_generator.cpp's WrapGridStrideLoop for where
		// this gets computed, and MaterializeColumn for why a subsequent selection-gather must carry it.
		result.valid = static_cast<bool *>(ctx.arena.Alloc(static_cast<size_t>(rows) * sizeof(bool)));
	}

	if (rows == 0) {
		return result; // Launch rejects n == 0 by contract; an empty result needs no kernel
	}
	// LATE MATERIALIZATION: hand the kernel the selection vector rather than gathering the inputs into
	// dense buffers first. Every column of one relation shares a selection, so taking it from the first
	// input is well-defined. The result is always dense -- output slot i belongs to logical row i.
	const uint32_t *selection = nullptr;
	for (auto &column : child_columns) {
		selection = column.selection;
		break;
	}
	auto kernel = ctx.generator.CompileOrFetch(expr);
	ctx.generator.Launch(kernel, kernel_inputs, result.data, static_cast<size_t>(rows), selection, ctx.stream,
	                     any_nullable ? kernel_input_valid : std::vector<const bool *> {}, result.valid);
	return result;
}

//! mask[i] = mask[i] && other[i], elementwise, in place.
__global__ void AndMasksKernel(bool *mask, const bool *other, uint64_t n) {
	for (uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
		mask[idx] = mask[idx] && other[idx];
	}
}

void AndMasks(bool *mask, const bool *other, uint64_t rows) {
	if (rows == 0) {
		return;
	}
	unsigned int block = 256;
	unsigned int grid = static_cast<unsigned int>((rows + block - 1) / block);
	if (grid > 65535u) {
		grid = 65535u; // grid-stride loop covers the remainder
	}
	AndMasksKernel<<<grid, block>>>(mask, other, rows);
	auto status = cudaDeviceSynchronize();
	if (status != cudaSuccess) {
		throw std::runtime_error(std::string("gpu_executor: mask combine kernel failed: ") +
		                         cudaGetErrorString(status));
	}
}

std::vector<DeviceColumn> ExecuteNode(const GpuPlanNode &node, ExecContext &ctx);

//! cudaMalloc's `bytes` directly (NOT via ctx.arena.Alloc) -- used only for buffers that might end up
//! owned by GpuColumnCache instead of this plan's DeviceArena. Whichever one ends up NOT owning it must
//! still free it exactly once: see RawAllocOutcome below and DeviceArena::AdoptExternal's comment for why
//! Alloc() itself must never be used for a cache-eligible buffer.
void *RawDeviceAlloc(size_t bytes) {
	if (bytes == 0) {
		bytes = 1;
	}
	void *ptr = nullptr;
	auto status = cudaMalloc(&ptr, bytes);
	if (status != cudaSuccess) {
		throw std::runtime_error(std::string("gpu_executor: cudaMalloc(") + std::to_string(bytes) +
		                         ") failed: " + cudaGetErrorString(status));
	}
	return ptr;
}

std::vector<DeviceColumn> ExecuteScan(const GpuPlanNode &node, ExecContext &ctx) {
	std::vector<DeviceColumn> columns;
	columns.reserve(node.column_names.size());

	// CACHE REPLAY PATH: PhysicalGpuExecute (extension layer) asked GpuColumnCache whether every column
	// this SCAN needs is already resident BEFORE deciding to scan the table at all. When it is, it runs
	// the plan once per cached chunk with GpuExecuteOptions::cache_chunk_index set, and the (dominant,
	// ~74% of GPU-path time per docs/SESSION_30_BENCHMARK_RESULTS.md) host-side scan is skipped entirely
	// upstream -- not just the H2D this function used to be the only place saving.
	//
	// One chunk per call, matching the chunk boundaries the populating scan used. Serving the whole
	// column at once would execute the plan over the entire table inside a routing decision that sized
	// VRAM for one MAX_CHUNK_ROWS chunk. See docs/GPU_RESIDENT_CACHE_DESIGN.md.
	if (ctx.cache_chunk_index >= 0) {
		auto chunk_index = static_cast<uint64_t>(ctx.cache_chunk_index);
		for (auto &column_name : node.column_names) {
			GpuColumnChunk cached;
			if (!GpuColumnCache::Instance().TryGetChunk(node.table, column_name, chunk_index, cached)) {
				// Race between PhysicalGpuExecute's "is it fully cached" peek and this actual fetch --
				// e.g. a concurrent write from another connection invalidated it in between. This engine
				// is documented as effectively single-GPU-serialized, not built for that race generally;
				// failing the query cleanly (not silently computing over missing/wrong data) is the safe
				// direction here, same posture as every other "declined variance" this file throws on.
				throw std::runtime_error("gpu_executor: chunk " + std::to_string(chunk_index) + " of column '" +
				                         column_name + "' of table '" + node.table.ToString() +
				                         "' was expected to be cache-resident but is no longer present "
				                         "(likely invalidated by a concurrent write) -- retry the query");
			}
			if (cached.device_validity != nullptr && !ctx.nulls_supported) {
				// Same refusal the scan path below makes, for the same reason: dropping the validity here
				// instead would turn NULLs into whatever bytes the data buffer happens to hold. Not
				// reachable today (only row-independent plans replay, and those set nulls_supported), so
				// this is a guard against a future caller, not a live path.
				throw std::runtime_error("gpu_executor: cached column '" + column_name +
				                         "' contains NULLs, which this operator does not model");
			}
			DeviceColumn column;
			column.name = column_name;
			column.type = cached.type;
			column.rows = cached.row_count;
			column.data = cached.device_data;
			if (cached.device_validity != nullptr) {
				column.valid = static_cast<bool *>(cached.device_validity);
			}
			VGPU_LOG(LogLevel::DEBUG, "cache_hit",
			         "\"table\":" + GpuLogger::Quote(node.table.ToString()) + ",\"column\":" +
			             GpuLogger::Quote(column_name) + ",\"chunk\":" + std::to_string(chunk_index) +
			             ",\"rows\":" + std::to_string(column.rows));
			columns.push_back(std::move(column));
		}
		return columns;
	}

	// WHOLE-SCAN CACHE PATH, for the operators that cannot be replayed chunk by chunk.
	//
	// GROUP_BY_AGGREGATE sorts across its ENTIRE input and HASH_JOIN needs a complete build side, so
	// handing either one chunk at a time computes per-chunk answers, not the query's. Those plans were
	// therefore the last shape still paying a full host scan plus H2D on every single run -- 0.04x of the
	// CPU on a 70M-row `SELECT SUM(a)` (docs/SESSION_32_CACHE_BENCHMARK.md), the worst number this engine
	// has ever measured.
	//
	// The chunks are already on the device; all that is missing is contiguity, which is one D2D copy per
	// chunk. Nothing crosses PCIe and nothing is read on the host, so both costs the benchmark blamed are
	// gone. The concatenated buffer comes from the plan's arena (freed with the plan), NOT from
	// RawDeviceAlloc: unlike the upload path below, this buffer is never a candidate for the cache to
	// adopt -- the cache already owns the chunks it was copied from.
	if (ctx.serve_whole_scan_from_cache) {
		auto chunk_rows = GpuColumnCache::Instance().ReplayableChunkRows(node.table, node.column_names);
		if (chunk_rows.empty()) {
			// Same race as the chunk-replay path above: the extension layer checked residency before
			// deciding not to scan, and something invalidated the table in between.
			throw std::runtime_error("gpu_executor: table '" + node.table.ToString() +
			                         "' was expected to be fully cache-resident but is no longer present "
			                         "(likely invalidated by a concurrent write) -- retry the query");
		}
		uint64_t total_rows = 0;
		for (auto rows : chunk_rows) {
			total_rows += rows;
		}
		for (auto &column_name : node.column_names) {
			// Fetched up front, before any copy is issued: TryGetChunk hands back a BORROW of the cache's
			// device pointers, and gathering them all first keeps the window in which they must stay valid
			// to this one loop. Nothing here can evict (no staging is open while serving), and ctx.stream is
			// synchronized before the plan returns, so every copy below has landed before those borrows
			// could go stale.
			std::vector<GpuColumnChunk> chunks;
			chunks.reserve(chunk_rows.size());
			bool any_validity = false;
			for (uint64_t index = 0; index < chunk_rows.size(); index++) {
				GpuColumnChunk cached;
				if (!GpuColumnCache::Instance().TryGetChunk(node.table, column_name, index, cached)) {
					throw std::runtime_error("gpu_executor: chunk " + std::to_string(index) + " of column '" +
					                         column_name + "' of table '" + node.table.ToString() +
					                         "' disappeared from the cache mid-gather -- retry the query");
				}
				if (cached.device_validity != nullptr) {
					if (!ctx.nulls_supported) {
						// Exactly the refusal the upload path makes: these operators have no null
						// semantics, and dropping the validity would turn NULLs into whatever bytes the
						// data buffer holds. Routing already refuses nullable input for them
						// (TableChecker::HasSafeNullHandling), so this is a backstop, not a live path.
						throw std::runtime_error("gpu_executor: cached column '" + column_name +
						                         "' contains NULLs, which this operator does not model");
					}
					any_validity = true;
				}
				chunks.push_back(cached);
			}
			auto element_size = TypeSize(chunks.front().type);
			DeviceColumn column;
			column.name = column_name;
			column.type = chunks.front().type;
			column.rows = total_rows;
			column.data = ctx.arena.Alloc(static_cast<size_t>(total_rows) * element_size);
			if (any_validity) {
				column.valid = static_cast<bool *>(ctx.arena.Alloc(static_cast<size_t>(total_rows) * sizeof(bool)));
				// Default every row to VALID, so a chunk that carried no validity buffer (the cache's
				// "this chunk has no nulls" convention) needs no copy of its own below.
				auto status = cudaMemsetAsync(column.valid, 1, static_cast<size_t>(total_rows) * sizeof(bool),
				                              ctx.stream);
				if (status != cudaSuccess) {
					throw std::runtime_error(std::string("gpu_executor: failed to initialise validity for cached "
					                                     "column '") +
					                         column_name + "': " + cudaGetErrorString(status));
				}
			}
			uint64_t row_offset = 0;
			for (auto &cached : chunks) {
				auto copy_bytes = static_cast<size_t>(cached.row_count) * element_size;
				if (cached.data_bytes != copy_bytes) {
					throw std::runtime_error("gpu_executor: cached chunk of column '" + column_name +
					                         "' holds " + std::to_string(cached.data_bytes) +
					                         " bytes for " + std::to_string(cached.row_count) +
					                         " rows, which does not match its declared type");
				}
				auto *destination = static_cast<char *>(column.data) + row_offset * element_size;
				auto status = cudaMemcpyAsync(destination, cached.device_data, copy_bytes,
				                              cudaMemcpyDeviceToDevice, ctx.stream);
				if (status != cudaSuccess) {
					throw std::runtime_error(std::string("gpu_executor: device-to-device gather of cached column '") +
					                         column_name + "' failed: " + cudaGetErrorString(status));
				}
				if (any_validity && cached.device_validity != nullptr) {
					status = cudaMemcpyAsync(column.valid + row_offset, cached.device_validity,
					                         static_cast<size_t>(cached.row_count) * sizeof(bool),
					                         cudaMemcpyDeviceToDevice, ctx.stream);
					if (status != cudaSuccess) {
						throw std::runtime_error(std::string("gpu_executor: device-to-device gather of cached "
						                                     "column '") +
						                         column_name + "'s validity failed: " + cudaGetErrorString(status));
					}
				}
				row_offset += cached.row_count;
			}
			VGPU_LOG(LogLevel::INFO, "cache_gather",
			         "\"table\":" + GpuLogger::Quote(node.table.ToString()) + ",\"column\":" +
			             GpuLogger::Quote(column_name) + ",\"chunks\":" + std::to_string(chunks.size()) +
			             ",\"rows\":" + std::to_string(total_rows));
			columns.push_back(std::move(column));
		}
		return columns;
	}

	for (size_t i = 0; i < node.column_names.size(); i++) {
		if (ctx.cursor >= ctx.inputs.size()) {
			throw std::runtime_error("gpu_executor: plan expects more scanned columns than were supplied");
		}
		auto &host = ctx.inputs[ctx.cursor++];
		if (host.validity != nullptr && !ctx.nulls_supported) {
			// GROUP_BY_AGGREGATE/HASH_JOIN/CROSS_PRODUCT (ctx.nulls_supported == false, see
			// ExecuteGpuPlan) still have no null semantics in their Thrust-based kernels, so nullable
			// input to one of them is still a clean refusal rather than a wrong answer -- unchanged from
			// before this prototype. Row-independent plans (SCAN/FILTER/PROJECTION) unpack it below
			// instead of refusing it.
			throw std::runtime_error("gpu_executor: column '" + host.name +
			                         "' contains NULLs, which this operator does not model");
		}
		DeviceColumn column;
		column.name = host.name;
		column.type = host.type;
		column.rows = host.row_count;
		auto bytes = static_cast<size_t>(host.row_count) * TypeSize(host.type);
		// RawDeviceAlloc, not ctx.arena.Alloc: this buffer might end up owned by GpuColumnCache instead
		// of this plan's DeviceArena (see CacheOrAdopt below) -- allocating it via the arena up front
		// would give it TWO owners the moment the cache also took it, and both would eventually free it.
		column.data = RawDeviceAlloc(bytes);
		if (bytes > 0 && host.data != nullptr) {
			auto copy_start = std::chrono::steady_clock::now();
			// Staged through a pooled pinned buffer so cudaMemcpyAsync is genuinely asynchronous: a
			// pageable source (what `host.data` still is -- the extension layer's plain heap buffers, see
			// docs/STREAMING_INGEST_DESIGN.md) forces the driver to bounce through its own internal pinned
			// staging anyway, synchronously, which is exactly the blocking cudaMemcpy this replaces. The
			// host-to-host memcpy into the pool slot is cheap (tens of GB/s) relative to the PCIe transfer
			// it unblocks.
			auto slot = PinnedBufferPool::Instance().Acquire(bytes);
			std::memcpy(slot.data(), host.data, bytes);
			auto status = cudaMemcpyAsync(column.data, slot.data(), bytes, cudaMemcpyHostToDevice, ctx.stream);
			VGPU_LOG(LogLevel::INFO, "h2d",
			         "\"column\":" + GpuLogger::Quote(host.name) + ",\"bytes\":" + std::to_string(bytes) +
			             ",\"us\":" +
			             std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(
			                                std::chrono::steady_clock::now() - copy_start)
			                                .count()));
			if (status != cudaSuccess) {
				throw std::runtime_error(std::string("gpu_executor: H2D upload of column '") + host.name +
				                         "' failed: " + cudaGetErrorString(status));
			}
			// `slot` must outlive the async copy it feeds. ctx.stream is synchronized once, at the end of
			// ExecuteGpuPlan, before any device buffer this chunk touched is freed or any host buffer this
			// chunk wrote is read -- see that function for why holding every slot until then is safe
			// rather than releasing it here.
			ctx.pinned_slots.push_back(std::move(slot));
		}
		uint64_t validity_bytes = 0;
		if (host.validity != nullptr) {
			// Nullable and ctx.nulls_supported: unpack the packed Arrow-style bitmap into a dense device
			// bool array. Staged through the pool for the same reason as the column data above; the buffer
			// is tiny (row_count/8 bytes) so this is not a meaningful addition to per-chunk H2D volume.
			auto packed_bytes = (static_cast<size_t>(host.row_count) + 7) / 8;
			auto packed_slot = PinnedBufferPool::Instance().Acquire(packed_bytes);
			std::memcpy(packed_slot.data(), host.validity, packed_bytes);
			auto *packed_device = static_cast<uint8_t *>(ctx.arena.Alloc(packed_bytes)); // transient scratch only
			auto copy_status = cudaMemcpyAsync(packed_device, packed_slot.data(), packed_bytes,
			                                   cudaMemcpyHostToDevice, ctx.stream);
			if (copy_status != cudaSuccess) {
				throw std::runtime_error(std::string("gpu_executor: H2D upload of column '") + host.name +
				                         "'s validity bitmap failed: " + cudaGetErrorString(copy_status));
			}
			ctx.pinned_slots.push_back(std::move(packed_slot));

			validity_bytes = static_cast<uint64_t>(host.row_count) * sizeof(bool);
			// RawDeviceAlloc for the same reason as column.data above: this is the buffer that would be
			// cached, not the transient packed scratch above.
			column.valid = static_cast<bool *>(RawDeviceAlloc(static_cast<size_t>(validity_bytes)));
			if (host.row_count > 0) {
				unsigned int block = 256;
				unsigned int grid = static_cast<unsigned int>((host.row_count + block - 1) / block);
				if (grid > 2147483647u) {
					grid = 2147483647u;
				}
				UnpackValidityKernel<<<grid, block, 0, ctx.stream>>>(packed_device, column.valid, host.row_count);
			}
		}

		// Offer this chunk to the cache's open staging slot, if the caller opened one. Staged, not
		// published: the cache accumulates chunks and only makes the column readable once the caller's
		// CommitStaging confirms they add up to the whole table (see gpu_column_cache.hpp -- publishing
		// per chunk is exactly the session-31 bug this replaced).
		bool cached_ok = false;
		if (ctx.stage_scan_to_cache) {
			GpuColumnChunk chunk;
			chunk.device_data = column.data;
			chunk.device_validity = column.valid;
			chunk.row_count = column.rows;
			chunk.data_bytes = bytes;
			chunk.validity_bytes = validity_bytes;
			chunk.type = column.type;
			cached_ok = GpuColumnCache::Instance().StageChunk(node.table, host.name, chunk);
		}
		if (!cached_ok) {
			// Not staged (caller isn't populating the cache, budget exceeded, no slot open) -- these
			// buffers have no other owner yet (RawDeviceAlloc, not the arena), so this plan's arena adopts
			// them for normal end-of-query cleanup. Exactly one owner either way: the cache on success,
			// the arena on decline, never both.
			ctx.arena.AdoptExternal(column.data);
			if (column.valid != nullptr) {
				ctx.arena.AdoptExternal(column.valid);
			}
		}
		// On success, the cache now owns column.data/column.valid -- this plan's DeviceColumn still points
		// at them (reads through a borrowed pointer for the remainder of THIS execution), same as the
		// replay path above; nothing here frees them, by design.
		columns.push_back(std::move(column));
	}
	return columns;
}

std::vector<DeviceColumn> ExecuteFilter(const GpuPlanNode &node, ExecContext &ctx) {
	if (node.children.size() != 1) {
		throw std::runtime_error("gpu_executor: FILTER expects exactly one child");
	}
	auto child_columns = ExecuteNode(*node.children[0], ctx);
	if (child_columns.empty()) {
		return child_columns;
	}
	// Composing one selection with another would need an extra indirection through the child's vector;
	// DuckDB folds consecutive predicates into a single LogicalFilter, so stacked filters are rare enough
	// that materializing here is the right trade for not having that complexity.
	if (child_columns.front().selection != nullptr) {
		child_columns = MaterializeAll(child_columns, ctx);
	}
	auto rows = child_columns.front().rows;
	if (node.expressions.empty() || rows == 0) {
		return child_columns; // nothing to filter by, or nothing to filter
	}

	// Evaluate predicate 0, then AND every remaining predicate into it (DuckDB splits a conjunctive
	// WHERE clause into one expression per LogicalFilter entry, so they combine with AND).
	auto mask_column = EvaluateExpression(node.expressions[0], child_columns, rows, ctx, "__mask");
	if (mask_column.type != GpuValueType::BOOLEAN) {
		throw std::runtime_error("gpu_executor: FILTER predicate did not evaluate to BOOLEAN");
	}
	auto *mask = static_cast<bool *>(mask_column.data);
	// SQL WHERE treats a NULL predicate as "not true", i.e. the row is excluded, same as false -- fold the
	// predicate's own nullability (from referencing a nullable column) into the mask itself rather than
	// exposing a nullable mask. AndMasks is exactly "mask &= other" over two bool arrays, so validity
	// combines through the identical kernel used for a second predicate.
	if (mask_column.valid != nullptr) {
		AndMasks(mask, mask_column.valid, rows);
	}
	for (size_t i = 1; i < node.expressions.size(); i++) {
		auto extra = EvaluateExpression(node.expressions[i], child_columns, rows, ctx, "__mask_extra");
		if (extra.type != GpuValueType::BOOLEAN) {
			throw std::runtime_error("gpu_executor: FILTER predicate did not evaluate to BOOLEAN");
		}
		AndMasks(mask, static_cast<const bool *>(extra.data), rows);
		if (extra.valid != nullptr) {
			AndMasks(mask, extra.valid, rows);
		}
	}

	// LATE MATERIALIZATION: compact the mask into a uint32 SELECTION VECTOR and hand it to the parent
	// attached to the (unchanged, still base-indexed) child columns. The previous version gathered every
	// carried column into a fresh dense buffer here, which is N allocations and N full copies per filter
	// — the dominant memory-bandwidth cost in a scan+filter+project pipeline. Consumers that cannot index
	// through a selection call MaterializeColumn.
	//
	// uint32 (not uint64) halves this array: row counts here are bounded by what fits in VRAM.
	auto *indices = static_cast<uint64_t *>(ctx.arena.Alloc(static_cast<size_t>(rows) * sizeof(uint64_t)));
	thrust::device_ptr<uint64_t> index_ptr(indices);
	thrust::device_ptr<const bool> mask_ptr(static_cast<const bool *>(mask));
	thrust::counting_iterator<uint64_t> first(0);
	auto index_end = thrust::copy_if(thrust::device, first, first + rows, mask_ptr, index_ptr,
	                                 [] __device__(bool keep) { return keep; });
	auto status = cudaDeviceSynchronize();
	if (status != cudaSuccess) {
		throw std::runtime_error(std::string("gpu_executor: filter compaction failed: ") +
		                         cudaGetErrorString(status));
	}
	auto kept = static_cast<uint64_t>(index_end - index_ptr);

	// Narrow the compacted indices to uint32 for the selection vector.
	auto *selection = static_cast<uint32_t *>(ctx.arena.Alloc(static_cast<size_t>(kept ? kept : 1) * sizeof(uint32_t)));
	if (kept > 0) {
		thrust::device_ptr<const uint64_t> wide(indices);
		thrust::copy(thrust::device, wide, wide + kept, thrust::device_ptr<uint32_t>(selection));
		auto narrow_status = cudaDeviceSynchronize();
		if (narrow_status != cudaSuccess) {
			throw std::runtime_error(std::string("gpu_executor: selection narrowing failed: ") +
			                         cudaGetErrorString(narrow_status));
		}
	}

	std::vector<DeviceColumn> result;
	result.reserve(child_columns.size());
	for (auto &column : child_columns) {
		DeviceColumn carried = column; // base data unchanged -- no copy, this is the whole point
		carried.rows = kept;
		carried.selection = selection;
		result.push_back(std::move(carried));
	}
	return result;
}

std::vector<DeviceColumn> ExecuteProjection(const GpuPlanNode &node, ExecContext &ctx) {
	if (node.children.size() != 1) {
		throw std::runtime_error("gpu_executor: PROJECTION expects exactly one child");
	}
	auto child_columns = ExecuteNode(*node.children[0], ctx);
	uint64_t rows = child_columns.empty() ? 0 : child_columns.front().rows;

	std::vector<DeviceColumn> result;
	result.reserve(node.expressions.size());
	for (size_t i = 0; i < node.expressions.size(); i++) {
		result.push_back(
		    EvaluateExpression(node.expressions[i], child_columns, rows, ctx, "col" + std::to_string(i)));
	}
	return result;
}

std::vector<DeviceColumn> ExecuteNode(const GpuPlanNode &node, ExecContext &ctx) {
	switch (node.op_type) {
	case GpuOpType::SCAN:
		return ExecuteScan(node, ctx);
	case GpuOpType::FILTER:
		return ExecuteFilter(node, ctx);
	case GpuOpType::PROJECTION:
		return ExecuteProjection(node, ctx);
	case GpuOpType::GROUP_BY_AGGREGATE: {
		if (node.children.size() != 1) {
			throw std::runtime_error("gpu_executor: GROUP_BY_AGGREGATE expects exactly one child");
		}
		// Thrust-based grouping indexes raw device pointers, so it needs dense input.
		auto child_columns = MaterializeAll(ExecuteNode(*node.children[0], ctx), ctx);
		return ExecuteGroupByNode(node, child_columns, ctx);
	}
	case GpuOpType::HASH_JOIN: {
		if (node.children.size() != 2) {
			throw std::runtime_error("gpu_executor: HASH_JOIN expects exactly two children");
		}
		// children[0] before children[1]: ExecContext::cursor hands out scanned inputs in traversal
		// order, and PhysicalGpuExecute::CollectScanInputs walks the plan in this same order.
		// Same as group-by: the sort-merge join indexes raw device pointers.
		auto left_columns = MaterializeAll(ExecuteNode(*node.children[0], ctx), ctx);
		auto right_columns = MaterializeAll(ExecuteNode(*node.children[1], ctx), ctx);
		return ExecuteJoinNode(node, left_columns, right_columns, ctx);
	}
	case GpuOpType::CROSS_PRODUCT: {
		if (node.children.size() != 2) {
			throw std::runtime_error("gpu_executor: CROSS_PRODUCT expects exactly two children");
		}
		// Same ordering and materialization contract as HASH_JOIN above: children[0] first so the
		// ExecContext::cursor hands out scanned inputs in the order CollectScanInputs gathered them, and
		// dense inputs because the expansion kernel indexes raw device pointers.
		auto left_columns = MaterializeAll(ExecuteNode(*node.children[0], ctx), ctx);
		auto right_columns = MaterializeAll(ExecuteNode(*node.children[1], ctx), ctx);
		return ExecuteCrossProductNode(node, left_columns, right_columns, ctx);
	}
	default:
		throw std::runtime_error("gpu_executor: unknown GpuOpType value " +
		                         std::to_string(static_cast<int>(node.op_type)));
	}
}

//! Process-wide code generator + kernel cache, so a repeated query shape skips NVRTC entirely.
CodeGenerator &SharedGenerator() {
	static KernelCache cache(256);
	static CodeGenerator generator(cache);
	return generator;
}

//! Process-wide stream for H2D/kernel/D2H (see ExecContext::stream for why one stream, not one per
//! chunk). Created once and reused for the life of the process, same lifetime rule as SharedGenerator.
cudaStream_t SharedStream() {
	static cudaStream_t stream = [] {
		cudaStream_t s = nullptr;
		auto status = cudaStreamCreate(&s);
		if (status != cudaSuccess) {
			throw std::runtime_error(std::string("gpu_executor: cudaStreamCreate failed: ") +
			                         cudaGetErrorString(status));
		}
		return s;
	}();
	return stream;
}

} // namespace

bool ExecuteGpuPlan(const GpuPlanNode &plan, const std::vector<GpuColumn> &inputs, const GpuExecuteOptions &options,
                    GpuExecutionResult &out) {
	try {
		DeviceArena arena;
		ExecContext ctx {inputs, 0, arena, SharedGenerator()};
		ctx.stream = SharedStream();
		// Mutually exclusive by contract (gpu_engine.hpp); enforced rather than assumed, because staging a
		// chunk that was itself read from the cache would hand the cache back its own buffers to own twice.
		const bool serving_from_cache = options.cache_chunk_index >= 0 || options.serve_whole_scan_from_cache;
		if (serving_from_cache && options.stage_scan_to_cache) {
			throw std::runtime_error("gpu_executor: cache replay and cache staging cannot be requested together");
		}
		if (options.cache_chunk_index >= 0 && options.serve_whole_scan_from_cache) {
			throw std::runtime_error("gpu_executor: cannot serve one cached chunk and the whole scan at once");
		}
		if (serving_from_cache && !inputs.empty()) {
			throw std::runtime_error("gpu_executor: cache replay expects no host inputs");
		}
		ctx.cache_chunk_index = options.cache_chunk_index;
		ctx.serve_whole_scan_from_cache = options.serve_whole_scan_from_cache;
		ctx.stage_scan_to_cache = options.stage_scan_to_cache;
		// Decided once, for the WHOLE plan, from the same predicate routing/chunking already agree on
		// (see IsRowIndependent's doc comment: routing, PhysicalGpuExecute's chunking choice, and this
		// null-support gate must all agree or a plan approved under one assumption executes under
		// another). GROUP_BY_AGGREGATE/HASH_JOIN/CROSS_PRODUCT plans always compute this false, so
		// ExecuteScan's refusal of nullable input for them is exactly as it was before this prototype.
		ctx.nulls_supported = IsRowIndependent(plan);

		// The result leaves the device as a dense host buffer, so any still-pending selection is applied
		// here -- exactly once, at the end, which is the point of deferring it.
		auto device_columns = MaterializeAll(ExecuteNode(plan, ctx), ctx);

		// Copy the final result back to host buffers owned by `out`. This runs in three passes rather than
		// one, deliberately: pass 1 pushes every buffer this result will ever need (data, and for a
		// nullable column, a dense validity scratch buffer plus its pre-sized packed-bitmap output) and
		// issues all the async D2H copies; pass 2 is the one cudaStreamSynchronize for the whole result;
		// pass 3 fills the now-synced buffers and captures pointers into them for `out.columns`. Pointers
		// into out.owned_buffers are only ever taken in pass 3, after every push in pass 1 has already
		// happened -- an earlier version of this function captured a buffer's address in the SAME loop
		// that kept calling push_back/emplace_back, which can reallocate the vector and dangle every
		// pointer captured before it (found by inspection, not on hardware: this is why pointer capture
		// and vector growth for the same container must never interleave).
		out.columns.reserve(device_columns.size());
		std::vector<PinnedBufferSlot> d2h_slots;
		struct ColumnBufferIndices {
			size_t data = SIZE_MAX;
			size_t data_slot = SIZE_MAX;
			size_t valid_dense = SIZE_MAX;
			size_t valid_slot = SIZE_MAX;
			size_t valid_packed = SIZE_MAX;
		};
		std::vector<ColumnBufferIndices> idx(device_columns.size());

		for (size_t i = 0; i < device_columns.size(); i++) {
			auto &device_column = device_columns[i];
			auto bytes = static_cast<size_t>(device_column.rows) * TypeSize(device_column.type);
			out.owned_buffers.emplace_back(bytes);
			idx[i].data = out.owned_buffers.size() - 1;
			auto copy_start = std::chrono::steady_clock::now();
			if (bytes > 0) {
				auto slot = PinnedBufferPool::Instance().Acquire(bytes);
				auto status = cudaMemcpyAsync(slot.data(), device_column.data, bytes, cudaMemcpyDeviceToHost,
				                              ctx.stream);
				if (status != cudaSuccess) {
					throw std::runtime_error(std::string("gpu_executor: D2H copy of result column '") +
					                         device_column.name + "' failed: " + cudaGetErrorString(status));
				}
				d2h_slots.push_back(std::move(slot));
				idx[i].data_slot = d2h_slots.size() - 1;
			}
			VGPU_LOG(LogLevel::INFO, "d2h",
			         "\"column\":" + GpuLogger::Quote(device_column.name) + ",\"bytes\":" +
			             std::to_string(bytes) + ",\"us\":" +
			             std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(
			                                std::chrono::steady_clock::now() - copy_start)
			                                .count()));

			if (device_column.valid != nullptr && device_column.rows > 0) {
				out.owned_buffers.emplace_back(static_cast<size_t>(device_column.rows)); // dense scratch
				idx[i].valid_dense = out.owned_buffers.size() - 1;
				auto &dense_scratch = out.owned_buffers[idx[i].valid_dense];
				auto slot = PinnedBufferPool::Instance().Acquire(dense_scratch.size());
				auto status = cudaMemcpyAsync(slot.data(), device_column.valid, dense_scratch.size(),
				                              cudaMemcpyDeviceToHost, ctx.stream);
				if (status != cudaSuccess) {
					throw std::runtime_error(std::string("gpu_executor: D2H copy of column '") +
					                         device_column.name + "'s validity failed: " + cudaGetErrorString(status));
				}
				d2h_slots.push_back(std::move(slot));
				idx[i].valid_slot = d2h_slots.size() - 1;

				// Pre-sized packed-bitmap output, all-valid by default (see FinalizeGpuColumnAccumulators
				// in vector_converter.cpp -- same packing convention: 1 bit/row, LSB-first, bit=1 valid).
				// Pushed here, in pass 1, so pass 3 only ever WRITES into it, never grows out.owned_buffers.
				out.owned_buffers.emplace_back((static_cast<size_t>(device_column.rows) + 7) / 8, 0xFFu);
				idx[i].valid_packed = out.owned_buffers.size() - 1;
			}
		}

		// One sync for every D2H issued above (H2D staging from ExecuteScan included -- ctx.pinned_slots
		// is only cleared after this point too). Every pinned slot involved is now guaranteed done being
		// read/written by the device, so it is safe to read from d2h_slots and to let every slot go out of
		// scope below, returning it to the pool for the NEXT chunk to reuse.
		auto sync_status = cudaStreamSynchronize(ctx.stream);
		if (sync_status != cudaSuccess) {
			throw std::runtime_error(std::string("gpu_executor: cudaStreamSynchronize failed: ") +
			                         cudaGetErrorString(sync_status));
		}

		// Pass 3: out.owned_buffers no longer grows past this point, so every pointer captured below into
		// it (column.data, column.validity) stays valid for the rest of `out`'s lifetime.
		for (size_t i = 0; i < device_columns.size(); i++) {
			auto &device_column = device_columns[i];
			auto bytes = static_cast<size_t>(device_column.rows) * TypeSize(device_column.type);
			auto &data_buffer = out.owned_buffers[idx[i].data];
			if (bytes > 0) {
				std::memcpy(data_buffer.data(), d2h_slots[idx[i].data_slot].data(), bytes);
			}

			GpuColumn column;
			column.name = device_column.name;
			column.type = device_column.type;
			column.data = data_buffer.empty() ? nullptr : data_buffer.data();
			column.row_count = device_column.rows;
			if (idx[i].valid_dense != SIZE_MAX) {
				auto &dense_host = out.owned_buffers[idx[i].valid_dense];
				std::memcpy(dense_host.data(), d2h_slots[idx[i].valid_slot].data(), dense_host.size());
				auto &packed = out.owned_buffers[idx[i].valid_packed];
				for (uint64_t r = 0; r < device_column.rows; r++) {
					if (dense_host[r] == 0) {
						packed[r / 8] &= ~(uint8_t(1) << (r % 8));
					}
				}
				column.validity = packed.data();
			} else {
				column.validity = nullptr;
			}
			out.columns.push_back(std::move(column));
		}
		out.success = true;
		return true;
	} catch (const std::exception &e) {
		// Any failure here is a clean "this plan did not execute on GPU" signal. The caller
		// (PhysicalGpuExecute) turns it into a NotImplementedException, which fails just this query
		// rather than invalidating the connection.
		out.success = false;
		out.columns.clear();
		out.owned_buffers.clear();
		out.error_message = e.what();
		return false;
	}
}

} // namespace vector_gpu
