// gpu_executor_internal.hpp
//
// Implementation-private interface shared between gpu_executor.cu (the plan walker) and the
// per-operator translation units under src/operators/. NOT part of cuda_engine's public API — external
// callers see only include/gpu_engine.hpp. It lives in src/ rather than include/ for exactly that
// reason.
//
// It exists so an operator big enough to deserve its own file (gpu_groupby.cu) can allocate device
// memory with the same plan-scoped lifetime, and reuse the same gather primitive, as the executor
// itself — instead of either duplicating them or collapsing every operator back into one file.

#pragma once

#include "gpu_engine.hpp"
#include "pinned_buffer_pool.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vector_gpu {

class CodeGenerator;

//! Byte width of one element of `type` on the device. Throws for types with no fixed device
//! representation (DICTIONARY_STRING).
size_t TypeSize(GpuValueType type);

//! Owns every device buffer allocated while executing one plan, freeing them all when execution ends
//! (success or exception). Deliberately cudaMalloc rather than RmmPool: intermediate result sizes here
//! are data-dependent and can exceed the pool arena, and a failed cudaMalloc gives a clean error where a
//! pool exhaustion would need its own fallback path. RmmPool remains the VRAM-headroom oracle used by
//! TableChecker to decide whether to offload at all.
class DeviceArena {
public:
	DeviceArena() = default;
	DeviceArena(const DeviceArena &) = delete;
	DeviceArena &operator=(const DeviceArena &) = delete;
	~DeviceArena();

	//! Throws std::runtime_error on allocation failure. A zero-byte request still returns a valid,
	//! distinct pointer.
	void *Alloc(size_t bytes);

	//! Adopts a device pointer this arena did NOT allocate (e.g. a raw cudaMalloc'd buffer a caller made
	//! for GpuColumnCache but the cache declined to keep) so it still gets freed exactly once, when this
	//! arena is destroyed -- same lifetime as everything Alloc'd normally. The point: a buffer that might
	//! end up cache-owned must not be allocated via Alloc() in the first place (the cache and the arena
	//! would then both think they own it, and both would try to free it -- a double-free). Allocating it
	//! separately and only adopting it HERE on the "cache declined" path keeps exactly one owner at a
	//! time, always.
	void AdoptExternal(void *ptr);

private:
	std::vector<void *> buffers_;
};

//! A column living in device memory during execution.
//!
//! LATE MATERIALIZATION: when `selection` is non-null, this column is NOT dense. `data` still points at
//! the full base column, `rows` is the number of SURVIVING rows, and logical row i lives at base row
//! `selection[i]`. Every column produced by one operator shares the same `selection` pointer.
//!
//! This is what lets a filter avoid copying every carried column: it publishes one uint32 index array
//! instead of N dense buffers. The trade is uncoalesced reads in whatever consumes it, which is far
//! cheaper than allocating and writing those buffers -- but it means any consumer that cannot index
//! through a selection (Thrust-based group-by and join, and the final D2H) must call MaterializeColumn
//! first.
struct DeviceColumn {
	std::string name;
	GpuValueType type = GpuValueType::FLOAT64;
	void *data = nullptr;
	uint64_t rows = 0;
	const uint32_t *selection = nullptr;
	//! Dense, one byte per BASE row (not per selection-vector row -- same indexing convention as `data`:
	//! read at `selection[idx]` under late materialization, at `idx` when dense), non-zero = valid.
	//! nullptr = "no nulls in this column", the same sentinel convention GpuColumn::validity uses on the
	//! host side. Populated by ExecuteScan only when ctx.nulls_supported (see below); PROJECTION/FILTER
	//! propagate it by ANDing every referenced input's bit per output row, matching SQL NULL propagation
	//! for scalar expressions. Unlike `data`, this is a plain bool array, not the packed Arrow bitmap
	//! format GpuColumn uses -- packing only happens once, at the host boundary (ExecuteScan's unpack and
	//! ExecuteGpuPlan's final pack), for the same reason GpuColumnAccumulator::null_flags is unpacked
	//! during accumulation: chunk/thread boundaries never land on a byte boundary, so packed bits would
	//! misalign the moment more than one write touches a shared byte.
	bool *valid = nullptr;
};

//! Returns a dense copy of `column`, gathering through its selection vector. A no-op (returns the column
//! unchanged) when it is already dense.


struct ExecContext {
	const std::vector<GpuColumn> &inputs;
	size_t cursor = 0; // next unconsumed entry of `inputs`, advanced by each SCAN in traversal order
	DeviceArena &arena;
	CodeGenerator &generator;
	//! H2D (ExecuteScan) and the final D2H (ExecuteGpuPlan) run on this stream via cudaMemcpyAsync instead
	//! of the legacy default stream, so a CPU-side producer (the streamed table scanner, see
	//! table_scanner.hpp) can keep filling the NEXT chunk's staging buffer while THIS chunk's transfer is
	//! in flight, without the CPU thread blocking on the copy. Deliberately still ONE stream, not one per
	//! chunk: PhysicalGpuExecute keeps only one GPU chunk resident on the device at a time (see
	//! MAX_CHUNK_ROWS's doc comment), and TableChecker::HasVramHeadroomForPlan sizes VRAM against exactly
	//! that invariant -- a second concurrently-resident chunk would silently invalidate a routing decision
	//! that already approved the plan. One stream still removes the "blocks the issuing CPU thread" cost
	//! of a synchronous cudaMemcpy; it does not add cross-chunk device concurrency, on purpose.
	cudaStream_t stream = nullptr;
	//! True when the whole plan is row-independent (SCAN/FILTER/PROJECTION only -- see IsRowIndependent).
	//! Gates whether ExecuteScan unpacks a nullable column's validity bitmap into device memory or refuses
	//! it outright. GROUP_BY_AGGREGATE/HASH_JOIN/CROSS_PRODUCT's Thrust-based kernels have no null
	//! semantics and are unchanged by this prototype, so a nullable column reaching one of them must still
	//! refuse exactly as before -- this flag is what keeps that refusal in place without threading parent-
	//! operator-type context through every ExecuteScan call.
	bool nulls_supported = false;
	//! GPU-resident column cache wiring for this execution, copied straight from the caller's
	//! GpuExecuteOptions (see gpu_engine.hpp for the full contract). Two independent modes:
	//!   cache_chunk_index >= 0  -- ExecuteScan serves this chunk index from GpuColumnCache and does no
	//!                              host upload at all; `inputs` is empty and `cursor` stays untouched.
	//!   stage_scan_to_cache     -- ExecuteScan offers every column it DOES upload to the cache's open
	//!                              staging slot for that table.
	//!   serve_whole_scan_from_cache -- ExecuteScan concatenates EVERY cached chunk of each column into
	//!                              one contiguous device buffer, for the whole-input operators that
	//!                              cannot be replayed chunk-by-chunk. Also does no host upload.
	//! Never both a read mode and the staging mode: the first reads what the second wrote, in a later
	//! query. The two read modes are likewise mutually exclusive -- one serves a chunk, the other the lot.
	int64_t cache_chunk_index = -1;
	bool serve_whole_scan_from_cache = false;
	bool stage_scan_to_cache = false;
	//! Pinned staging slots acquired by ExecuteScan, held for the WHOLE plan execution rather than
	//! released right after each cudaMemcpyAsync call returns. Releasing early would let the pool hand the
	//! same pinned memory to a concurrent Acquire() while this chunk's async copy might still be reading
	//! it -- the copy returning does not mean the copy has finished, only that it has been queued. Cleared
	//! (releasing every slot back to the pool) only after ExecuteGpuPlan's one cudaStreamSynchronize, by
	//! which point every H2D that read from them is guaranteed complete.
	std::vector<PinnedBufferSlot> pinned_slots;
};

//! Throws std::runtime_error if no column of that name is present.
const DeviceColumn &FindColumn(const std::vector<DeviceColumn> &columns, const std::string &name);

//! Returns a dense copy of `column`, gathering through its selection vector. Returns it unchanged when
//! already dense. Consumers that index raw device pointers (Thrust group-by/join, the final D2H) must
//! call this; expression kernels do not, because they take the selection vector directly.
DeviceColumn MaterializeColumn(const DeviceColumn &column, ExecContext &ctx);

//! Materializes every column of a relation. Convenience for operators that cannot consume a selection.
std::vector<DeviceColumn> MaterializeAll(const std::vector<DeviceColumn> &columns, ExecContext &ctx);

//! Returns a dense copy of `column`, gathering through its selection vector. Returns it unchanged when
//! already dense. Consumers that index raw device pointers (Thrust group-by/join, the final D2H) must
//! call this; expression kernels do not, because they take the selection vector directly.


//! Materializes every column of a relation. Convenience for operators that cannot consume a selection.
std::vector<DeviceColumn> MaterializeAll(const std::vector<DeviceColumn> &columns, ExecContext &ctx);

//! Gathers `column` at the given device row indices into a new device column of `count` rows.
DeviceColumn GatherColumn(const DeviceColumn &column, const uint64_t *indices, uint64_t count, ExecContext &ctx);

//! GROUP_BY_AGGREGATE, implemented in operators/gpu_groupby.cu. `child_columns` is the already-executed
//! input operator's output. Returns the group key columns followed by the aggregate columns.
std::vector<DeviceColumn> ExecuteGroupByNode(const GpuPlanNode &node, const std::vector<DeviceColumn> &child_columns,
                                             ExecContext &ctx);

//! HASH_JOIN, implemented in operators/gpu_hash_join.cu. `left_columns`/`right_columns` are the
//! already-executed children[0]/children[1] outputs. Returns exactly `node.output_columns`, in order.
std::vector<DeviceColumn> ExecuteJoinNode(const GpuPlanNode &node, const std::vector<DeviceColumn> &left_columns,
                                          const std::vector<DeviceColumn> &right_columns, ExecContext &ctx);

//! CROSS_PRODUCT, implemented in operators/gpu_cross_join.cu. Same input/output contract as
//! ExecuteJoinNode, but every left row is paired with every right row.
std::vector<DeviceColumn> ExecuteCrossProductNode(const GpuPlanNode &node,
                                                  const std::vector<DeviceColumn> &left_columns,
                                                  const std::vector<DeviceColumn> &right_columns, ExecContext &ctx);

//! Resolves one requested output column of a two-input operator to whichever side actually produces it,
//! setting `from_left` accordingly. Throws if the name is present on BOTH sides rather than silently
//! taking the left one -- columns are selected by name, so an ambiguous name could otherwise return the
//! wrong side's data. Shared by the join and cross-product operators so that rule has one implementation.
const DeviceColumn &ResolveSideColumn(const std::string &name, const std::vector<DeviceColumn> &left_columns,
                                      const std::vector<DeviceColumn> &right_columns, bool &from_left);

} // namespace vector_gpu
