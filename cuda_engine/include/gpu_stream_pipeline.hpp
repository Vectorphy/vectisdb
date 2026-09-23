#pragma once

#include "gpu_engine.hpp"
#include "gpu_memory_pool.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// A THREE-STREAM, THREE-SLOT execution pipeline for a row-independent projection: upload of batch N+2,
// kernel execution of batch N+1, and download of batch N all in flight on the device at the same time,
// ordered against each other only by CUDA events. It is the piece that turns the existing "one chunk on
// the device at a time" executor (gpu_executor.cu) into something that can actually hide PCIe latency
// behind compute rather than paying the two in sequence.
//
// WHAT IT DOES NOT DO, AND WHY THAT IS DELIBERATE. This is not a second general-purpose executor.
// GpuEngine::ExecutePlan stays the entry point for every plan shape; this class runs exactly one family --
// a chain of one or more PROJECTIONs over a SCAN, no filter, no aggregate, no join (see Supports()) --
// because that is the only shape whose OUTPUT ROW COUNT IS KNOWN BEFORE THE KERNEL RUNS. A FILTER's output
// size is data-dependent, so a pipelined D2H would have to either copy back a worst-case-sized buffer or
// stall the download stream on a device-to-host read of the surviving-row count -- and that stall is
// exactly the serialization this class exists to remove. Anything outside the supported shape must keep
// using PhysicalGpuExecute; the caller checks Supports() and routes accordingly rather than this class
// silently degrading.
//
// A CHAIN of projections rather than a single one, because that is what real plans look like: DuckDB
// routinely emits PROJECTION -> PROJECTION -> GET (a narrowing projection over the scan, with the
// computed one above it), so accepting only a direct SCAN child would leave this class unreachable on
// the queries it was written for -- verified by watching it decline a live query, not assumed. Each level
// runs on stream_exec in turn, so one stream provides the ordering between them and no host wait is
// involved; only the top level's outputs are downloaded.
//
// WHY THE THREE STREAMS ARE NON-BLOCKING (cudaStreamNonBlocking), unlike gpu_executor.cu's single
// SharedStream. A stream created with plain cudaStreamCreate implicitly synchronizes with the legacy
// default stream, which is what let the streaming-ingest redesign (docs/STREAMING_INGEST_DESIGN.md, point
// 3) add one async stream without re-auditing every Thrust call in the engine. Three streams that all
// implicitly sync with the default stream would also implicitly sync with EACH OTHER through it, which
// would leave the H2D/kernel/D2H of different batches serialized -- i.e. this class would compile, run,
// and produce correct answers while delivering none of the overlap it exists for. So these three are
// non-blocking, and the consequence is accepted explicitly: nothing in this pipeline may touch the legacy
// default stream, and it therefore uses NO Thrust algorithm and NO other operator's kernels. Everything
// it launches, it launches on stream_exec.
//
// DEVICE MEMORY IS ALLOCATED ONCE, IN THE CONSTRUCTOR, AND NEVER DURING A SUBMISSION. cudaMalloc and
// cudaFree synchronize the whole device: a single one of either, issued between two batches, drains all
// three streams and collapses the pipeline back to serial execution for that batch. So every buffer a
// stage will ever need is sized for max_rows_per_batch and allocated up front (the same
// pay-for-a-bounded-footprint-in-advance rule GpuMemoryPool applies to pinned host memory, and for the
// same reason). Peak device use is kDepth times one batch's working set, NOT one chunk's -- which is why
// this class is sized against GpuBatchAccumulator::kBatchReadyRows (32,768 rows) rather than
// PhysicalGpuExecute::MAX_CHUNK_ROWS (2,097,152): three batches of 32k rows is a few MB and cannot
// invalidate a routing decision that already approved a 2M-row chunk, whereas three chunks of 2M rows
// would triple the footprint routing sized the plan against.
//
// EXTERNALLY SERIALIZED BY CONTRACT. Calls must never overlap: one at a time, from whichever thread, and
// interleaved with nothing but the GpuBatchAccumulator feeding it. (Not the same as same-thread-only --
// DuckDB may drive one source operator from different task threads across calls, which CUDA is fine with
// as long as the calls do not overlap.) There is no lock anywhere in this class, and that is not an
// oversight to fix later: the whole point is that the CALLING thread never blocks while the device works,
// so it returns to DuckDB and keeps scanning; a second concurrent caller would buy nothing and would need
// a lock on every event query.

namespace vector_gpu {

struct GpuStreamPipelineImpl;

//! One batch of already-pinned host rows handed to the pipeline.
//!
//! `slot` owns the pinned memory every column's `data`/`validity` points into (this is exactly what
//! GpuBatchAccumulator::TakeReadyBatch hands back). Submit() takes ownership of it and releases it back to
//! GpuMemoryPool as soon as that batch's H2D has actually COMPLETED -- not when the batch's results are
//! consumed. That early release is load-bearing: the ring has only GpuMemoryPool::kSlotCount slots, and
//! the producer needs one of them to fill the next batch into while up to kDepth batches are still in
//! flight on the device.
struct GpuStreamBatch {
	GpuMemoryPoolSlot slot;
	//! One entry per column of the pipeline's declared input schema, in that exact order. Names are
	//! checked against the schema on submission -- a positional mismatch would otherwise reinterpret one
	//! column's bytes as another's.
	std::vector<GpuColumn> columns;
	uint64_t row_count = 0;
};

//! One completed batch's results, in PINNED host memory owned by the pipeline stage that produced it.
//!
//! RAII: the stage (its device buffers, its pinned output buffer, its events) is returned to the pipeline
//! when this object is destroyed or overwritten, and only then can a later Submit() reuse it. Hold it for
//! exactly as long as you are reading `columns()` -- every pointer in there dangles the moment it goes.
class GpuStreamResult {
public:
	GpuStreamResult() = default;
	GpuStreamResult(const GpuStreamResult &) = delete;
	GpuStreamResult &operator=(const GpuStreamResult &) = delete;
	GpuStreamResult(GpuStreamResult &&other) noexcept;
	GpuStreamResult &operator=(GpuStreamResult &&other) noexcept;
	~GpuStreamResult();

	//! One column per expression of the TOP projection level, in expression order, named "col0", "col1",
	//! ... -- the same naming gpu_executor.cu's ExecuteProjection uses, so a caller packing these into a
	//! DuckDB operator's output types positionally behaves identically on both paths.
	const std::vector<GpuColumn> &columns() const {
		return columns_;
	}
	uint64_t row_count() const {
		return row_count_;
	}
	bool valid() const {
		return pipeline_ != nullptr;
	}
	//! Releases the stage early, before destruction. Safe to call on an already-released result.
	void Reset();

private:
	friend class GpuStreamPipeline;
	class GpuStreamPipeline *pipeline_ = nullptr;
	int stage_ = -1;
	std::vector<GpuColumn> columns_;
	uint64_t row_count_ = 0;
};

class GpuStreamPipeline {
public:
	//! Batches in flight on the device at once. Three because that is what it takes to have all three
	//! stages busy simultaneously -- one uploading, one computing, one downloading -- which is the entire
	//! claim this class makes. Two would leave one of the three streams idle at every instant.
	static constexpr size_t kDepth = 3;

	//! Sets of buffers, which is ONE MORE than kDepth. The extra one is for the batch the consumer is
	//! currently reading: a GpuStreamResult keeps its stage checked out for as long as the caller holds it,
	//! and a DuckDB consumer holds one across sixteen GetData calls while it emits 32,768 rows 2,048 at a
	//! time. Sized at exactly kDepth, that permanently-occupied stage came straight out of the in-flight
	//! budget -- measured, not theorised: a debug log of every submission showed in_flight reaching 3 once
	//! and 2 on the other ninety submissions, i.e. the pipeline ran a slot short for its entire life.
	static constexpr size_t kStageCount = kDepth + 1;

	//! True when `plan` is a shape this pipeline runs: a chain of one or more PROJECTIONs ending in a
	//! SCAN, every level carrying at least one expression, every expression's inputs resolvable against
	//! what its input produces (the SCAN's columns at the bottom, the level below's "col<i>" outputs above
	//! it), and every output type having a fixed device width. False for everything else -- including
	//! plans this engine executes perfectly well through PhysicalGpuExecute. See the file header for why
	//! the shape is this narrow.
	static bool Supports(const GpuPlanNode &plan);

	//! Allocates every device buffer (including one set of intermediates per projection level), every
	//! pinned output buffer, the three streams and the per-stage events, and compiles (or fetches from the
	//! process-wide kernel cache) one fused kernel per expression at every level -- all of it before a
	//! single row is accepted, so an unsupported expression or
	//! an out-of-memory device fails HERE, while the caller can still decline the plan cleanly, rather
	//! than halfway through a scan.
	//!
	//! `input_names`/`input_types` describe the columns every submitted batch will carry, in order (take
	//! them from the GpuBatchAccumulator that will feed this pipeline, so the two cannot disagree about
	//! what the bytes in a slot mean). Throws std::runtime_error if the plan shape is unsupported or any
	//! resource cannot be acquired.
	GpuStreamPipeline(const GpuPlanNode &plan, const std::vector<std::string> &input_names,
	                  const std::vector<GpuValueType> &input_types, uint64_t max_rows_per_batch);
	~GpuStreamPipeline();
	GpuStreamPipeline(const GpuStreamPipeline &) = delete;
	GpuStreamPipeline &operator=(const GpuStreamPipeline &) = delete;

	//! Output type per column of the TOP projection level, in order -- i.e. of the result. The caller must
	//! check these agree with whatever it is packing the results into.
	const std::vector<GpuValueType> &output_types() const;

	//! Non-blocking housekeeping: releases the pinned input slot of every in-flight batch whose H2D has
	//! completed (cudaEventQuery, never cudaEventSynchronize). Cheap; call it whenever you are about to
	//! ask whether the pipeline can take more work. Never blocks, never throws on "not ready yet".
	void Poll();

	//! True when Submit() would neither exceed kDepth in-flight batches NOR leave the producer without a
	//! free GpuMemoryPool slot to fill the next batch into. Calls Poll() first, so a caller does not have
	//! to. See the .cu for the slot accounting -- getting the second half of this condition wrong
	//! deadlocks the producer inside GpuBatchAccumulator::TakeReadyBatch, which waits on a pool slot that
	//! only this class can free.
	bool CanSubmit();

	//! Queues one batch: H2D on stream_h2d, then (gated on an event, not a sync) the fused kernels on
	//! stream_exec, then (gated on another event) the D2H into pinned host memory on stream_d2h. Returns
	//! as soon as the work is QUEUED -- it does not wait for any of it. Throws std::runtime_error if the
	//! batch does not match the declared schema, holds more than max_rows_per_batch rows, or a CUDA call
	//! fails.
	//!
	//! Precondition: CanSubmit(). Submitting anyway throws rather than silently overwriting a stage whose
	//! results have not been read.
	void Submit(GpuStreamBatch batch);

	//! Hands back the OLDEST in-flight batch's results if its D2H has already finished, without waiting.
	//! Returns false (leaving `out` untouched) when nothing is in flight or the oldest is not done yet.
	//! Batches complete in submission order, so this is FIFO and the caller's row order is the scan's.
	bool TryTake(GpuStreamResult &out);

	//! Same, but WAITS for the oldest in-flight batch (cudaEventSynchronize on its D2H-complete event).
	//! Returns false only when nothing is in flight at all. This is the one place the calling thread ever
	//! blocks, and a caller should reach it only when it has genuinely nothing else to do -- no scanning
	//! left to overlap with, and no already-finished batch waiting via TryTake.
	bool TakeBlocking(GpuStreamResult &out);

	//! Batches submitted but not yet taken.
	size_t in_flight() const;

private:
	friend class GpuStreamResult;
	//! Returns a stage to the free list once its GpuStreamResult is dropped.
	void ReleaseStage(int stage);

	std::unique_ptr<GpuStreamPipelineImpl> impl_;
};

} // namespace vector_gpu
