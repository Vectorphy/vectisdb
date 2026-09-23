#pragma once

#include <array>
#include <condition_variable>
#include <cstddef>
#include <mutex>

// Fixed-size, cyclic pool of page-locked (cudaHostAlloc) host memory, for a PRODUCER/CONSUMER pipeline
// that fills one slot while a previous one is still draining (H2D transfer in flight) -- as opposed to
// PinnedBufferPool (pinned_buffer_pool.hpp), which grows on demand to however many slots of however many
// bytes a caller asks for and is consumed synchronously, memcpy-then-transfer, at a single call site in
// gpu_executor.cu.
//
// WHY THIS IS A SEPARATE CLASS RATHER THAN A PinnedBufferPool VARIANT. The two solve different problems:
//   - PinnedBufferPool answers "give me a pinned staging buffer for THIS transfer" -- elastic, sized to
//     the request, freed back to an unordered pool the instant the caller is done. Right for ExecuteScan,
//     which needs one slot per column, sizes vary by column width, and there is no fixed cadence.
//   - GpuMemoryPool answers "hand me the NEXT slot in a fixed rotation, waiting if it is still in use" --
//     exactly the discipline a ring buffer needs: a producer (GpuBatchAccumulator, filling a slot from
//     DataChunks) must never be handed a slot a consumer (an H2D transfer) has not finished draining, and
//     with only kSlotCount slots that means blocking, not growing.
// Sharing one class for both would either force PinnedBufferPool's every-caller-picks-its-own-size
// flexibility onto a fixed ring (defeating the deterministic memory footprint a ring buffer is for), or
// force this pool's fixed rotation onto ExecuteScan's per-column, per-transfer usage (which needs MORE
// than 3 slots live at once whenever a query touches more than 3 columns).
//
// cudaHostAllocPortable, not PinnedBufferPool's cudaHostAllocDefault: memory allocated Default is pinned
// only with respect to the CUDA context current when cudaHostAlloc was called. This pool's whole point is
// a cross-thread handoff -- one thread (a DuckDB scan) fills a slot, a DIFFERENT thread (issuing the H2D
// copy, mirroring the existing std::async chunk-execution pattern in physical_gpu_execute.cpp) drains it
// -- so the memory must stay pinned from every context's point of view, not just the one that allocated
// it. Portable is the flag that promises that.
//
// Slots are allocated ALL AT ONCE, eagerly, IN THE CONSTRUCTOR -- not grown lazily one at a time like
// PinnedBufferPool, and not deferred to a separate "ensure ready" call either. A ring buffer's entire
// value proposition is a bounded, predictable footprint known up front; growing it on demand would defeat
// that, and unlike RmmPool's VRAM budget (deliberately deferred -- see rmm_pool.hpp -- because reserving
// gigabytes of VRAM up front costs real startup time for a process that may never touch the GPU),
// kSlotCount * kSlotBytes here is a fixed, small (~256 MiB), known-in-advance host allocation with
// nothing to size against.
//
// PINNING FROM THE CONSTRUCTOR IS LOAD-BEARING, NOT STYLE. An earlier version pinned lazily from a
// separate EnsureInitialized() method called from AcquireNext() (mirroring RmmPool's EnsureInitialized
// pattern) -- constructing the function-local static singleton itself trivially, then touching CUDA only
// later, from an ordinary function call. That version SEGFAULTED (0xC0000005) during normal process exit,
// inside cudaFreeHost, on the very first slot freed -- reproduced repeatedly with a minimal, isolated
// build (just this class + cudart.lib, no other engine code linked). Moving the identical cudaSetDevice +
// cudaHostAlloc calls INTO the constructor -- with nothing else changed -- made the crash disappear,
// confirmed by the same repro.
//
// The mechanism (verified by A/B testing the ONE variable, not guessed): a function-local static's
// destructor is registered for process-exit teardown the moment its constructor COMPLETES. The CUDA
// Runtime registers its OWN internal exit-time teardown the moment its FIRST real API call happens
// (lazy init). Exit-time cleanup runs in reverse registration order. Touch CUDA lazily, well after
// Instance() has already returned, and cudart's internal registration happens AFTER ours -- so at exit,
// cudart tears itself down FIRST, and our destructor's cudaFreeHost then runs against an already-dead
// runtime. Touch CUDA from INSIDE the constructor instead, and cudart's registration happens WHILE our
// constructor is still running, i.e. BEFORE ours completes and gets registered -- so at exit, OUR
// destructor runs first, while the runtime is still alive. Neither the C++ standard nor CUDA's docs
// promise this ordering (this is exactly the kind of thing cudaErrorCudartUnloading exists to let a
// caller detect gracefully) -- constructor-time initialization does not turn "unspecified" into
// "guaranteed," it just wins the race reliably enough to be the pattern this codebase should USE, not
// discover by accident. PinnedBufferPool (pinned_buffer_pool.cpp) already follows this pattern -- its
// constructor calls cudaSetDevice(0) directly -- which is very likely WHY it was never observed crashing
// in this codebase's existing test runs, though nothing there documented it as deliberate before now.
//
// Consequence: there is no public EnsureInitialized() here (unlike RmmPool). Instance() alone is enough
// -- by the time it returns, every slot is pinned -- matching PinnedBufferPool's own public API shape
// (Instance() + Acquire(), no separate readiness call) rather than RmmPool's lazy-budget shape, because
// THIS class's whole reason to exist is a footprint known and paid for up front, not deferred.

namespace vector_gpu {

//! One ring slot, checked out from GpuMemoryPool. Non-copyable; returns itself to the pool (and wakes
//! whichever producer thread is waiting for this exact slot to free up) on destruction. This is the
//! entirety of "cleans up safely on query/task destruction" for a caller holding one: whether it is
//! dropped by falling out of scope normally or by an exception unwinding past it, the slot returns to
//! the ring exactly once, never leaked, never freed twice.
class GpuMemoryPoolSlot {
public:
	GpuMemoryPoolSlot() = default;
	GpuMemoryPoolSlot(const GpuMemoryPoolSlot &) = delete;
	GpuMemoryPoolSlot &operator=(const GpuMemoryPoolSlot &) = delete;
	GpuMemoryPoolSlot(GpuMemoryPoolSlot &&other) noexcept;
	GpuMemoryPoolSlot &operator=(GpuMemoryPoolSlot &&other) noexcept;
	~GpuMemoryPoolSlot();

	//! Pinned host memory, exactly GpuMemoryPool::kSlotBytes long. nullptr for a default-constructed or
	//! moved-from slot -- callers that might hold one of those must check valid() first.
	void *data() const {
		return data_;
	}
	size_t capacity() const {
		return capacity_;
	}
	bool valid() const {
		return data_ != nullptr;
	}

private:
	friend class GpuMemoryPool;
	GpuMemoryPoolSlot(class GpuMemoryPool *pool, int index, void *data, size_t capacity)
	    : pool_(pool), index_(index), data_(data), capacity_(capacity) {
	}

	class GpuMemoryPool *pool_ = nullptr;
	int index_ = -1;
	void *data_ = nullptr;
	size_t capacity_ = 0;
};

//! Process-wide singleton owning exactly kSlotCount pinned slots, handed out in a fixed cyclic order.
class GpuMemoryPool {
public:
	//! ONE PER CONCURRENTLY-DRAINING SLOT, PLUS ONE FOR THE PRODUCER. With only two slots a producer
	//! finishing slot N has nowhere to go but slot N+1, which is almost certainly still draining from the
	//! transfer TWO batches ago -- there is no room for the producer to ever get ahead of a consumer that
	//! has not caught up yet, so the ring degrades to fully synchronous double-buffering.
	//!
	//! Four, not three, and the fourth was earned on a profiler rather than reasoned into existence.
	//! GpuStreamPipeline (gpu_stream_pipeline.hpp) keeps kDepth = 3 batches in flight and holds each
	//! batch's slot until its H2D actually COMPLETES -- which, on Windows/WDDM, is around a millisecond
	//! after the copy is issued, because the driver batches command submission. With three slots the
	//! producer's one plus at most two held by the pipeline meant the third submission was always refused,
	//! the pipeline capped out at two in flight, and an `nsys` capture of a real 3M-row query showed the
	//! H2D, kernel and D2H of every batch running strictly one after another: zero overlap, which is the
	//! entire thing the pipeline exists to produce. gpu_stream_pipeline.cu static_asserts this relationship
	//! so the two constants cannot drift apart silently.
	static constexpr size_t kSlotCount = 4;
	//! ~64 MiB/slot. At GpuBatchAccumulator::kBatchReadyRows (32,768) rows, this comfortably covers every
	//! column of a realistically wide query (roughly 200 DOUBLE columns' worth of dense data plus their
	//! validity scratch -- see gpu_batch_accumulator.hpp's layout comment for the exact accounting) with
	//! headroom to spare, while keeping the whole ring's footprint (~192 MiB) small and predictable.
	static constexpr size_t kSlotBytes = 350ull * 1024 * 1024;

	//! Constructs (pinning all kSlotCount slots -- see the file header for why that happens here and not
	//! in a separate lazy-init call) the first time anything calls this. Throws std::runtime_error if a
	//! slot cannot be pinned; per ordinary C++ magic-statics semantics, a throwing constructor leaves the
	//! static uninitialized, so the NEXT call to Instance() attempts construction again rather than
	//! wedging in a half-built state.
	static GpuMemoryPool &Instance();

	GpuMemoryPool(const GpuMemoryPool &) = delete;
	GpuMemoryPool &operator=(const GpuMemoryPool &) = delete;

	//! Blocks until the NEXT slot in cyclic order (0, 1, 2, 0, 1, 2, ...) is free, then checks it out and
	//! advances the cursor.
	GpuMemoryPoolSlot AcquireNext();

	//! Test-only: rewinds the cyclic cursor to slot 0. The slots themselves are never freed or resized
	//! (there is nothing to vary between tests the way RmmPool's ResetForTesting varies budget size) --
	//! this exists only so each test starts from a known acquisition order rather than wherever a prior
	//! test's run left the cursor. Caller's responsibility to ensure no slot is currently checked out
	//! first (i.e. every GpuMemoryPoolSlot from a previous test has already gone out of scope) --
	//! resetting the cursor while a slot is still in use would let AcquireNext() hand that same slot to
	//! two holders at once.
	void ResetCursorForTesting();

private:
	//! Pins all kSlotCount slots (kSlotCount separate cudaHostAlloc(..., cudaHostAllocPortable) calls).
	//! Throws std::runtime_error if a slot cannot be pinned, freeing whatever earlier slots in THIS call
	//! did succeed first, rather than leaking them. See the file header for why this work happens here
	//! and not in a separate lazily-called method.
	GpuMemoryPool();
	~GpuMemoryPool();

	friend class GpuMemoryPoolSlot;
	//! Marks slot `index` free and wakes every waiter to re-check its own wait condition. notify_all, not
	//! notify_one: unlike PinnedBufferPool's "any free slot will do" wait (where waking one waiter is
	//! always productive), a waiter here blocks on one SPECIFIC index, so the one release that would
	//! satisfy it must not be swallowed by waking some other, unrelated waiter instead.
	void Release(int index);

	struct Slot {
		void *data = nullptr;
		bool in_use = false;
	};

	std::mutex mutex_;
	std::condition_variable cv_;
	std::array<Slot, kSlotCount> slots_ {};
	size_t next_index_ = 0; // cyclic cursor: the slot index AcquireNext() will try to check out next
};

} // namespace vector_gpu
