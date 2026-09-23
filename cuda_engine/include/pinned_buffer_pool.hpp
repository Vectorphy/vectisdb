#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <vector>

// Pooled, reused page-locked (cudaHostAlloc) staging memory for H2D transfers.
//
// docs/TODO.md P1 item 5 measured that pinning is only worth it with a POOLED, REUSED allocator --
// page-locking a fresh buffer per query costs more than the transfer it speeds up, because
// cudaHostAlloc/cudaFreeHost are themselves synchronizing, expensive calls. This pool pins a small,
// fixed number of slots ONCE (grown, never shrunk, exactly like CodeGenerator::input_ptrs_device_) and
// hands them out for the lifetime of the process, so the pinning cost is amortized across every chunk of
// every query rather than paid per chunk.
//
// Deliberately NOT exposed through gpu_engine.hpp: the extension layer (physical_gpu_execute.cpp,
// table_scanner.cpp) must stay CUDA-unaware by design (see gpu_engine.hpp's header comment). This pool is
// consumed only from inside cuda_engine, at the H2D call site in gpu_executor.cu -- host data scanned by
// the (plain, unpinned) extension-side buffers is memcpy'd into a pinned slot immediately before the
// async device upload, so the DMA-eligible transfer never requires the extension to know what pinning is.

namespace vector_gpu {

//! One pinned slot, checked out from the pool. Non-copyable; returns itself to the pool on destruction.
//! Moveable so it can be returned by value from Acquire().
class PinnedBufferSlot {
public:
	PinnedBufferSlot() = default;
	PinnedBufferSlot(const PinnedBufferSlot &) = delete;
	PinnedBufferSlot &operator=(const PinnedBufferSlot &) = delete;
	PinnedBufferSlot(PinnedBufferSlot &&other) noexcept;
	PinnedBufferSlot &operator=(PinnedBufferSlot &&other) noexcept;
	~PinnedBufferSlot();

	//! Pinned host memory, valid for at least the `bytes` requested in Acquire(). May be larger: slots
	//! are grown to the largest request ever made and never shrunk, so a later smaller request reuses
	//! the same allocation instead of re-pinning.
	void *data() const {
		return data_;
	}

private:
	friend class PinnedBufferPool;
	PinnedBufferSlot(class PinnedBufferPool *pool, int index, void *data) : pool_(pool), index_(index), data_(data) {
	}

	class PinnedBufferPool *pool_ = nullptr;
	int index_ = -1;
	void *data_ = nullptr;
};

//! Process-wide pool of pinned slots, grown on demand up to a small cap.
//!
//! One SCAN of a chunk needs one staging slot PER COLUMN concurrently -- each column's async H2D must
//! keep its own staging buffer alive until that specific transfer completes, so a single shared slot
//! reused across columns would force column 2's copy to wait for column 1's to finish, defeating the
//! point. The pool therefore starts empty and grows a new slot (one more cudaHostAlloc) whenever every
//! existing slot is checked out, up to kMaxSlots -- typical queries touch a handful of columns, so this
//! settles after the first chunk and never grows again for the rest of the query. Beyond the cap, Acquire
//! blocks until a slot frees, which only degrades a pathologically wide scan back toward the old
//! synchronous-copy behavior for its excess columns rather than growing pinned memory without bound.
//!
//! Slots persist and are reused across every chunk and every query for the process's lifetime (never
//! pinned/unpinned per query) -- the amortization docs/TODO.md P1 item 5 required before pinning could
//! pay for itself at all.
class PinnedBufferPool {
public:
	static constexpr size_t kMaxSlots = 16;

	static PinnedBufferPool &Instance();

	//! Grows the slot count (a fresh cudaHostAlloc) if every existing slot is checked out and the pool is
	//! below kMaxSlots; otherwise blocks until a slot frees. Grows the chosen slot's allocation if it is
	//! smaller than `bytes` (cudaHostAlloc + cudaFreeHost of the old allocation) -- amortized to
	//! O(log(max chunk size)) regrows over a ramping chunk-size scan, not one regrow per chunk.
	PinnedBufferSlot Acquire(size_t bytes);

private:
	PinnedBufferPool();
	~PinnedBufferPool();
	PinnedBufferPool(const PinnedBufferPool &) = delete;
	PinnedBufferPool &operator=(const PinnedBufferPool &) = delete;

	friend class PinnedBufferSlot;
	void Release(int index);

	struct Slot {
		void *data = nullptr;
		size_t capacity = 0;
		bool in_use = false;
	};

	std::mutex mutex_;
	std::condition_variable cv_;
	std::vector<Slot> slots_; // starts empty; grows lazily in Acquire, up to kMaxSlots
};

} // namespace vector_gpu
