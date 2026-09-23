#include "pinned_buffer_pool.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace vector_gpu {

PinnedBufferSlot::PinnedBufferSlot(PinnedBufferSlot &&other) noexcept
    : pool_(other.pool_), index_(other.index_), data_(other.data_) {
	other.pool_ = nullptr;
	other.index_ = -1;
	other.data_ = nullptr;
}

PinnedBufferSlot &PinnedBufferSlot::operator=(PinnedBufferSlot &&other) noexcept {
	if (this != &other) {
		if (pool_ != nullptr) {
			pool_->Release(index_);
		}
		pool_ = other.pool_;
		index_ = other.index_;
		data_ = other.data_;
		other.pool_ = nullptr;
		other.index_ = -1;
		other.data_ = nullptr;
	}
	return *this;
}

PinnedBufferSlot::~PinnedBufferSlot() {
	if (pool_ != nullptr) {
		pool_->Release(index_);
	}
}

PinnedBufferPool::PinnedBufferPool() {
	// cudaSetDevice(0) matches this codebase's single-device convention (see
	// code_generator.cpp's GetSharedPrimaryContext, which retains device 0's primary context). The CUDA
	// Runtime lazily creates its own primary context on first Runtime API call and shares it with the
	// driver API's cuDevicePrimaryCtxRetain (same physical context, whichever side initializes it first),
	// so this pool needs no driver-API context handling of its own -- it only ever calls Runtime API.
	//
	// DO NOT REMOVE OR DEFER THIS CALL -- it is also what makes ~PinnedBufferPool's cudaFreeHost calls
	// below safe at process exit, not just device selection. Verified directly (gpu_memory_pool.hpp/.cpp,
	// session 38): a sibling pool whose constructor did NOT touch CUDA (all real CUDA calls deferred to a
	// separate lazily-invoked method) segfaulted inside its own equivalent destructor during ordinary
	// process exit -- reproduced repeatedly with a minimal, isolated build. The mechanism: a function-
	// local static's destructor is registered for exit-time teardown the moment its CONSTRUCTOR completes;
	// the CUDA Runtime registers its own internal exit-time teardown the moment its first real API call
	// happens; exit-time cleanup runs in reverse registration order. This cudaSetDevice(0) call is that
	// first real API call, made WHILE this constructor is still running -- so cudart's registration lands
	// before this object's destructor's does, and at exit this destructor runs first, while the runtime is
	// still alive. Moving or removing this call (e.g. "it looks redundant, Acquire() doesn't need it")
	// reopens the same crash cudaFreeHost below would otherwise hit. See gpu_memory_pool.hpp's file header
	// for the full A/B-tested writeup of this mechanism.
	cudaSetDevice(0);
}

PinnedBufferPool::~PinnedBufferPool() {
	// Safe to call cudaFreeHost here specifically BECAUSE the constructor above already made a real CUDA
	// call -- see its comment for why that ordering, not luck, is what keeps this from segfaulting at
	// process exit the way an otherwise-identical destructor was found to when its object's first CUDA
	// touch happened lazily instead. "Best-effort" is not the operative property; ORDER is.
	for (auto &slot : slots_) {
		if (slot.data != nullptr) {
			cudaFreeHost(slot.data);
		}
	}
}

PinnedBufferPool &PinnedBufferPool::Instance() {
	static PinnedBufferPool instance;
	return instance;
}

PinnedBufferSlot PinnedBufferPool::Acquire(size_t bytes) {
	std::unique_lock<std::mutex> lock(mutex_);
	int chosen = -1;
	cv_.wait(lock, [&] {
		for (size_t i = 0; i < slots_.size(); i++) {
			if (!slots_[i].in_use) {
				chosen = static_cast<int>(i);
				return true;
			}
		}
		if (slots_.size() < kMaxSlots) {
			slots_.emplace_back();
			chosen = static_cast<int>(slots_.size() - 1);
			return true;
		}
		return false; // every slot in use and at the cap -- wait for one to free
	});

	auto &slot = slots_[static_cast<size_t>(chosen)];
	if (bytes > slot.capacity) {
		// Grow, never shrink -- a later smaller chunk (e.g. the trailing partial chunk of a ramping scan)
		// reuses this same pinned allocation rather than re-pinning. cudaFreeHost/cudaHostAlloc are
		// synchronizing device-wide calls, so this path is only hit O(log(max_chunk_rows)) times over a
		// doubling ramp, not once per chunk.
		void *grown = nullptr;
		auto status = cudaHostAlloc(&grown, bytes, cudaHostAllocDefault);
		if (status != cudaSuccess) {
			throw std::runtime_error(std::string("PinnedBufferPool::Acquire: cudaHostAlloc(") +
			                         std::to_string(bytes) + ") failed: " + cudaGetErrorString(status));
		}
		if (slot.data != nullptr) {
			cudaFreeHost(slot.data);
		}
		slot.data = grown;
		slot.capacity = bytes;
	}
	slot.in_use = true;
	return PinnedBufferSlot(this, chosen, slot.data);
}

void PinnedBufferPool::Release(int index) {
	{
		std::lock_guard<std::mutex> guard(mutex_);
		slots_[static_cast<size_t>(index)].in_use = false;
	}
	cv_.notify_one();
}

} // namespace vector_gpu
