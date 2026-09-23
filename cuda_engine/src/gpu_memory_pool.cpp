#include "gpu_memory_pool.hpp"

#include <cuda_runtime.h>

#include "gpu_logger.hpp"

#include <stdexcept>
#include <string>

namespace vector_gpu {

GpuMemoryPoolSlot::GpuMemoryPoolSlot(GpuMemoryPoolSlot &&other) noexcept
    : pool_(other.pool_), index_(other.index_), data_(other.data_), capacity_(other.capacity_) {
	other.pool_ = nullptr;
	other.index_ = -1;
	other.data_ = nullptr;
	other.capacity_ = 0;
}

GpuMemoryPoolSlot &GpuMemoryPoolSlot::operator=(GpuMemoryPoolSlot &&other) noexcept {
	if (this != &other) {
		if (pool_ != nullptr) {
			pool_->Release(index_);
		}
		pool_ = other.pool_;
		index_ = other.index_;
		data_ = other.data_;
		capacity_ = other.capacity_;
		other.pool_ = nullptr;
		other.index_ = -1;
		other.data_ = nullptr;
		other.capacity_ = 0;
	}
	return *this;
}

GpuMemoryPoolSlot::~GpuMemoryPoolSlot() {
	if (pool_ != nullptr) {
		pool_->Release(index_);
	}
}

GpuMemoryPool &GpuMemoryPool::Instance() {
	static GpuMemoryPool instance; // function-local static: thread-safe, one-time construction
	return instance;
}

GpuMemoryPool::GpuMemoryPool() {
	// cudaSetDevice(0) matches this codebase's single-device convention (see PinnedBufferPool's
	// constructor and code_generator.cpp's GetSharedPrimaryContext) -- the Runtime API lazily creates its
	// own primary context on first call and shares it with the driver API's primary context, so no
	// explicit driver-API context handling is needed here either.
	//
	// Calling it HERE, before any cudaHostAlloc, is also what makes ~GpuMemoryPool's cudaFreeHost calls
	// below safe at process exit -- see the file header for the full mechanism. Short version: this is
	// the first real CUDA Runtime call this object makes, and making it from inside the constructor means
	// the runtime's own internal exit-time teardown gets registered before this object's destructor does,
	// so at exit this destructor runs first (LIFO), while the runtime is still alive.
	cudaSetDevice(0);
	for (size_t i = 0; i < kSlotCount; i++) {
		void *ptr = nullptr;
		auto status = cudaHostAlloc(&ptr, kSlotBytes, cudaHostAllocPortable);
		if (status != cudaSuccess) {
			// Free whatever slots THIS call already pinned before throwing, rather than leaking them --
			// the constructor not completing means the static never finishes initializing (ordinary magic-
			// statics semantics), so there is no partially-built instance left around to clean up later.
			for (size_t j = 0; j < i; j++) {
				cudaFreeHost(slots_[j].data);
				slots_[j].data = nullptr;
			}
			throw std::runtime_error("GpuMemoryPool: cudaHostAlloc(" + std::to_string(kSlotBytes) +
			                         ", cudaHostAllocPortable) failed for slot " + std::to_string(i) + ": " +
			                         cudaGetErrorString(status));
		}
		slots_[i].data = ptr;
	}
}

GpuMemoryPool::~GpuMemoryPool() {
	// Safe to call cudaFreeHost here -- unlike an earlier version of this destructor, which touched CUDA
	// only lazily (a separate EnsureInitialized() called from AcquireNext(), well after this object's own
	// construction had already completed) and SEGFAULTED at process exit calling this exact function, on
	// the first slot, reproduced repeatedly. See the constructor above and the file header for why moving
	// the first CUDA touch into the constructor is what fixes it, not a guess that this happens to work.
	for (auto &slot : slots_) {
		if (slot.data != nullptr) {
			cudaFreeHost(slot.data);
		}
	}
}

GpuMemoryPoolSlot GpuMemoryPool::AcquireNext() {
	std::unique_lock<std::mutex> lock(mutex_);
	auto index = next_index_;
	// Blocks here, not grows: with a FIXED ring, the only way to honor "hand me the next slot in
	// rotation" is to wait for whoever still holds it to finish -- see the header for why that backpressure
	// is the point rather than a limitation to work around.
	cv_.wait(lock, [&] { return !slots_[index].in_use; });
	slots_[index].in_use = true;
	next_index_ = (index + 1) % kSlotCount;
	
	size_t active_count = 0;
	for (const auto &s : slots_) if (s.in_use) active_count++;
	VGPU_LOG(LogLevel::DEBUG, "gpu_memory_pool", "\"active_slots\":" + std::to_string(active_count) + ",\"total_slots\":" + std::to_string(kSlotCount));

	return GpuMemoryPoolSlot(this, static_cast<int>(index), slots_[index].data, kSlotBytes);
}

void GpuMemoryPool::ResetCursorForTesting() {
	std::lock_guard<std::mutex> guard(mutex_);
	next_index_ = 0;
}

void GpuMemoryPool::Release(int index) {
	{
		std::lock_guard<std::mutex> guard(mutex_);
		slots_[static_cast<size_t>(index)].in_use = false;
		size_t active_count = 0;
		for (const auto &s : slots_) if (s.in_use) active_count++;
		VGPU_LOG(LogLevel::DEBUG, "gpu_memory_pool", "\"active_slots\":" + std::to_string(active_count) + ",\"total_slots\":" + std::to_string(kSlotCount));
	}
	cv_.notify_all();
}

} // namespace vector_gpu
