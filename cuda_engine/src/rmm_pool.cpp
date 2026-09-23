#include "rmm_pool.hpp"
#include "gpu_logger.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace vector_gpu {

namespace {
//! GPU allocations are aligned to 256 bytes so sub-allocated buffers stay coalescing-friendly for any
//! kernel that reads them with a vectorized (float4/int4-style) access pattern.
constexpr size_t kAlignment = 256;

//! NOTE (session-7 KI-1/KI-2): for n in [SIZE_MAX-254, SIZE_MAX] the `n + kAlignment - 1` addition
//! wraps and this returns 0 — which the old code then treated as "fits anywhere" / "allocate zero
//! bytes but hand out a pointer". Every caller must therefore reject n > total_pool_bytes_ (or an
//! equivalent bound far below the wrap band) BEFORE calling AlignUp. AllocateArenaLocked additionally
//! throws on a computed size of 0, so the wrap can't silently produce an empty arena either.
size_t AlignUp(size_t n) {
	return (n + kAlignment - 1) & ~(kAlignment - 1);
}
} // namespace

RmmPool &RmmPool::Instance() {
	static RmmPool instance;
	return instance;
}

RmmPool::RmmPool() {
	// A single, cheap cudaSetDevice(0) -- NOT the real cudaMemGetInfo/cudaMalloc setup, which stays in
	// EnsureInitialized exactly as deliberately deferred as before (see its own comment: eager arena
	// reservation cost ~83ms of startup for memory a routing decision that declines the query never
	// uses). This call's job is narrower: it is a real CUDA Runtime call made from INSIDE this
	// constructor, which is what keeps ~RmmPool's cudaFree below from segfaulting at process exit.
	//
	// Verified directly, not assumed (gpu_memory_pool.hpp/.cpp, session 38): a trivial `= default`
	// constructor here, with every CUDA touch deferred to EnsureInitialized as an ordinary later call,
	// reproduced the SAME crash pattern found and fixed in GpuMemoryPool -- confirmed with an isolated
	// repro of this exact class (EnsureInitialized(0.1) -> Allocate -> Free -> scope exit -> singleton
	// teardown), 0xC0000005 inside this destructor's cudaFree. Adding just this one line, leaving
	// EnsureInitialized untouched, fixed it -- confirmed with the same repro, clean exit.
	//
	// The mechanism: a function-local static's destructor is registered for exit-time teardown the
	// moment its constructor completes; the CUDA Runtime registers its own internal exit-time teardown
	// the moment its first real API call happens; exit-time cleanup runs in reverse registration order.
	// Touch CUDA only lazily, well after this constructor has already returned, and cudart's
	// registration lands AFTER this destructor's -- so at exit cudart tears itself down FIRST, and
	// cudaFree then runs against an already-dead runtime. Make one real call from inside the constructor
	// instead, and cudart's registration happens WHILE this constructor is still running, i.e. BEFORE
	// this object's own destructor gets registered -- so at exit this destructor runs first, while the
	// runtime is still alive. See gpu_memory_pool.hpp's file header for the full writeup and
	// pinned_buffer_pool.cpp's constructor for the same pattern (already present there, now documented as
	// deliberate rather than incidental).
	//
	// DO NOT REMOVE THIS CALL, even though EnsureInitialized never reads anything it sets: removing it
	// reopens the exit crash above.
	cudaSetDevice(0);
}

RmmPool::~RmmPool() {
	// Safe here specifically because the constructor above already made a real CUDA call -- see its
	// comment. "Best-effort" was the old framing; ORDER, not effort, is what makes this not crash.
	if (arena_) {
		cudaFree(arena_);
	}
}

void RmmPool::AllocateArenaLocked(size_t requested_bytes) {
	auto aligned = AlignUp(requested_bytes);
	if (aligned == 0) {
		throw std::runtime_error("RmmPool: computed pool size is 0 bytes");
	}
	auto status = cudaMalloc(&arena_, aligned);
	if (status != cudaSuccess) {
		throw std::runtime_error(std::string("RmmPool: cudaMalloc(") + std::to_string(aligned) +
		                         ") failed: " + cudaGetErrorString(status));
	}
	total_pool_bytes_ = aligned;
	free_blocks_.clear();
	free_blocks_.push_back({0, aligned});
	live_blocks_.clear();
	initialized_ = true;
}

void RmmPool::EnsureInitialized(double fraction_of_vram) {
	std::lock_guard<std::mutex> guard(mutex_);
	if (initialized_) {
		return;
	}
	// VECTOR_GPU_VRAM_FRACTION overrides the default so the budget can be swept per workload without a
	// rebuild -- the value that maximises throughput depends on how much memory traffic the admitted plan
	// does, not on the device alone (see the header). Parsed once; a malformed or out-of-range value is
	// ignored rather than throwing, because this runs inside a routing decision.
	if (const char *env = std::getenv("VECTOR_GPU_VRAM_FRACTION")) {
		char *end = nullptr;
		auto parsed = std::strtod(env, &end);
		if (end != env && parsed > 0.0 && parsed <= 1.0) {
			fraction_of_vram = parsed;
		}
	}
	if (fraction_of_vram <= 0.0 || fraction_of_vram > 1.0) {
		throw std::invalid_argument("RmmPool::EnsureInitialized: fraction_of_vram must be in (0, 1]");
	}

	size_t free_bytes = 0, total_bytes = 0;
	auto status = cudaMemGetInfo(&free_bytes, &total_bytes);
	if (status != cudaSuccess) {
		throw std::runtime_error(std::string("RmmPool: cudaMemGetInfo failed: ") + cudaGetErrorString(status));
	}

	// Reserve against currently-free memory, not total device memory: total_bytes includes whatever the
	// driver/other processes already hold, and asking for a fraction of *that* can request more than is
	// actually available (e.g. another process already holding VRAM on a shared/dev machine).
	size_t requested = static_cast<size_t>(static_cast<double>(free_bytes) * fraction_of_vram);

	// P1: record the budget WITHOUT cudaMalloc'ing it. Reserving here cost ~83 ms of startup in every
	// process — measured, it was the entire `optimize` phase in `duckdb_gpu --gpu-trace` — for memory
	// that nothing at execution time actually uses: gpu_executor.cu deliberately allocates its own
	// cudaMalloc arena because intermediate sizes are data-dependent and can exceed this pool. The
	// pool's only production role is answering TableChecker's "would N bytes fit?", which needs the
	// budget, not the allocation. The arena is cudaMalloc'd lazily by EnsureArenaLocked() the first time
	// anything really calls Allocate(); reserving it also used to squeeze the executor's genuine
	// allocations into whatever VRAM was left over.
	auto aligned = AlignUp(requested);
	if (aligned == 0) {
		throw std::runtime_error("RmmPool: computed pool size is 0 bytes");
	}
	total_pool_bytes_ = aligned;
	free_blocks_.clear();
	free_blocks_.push_back({0, aligned});
	live_blocks_.clear();
	initialized_ = true;

	// This budget is a ONE-TIME snapshot: it is locked in on whichever query first touches the routing
	// VRAM check and never re-measured for the rest of the process (RmmPool is a process-wide
	// singleton). Two runs of the same query in fresh processes can route differently purely because
	// ambient VRAM use differed at each process's cold start. Log it once so "why did this route
	// differently than last time" is answerable from the trace rather than being a silent constant.
	// See docs/KNOWN_ISSUES.md (session 29). A periodic re-measure is the fuller fix and still open.
	VGPU_LOG(LogLevel::INFO, "vram_budget",
	         "\"locked_mib\":" + std::to_string(aligned / (1024 * 1024)) + ",\"free_mib\":" +
	             std::to_string(free_bytes / (1024 * 1024)) + ",\"total_mib\":" +
	             std::to_string(total_bytes / (1024 * 1024)) + ",\"fraction\":" +
	             std::to_string(fraction_of_vram) + ",\"one_time_snapshot\":true");
}

bool RmmPool::EnsureArenaLocked() {
	if (arena_ != nullptr) {
		return true;
	}
	if (total_pool_bytes_ == 0) {
		return false;
	}
	// Deferred from EnsureInitialized (see there). Failure is reported as "no memory" rather than an
	// exception so Allocate keeps its documented "returns nullptr on failure" contract.
	return cudaMalloc(&arena_, total_pool_bytes_) == cudaSuccess && arena_ != nullptr;
}

void RmmPool::EnsureInitializedExactBytesForTesting(size_t bytes) {
	std::lock_guard<std::mutex> guard(mutex_);
	if (initialized_) {
		return;
	}
	AllocateArenaLocked(bytes);
}

bool RmmPool::HasHeadroomFor(size_t estimated_bytes) const {
	std::lock_guard<std::mutex> guard(mutex_);
	if (!initialized_ || estimated_bytes == 0) {
		return false;
	}
	// KI-1 fix: anything larger than the whole arena can never fit, and rejecting it here — before
	// AlignUp — also keeps the wrap band near SIZE_MAX (where AlignUp returns 0) from turning
	// "largest request representable" into "fits in the first free block". This is exactly the value
	// TableChecker's SaturatingMultiply produces for an overflowing cardinality estimate.
	if (estimated_bytes > total_pool_bytes_) {
		return false;
	}
	auto aligned = AlignUp(estimated_bytes);
	for (auto &block : free_blocks_) {
		if (block.size >= aligned) {
			return true;
		}
	}
	return false;
}

void *RmmPool::Allocate(size_t bytes) {
	if (bytes == 0) {
		return nullptr;
	}
	std::lock_guard<std::mutex> guard(mutex_);
	if (!initialized_) {
		return nullptr;
	}
	// The arena is reserved lazily, so the first real allocation is what pays for it (see
	// EnsureInitialized). Headroom checks never get here, which is the whole point.
	if (!EnsureArenaLocked()) {
		return nullptr;
	}
	// KI-2 fix: reject anything larger than the whole arena before AlignUp — in the wrap band near
	// SIZE_MAX, AlignUp returns 0, and the old code then "found" a fit, consumed zero bytes from the
	// free list, and handed the same arena pointer to every subsequent caller (aliasing).
	if (bytes > total_pool_bytes_) {
		return nullptr;
	}
	auto aligned = AlignUp(bytes);

	// First-fit: scan for the first free block large enough. Simpler and less memory-efficient than
	// RMM's best-fit coalescing allocator, but correct, and sufficient for phase-1 validation on a single
	// GPU with a handful of concurrent intermediate buffers.
	for (size_t i = 0; i < free_blocks_.size(); i++) {
		auto &block = free_blocks_[i];
		if (block.size < aligned) {
			continue;
		}
		size_t offset = block.offset;
		if (block.size == aligned) {
			free_blocks_.erase(free_blocks_.begin() + static_cast<long>(i));
		} else {
			block.offset += aligned;
			block.size -= aligned;
		}
		live_blocks_[offset] = aligned; // KI-3/KI-4: remember the real size while the block is live
		return static_cast<uint8_t *>(arena_) + offset;
	}
	return nullptr; // pool exhausted / too fragmented for this request
}

void RmmPool::Free(void *ptr, size_t bytes) {
	if (!ptr || bytes == 0) {
		return;
	}
	std::lock_guard<std::mutex> guard(mutex_);
	if (!initialized_ || !arena_) {
		return; // defensive: freeing into a pool that no longer exists is a no-op, not a crash
	}
	auto byte_ptr = static_cast<uint8_t *>(ptr);
	auto arena_start = static_cast<uint8_t *>(arena_);
	if (byte_ptr < arena_start || byte_ptr >= arena_start + total_pool_bytes_) {
		return; // not a pointer this pool owns — ignore rather than corrupt free_blocks_
	}
	size_t offset = static_cast<size_t>(byte_ptr - arena_start);

	// KI-3/KI-4 fix: only a currently-live allocation may be freed, and the size returned to the free
	// list is the tracked one, not the caller's. This turns a double-free or interior-pointer free into
	// a no-op (previously: duplicate/overlapping free blocks, AllocatedBytes underflow, two live
	// allocations aliasing one address) and makes a wrong `bytes` harmless (previously: the free block
	// could swallow the live neighbouring allocation).
	auto live = live_blocks_.find(offset);
	if (live == live_blocks_.end()) {
		return; // double-free or pointer not at a live allocation start — no-op
	}
	auto aligned = live->second;
	live_blocks_.erase(live);

	auto insert_pos = std::lower_bound(free_blocks_.begin(), free_blocks_.end(), offset,
	                                   [](const FreeBlock &b, size_t off) { return b.offset < off; });
	auto it = free_blocks_.insert(insert_pos, {offset, aligned});

	// Coalesce with the next block first (iterator stays valid), then with the previous block.
	auto next = it + 1;
	if (next != free_blocks_.end() && it->offset + it->size == next->offset) {
		it->size += next->size;
		free_blocks_.erase(next);
	}
	if (it != free_blocks_.begin()) {
		auto prev = it - 1;
		if (prev->offset + prev->size == it->offset) {
			prev->size += it->size;
			free_blocks_.erase(it);
		}
	}
}

size_t RmmPool::AllocatedBytes() const {
	std::lock_guard<std::mutex> guard(mutex_);
	size_t free_total = 0;
	for (auto &block : free_blocks_) {
		free_total += block.size;
	}
	return total_pool_bytes_ - free_total;
}

void RmmPool::ResetForTesting() {
	std::lock_guard<std::mutex> guard(mutex_);
	if (arena_) {
		cudaFree(arena_);
		arena_ = nullptr;
	}
	total_pool_bytes_ = 0;
	free_blocks_.clear();
	live_blocks_.clear();
	initialized_ = false;
}

} // namespace vector_gpu
