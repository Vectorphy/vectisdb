#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace vector_gpu {

//! NOTE ON PLATFORM: real RAPIDS librmm has no native Windows build (RAPIDS officially supports
//! Linux/WSL2 only) and this dev machine is native Windows with no WSL installed. Rather than block all
//! GPU work on that, this class implements a minimal drop-in pool allocator over the CUDA Runtime API
//! (one big cudaMalloc'd arena, first-fit free-list sub-allocation) that satisfies the same interface
//! RmmPool always had. This is the thing to delete and replace with real
//! rmm::mr::pool_memory_resource once building on Linux/WSL2 — see docs/KNOWN_ISSUES.md. It still
//! achieves the goal that matters for phase 1 (avoid a cudaMalloc/cudaFree round-trip, i.e. a driver
//! sync, for every intermediate buffer), just with a simpler allocation policy than RMM's.
//! ACTUAL PRODUCTION ROLE (verified by grep, session 12): this class is a VRAM **headroom oracle**, not
//! an allocator anyone calls. The only production entry points are EnsureInitialized() and
//! HasHeadroomFor(), both from TableChecker's routing decision; Allocate()/Free() have **no callers
//! outside tests**. cuda_engine/src/gpu_executor.cu deliberately uses its own cudaMalloc arena instead,
//! because intermediate result sizes are data-dependent and can exceed this pool.
//!
//! Since the P1 pass the arena is not even reserved up front (see EnsureInitialized) -- only the byte
//! budget is recorded, and the cudaMalloc is deferred to the first real Allocate() that never comes.
//!
//! Consequence, so nobody spends time on it: the first-fit allocator's fragmentation behaviour and its
//! lack of an eviction/pressure policy are NOT reachable in production today. Do not build an eviction
//! policy here without first giving the pool a real caller (e.g. making gpu_executor.cu allocate from it),
//! which is the change that would make any of that matter.
class RmmPool {
public:
	static RmmPool &Instance();

	//! Records a budget of `fraction_of_vram` (0.0-1.0) of currently-FREE device memory. No-op if already
	//! initialized. Called from TableChecker's routing path (and by GpuEngine before dispatch).
	//! Does NOT cudaMalloc the arena: reserving eagerly cost ~83 ms of startup in every process for memory
	//! nothing at execution time uses (see the P1 note in rmm_pool.cpp). The arena is allocated lazily by
	//! the first real Allocate(), which in production never happens.
	//! Throws std::runtime_error if the device cannot be queried (cudaMemGetInfo) or the computed budget
	//! rounds to 0 bytes.
	//! A fraction of FREE memory, not total -- and read AFTER the CUDA context exists, so the budget is
	//! what this process can still obtain, already net of the context and everything else resident.
	//!
	//! WHY 0.85 AND NOT 1.00, since the display runs on the integrated GPU and there is no framebuffer
	//! on this device to starve. Because the constraint turned out not to be the display. Measured in
	//! session 27: at 1.00 the budget admits a 36M-pair
	//! pairwise plan needing ~3296 MiB, which then peaks at 3654 MiB and runs **1.33x SLOWER than the
	//! CPU** (4946 ms vs 3710 ms). At 0.85 the same plan is declined and falls back, matching CPU instead
	//! of losing to it. Admitting a plan that needs essentially all of VRAM does not make it fast; it
	//! makes it page.
	//!
	//! This only became a real knob once the working-set estimate was fixed (session 26 measured it 2.35x
	//! low, so plans were admitted on an undercount and the fraction was decoration). Now that
	//! EstimateWorkingSet lands within 0.92-1.24 of real device peak, this number genuinely decides, and
	//! the right value is workload-dependent -- a plan with fewer carried columns tolerates a higher
	//! fraction than one gathering 7 DOUBLEs per pair. Measure before changing it.
	//!
	//! Note that being wrong here costs performance only. Over-committing does NOT raise
	//! CUDA_ERROR_OUT_OF_MEMORY on WDDM -- it pages to host RAM silently, ~12x on the worst case measured
	//! (session 26) -- so there is no crash and no wrong answer to protect against, just a slow query.
	//! Overridden at runtime by VECTOR_GPU_VRAM_FRACTION (see rmm_pool.cpp), matching the VECTOR_GPU_*
	//! convention used by the logger, STRICT_FP and NO_CHUNK. The right value is workload-dependent and
	//! wants measurement, not guessing.
	void EnsureInitialized(double fraction_of_vram = 0.85);

	//! Best-effort headroom check used by TableChecker::HasVramHeadroom before committing to GPU
	//! execution, so we reject oversized queries before the pool would fail mid-execution.
	//! Returns false (never throws) if the pool isn't initialized yet, if `estimated_bytes` is 0
	//! (nothing to allocate is not "has headroom", it's a meaningless request), or if `estimated_bytes`
	//! exceeds the whole arena — the last check also guards the alignment arithmetic against the
	//! session-7 KI-1 wraparound (AlignUp of the top 255 size_t values wraps to 0, which used to make
	//! TableChecker's saturated "clearly too big" estimate read as "fits"; see docs/KNOWN_ISSUES.md).
	bool HasHeadroomFor(size_t estimated_bytes) const;

	//! Sub-allocates `bytes` from the pool arena. Returns nullptr on failure (no free block big enough,
	//! or `bytes` larger than the whole arena — including the AlignUp wrap band near SIZE_MAX, see KI-2)
	//! — callers must check, this does not throw for the common "pool exhausted" case.
	//! Returns nullptr immediately (no allocation attempted) if bytes == 0.
	void *Allocate(size_t bytes);

	//! Returns a block previously obtained from Allocate(). The pool tracks each live allocation's real
	//! size internally, so `bytes` is accepted for interface compatibility but the *tracked* size is
	//! what gets returned to the free list — a caller passing the wrong size can no longer corrupt
	//! neighbouring live allocations (session-7 KI-4). Freeing nullptr, a pointer not owned by this
	//! pool, or a pointer that is not a currently-live allocation (double-free, interior pointer —
	//! session-7 KI-3) is a safe no-op rather than undefined behavior.
	void Free(void *ptr, size_t bytes);

	size_t TotalPoolBytes() const {
		std::lock_guard<std::mutex> guard(mutex_);
		return total_pool_bytes_;
	}
	size_t AllocatedBytes() const;

	//! Test-only: releases the arena so a fresh EnsureInitialized() can run with different parameters.
	//! Never call this from production code paths — GpuEngine assumes the pool lives for the process
	//! lifetime.
	void ResetForTesting();

	//! Test-only: initializes the pool to an exact byte size instead of a fraction of device VRAM, so
	//! tests can exercise OOM/fragmentation edge cases without allocating gigabytes of real VRAM. No-op
	//! if already initialized (same contract as EnsureInitialized) — call ResetForTesting() first if a
	//! test needs a fresh pool size.
	void EnsureInitializedExactBytesForTesting(size_t bytes);

private:
	//! Makes exactly one cheap CUDA call (cudaSetDevice(0)) and nothing else -- deliberately NOT where the
	//! real budget/arena setup happens; see EnsureInitialized for why that stays deferred. This call's
	//! purpose is narrower and load-bearing for a different reason: see rmm_pool.cpp's definition.
	RmmPool();
	~RmmPool();
	RmmPool(const RmmPool &) = delete;
	RmmPool &operator=(const RmmPool &) = delete;

	//! Shared init logic. Caller must already hold mutex_. Throws on cudaMalloc failure or a computed
	//! size of 0.
	void AllocateArenaLocked(size_t requested_bytes);

	//! cudaMalloc's the arena the first time it is genuinely needed. EnsureInitialized only records the
	//! byte budget (reserving it eagerly cost ~83 ms of startup for memory the executor never uses — see
	//! the comment there), so every path that hands out real device memory must call this first.
	//! Caller must already hold mutex_. Returns false if the reservation could not be satisfied.
	bool EnsureArenaLocked();

	struct FreeBlock {
		size_t offset;
		size_t size;
	};

	mutable std::mutex mutex_;
	bool initialized_ = false;
	void *arena_ = nullptr;
	size_t total_pool_bytes_ = 0;
	std::vector<FreeBlock> free_blocks_; // sorted by offset, coalesced on Free()
	//! offset -> aligned size of every live (allocated, not yet freed) block. This is what makes
	//! double-free and wrong-size-free detectable no-ops instead of silent free-list corruption
	//! (session-7 KI-3/KI-4): Free() consults this map, not the caller's byte count.
	std::unordered_map<size_t, size_t> live_blocks_;
};

} // namespace vector_gpu
