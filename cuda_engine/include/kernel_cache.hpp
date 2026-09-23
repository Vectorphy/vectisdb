#pragma once

#include <list>
#include <mutex>
#include <string>
#include <unordered_map>

namespace vector_gpu {

//! Opaque handle to a compiled, loaded CUDA function (in practice a heap CompiledKernel* holding a
//! CUmodule + CUfunction — see code_generator.cpp).
using CompiledKernelHandle = void *;

//! SHA-256-keyed LRU cache of compiled kernels, so recurring queries skip NVRTC compilation entirely
//! (per docs/ARCHITECTURE.md — JIT compilation can cost tens to hundreds of milliseconds, which is fatal
//! to interactive OLAP latency if paid on every execution).
//!
//! Ownership (session-7 fix): once a HandleDeleter is registered (CodeGenerator does this in its
//! constructor), the cache OWNS every inserted handle — eviction, same-key replacement, and cache
//! destruction all invoke the deleter (which unloads the CUmodule and frees the struct), closing the
//! per-eviction VRAM leak. Consequence for callers: a handle obtained from Get()/CompileOrFetch is a
//! BORROW, valid only until the next Insert() into this cache — use it immediately, don't store it.
//! With no deleter registered (standalone unit tests using dummy pointers), all of that is a no-op and
//! the old non-owning behavior holds.
//!
//! Thread safety (session-7 fix): Get/Insert are internally locked (Get mutates the LRU list, so
//! concurrent readers were never safe). Non-copyable: entries_ holds iterators into lru_order_, so a
//! copy's map would point into the source's list and corrupt it on first use.
class KernelCache {
public:
	//! Frees one cache-owned handle. Registered by CodeGenerator (the only code that knows the concrete
	//! type behind CompiledKernelHandle); a plain function pointer so registration is idempotent.
	using HandleDeleter = void (*)(CompiledKernelHandle);

	explicit KernelCache(size_t max_entries = 256) : max_entries_(max_entries) {
	}
	~KernelCache();

	KernelCache(const KernelCache &) = delete;
	KernelCache &operator=(const KernelCache &) = delete;

	//! Registers the deleter invoked for evicted/replaced/destroyed entries. Idempotent; must be the
	//! same function on every call (throws std::logic_error on an attempt to change it once set).
	void SetHandleDeleter(HandleDeleter deleter);

	//! Returns the cached kernel for `source_hash`, or nullptr if not present (cache miss — caller must
	//! compile via NVRTC and call Insert()). The returned handle is a borrow — see class comment.
	CompiledKernelHandle Get(const std::string &source_hash);

	//! Inserts a newly compiled kernel, evicting the least-recently-used entry if over capacity.
	//! Evicted/replaced handles are freed via the registered deleter (if any). Inserting the same
	//! handle again under the same key just refreshes recency.
	void Insert(const std::string &source_hash, CompiledKernelHandle handle);

	//! Computes the SHA-256 hex digest of `cuda_source`, used as the cache key.
	static std::string HashSource(const std::string &cuda_source);

private:
	void DeleteHandleLocked(CompiledKernelHandle handle);

	std::mutex mutex_;
	HandleDeleter deleter_ = nullptr;
	size_t max_entries_;
	std::list<std::string> lru_order_; // front = most recently used
	std::unordered_map<std::string, std::pair<CompiledKernelHandle, std::list<std::string>::iterator>> entries_;
};

} // namespace vector_gpu
