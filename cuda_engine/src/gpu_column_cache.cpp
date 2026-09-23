#include "gpu_column_cache.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <vector>

namespace vector_gpu {

namespace {
//! Frees a chunk's device buffers. Never throws (matches cudaFree's own no-throw contract) -- called
//! from eviction and the destructor, neither of which has anywhere useful to report a failure to.
void FreeChunkDeviceMemory(const GpuColumnChunk &chunk) {
	if (chunk.device_data != nullptr) {
		cudaFree(chunk.device_data);
	}
	if (chunk.device_validity != nullptr) {
		cudaFree(chunk.device_validity);
	}
}

void FreeChunksDeviceMemory(const std::vector<GpuColumnChunk> &chunks) {
	for (auto &chunk : chunks) {
		FreeChunkDeviceMemory(chunk);
	}
}
} // namespace

GpuColumnCache &GpuColumnCache::Instance() {
	static GpuColumnCache instance;
	return instance;
}

GpuColumnCache::GpuColumnCache() {
	// One cheap cudaSetDevice(0) -- not the real budget/staging setup, which stays in EnsureInitialized,
	// deferred exactly as deliberately as before. This call's job is narrower: made from INSIDE this
	// constructor, it is what keeps ~GpuColumnCache's cudaFree calls below from segfaulting at process
	// exit, by the same verified mechanism documented in full in gpu_memory_pool.hpp's file header and
	// applied identically to RmmPool (rmm_pool.cpp) -- a trivial `= default` constructor here, with every
	// CUDA touch deferred to EnsureInitialized as an ordinary later call, reproduces the same 0xC0000005
	// crash pattern found in GpuMemoryPool and confirmed again in RmmPool. Do not remove this call.
	cudaSetDevice(0);
}

GpuColumnCache::~GpuColumnCache() {
	// No lock: by the time the destructor runs (process teardown), concurrent access is already a caller
	// bug, same convention as KernelCache's destructor. Safe to call cudaFree here specifically because
	// the constructor above already made a real CUDA call -- see its comment. "Best-effort" was the old
	// framing here; ORDER, not effort, is what makes this not crash (verified: gpu_memory_pool.hpp).
	for (auto &kv : entries_) {
		FreeChunksDeviceMemory(kv.second.first.chunks);
	}
	for (auto &table : staging_) {
		for (auto &column : table.second) {
			FreeChunksDeviceMemory(column.second.chunks);
		}
	}
}

void GpuColumnCache::EnsureInitialized(double fraction_of_vram) {
	std::lock_guard<std::mutex> guard(mutex_);
	if (initialized_) {
		return;
	}
	// VECTOR_GPU_CACHE_FRACTION overrides the default, matching this project's VECTOR_GPU_* convention
	// (VECTOR_GPU_VRAM_FRACTION on RmmPool, VECTOR_GPU_MIN_AMPLIFICATION on TableChecker, ...). Parsed
	// once; a malformed or out-of-range value is ignored rather than thrown, since this can run from a
	// routing/execution path that must not fail a query over a bad env var.
	if (const char *env = std::getenv("VECTOR_GPU_CACHE_FRACTION")) {
		char *end = nullptr;
		auto parsed = std::strtod(env, &end);
		if (end != env && parsed > 0.0 && parsed <= 1.0) {
			fraction_of_vram = parsed;
		}
	}
	if (fraction_of_vram <= 0.0 || fraction_of_vram > 1.0) {
		fraction_of_vram = 0.5;
	}

	// VECTOR_GPU_CACHE_MB pins the budget to an absolute size in MiB and wins over the fraction. A
	// fraction of "whatever happened to be free at first use" cannot express a budget reproducibly: the
	// same fraction is a different number of bytes on every run, depending on what else is on the card.
	// Benchmarks and capacity planning both need to state an actual size.
	if (const char *env = std::getenv("VECTOR_GPU_CACHE_MB")) {
		char *end = nullptr;
		auto parsed = std::strtoull(env, &end, 10);
		if (end != env && parsed > 0) {
			budget_bytes_ = static_cast<uint64_t>(parsed) * 1024ull * 1024ull;
			initialized_ = true;
			return;
		}
	}

	size_t free_bytes = 0, total_bytes = 0;
	auto status = cudaMemGetInfo(&free_bytes, &total_bytes);
	if (status != cudaSuccess) {
		// Same posture as RmmPool::HasVramHeadroom on a query failure: no GPU visible, so there is
		// nothing to cache into -- budget 0 means BeginStaging below never opens a slot, which is a
		// correctness-neutral no-op (falls back to "always re-scan"), not an error.
		budget_bytes_ = 0;
		initialized_ = true;
		return;
	}
	budget_bytes_ = static_cast<uint64_t>(static_cast<double>(free_bytes) * fraction_of_vram);
	initialized_ = true;
}

void GpuColumnCache::TouchLocked(const SlotKey &slot) {
	auto it = entries_.find(slot);
	if (it == entries_.end()) {
		return;
	}
	// Move to front (most recently used) -- same recency convention as KernelCache::Get.
	lru_order_.erase(it->second.second);
	lru_order_.push_front(slot);
	it->second.second = lru_order_.begin();
}

void GpuColumnCache::EvictLocked(const SlotKey &slot) {
	auto it = entries_.find(slot);
	if (it == entries_.end()) {
		return;
	}
	FreeChunksDeviceMemory(it->second.first.chunks);
	committed_bytes_ -= it->second.first.total_bytes;
	lru_order_.erase(it->second.second);
	entries_.erase(it);
}

void GpuColumnCache::EvictTableLocked(const GpuTableRef &table) {
	// Collect matching slots first rather than erasing while iterating entries_ (EvictLocked mutates
	// entries_/lru_order_ itself). Compares the table identity exactly -- the earlier string-prefix form
	// could both miss an entry it had to drop and drop one it should not have.
	std::vector<SlotKey> to_evict;
	for (auto &kv : entries_) {
		if (kv.first.table == table) {
			to_evict.push_back(kv.first);
		}
	}
	for (auto &slot : to_evict) {
		EvictLocked(slot);
	}
}

void GpuColumnCache::FailStagedColumnLocked(StagedColumn &staged) {
	FreeChunksDeviceMemory(staged.chunks);
	staged.chunks.clear();
	staged_bytes_ -= staged.total_bytes;
	staged.total_bytes = 0;
	staged.row_count = 0;
	staged.failed = true;
}

void GpuColumnCache::AbortStagingLocked(const GpuTableRef &table) {
	auto it = staging_.find(table);
	if (it == staging_.end()) {
		return;
	}
	for (auto &column : it->second) {
		FreeChunksDeviceMemory(column.second.chunks);
		staged_bytes_ -= column.second.total_bytes;
	}
	staging_.erase(it);
}

void GpuColumnCache::BeginStaging(const GpuTableRef &table) {
	EnsureInitialized(); // takes and releases mutex_ itself; must not be called under the lock below
	std::lock_guard<std::mutex> guard(mutex_);
	AbortStagingLocked(table);
	// Existing committed columns of this table are preserved so that frequently queried column
	// subsets remain resident even when a table exceeds total VRAM budget. If an existing column
	// is re-staged, CommitStaging will replace that specific column entry. If the table is modified,
	// InvalidateTable drops all columns.
	if (budget_bytes_ == 0) {
		return; // no usable budget -- leave no slot open, so every StageChunk simply declines
	}
	staging_[table]; // default-construct the (empty) column map
}

bool GpuColumnCache::StageChunk(const GpuTableRef &table, const std::string &column_name,
                                const GpuColumnChunk &chunk) {
	std::lock_guard<std::mutex> guard(mutex_);
	auto table_it = staging_.find(table);
	if (table_it == staging_.end()) {
		return false; // no scan is staging this table (whole-input path, zero budget, already aborted)
	}
	auto &staged = table_it->second[column_name];
	if (staged.failed) {
		return false; // an earlier chunk of this column was refused; publishing it now would leave a hole
	}
	if (!staged.type_known) {
		staged.type = chunk.type;
		staged.type_known = true;
	} else if (staged.type != chunk.type) {
		// Two chunks of one column disagreeing on type means the producer changed its mind mid-scan --
		// not something to paper over by picking one, since every reader indexes these bytes as that type.
		FailStagedColumnLocked(staged);
		return false;
	}

	auto needed = chunk.TotalBytes();
	// Evict least-recently-used COMMITTED entries until this chunk fits alongside everything else that is
	// resident. Staged bytes are not evictable -- dropping half a column in progress only guarantees it
	// never publishes, which costs the memory anyway and buys nothing.
	while (committed_bytes_ + staged_bytes_ + needed > budget_bytes_ && !lru_order_.empty()) {
		EvictLocked(lru_order_.back());
	}
	if (committed_bytes_ + staged_bytes_ + needed > budget_bytes_) {
		// This column will never fit. Fail just this column (freeing what it already holds) rather than
		// the whole table: a partial column SET is still a usable cache, and the caller falls back to
		// re-scanning this one column, which is correctness-neutral.
		FailStagedColumnLocked(staged);
		return false;
	}

	staged.chunks.push_back(chunk);
	staged.row_count += chunk.row_count;
	staged.total_bytes += needed;
	staged_bytes_ += needed;
	return true;
}

void GpuColumnCache::CommitStaging(const GpuTableRef &table, uint64_t scanned_rows) {
	std::lock_guard<std::mutex> guard(mutex_);
	auto table_it = staging_.find(table);
	if (table_it == staging_.end()) {
		return;
	}
	auto generation = generation_[table];
	for (auto &kv : table_it->second) {
		auto &staged = kv.second;
		// THE completeness gate. A column publishes only if its chunks account for every row the scan
		// produced -- anything else (a refused chunk, a scan that ended early, a producer that skipped a
		// batch) is dropped, because a cached column that is missing rows is a wrong answer, not a
		// smaller cache. scanned_rows == 0 covers the empty-table case: nothing worth publishing.
		if (staged.failed || staged.chunks.empty() || scanned_rows == 0 || staged.row_count != scanned_rows) {
			FreeChunksDeviceMemory(staged.chunks);
			staged_bytes_ -= staged.total_bytes;
			continue;
		}
		SlotKey slot {table, kv.first};
		EvictLocked(slot); // no-op unless something raced in after BeginStaging cleared the table

		CachedColumn column;
		column.chunks = std::move(staged.chunks);
		column.row_count = staged.row_count;
		column.total_bytes = staged.total_bytes;
		column.generation = generation;

		staged_bytes_ -= staged.total_bytes;
		committed_bytes_ += column.total_bytes;
		lru_order_.push_front(slot);
		entries_[slot] = {std::move(column), lru_order_.begin()};
	}
	staging_.erase(table_it);
}

void GpuColumnCache::AbortStaging(const GpuTableRef &table) {
	std::lock_guard<std::mutex> guard(mutex_);
	AbortStagingLocked(table);
}

std::vector<uint64_t> GpuColumnCache::ReplayableChunkRows(const GpuTableRef &table,
                                                          const std::vector<std::string> &column_names) {
	std::lock_guard<std::mutex> guard(mutex_);
	std::vector<uint64_t> chunk_rows;
	if (column_names.empty()) {
		return chunk_rows; // a SCAN producing no columns has nothing to serve from the cache
	}
	for (auto &column_name : column_names) {
		auto it = entries_.find(SlotKey {table, column_name});
		if (it == entries_.end()) {
			return {}; // miss on any one column: the caller must re-scan the table anyway
		}
		auto &chunks = it->second.first.chunks;
		if (chunks.empty()) {
			return {};
		}
		if (chunk_rows.empty()) {
			chunk_rows.reserve(chunks.size());
			for (auto &chunk : chunks) {
				chunk_rows.push_back(chunk.row_count);
			}
			continue;
		}
		// Every column must be chunked identically, or chunk i of column A and chunk i of column B are
		// different row ranges and executing them together silently mixes rows across the table.
		if (chunks.size() != chunk_rows.size()) {
			return {};
		}
		for (size_t i = 0; i < chunks.size(); i++) {
			if (chunks[i].row_count != chunk_rows[i]) {
				return {};
			}
		}
	}
	// Only touch recency once the whole set is confirmed usable -- a partial match is a miss, and a miss
	// should not promote the columns that happened to be there.
	for (auto &column_name : column_names) {
		TouchLocked(SlotKey {table, column_name});
	}
	return chunk_rows;
}

bool GpuColumnCache::TryGetChunk(const GpuTableRef &table, const std::string &column_name,
                                 uint64_t chunk_index, GpuColumnChunk &out_chunk) {
	std::lock_guard<std::mutex> guard(mutex_);
	SlotKey slot {table, column_name};
	auto it = entries_.find(slot);
	if (it == entries_.end()) {
		return false; // miss -- see file header for why presence alone means "current"
	}
	auto &chunks = it->second.first.chunks;
	if (chunk_index >= chunks.size()) {
		return false;
	}
	out_chunk = chunks[static_cast<size_t>(chunk_index)];
	TouchLocked(slot);
	return true;
}

void GpuColumnCache::InvalidateTable(const GpuTableRef &table) {
	std::lock_guard<std::mutex> guard(mutex_);
	AbortStagingLocked(table);
	EvictTableLocked(table);
	generation_[table]++; // diagnostic only; also seeds the map for a table never cached before
}

void GpuColumnCache::InvalidateAll() {
	std::lock_guard<std::mutex> guard(mutex_);
	std::vector<GpuTableRef> staged_tables;
	staged_tables.reserve(staging_.size());
	for (auto &kv : staging_) {
		staged_tables.push_back(kv.first);
	}
	for (auto &table : staged_tables) {
		AbortStagingLocked(table);
	}

	std::vector<SlotKey> all_slots;
	all_slots.reserve(entries_.size());
	for (auto &kv : entries_) {
		all_slots.push_back(kv.first);
	}
	for (auto &slot : all_slots) {
		EvictLocked(slot);
	}
	for (auto &kv : generation_) {
		kv.second++;
	}
}

uint64_t GpuColumnCache::CurrentBytes() const {
	std::lock_guard<std::mutex> guard(mutex_);
	return committed_bytes_ + staged_bytes_;
}

uint64_t GpuColumnCache::TableGeneration(const GpuTableRef &table) const {
	std::lock_guard<std::mutex> guard(mutex_);
	auto it = generation_.find(table);
	return it == generation_.end() ? 0 : it->second;
}

} // namespace vector_gpu
