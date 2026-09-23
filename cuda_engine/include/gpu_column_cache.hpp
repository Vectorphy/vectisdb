#pragma once

#include <cstdint>
#include <functional>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "gpu_engine.hpp" // GpuValueType

// Persistent, device-resident cache of scanned columns, so a repeat query against an UNCHANGED table
// skips the scan+H2D upload entirely (see docs/GPU_RESIDENT_CACHE_DESIGN.md, branch gpu-resident-cache).
//
// WHAT A CACHE ENTRY IS: a COMPLETE column, stored as the ORDERED SEQUENCE OF CHUNKS the streaming
// scanner produced it in -- not one contiguous device buffer, and never a partial column.
//
//   Session 31 shipped a version keyed only by (table, column) with ONE buffer per slot, populated from
//   StreamingTableScanner's per-chunk ExecuteScan calls. Each chunk's Put() overwrote the previous one,
//   so a multi-chunk table ended up cached as just its LAST chunk -- and the next query served that as
//   the whole table. Silently wrong results, no error: a 20M-row table came back as 1,191,168 rows.
//   Staging (BeginStaging/StageChunk/CommitStaging) exists precisely so that state is unreachable: a
//   column is invisible to readers until every one of its chunks has landed AND the row counts add up.
//
//   Chunks rather than one contiguous buffer, on purpose, two reasons:
//     1. No total row count is needed up front. DataTable::GetTotalRows() counts appended rows including
//        tombstoned ones and excludes this transaction's uncommitted appends, so it is neither an exact
//        size nor a reliable upper bound -- sizing a buffer from it means either overflow or a growth
//        path. Accumulating the chunks the scanner actually produced needs neither.
//     2. A cache HIT then replays chunk-by-chunk, with exactly the chunk sizes the populating scan used
//        (<= PhysicalGpuExecute::MAX_CHUNK_ROWS). That is the footprint TableChecker::HasVramHeadroom-
//        ForPlan sized the plan against; serving a cache hit as one whole-table chunk instead would run
//        a 20M-row plan inside a routing decision made for 2M rows.
//
// CORRECTNESS MODEL (verified against real DuckDB source + an empirical test, not assumed):
//   Invalidation is PUSH-based, driven from gpu_offload_extension.cpp's already-registered
//   OptimizerExtension callback -- DuckDB calls it for every statement, including INSERT/UPDATE/DELETE/
//   DDL, strictly before that statement executes or commits (verified: Optimizer::Optimize() has no
//   statement-type gate, and always runs before PhysicalPlanGenerator::Plan()). So a write to table T is
//   guaranteed to call InvalidateTable("T") -- removing every cached entry for T from the map below --
//   before any LATER query could read a stale one. That makes simple MAP PRESENCE the correctness
//   signal: a committed entry IS the current data, no separate version comparison needed.
//
//   An earlier design (see the design doc) tried DuckTransactionManager::GetLastCommit() as a pull-based
//   "has this changed" signal instead. Empirically disproven, not just theoretically risky: it changed
//   between two back-to-back READ-ONLY queries with zero writes in between (confirmed live via
//   duckdb_gpu.exe --serve) -- it counts transactions, not writes, and would have invalidated on every
//   single query. Kept as a documented dead end so it isn't tried again.
//
//   Known, accepted gap: DuckDB's low-level InternalAppender/DataTable::LocalAppend path writes without
//   ever constructing a LogicalOperator, so it's invisible to the optimizer hook this cache relies on. In
//   this DuckDB tree that path is reachable only via dbgen()/dsdgen() (TPC-H/TPC-DS synthetic data
//   generation), not any SQL write. Not closed here -- doing so would mean hooking DataTable itself,
//   which starts to blur into modifying how core is used.
//
// Mirrors KernelCache's LRU shape (list + map, front = most recently used) rather than inventing a new
// pattern -- the difference is eviction here is BYTE-BUDGET-bounded, not count-bounded, since columns
// vary hugely in size and a byte cap is what actually protects the VRAM budget TableChecker reasons
// about. Only COMMITTED entries are evictable; staging in progress is not, since evicting half a column
// mid-scan would just guarantee it never publishes.
//
// Deliberately a SEPARATE allocator from RmmPool and DeviceArena, not layered on either:
//   - RmmPool is a headroom ORACLE in production today (Allocate()/Free() have no real caller) -- giving
//     it a real, PERMANENT caller here would make its arena hold memory for the process's lifetime,
//     silently shrinking what DeviceArena's own independent cudaMalloc calls can actually get, in a way
//     HasVramHeadroomForPlan's existing accounting does not model. See the design doc for the reasoning.
//   - DeviceArena is explicitly per-query and freed when the plan finishes; a cache needs the opposite
//     lifetime.
// So this class owns its own cudaMalloc'd budget, sized once (mirrors RmmPool::EnsureInitialized's own
// "measure free VRAM once, at first use" pattern), and TableChecker::HasVramHeadroomForPlan is updated
// (see table_checker.cpp) to add CurrentBytes() to what a new plan demands -- otherwise routing could
// approve a plan whose transient working set, added to what the cache already holds permanently, exceeds
// the card.
//
// THREADING: every public method takes mutex_. The engine is documented as effectively
// single-GPU-serialized, and PhysicalGpuExecute keeps at most one chunk in flight, so the realistic
// concurrency here is "one background chunk calling StageChunk while the consumer thread scans the next
// batch". A write from a SECOND connection invalidating mid-scan is the one race not closed: it would
// free buffers an in-flight H2D is still writing. Same posture as ExecuteScan's cache-miss-mid-replay
// throw -- documented, not solved, because this engine is not built for concurrent connections generally.

namespace vector_gpu {

//! One streamed chunk of a column, exactly as ExecuteScan uploaded it: the device buffers holding
//! `row_count` consecutive rows. A cached column is a SEQUENCE of these in scan order (see the file
//! header for why chunks and not one contiguous buffer).
//!
//! Ownership passes to GpuColumnCache on a successful StageChunk and never comes back -- the cache frees
//! these buffers on eviction/invalidation. On a DECLINED StageChunk the caller still owns them (and, in
//! ExecuteScan, hands them to the plan's DeviceArena instead, so there is exactly one owner either way).
struct GpuColumnChunk {
	void *device_data = nullptr;     // cudaMalloc'd by the caller; owned by this cache once staged
	void *device_validity = nullptr; // nullptr if the chunk has no nulls, matching GpuColumn's convention
	uint64_t row_count = 0;
	uint64_t data_bytes = 0;
	uint64_t validity_bytes = 0;
	//! The column's value type. Carried per chunk because that is where the producer knows it; StageChunk
	//! rejects a chunk whose type disagrees with the column's earlier chunks.
	GpuValueType type = GpuValueType::FLOAT64;

	uint64_t TotalBytes() const {
		return data_bytes + validity_bytes;
	}
};

class GpuColumnCache {
public:
	static GpuColumnCache &Instance();

	GpuColumnCache(const GpuColumnCache &) = delete;
	GpuColumnCache &operator=(const GpuColumnCache &) = delete;

	//! Records a budget of `fraction_of_vram` (0.0-1.0) of currently-free device memory, measured once
	//! (mirrors RmmPool::EnsureInitialized). No-op if already initialized. Does NOT cudaMalloc anything
	//! up front -- like RmmPool, the budget is a number until the first real StageChunk needs space, so a
	//! process that never populates the cache never pays for it.
	//! VECTOR_GPU_CACHE_FRACTION overrides the fraction; VECTOR_GPU_CACHE_MB pins an absolute budget in
	//! MiB and wins over both, which is the only form that means the same thing across runs.
	void EnsureInitialized(double fraction_of_vram = 0.5);

	// ---- Population: staging. A column becomes readable only via CommitStaging, never via StageChunk.

	//! Opens a staging slot for `table`, ready to accumulate chunks. Discards anything previously
	//! staged for this table AND drops its committed entries: every cached column of a table must come
	//! from ONE scan, so that chunk boundaries are identical across columns (what a replay requires) and
	//! so a table being re-scanned never holds two generations of itself in VRAM at once.
	//! Calls EnsureInitialized() itself; a zero budget simply leaves no slot open, so every StageChunk
	//! below declines.
	void BeginStaging(const GpuTableRef &table);

	//! Appends `chunk` as the next chunk of `table`'s `column_name`. Returns true iff the cache took
	//! ownership of the chunk's device buffers.
	//!
	//! Returns false (caller keeps ownership, nothing leaks) when: no staging slot is open for the table,
	//! the chunk's type disagrees with the column's earlier chunks, or the column cannot fit the budget
	//! even after evicting every committed entry. A budget failure fails only THAT column -- other
	//! columns of the same table keep staging, since a partial column set is still a usable cache.
	bool StageChunk(const GpuTableRef &table, const std::string &column_name, const GpuColumnChunk &chunk);

	//! Publishes every fully-staged column of `table` as a readable entry, then closes the slot.
	//! `scanned_rows` is how many rows the scan actually produced; a column whose staged chunks do not sum
	//! to exactly that is DROPPED rather than published. That check is the whole point of staging -- it is
	//! what makes "an entry exists" mean "the complete column is resident", which the session-31 bug's
	//! per-chunk overwrite could not promise. No-op if no slot is open.
	void CommitStaging(const GpuTableRef &table, uint64_t scanned_rows);

	//! Discards everything staged for `table` without publishing any of it, freeing its device
	//! memory. MUST be called if a query that opened a slot ends without committing (early-terminated
	//! scan, exception) or the staged bytes stay resident for the life of the process. No-op if no slot
	//! is open, so it is safe as an unconditional cleanup call.
	void AbortStaging(const GpuTableRef &table);

	// ---- Reading

	//! Row count of every chunk `column_names` are cached in, in scan order -- the caller replays chunk i
	//! by executing its plan with GpuExecuteOptions::cache_chunk_index == i.
	//!
	//! Returns EMPTY unless EVERY requested column is committed AND they all agree on chunk count and on
	//! each chunk's row count. That agreement is guaranteed by construction (BeginStaging wipes the table,
	//! so one scan populates them all) but is re-checked here rather than assumed: serving columns whose
	//! chunk boundaries disagree would silently mis-align rows across columns, which is exactly the class
	//! of failure this whole file exists to make impossible.
	std::vector<uint64_t> ReplayableChunkRows(const GpuTableRef &table,
	                                          const std::vector<std::string> &column_names);

	//! Fetches chunk `chunk_index` of `table`'s `column_name` into `out_chunk`. Returns false on a miss
	//! or an out-of-range index. Touches LRU recency. The device pointers in `out_chunk` are a BORROW
	//! valid only until the next call that could evict (StageChunk/BeginStaging/Invalidate*) -- callers
	//! must finish using them (issue their kernels, synchronize) before calling those again, the same
	//! contract as KernelCache's Get().
	bool TryGetChunk(const GpuTableRef &table, const std::string &column_name, uint64_t chunk_index,
	                 GpuColumnChunk &out_chunk);

	// ---- Invalidation

	//! Drops every cached entry for `table` (all columns) and anything staged for it, freeing their
	//! device memory, and bumps that table's generation counter. THE correctness backstop this whole cache
	//! depends on: called from gpu_offload_extension.cpp's OptimizerExtension callback whenever it sees a
	//! write/DDL plan targeting this table, before that statement executes. See the file header for why
	//! this is provably sufficient (not just "probably catches it").
	void InvalidateTable(const GpuTableRef &table);

	//! Drops every cached entry and every staging slot, for every table. Used for DDL/maintenance
	//! operators (ATTACH, VACUUM, ...) where identifying the one affected table isn't worth the extra
	//! verified-API surface for something this rare -- see the design doc.
	void InvalidateAll();

	//! Total bytes currently resident -- committed entries plus whatever is mid-staging, since both hold
	//! real device memory. TableChecker::HasVramHeadroomForPlan adds this to what a new plan demands.
	uint64_t CurrentBytes() const;

	//! How many times `table` has been invalidated. Diagnostic only (logged with a cache hit so a
	//! stale-looking answer can be traced to an invalidation era); never compared for correctness -- see
	//! the file header for why map presence alone is sufficient.
	uint64_t TableGeneration(const GpuTableRef &table) const;

private:
	//! Makes exactly one cheap CUDA call (cudaSetDevice(0)) and nothing else -- the real budget/staging
	//! setup stays in EnsureInitialized, deferred exactly as before. See gpu_column_cache.cpp's definition
	//! for why this call exists and must not be removed.
	GpuColumnCache();
	~GpuColumnCache();

	//! A complete, readable column: every chunk of one table scan, in order.
	struct CachedColumn {
		std::vector<GpuColumnChunk> chunks;
		uint64_t row_count = 0;   // sum over chunks
		uint64_t total_bytes = 0; // sum over chunks
		uint64_t generation = 0;
	};

	//! A column mid-accumulation. Not readable, not evictable, and dropped wholesale unless its rows add
	//! up at CommitStaging time.
	struct StagedColumn {
		std::vector<GpuColumnChunk> chunks;
		uint64_t row_count = 0;
		uint64_t total_bytes = 0;
		GpuValueType type = GpuValueType::FLOAT64;
		bool type_known = false;
		//! Set once a chunk is refused (budget, type mismatch). Every later chunk for this column is
		//! refused too, so a column can never publish with a hole in the middle.
		bool failed = false;
	};

	//! Identity of one cached column. A STRUCT, not a "table.column" string: the previous string form was
	//! prefix-matched to find a table's entries, so a quoted identifier containing a dot could make one
	//! table's slot look like another's, and an invalidation could miss the entry it had to drop. Nothing
	//! about this cache is worth leaving to string parsing.
	struct SlotKey {
		GpuTableRef table;
		std::string column;

		bool operator==(const SlotKey &other) const {
			return table == other.table && column == other.column;
		}
	};

	struct TableRefHash {
		size_t operator()(const GpuTableRef &ref) const {
			auto h = std::hash<std::string> {};
			// Ordinary hash_combine; the mix constant is boost's, used here only to avoid the trivial
			// collisions plain XOR gives for swapped components.
			size_t seed = h(ref.catalog);
			seed ^= h(ref.schema) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= h(ref.table) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			return seed;
		}
	};

	struct SlotKeyHash {
		size_t operator()(const SlotKey &key) const {
			size_t seed = TableRefHash {}(key.table);
			seed ^= std::hash<std::string> {}(key.column) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			return seed;
		}
	};

	//! Caller must hold mutex_. Frees the entry's device buffers and removes it from entries_/lru_order_.
	void EvictLocked(const SlotKey &slot);
	//! Caller must hold mutex_. Evicts every committed entry belonging to `table`.
	void EvictTableLocked(const GpuTableRef &table);
	//! Caller must hold mutex_. Frees and forgets everything staged for `table`.
	void AbortStagingLocked(const GpuTableRef &table);
	//! Caller must hold mutex_. Frees a staged column's chunks and marks it failed.
	void FailStagedColumnLocked(StagedColumn &staged);
	//! Caller must hold mutex_. Moves `slot` to the front of lru_order_. No-op if not committed.
	void TouchLocked(const SlotKey &slot);

	mutable std::mutex mutex_;
	bool initialized_ = false;
	uint64_t budget_bytes_ = 0;
	uint64_t committed_bytes_ = 0;
	uint64_t staged_bytes_ = 0;
	std::list<SlotKey> lru_order_; // front = most recently used, matches KernelCache's convention
	std::unordered_map<SlotKey, std::pair<CachedColumn, std::list<SlotKey>::iterator>, SlotKeyHash> entries_;
	//! table -> column_name -> in-progress accumulation. At most one open slot per table.
	std::unordered_map<GpuTableRef, std::unordered_map<std::string, StagedColumn>, TableRefHash> staging_;
	//! table -> how many times it has been invalidated. Diagnostic only (see TableGeneration).
	std::unordered_map<GpuTableRef, uint64_t, TableRefHash> generation_;
};

} // namespace vector_gpu
