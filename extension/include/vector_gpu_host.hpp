#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// In-process C++ API for embedding the GPU-offload engine directly, instead of spawning duckdb_gpu.exe
// and talking to it over pipes.
//
// WHY THIS EXISTS. Every query through the subprocess path pays per-process CUDA startup -- ~80 ms warm,
// ~155 ms cold (measured, docs/TODO.md P1 item 3) -- plus NVRTC compilation of the first kernel, ~125 ms,
// because the kernel cache dies with the process. A host that keeps one process alive pays both exactly
// once. It also removes the pipe protocol, its --END--/--DONE-- framing, and the timeout/restart handling
// that a child process forces on every caller.
//
// It ALSO fixes a resource bug that the subprocess design cannot: RmmPool computes its VRAM budget as a
// fraction of free device memory PER PROCESS. Two child processes each independently conclude they may
// use ~85% of the card and then both page -- silently, since WDDM raises no out-of-memory error. One
// process means one pool and one honest budget.
//
// DELIBERATELY NO duckdb.hpp HERE. This header is what a GUI includes; dragging DuckDB's headers (and
// their compile cost and macro surface) into that build is exactly what a boundary is for. Everything
// DuckDB-shaped lives behind Session::Impl.

namespace vector_gpu {

//! What the engine did with one query. All values are for THAT query alone, obtained by differencing the
//! engine's process-global counters around it -- see the note on Session::Query about serialization.
struct QueryStats {
	//! Subtrees the optimizer actually rewrote to GpuExecuteOperator. 0 means the query ran on CPU.
	uint64_t offloaded_plans = 0;
	//! Reading DuckDB storage and packing it into host buffers for upload.
	uint64_t scan_us = 0;
	//! H2D + kernels + D2H.
	uint64_t execute_us = 0;
	//! The routing pass itself, paid on every query whether or not it offloads.
	uint64_t optimize_us = 0;
	uint64_t input_rows = 0;
	//! End-to-end, including DuckDB's own CPU work.
	uint64_t wall_us = 0;
};

struct QueryResult {
	bool success = false;
	std::string error;
	std::vector<std::string> column_names;
	std::vector<std::string> column_types;
	//! Row-major, already stringified. Deliberately not DuckDB Values: the point of this API is that a
	//! caller needs no DuckDB types to use it.
	std::vector<std::vector<std::string>> rows;
	//! True when `rows` was truncated by the caller's max_rows.
	bool truncated = false;
	QueryStats stats;
};

//! One isolated database. A caller wanting N independent datasets constructs N Sessions: each owns its
//! own duckdb::DuckDB instance, so a query in one cannot name a table in another -- the same isolation
//! the process-per-tab design bought, without the processes.
class Session {
public:
	//! `db_path` empty opens an in-memory database. When `enable_gpu` is false the optimizer extension is
	//! simply not registered, so the session behaves exactly like stock DuckDB -- a true A/B rather than a
	//! flag consulted at run time.
	Session(const std::string &db_path, bool enable_gpu);
	~Session();

	Session(const Session &) = delete;
	Session &operator=(const Session &) = delete;

	//! Runs `sql`, returning at most `max_rows` materialized rows.
	//!
	//! THREADING: the per-query statistics are differenced from counters that are process-global (see
	//! gpu_offload_extension.cpp), so concurrent Query() calls across Sessions would attribute each
	//! other's work. Results stay correct -- DuckDB handles the concurrency -- but the stats would not.
	//! Callers running several sessions must serialize their queries, which is independently required
	//! anyway: the GPU is one device with one VRAM budget.
	QueryResult Query(const std::string &sql, uint64_t max_rows = 10000);

	//! Pays the one-time device costs (CUDA context, first NVRTC compile) up front, so the first real
	//! query does not. Safe to call more than once; later calls are cheap.
	void Warmup();

	bool gpu_enabled() const;
	const std::string &db_path() const;

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

} // namespace vector_gpu
