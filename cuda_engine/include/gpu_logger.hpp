#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>

// Bounded GPU/debug activity logger.
//
// Answers the question this project kept having to answer by bisecting: "what did the GPU path actually
// DO for this query, and if it did nothing, why?". The single highest-value record is the `decline`
// event, which carries the reason an operator was refused — the difference between "my query is slow"
// and "my query was never offloaded because ORDER BY is not in the whitelist".
//
// Lives in cuda_engine (not extension) so both sides can log, and has NO DuckDB dependency, matching the
// rest of this library.
//
// OFF BY DEFAULT. When disabled, a call site costs one relaxed atomic load and nothing else — that is
// why the Enabled() check is inlined here and the formatting happens only inside the branch. Always go
// through the VGPU_LOG macro rather than calling Log() directly, so the arguments are not even evaluated
// when logging is off.
//
// STORAGE / RETENTION CONTRACT. Records go to two files (`gpu_log.0.jsonl`, `gpu_log.1.jsonl`), each
// capped at half the configured cap. Writes append to the active file; when it fills, the other file is
// truncated and becomes active. So the newest data is ALWAYS retained, total on-disk stays under the cap,
// and what is kept oscillates between half the cap and the full cap as the newest file fills. That band
// is the documented contract — a byte-exact ring would need to rewrite file contents on every rollover
// for no practical gain. Oldest entries are the ones discarded, per the spec.
//
// CONFIG (environment, read once on first use):
//   VECTOR_GPU_LOG           0/1     enable (default 0)
//   VECTOR_GPU_LOG_LEVEL     error|info|debug  (default info)
//   VECTOR_GPU_LOG_CAP_MIB   integer, total across both files (default 50)
//   VECTOR_GPU_LOG_PATH      directory for the log files (default: current directory)

namespace vector_gpu {

enum class LogLevel : int {
	ERROR = 0,
	INFO = 1,
	DEBUG = 2,
};

class GpuLogger {
public:
	static GpuLogger &Instance();

	//! Cheap enough to call unconditionally: one relaxed atomic load when logging is off.
	bool Enabled(LogLevel level) const {
		return enabled_.load(std::memory_order_relaxed) && static_cast<int>(level) <= level_.load(std::memory_order_relaxed);
	}

	//! Writes one JSON object line: {"ts_us":..,"thread":..,"lvl":"..","evt":"<event>",<fields>}.
	//! `fields` must be a valid JSON object body WITHOUT the enclosing braces and without a leading
	//! comma, e.g. R"("bytes":4096,"us":12)". Pass an empty string for no extra fields.
	//! Thread-safe. Never throws — a logger that takes down a query is worse than no logger.
	void Log(LogLevel level, const char *event, const std::string &fields) noexcept;

	//! Escapes a string for embedding as a JSON value (quotes included in the result).
	static std::string Quote(const std::string &value);

	//! Test-only: forces configuration instead of reading the environment, and restarts the log files.
	//! `cap_bytes` is the total across both files.
	void ConfigureForTesting(bool enabled, LogLevel level, size_t cap_bytes, const std::string &directory);

	//! Test-only: flushes and closes the current file so a test can read it back.
	void FlushForTesting();

	//! Total bytes currently on disk across both log files (test/observability helper).
	size_t OnDiskBytesForTesting() const;

private:
	//! Reads the environment on construction. This MUST happen here rather than lazily inside Log():
	//! the VGPU_LOG macro gates on Enabled(), so configuring inside Log() means Enabled() is false
	//! forever and the logger can never switch itself on. (Found the hard way — the first end-to-end
	//! run produced an empty log.)
	GpuLogger();
	~GpuLogger();
	GpuLogger(const GpuLogger &) = delete;
	GpuLogger &operator=(const GpuLogger &) = delete;

	void EnsureConfigured();
	//! Caller must hold mutex_. Rotates to the other file when the active one reaches its half-cap.
	void RotateIfNeededLocked(size_t incoming_bytes);
	void OpenActiveLocked(bool truncate);

	std::atomic<bool> enabled_ {false};
	std::atomic<int> level_ {static_cast<int>(LogLevel::INFO)};

	mutable std::mutex mutex_;
	bool configured_ = false;
	std::string directory_;
	size_t half_cap_bytes_ = 25u * 1024u * 1024u;
	int active_file_ = 0;
	size_t active_bytes_ = 0;
	void *file_ = nullptr; // FILE*, kept as void* so <cstdio> stays out of this header
};

} // namespace vector_gpu

//! Use this, not GpuLogger::Log directly: the `fields` expression is not evaluated when logging is off.
#define VGPU_LOG(level, event, fields_expr)                                                                          \
	do {                                                                                                             \
		auto &_vgpu_logger = ::vector_gpu::GpuLogger::Instance();                                                     \
		if (_vgpu_logger.Enabled(level)) {                                                                           \
			_vgpu_logger.Log((level), (event), (fields_expr));                                                        \
		}                                                                                                            \
	} while (0)
