#include "vector_gpu_host.hpp"

#include "duckdb.hpp"
#include "gpu_offload_extension.hpp"

#include <chrono>

namespace vector_gpu {

namespace {

//! The phase timers and the offload counter are process-global atomics that only ACCUMULATE (see
//! gpu_offload_extension.cpp -- they are written from whichever thread plans or materializes, so they
//! were never per-query). Differencing a snapshot around one query is therefore the only way to get a
//! per-query figure without changing that contract, and it is exact as long as queries are serialized.
struct Snapshot {
	uint64_t offloads;
	GpuPhaseTimings timings;
};

Snapshot TakeSnapshot() {
	return Snapshot {GetGpuOffloadCount(), GetGpuPhaseTimings()};
}

uint64_t Delta(uint64_t after, uint64_t before) {
	// Saturating: a counter reset underneath us must read as zero rather than as 18 quintillion.
	return after >= before ? after - before : 0;
}

} // namespace

struct Session::Impl {
	std::string db_path;
	bool enable_gpu;
	duckdb::unique_ptr<duckdb::DuckDB> database;
	duckdb::unique_ptr<duckdb::Connection> connection;
	bool warmed = false;
};

Session::Session(const std::string &db_path, bool enable_gpu) : impl_(new Impl()) {
	impl_->db_path = db_path;
	impl_->enable_gpu = enable_gpu;

	duckdb::DBConfig config;
	if (enable_gpu) {
		// Must happen BEFORE the instance is constructed: the optimizer extension list is read when the
		// DatabaseInstance is built, so registering afterwards silently does nothing.
		RegisterGpuOffloadOptimizer(config);
	}
	// Tracing gates the phase timers (RecordGpuPhaseTimings returns immediately when it is off), so it
	// has to be on for QueryStats to contain anything. It costs a few atomic adds per query.
	SetGpuOffloadTrace(true);

	impl_->database = duckdb::make_uniq<duckdb::DuckDB>(db_path.empty() ? nullptr : db_path.c_str(), &config);
	impl_->connection = duckdb::make_uniq<duckdb::Connection>(*impl_->database);
}

Session::~Session() = default;

bool Session::gpu_enabled() const {
	return impl_->enable_gpu;
}

const std::string &Session::db_path() const {
	return impl_->db_path;
}

void Session::Warmup() {
	if (impl_->warmed) {
		return;
	}
	impl_->warmed = true;
	// A trivial statement is enough to build the DuckDB side, but NOT to create the CUDA context: the
	// VRAM check is what does that, and it only runs for a plan that survives translation. Nothing here
	// can force that without inventing a table, so this warms the DuckDB half and leaves the device half
	// to the first offloadable query. Documented rather than papered over.
	impl_->connection->Query("SELECT 1;");
}

QueryResult Session::Query(const std::string &sql, uint64_t max_rows) {
	QueryResult out;
	auto before = TakeSnapshot();
	auto started = std::chrono::steady_clock::now();

	auto result = impl_->connection->Query(sql);

	auto elapsed = std::chrono::steady_clock::now() - started;
	auto after = TakeSnapshot();

	out.stats.wall_us =
	    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
	out.stats.offloaded_plans = Delta(after.offloads, before.offloads);
	out.stats.scan_us = Delta(after.timings.scan_us, before.timings.scan_us);
	out.stats.execute_us = Delta(after.timings.execute_us, before.timings.execute_us);
	out.stats.optimize_us = Delta(after.timings.optimize_us, before.timings.optimize_us);
	out.stats.input_rows = Delta(after.timings.input_rows, before.timings.input_rows);

	if (result->HasError()) {
		out.success = false;
		out.error = result->GetError();
		return out;
	}

	out.success = true;
	for (auto &name : result->names) {
		out.column_names.push_back(name);
	}
	for (auto &type : result->types) {
		out.column_types.push_back(type.ToString());
	}

	auto &materialized = result->Cast<duckdb::MaterializedQueryResult>();
	auto total = materialized.RowCount();
	auto emit = total > max_rows ? max_rows : total;
	out.truncated = total > emit;
	out.rows.reserve(static_cast<size_t>(emit));

	auto &collection = materialized.Collection();
	duckdb::ColumnDataScanState scan_state;
	collection.InitializeScan(scan_state);
	duckdb::DataChunk chunk;
	collection.InitializeScanChunk(chunk);
	uint64_t emitted = 0;
	while (emitted < emit && collection.Scan(scan_state, chunk)) {
		for (duckdb::idx_t row = 0; row < chunk.size() && emitted < emit; row++, emitted++) {
			std::vector<std::string> values;
			values.reserve(chunk.ColumnCount());
			for (duckdb::idx_t col = 0; col < chunk.ColumnCount(); col++) {
				auto value = chunk.GetValue(col, row);
				values.push_back(value.IsNull() ? std::string("NULL") : value.ToString());
			}
			out.rows.push_back(std::move(values));
		}
	}
	return out;
}

} // namespace vector_gpu
