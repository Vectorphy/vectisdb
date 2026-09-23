// duckdb_gpu — a DuckDB shell with the VectisDB optimizer extension statically linked in.
//
// This is the piece that makes GPU offload actually happen for real SQL. Registering
// RegisterGpuOffloadOptimizer on the DBConfig *before* the database is opened installs the optimizer
// extension, so every query's logical plan passes through TableChecker::ShouldOffload and, when
// approved, is rewritten to a GpuExecuteOperator whose physical operator executes on the device
// (cuda_engine/src/gpu_executor.cu). Plans that aren't offloadable are left untouched and run on
// DuckDB's normal CPU path.
//
// CLI is a subset of duckdb.exe's, with an optional persistent query mode:
//   duckdb_gpu <db-path> [-json] -c "<sql>"
//   duckdb_gpu <db-path> [-json] --gpu-trace -c "<sql>"   (adds a GPU-offload report to stderr)
//   duckdb_gpu <db-path> [-json] --serve                  (persistent loop; blocks ended by --END--)
//   duckdb_gpu --gpu-probe                                (reports GPU/engine availability)
//
// -json prints one JSON array per result-producing statement, in DuckDB CLI-style output.

#include "duckdb.hpp"
#include "gpu_offload_extension.hpp"
#include "table_checker.hpp"

#include "rmm_pool.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace duckdb;

namespace {

//! Escapes a string for JSON output.
std::string JsonEscape(const std::string &input) {
	std::ostringstream oss;
	for (unsigned char c : input) {
		switch (c) {
		case '"':
			oss << "\\\"";
			break;
		case '\\':
			oss << "\\\\";
			break;
		case '\n':
			oss << "\\n";
			break;
		case '\r':
			oss << "\\r";
			break;
		case '\t':
			oss << "\\t";
			break;
		default:
			if (c < 0x20) {
				char buf[8];
				snprintf(buf, sizeof(buf), "\\u%04x", c);
				oss << buf;
			} else {
				oss << c;
			}
		}
	}
	return oss.str();
}

//! Renders one materialized result as a JSON array of row objects (duckdb.exe -json's shape).
//! `max_rows` (0 = unlimited) bounds how many rows are PRINTED — the query itself still executes in
//! full. Callers can request a bounded page without rewriting the user's SQL: wrapping the query in a
//! LIMIT would change the plan shape and, in particular,
//! prevent the optimizer from offloading it to the GPU.
void PrintJson(MaterializedQueryResult &result, idx_t max_rows) {
	auto &names = result.names;
	std::ostringstream oss;
	oss << "[";
	bool first_row = true;
	idx_t printed = 0;
	for (auto &row : result.Collection().GetRows()) {
		if (max_rows > 0 && printed >= max_rows) {
			break;
		}
		printed++;
		if (!first_row) {
			oss << ",\n";
		}
		first_row = false;
		oss << "{";
		for (idx_t col = 0; col < names.size(); col++) {
			if (col > 0) {
				oss << ",";
			}
			oss << "\"" << JsonEscape(names[col]) << "\":";
			auto value = row.GetValue(col);
			if (value.IsNull()) {
				oss << "null";
			} else {
				switch (result.types[col].id()) {
				case LogicalTypeId::TINYINT:
				case LogicalTypeId::SMALLINT:
				case LogicalTypeId::INTEGER:
				case LogicalTypeId::BIGINT:
				case LogicalTypeId::UTINYINT:
				case LogicalTypeId::USMALLINT:
				case LogicalTypeId::UINTEGER:
				case LogicalTypeId::UBIGINT:
				case LogicalTypeId::FLOAT:
				case LogicalTypeId::DOUBLE:
					oss << value.ToString();
					break;
				case LogicalTypeId::BOOLEAN:
					oss << (value.GetValue<bool>() ? "true" : "false");
					break;
				default:
					oss << "\"" << JsonEscape(value.ToString()) << "\"";
					break;
				}
			}
		}
		oss << "}";
	}
	oss << "]\n";
	std::cout << oss.str();
	// Report the true size on stderr so the caller can tell the user their result was larger than
	// the page it received, without having to count rows itself.
	if (max_rows > 0 && result.Collection().Count() > printed) {
		std::cerr << "[rows] printed " << printed << " of " << result.Collection().Count() << "\n";
	}
}

int RunProbe() {
	std::cout << "VectisDB GPU probe\n";
	try {
		DBConfig config;
		vector_gpu::RegisterGpuOffloadOptimizer(config);
		DuckDB db(nullptr, &config);
		Connection con(db);
		auto version = con.Query("SELECT version()");
		std::cout << "  duckdb           : "
		          << (version->HasError() ? version->GetError() : version->GetValue(0, 0).ToString()) << "\n";
		std::cout << "  gpu optimizer    : registered\n";
		std::cout << "  offload threshold: " << vector_gpu::TableChecker::MIN_ROW_COUNT_THRESHOLD << " rows\n";
		return 0;
	} catch (const std::exception &e) {
		std::cout << "  ERROR: " << e.what() << "\n";
		return 1;
	}
}

} // namespace

int main(int argc, char **argv) {
	std::string db_path;
	std::string sql;
	bool json_output = false;
	bool gpu_trace = false;
	bool gpu_enabled = true;
	bool serve = false;
	idx_t max_rows = 0; // 0 = print everything

	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		if (arg == "--gpu-probe") {
			return RunProbe();
		} else if (arg == "-json") {
			json_output = true;
		} else if (arg == "--max-rows") {
			if (i + 1 >= argc) {
				std::cerr << "duckdb_gpu: --max-rows requires an argument\n";
				return 1;
			}
			max_rows = static_cast<idx_t>(std::strtoull(argv[++i], nullptr, 10));
		} else if (arg == "--serve") {
			// Persistent query loop -- see the protocol note where it is implemented.
			serve = true;
		} else if (arg == "--gpu-trace") {
			gpu_trace = true;
		} else if (arg == "--no-gpu") {
			// Skips registering the optimizer entirely, so nothing can be offloaded. This is a true
			// A/B switch — the same binary, same data, same SQL, with the GPU path simply absent —
			// which makes --no-gpu a CPU comparison using the same database and SQL.
			gpu_enabled = false;
		} else if (arg == "-c") {
			if (i + 1 >= argc) {
				std::cerr << "duckdb_gpu: -c requires an argument\n";
				return 1;
			}
			sql = argv[++i];
		} else if (arg == "-f") {
			if (i + 1 >= argc) {
				std::cerr << "duckdb_gpu: -f requires a filename\n";
				return 1;
			}
			std::ifstream file(argv[++i]);
			if (!file) {
				std::cerr << "duckdb_gpu: failed to open " << argv[i] << "\n";
				return 1;
			}
			std::stringstream buffer;
			buffer << file.rdbuf();
			sql = buffer.str();
		} else if (db_path.empty()) {
			db_path = arg;
		}
	}

	if (sql.empty() && !serve) {
		std::cerr << "usage: duckdb_gpu <db-path> [-json] [--gpu-trace] [--no-gpu] -c \"<sql>\"\n"
		             "       duckdb_gpu --gpu-probe\n";
		return 1;
	}

	try {
		DBConfig config;
		// THE line that enables GPU offload: installs the optimizer extension that rewrites approved
		// plans into GpuExecuteOperator. Must happen before the DuckDB instance is constructed.
		// With --no-gpu it is skipped, so the process behaves exactly like stock DuckDB.
		if (gpu_enabled) {
			vector_gpu::RegisterGpuOffloadOptimizer(config);
		}
		if (gpu_trace) {
			vector_gpu::SetGpuOffloadTrace(true);
		}

		// Create the CUDA context concurrently with opening the database and planning the query, instead
		// of paying for it serially on the routing path. Measured: creating the context is ~100ms and was
		// much of cold-start latency once the VRAM arena stopped being reserved eagerly; opening the DB and
		// planning is comparable work that needs no GPU, so the two overlap. RmmPool::EnsureInitialized is
		// idempotent and mutex-guarded, so the optimizer thread finds the work done or waits briefly for it.
		// Joined below rather than detached: a detached thread could still be touching the RmmPool
		// singleton while static destructors run at process exit.
		// OPT-IN as of session 15, and off by default. Warming the context concurrently helps only when
		// the query actually offloads; when it does not, it creates a CUDA context nobody uses and costs
		// ~79 ms of wall clock (measured). Deferring the routing VRAM check past translation was supposed
		// to avoid that context entirely — this thread defeated it, because it called EnsureInitialized
		// unconditionally. Today almost nothing offloads, so not creating it wins on expected value.
		// Set VECTOR_GPU_WARMUP=1 for workloads known to offload, where the overlap is worth ~40 ms.
		const char *warmup_env = std::getenv("VECTOR_GPU_WARMUP");
		const bool warmup_requested = warmup_env != nullptr && std::strcmp(warmup_env, "0") != 0;
		std::thread cuda_warmup;
		if (gpu_enabled && warmup_requested) {
			cuda_warmup = std::thread([] {
				try {
					vector_gpu::RmmPool::Instance().EnsureInitialized();
				} catch (...) {
					// No GPU / no driver: the routing path re-checks and falls back to CPU on its own.
				}
			});
		}
		struct ThreadJoiner {
			std::thread &t;
			~ThreadJoiner() {
				if (t.joinable()) {
					t.join();
				}
			}
		} joiner {cuda_warmup};

		DuckDB db(db_path.empty() ? nullptr : db_path.c_str(), &config);
		Connection con(db);

		// Split and run each statement, so multi-statement input behaves like duckdb.exe -c.
		auto run_sql = [&](const std::string &text) {
			auto statements = con.ExtractStatements(text);
			for (auto &statement : statements) {
				auto result = con.Query(std::move(statement));
				if (result->HasError()) {
					std::cerr << result->GetError() << "\n";
					return false;
				}
				if (result->properties.return_type != StatementReturnType::QUERY_RESULT) {
					continue; // DDL/DML produce no rows, same as duckdb.exe -json
				}
				if (json_output) {
					PrintJson(*result, max_rows);
				} else {
					std::cout << result->ToString();
				}
			}
			return true;
		};

		if (serve) {
			// Persistent mode: keep the database, the CUDA context and the NVRTC kernel cache alive across
			// many queries, so per-process startup is paid ONCE instead of once per query. That startup is
			// so callers issuing many queries reuse database setup, the CUDA context and compiled kernels.
			//
			// Protocol, deliberately trivial and newline-safe: read lines until one is exactly "--END--",
			// run the accumulated SQL, then print a line that is exactly "--DONE--" and flush. Errors go to
			// stderr as usual and are followed by "--DONE--" too, so a client never blocks on a failure.
			std::string line;
			std::string pending;
			while (std::getline(std::cin, line)) {
				if (!line.empty() && line.back() == '\r') {
					line.pop_back(); // tolerate CRLF clients on Windows
				}
				if (line != "--END--") {
					pending += line;
					pending += '\n';
					continue;
				}
				run_sql(pending);
				pending.clear();
				std::cout << "--DONE--" << std::endl; // endl: the flush is the point
			}
		} else if (!run_sql(sql)) {
			return 1;
		}

		if (gpu_trace) {
			std::cerr << "[gpu-trace] plans offloaded to GPU: " << vector_gpu::GetGpuOffloadCount() << "\n";
			auto timings = vector_gpu::GetGpuPhaseTimings();
			auto total_us = timings.scan_us + timings.execute_us + timings.pack_us;
			if (total_us > 0) {
				// Phase breakdown, so "the GPU path is slow" can be attributed to a phase instead of
				// guessed at. scan = DuckDB storage read + host conversion, execute = H2D + kernels + D2H,
				// pack = result back into a ColumnDataCollection.
				std::cerr << "[gpu-trace] rows in: " << timings.input_rows << "  optimize: "
				          << timings.optimize_us / 1000.0 << " ms  scan: " << timings.scan_us / 1000.0
				          << " ms  execute: " << timings.execute_us / 1000.0 << " ms  pack: "
				          << timings.pack_us / 1000.0 << " ms  (total " << (total_us + timings.optimize_us) / 1000.0
				          << " ms)\n";
			}
		}
		return 0;
	} catch (const std::exception &e) {
		std::cerr << e.what() << "\n";
		return 1;
	}
}
