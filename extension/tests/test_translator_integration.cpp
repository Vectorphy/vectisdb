// Real end-to-end integration test: links statically against duckdb_static (not a loadable
// .duckdb_extension -- that build mode needs extension-ci-tools, see docs/EXECUTION_TRACKER.md #13),
// registers the real GpuOffloadOptimizer via RegisterGpuOffloadOptimizer, and runs real SQL through a
// real in-memory DuckDB database. This is the strongest verification available right now for
// TableChecker + TranslateToGpuPlan + expression_translator.cpp + GpuExecuteOperator +
// PhysicalGpuExecute, short of a full extension-template build.
//
// STALE AS OF SESSION 10 -- DO NOT TRUST THIS FILE'S ASSERTIONS UNTIL IT IS REWRITTEN.
// Its whole premise below (that an offloaded query must FAIL at physical execution, and that reaching
// that failure proves the pipeline works) stopped being true once the operators became real: sessions
// 8-10 landed genuine SCAN/FILTER/PROJECTION, GROUP BY and INNER-JOIN execution, so a query that gets
// through the optimizer pass now SUCCEEDS and returns real GPU-computed rows. Cases here that assert the
// GPU_STUB_ERROR_PREFIX are therefore asserting the opposite of the correct behaviour.
//
// It is left in place rather than half-fixed because it is not wired into CMake (tracker #13) and so was
// not run this session -- editing assertions without executing them would be worse than flagging it.
// The end-to-end coverage it used to provide now lives in gpu_shell/verify_gpu_vs_cpu.sh, which diffs
// real SQL results between GPU and --no-gpu runs of the same binary and IS runnable today.
//
// Original premise, retained for context:
// Because GpuEngine::ExecutePlan's operator stubs all return success=false (blocked on libcudf --
// docs/KNOWN_ISSUES.md), a query that gets all the way through our optimizer pass is expected to fail at
// physical execution with a specific, recognizable error message. That's not a bug: reaching that error
// message IS the proof the whole pipeline up to (not including) real GPU execution works correctly.
//
// Build (no CMake wiring for this yet -- run directly, see comment at bottom of this file). Links the
// REAL vector_gpu_cuda_engine.lib now (table_checker.cpp calls into RmmPool for the VRAM headroom check)
// -- this only works because cuda_engine/CMakeLists.txt now forces the /MT CRT (CMP0091 +
// CMAKE_MSVC_RUNTIME_LIBRARY) to match duckdb_static.lib's; before that fix this needed a throwaway
// link_stubs.cpp providing fake GpuEngine/KernelCache symbols to sidestep the CRT mismatch entirely.
//   cl /nologo /std:c++17 /EHsc /MT /I ../include /I ../../cuda_engine/include
//      /I <duckdb>/src/include /I <duckdb>/third_party/fmt/include
//      ../src/table_checker.cpp ../src/gpu_logical_operator.cpp ../src/physical_gpu_execute.cpp
//      ../src/gpu_offload_extension.cpp ../src/expression_translator.cpp
//      test_translator_integration.cpp
//      /link <duckdb>/build_core/src/duckdb_static.lib
//            <duckdb>/build_core/extension/dummy_static_extension_loader.lib
//            <duckdb>/build_core/extension/duckdb_generated_extension_loader.lib
//            <duckdb>/build_core/extension/core_functions/core_functions_extension.lib
//            <duckdb>/build_core/extension/parquet/parquet_extension.lib
//            ../../cuda_engine/build/vector_gpu_cuda_engine.lib
//            Rstrtmgr.lib bcrypt.lib

#include "duckdb.hpp"
#include "gpu_offload_extension.hpp"
#include "rmm_pool.hpp"

#include <iostream>
#include <string>

using namespace duckdb;

namespace {

int g_checks = 0;
int g_failures = 0;

void Check(bool condition, const std::string &description) {
	g_checks++;
	if (!condition) {
		g_failures++;
		std::cerr << "[FAIL] " << description << "\n";
	} else {
		std::cerr << "[PASS] " << description << "\n";
	}
}

//! The exact fixed prefix PhysicalGpuExecute::MaterializeOnce throws when it reaches
//! GpuEngine::ExecutePlan and gets back success=false (see physical_gpu_execute.cpp). Matching against
//! this string is how these tests distinguish "reached real GPU execution and it's stubbed" (expected,
//! proves the whole pass wired up correctly) from any other, unexpected failure.
const char *GPU_STUB_ERROR_PREFIX = "PhysicalGpuExecute: GPU execution failed";

Connection MakeConnection(DBConfig &config) {
	static DuckDB db(nullptr, &config);
	return Connection(db);
}

} // namespace

int main() {
	DBConfig config;
	vector_gpu::RegisterGpuOffloadOptimizer(config);
	auto con = MakeConnection(config);

	// Large table: comfortably clears TableChecker::MIN_ROW_COUNT_THRESHOLD (100,000) even after a
	// selective filter's estimated (not actual) selectivity, or a low-cardinality GROUP BY's estimated
	// (not actual) output size -- both of those cost-estimator effects turned out to matter, see below.
	auto create_result =
	    con.Query("CREATE TABLE big AS SELECT range::INTEGER AS a, (range % 17)::INTEGER AS b, "
	              "(range % 5)::INTEGER AS c, range::DOUBLE AS d FROM range(2000000)");
	Check(!create_result->HasError(), "create large table (2M rows) succeeds: " +
	                                      (create_result->HasError() ? create_result->GetError() : "ok"));
	// Without this, DuckDB has no column statistics/histograms for a freshly created table and falls
	// back to a flat default filter selectivity guess (independent of the actual predicate literal) when
	// estimating cardinality -- which can put a post-filter estimate below MIN_ROW_COUNT_THRESHOLD even
	// for a 95%-selective filter over 200k rows. ANALYZE gives the optimizer real stats to work with.
	con.Query("ANALYZE big");

	// Small table: same schema, below the row threshold -> Table Checker must reject and let this run on
	// CPU normally, producing a correct, verifiable result.
	auto create_small = con.Query("CREATE TABLE small AS SELECT range::INTEGER AS a, (range % 17)::INTEGER AS "
	                              "b, (range % 5)::INTEGER AS c, range::DOUBLE AS d FROM range(10)");
	Check(!create_small->HasError(), "create small table (10 rows) succeeds");

	// --- Case 1: simple arithmetic filter+projection over the large table -----------------------------
	// Table Checker approves (row count, supported types/operators), translation succeeds (column refs +
	// comparison + arithmetic are all supported), physical execution reaches the GPU stub.
	// NOTE: the filter must stay low-selectivity (a < 190000 keeps ~95% of 200k rows) -- an aggressive
	// filter like `a > 100` drops DuckDB's cardinality *estimate* for the post-filter row count below
	// MIN_ROW_COUNT_THRESHOLD even though the real result set would be large, causing a correct Table
	// Checker rejection (not a bug) that this test would misread as a translator failure.
	{
		auto result = con.Query("SELECT a * b - c AS val FROM big WHERE a < 190000");
		Check(result->HasError(), "large table, simple arithmetic+filter: routes to GPU stub (has error)");
		if (result->HasError()) {
			Check(result->GetError().find(GPU_STUB_ERROR_PREFIX) != std::string::npos,
			      "error message is the expected GPU-stub failure, not something else: " + result->GetError());
		}
	}

	// --- Case 2: same query shape, but table is below the row threshold ------------------------------
	// Must NOT be offloaded -- Table Checker rejects on cardinality, CPU executes normally, and the
	// result must be numerically correct (this is the real fallback-correctness check, not just
	// "didn't crash").
	{
		auto result = con.Query("SELECT a * b - c AS val FROM small WHERE a > 2 ORDER BY a");
		Check(!result->HasError(), "small table falls back to CPU and executes without error: " +
		                               (result->HasError() ? result->GetError() : "ok"));
		if (!result->HasError()) {
			// range(10) -> a in [0,9], b = a % 17 = a, c = a % 5. Row a=3: val = 3*3-3 = 6.
			auto chunk = result->Fetch();
			Check(chunk && chunk->size() > 0, "small table fallback query returns at least one row");
			if (chunk && chunk->size() > 0) {
				auto val = chunk->GetValue(0, 0).GetValue<int32_t>();
				Check(val == 6, "small table fallback query result is numerically correct (expected 6, got " +
				                    std::to_string(val) + ")");
			}
		}
	}

	// --- Case 3: unsupported construct (CASE expression) must fall back, not crash or mistranslate -----
	{
		auto result = con.Query("SELECT CASE WHEN a > 100 THEN a ELSE b END AS val FROM big");
		Check(!result->HasError(),
		      "CASE expression (unsupported by translator) falls back to CPU cleanly: " +
		          (result->HasError() ? result->GetError() : "ok"));
	}

	// --- Case 4: hash join on the large table reaches the GPU stub (join-key extraction path) ---------
	// Self-join on `a` (2M distinct values, essentially a key) keeps the output estimate close to 1:1
	// with the input (~2M rows) -- comfortably above MIN_ROW_COUNT_THRESHOLD and comfortably under the
	// RMM pool's VRAM budget. An earlier version of this test self-joined on `c` (only 5 distinct
	// values), which multiplies out to an ~800-BILLION-row estimate: after wiring TableChecker's VRAM
	// check to the real RmmPool (see docs/EXECUTION_TRACKER.md), that plan is now correctly REJECTED for
	// GPU offload -- and DuckDB then actually attempts to run that genuinely enormous join on the CPU,
	// which doesn't error quickly, it hangs/thrashes for a very long time before finally OOMing. Nothing
	// wrong with the code; just the wrong query to prove "reaches the GPU stub" with once the VRAM check
	// became real.
	{
		auto result = con.Query("SELECT b1.a FROM big b1 JOIN big b2 ON b1.a = b2.a");
		Check(result->HasError(), "equi-join over large table routes to GPU stub (has error)");
		if (result->HasError()) {
			Check(result->GetError().find(GPU_STUB_ERROR_PREFIX) != std::string::npos,
			      "join error message is the expected GPU-stub failure: " + result->GetError());
		}
	}

	// --- Case 5: group-by with a bare column key reaches the GPU stub (group-key extraction path) -----
	// A bare `GROUP BY a` with no aggregate call (equivalent to SELECT DISTINCT a) turned out to plan as
	// a LOGICAL_DISTINCT operator instead of LOGICAL_AGGREGATE_AND_GROUP_BY -- a different, unsupported
	// operator type that TableChecker correctly rejects. Real aggregate functions (SUM/COUNT/...) live in
	// the core_functions extension, which this hand-built DBConfig doesn't auto-load; load the already
	//-compiled local binary directly (produced by this same build) instead of hitting the network.
	con.Query("LOAD '" +
	          std::string("C:\\Users\\Vector\\OneDrive\\Desktop\\CR\\duckdb\\build_core\\repository\\"
	                      "af1b4a9bd2\\windows_amd64\\core_functions.duckdb_extension") +
	          "'");
	{
		auto result = con.Query("SELECT a, COUNT(*) FROM big GROUP BY a");
		Check(result->HasError(), "group-by over large table routes to GPU stub (has error)");
		if (result->HasError()) {
			Check(result->GetError().find(GPU_STUB_ERROR_PREFIX) != std::string::npos,
			      "group-by error message is the expected GPU-stub failure: " + result->GetError());
		}
	}

	// --- Case 6: group-by on a computed expression (not a bare column) must fall back ------------------
	// Uses COUNT(*) (now loaded) to force a real LOGICAL_AGGREGATE_AND_GROUP_BY node -- specifically
	// exercising ResolveKeyColumnName's rejection of a non-column-reference group key, rather than
	// incidentally falling back for the unrelated reason case 5 initially did (GROUP BY with no
	// aggregate call planning as LOGICAL_DISTINCT instead).
	{
		auto result = con.Query("SELECT a + 1, COUNT(*) FROM big GROUP BY a + 1");
		Check(!result->HasError(),
		      "group-by on a computed key (unsupported) falls back to CPU cleanly: " +
		          (result->HasError() ? result->GetError() : "ok"));
	}

	// --- Case 7: an operator our Table Checker doesn't even attempt (LOGICAL_ORDER_BY) must behave
	// exactly like normal DuckDB, unaffected by our extension being registered -------------------------
	{
		auto result = con.Query("SELECT a FROM small ORDER BY a DESC LIMIT 3");
		Check(!result->HasError(), "ORDER BY/LIMIT (unsupported operator) executes normally: " +
		                               (result->HasError() ? result->GetError() : "ok"));
		if (!result->HasError()) {
			auto chunk = result->Fetch();
			Check(chunk && chunk->size() == 3, "ORDER BY/LIMIT returns the expected row count");
			if (chunk && chunk->size() == 3) {
				Check(chunk->GetValue(0, 0).GetValue<int32_t>() == 9,
				      "ORDER BY DESC returns correctly-ordered results (expected top value 9)");
			}
		}
	}

	// --- Case 8: a LEFT JOIN must fall back to CPU, not be mistranslated as an INNER-shaped hash join ---
	// Correctness fix, not just a translation gap: TranslateToGpuPlan now rejects any join_type other
	// than INNER (LEFT/RIGHT/OUTER/SEMI/ANTI/MARK/... have different null-handling or output-shape
	// semantics a plain build_keys/probe_keys node can't represent). Executes to completion (unlike case
	// 9 below) because the output is naturally bounded: `small` only has 10 rows, `c` only 5 distinct
	// values, so the LEFT JOIN can't produce more than a few tens of rows either way.
	{
		auto result = con.Query("SELECT big.a FROM big LEFT JOIN small ON big.c = small.c LIMIT 100");
		Check(!result->HasError(), "LEFT JOIN (unsupported join_type) falls back to CPU cleanly: " +
		                               (result->HasError() ? result->GetError() : "ok"));
	}

	// --- Case 9: VRAM headroom genuinely blocks offload when the pool doesn't have room ----------------
	// EXPLAIN cannot be used to test this (discovered the hard way -- see docs/EXECUTION_TRACKER.md):
	// DuckDB wraps any EXPLAIN'd query in a LOGICAL_EXPLAIN node, which TableChecker's operator whitelist
	// rejects immediately, before ever reaching the real inner plan -- so EXPLAIN output can NEVER show
	// GPU_EXECUTE regardless of whether the underlying query would actually be offloaded. That is a
	// property of this test technique, not of the VRAM check. Actually executing a query whose estimated
	// output is astronomically large (e.g. a genuine 2M x 2M cross join) is also unsafe: once correctly
	// rejected for GPU offload, DuckDB would actually attempt to run that join on the CPU, which doesn't
	// error quickly -- it hangs for a very long time before finally OOMing (hit this for real while
	// developing this test). Instead, this shrinks the real RmmPool to a size too small for an ordinary,
	// safely-executable query, using the same pool the extension itself calls into (this test links the
	// real cuda_engine, not stubs) -- proving the rejection path fires for a real, boundable query rather
	// than only for a hypothetical one this test can't safely run.
	{
		auto result_before = con.Query("SELECT a * b - c AS val FROM big WHERE a < 190000");
		Check(result_before->HasError(),
		      std::string("sanity: case-1-shaped query offloads with the default-size pool (has error): ") +
		          (result_before->HasError() ? "ok" : "unexpectedly succeeded without offloading"));

		vector_gpu::RmmPool::Instance().ResetForTesting();
		vector_gpu::RmmPool::Instance().EnsureInitializedExactBytesForTesting(1024); // 1 KiB: too small for any real query

		auto result_after = con.Query("SELECT a * b - c AS val FROM big WHERE a < 190000");
		Check(!result_after->HasError(),
		      "same query falls back to CPU and executes correctly once the pool has no headroom: " +
		          (result_after->HasError() ? result_after->GetError() : "ok"));

		// Restore a normal-sized pool in case any test is ever added after this one.
		vector_gpu::RmmPool::Instance().ResetForTesting();
	}

	std::cout << "\n" << (g_checks - g_failures) << "/" << g_checks << " checks passed\n";
	return g_failures == 0 ? 0 : 1;
}
