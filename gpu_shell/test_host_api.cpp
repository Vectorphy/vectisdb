// test_host_api.cpp — proves the in-process API does what the subprocess design did, in one process.
//
// The three properties that matter, none of which can be assumed:
//   1. Two Sessions in ONE process are isolated: a query in A cannot name a table in B.
//   2. Per-query stats are attributable, even though the underlying counters are process-global.
//   3. A GPU-eligible query still offloads through this path, and a non-amplifying one still declines.
//
// It also measures the thing the whole exercise is for: the second query in a process should not pay the
// CUDA/NVRTC startup the first one did.

#include "vector_gpu_host.hpp"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using vector_gpu::QueryResult;
using vector_gpu::Session;

namespace {

int failures = 0;

void Check(const char *label, bool ok, const std::string &detail = "") {
	std::printf("  %s  %s%s%s\n", ok ? "PASS" : "FAIL", label, detail.empty() ? "" : "   -> ",
	            detail.c_str());
	if (!ok) {
		failures++;
	}
}

std::string FirstCell(const QueryResult &result) {
	if (!result.success || result.rows.empty() || result.rows[0].empty()) {
		return "<none>";
	}
	return result.rows[0][0];
}

} // namespace

int main() {
	std::printf("== two isolated Sessions in ONE process ==\n");
	Session a("", /* enable_gpu = */ true);   // in-memory
	Session b("", /* enable_gpu = */ true);

	a.Query("CREATE TABLE alpha AS SELECT 111 AS v;");
	b.Query("CREATE TABLE beta  AS SELECT 222 AS v;");

	auto a_own = a.Query("SELECT v FROM alpha;");
	Check("session A reads its own table", a_own.success && FirstCell(a_own) == "111", FirstCell(a_own));

	auto a_cross = a.Query("SELECT v FROM beta;");
	Check("session A CANNOT see session B's table", !a_cross.success,
	      a_cross.success ? "LEAKED: " + FirstCell(a_cross) : "refused as expected");

	auto b_own = b.Query("SELECT v FROM beta;");
	Check("session B reads its own table", b_own.success && FirstCell(b_own) == "222", FirstCell(b_own));

	std::printf("\n== a GPU-eligible plan still offloads through the in-process path ==\n");
	Session gpu("", true);
	gpu.Warmup();
	gpu.Query("CREATE TABLE ta AS SELECT (i%64)::DOUBLE AS av FROM range(4472) t(i);");
	gpu.Query("CREATE TABLE tb AS SELECT (i%32)::DOUBLE AS bv FROM range(4472) t(i);");

	const char *cross =
	    "SELECT AVG(d) a FROM (SELECT sin(av)*cos(bv)+sqrt(av*bv+1) AS d FROM ta CROSS JOIN tb) q;";

	auto first = gpu.Query(cross);
	Check("cross-product query succeeded", first.success, first.error);
	Check("cross-product query OFFLOADED", first.stats.offloaded_plans > 0,
	      "offloaded_plans=" + std::to_string(first.stats.offloaded_plans));
	std::printf("       first run : wall %llu ms, scan %llu us, execute %llu us, optimize %llu us\n",
	            (unsigned long long)(first.stats.wall_us / 1000), (unsigned long long)first.stats.scan_us,
	            (unsigned long long)first.stats.execute_us, (unsigned long long)first.stats.optimize_us);

	auto second = gpu.Query(cross);
	std::printf("       second run: wall %llu ms, scan %llu us, execute %llu us, optimize %llu us\n",
	            (unsigned long long)(second.stats.wall_us / 1000),
	            (unsigned long long)second.stats.scan_us, (unsigned long long)second.stats.execute_us,
	            (unsigned long long)second.stats.optimize_us);
	Check("second run reuses the warm context (not slower than the first)",
	      second.stats.wall_us <= first.stats.wall_us,
	      std::to_string(first.stats.wall_us / 1000) + " ms -> " +
	          std::to_string(second.stats.wall_us / 1000) + " ms");

	std::printf("\n== stats are per-query, not cumulative ==\n");
	auto trivial = gpu.Query("SELECT 1;");
	Check("a trivial query reports zero offloads", trivial.stats.offloaded_plans == 0,
	      "offloaded_plans=" + std::to_string(trivial.stats.offloaded_plans));

	std::printf("\n== routing follows expression density, not row amplification ==\n");
	gpu.Query("CREATE TABLE flat AS SELECT (i%64)::DOUBLE AS v FROM range(2000000) t(i);");

	// One addition per row: 1.0 weighted op per scanned row, under kMinOpsPerScannedRow. Measured at
	// 0.75-0.85x of DuckDB's CPU path even with a warm GPU-resident cache (session 32), so it must
	// decline.
	auto cheap = gpu.Query("SELECT v + v AS r FROM flat;");
	Check("2M-row cheap projection declines (1 op/row)", cheap.stats.offloaded_plans == 0,
	      "offloaded_plans=" + std::to_string(cheap.stats.offloaded_plans));

	// ~132 ops per row over the SAME 2M rows. Its row amplification is 1.0, identical to the query above
	// -- which is precisely what the old metric could not see, and why every heavy-math projection used
	// to need VECTOR_GPU_MIN_AMPLIFICATION=0 to route at all. Nothing sets that variable in this process.
	auto heavy = gpu.Query("SELECT sin(v) + cos(v) + sqrt(v) + ln(abs(v) + 1) + exp(v / 100) AS r FROM flat;");
	Check("2M-row heavy-math projection offloads with no env override", heavy.stats.offloaded_plans > 0,
	      "offloaded_plans=" + std::to_string(heavy.stats.offloaded_plans));

	std::printf("\n== a GPU-disabled Session is a true A/B ==\n");
	Session cpu("", /* enable_gpu = */ false);
	cpu.Query("CREATE TABLE ta AS SELECT (i%64)::DOUBLE AS av FROM range(4472) t(i);");
	cpu.Query("CREATE TABLE tb AS SELECT (i%32)::DOUBLE AS bv FROM range(4472) t(i);");
	auto cpu_result = cpu.Query(cross);
	Check("optimizer absent -> never offloads", cpu_result.success && cpu_result.stats.offloaded_plans == 0,
	      "offloaded_plans=" + std::to_string(cpu_result.stats.offloaded_plans));

	std::printf("\n%s\n", failures == 0 ? "ALL CHECKS PASSED" : (std::to_string(failures) + " FAILED").c_str());
	return failures == 0 ? 0 : 1;
}
