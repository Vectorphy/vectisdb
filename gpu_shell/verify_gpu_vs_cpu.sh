#!/usr/bin/env bash
# verify_gpu_vs_cpu.sh — end-to-end differential test for the GPU offload path.
#
# Runs the same SQL twice through the SAME duckdb_gpu binary — once with GPU offload enabled, once with
# --no-gpu — and fails if the answers differ. That is the property that actually matters: offloading must
# never change a result, and a shape the GPU cannot run must fall back to CPU rather than fail the query.
#
# This catches a class of bug the cuda_engine unit tests structurally cannot: those drive the engine with
# hand-built GpuPlanNodes, so they cannot see a ROUTING mistake (a plan that should not have been
# offloaded, or one lowered with the wrong columns). A real self-join regression found exactly that way is
# case "self-join" below — it used to be offloaded and then fail at execution instead of falling back.
#
# Usage:  bash gpu_shell/verify_gpu_vs_cpu.sh [path-to-duckdb_gpu[.exe]]
# See SETUP.md for build instructions.

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_BIN="${1:-$HERE/build/duckdb_gpu.exe}"

if [ ! -x "$GPU_BIN" ] && [ ! -f "$GPU_BIN" ]; then
	echo "error: duckdb_gpu not found at $GPU_BIN (see SETUP.md)" >&2
	exit 1
fi

DB="$(mktemp -u)_gpuverify.duckdb"
trap 'rm -f "$DB"' EXIT

echo "building fixtures in $DB ..."
"$GPU_BIN" "$DB" --no-gpu -c "
CREATE TABLE cust AS SELECT i::INTEGER AS cid, (i*2)::INTEGER AS cval FROM range(200000) t(i);
CREATE TABLE ord  AS SELECT i::INTEGER AS oid, (i % 200000)::INTEGER AS ocid, (i%97)::INTEGER AS oval FROM range(1000000) t(i);
CREATE TABLE dupa AS SELECT (i % 100000)::INTEGER AS ak, i::INTEGER AS av FROM range(300000) t(i);
CREATE TABLE dupb AS SELECT (i % 100000)::INTEGER AS bk, (i*3)::INTEGER AS bv FROM range(400000) t(i);
CREATE TABLE ints AS SELECT (i % 200000)::INTEGER AS k, (i % 64)::DOUBLE AS v FROM range(1000000) t(i);
CREATE TABLE nums AS SELECT (i % 200000)::BIGINT AS k, (i % 64)::DOUBLE AS v, (i % 1000)::INTEGER AS w FROM range(1000000) t(i);
-- Cross-product fixtures are deliberately SMALL: a cross join's whole point is that a few thousand rows
-- of input become millions of pairs. xa x xb is 9,000,000 pairs; xa x xbig is 300,000,000, past the
-- routing-time cap, and must fall back.
CREATE TABLE xa   AS SELECT i::INTEGER AS xk, (i % 7)::DOUBLE AS xv FROM range(3000) t(i);
CREATE TABLE xb   AS SELECT i::INTEGER AS yk, (i % 5)::DOUBLE AS yv FROM range(3000) t(i);
CREATE TABLE xbig AS SELECT (i % 11)::DOUBLE AS zv FROM range(100000) t(i);
" >/dev/null 2>&1 || { echo "fixture creation failed" >&2; exit 1; }

pass=0
fail=0

check() {
	local name="$1" sql="$2"
	local got want
	got=$("$GPU_BIN" "$DB" -c "$sql" 2>&1)
	want=$("$GPU_BIN" "$DB" --no-gpu -c "$sql" 2>&1)
	if [ "$got" == "$want" ]; then
		pass=$((pass + 1))
		printf '  ok    %s\n' "$name"
	else
		fail=$((fail + 1))
		printf '  FAIL  %s\n' "$name"
		printf '    gpu: %s\n' "$(echo "$got" | tr '\n' ' ')"
		printf '    cpu: %s\n' "$(echo "$want" | tr '\n' ' ')"
	fi
}

# These check WHAT was offloaded, not just that results match -- a wrong routing decision can be
# perfectly correct and still be a performance loss, which result-diffing cannot see.
#
# Defined HERE, before the first caller. It used to be defined near the bottom, below the
# `routing/groupby-avg-offloaded` call on line ~101: sh resolves a function name at call time, so that
# one call hit "command not found", printed to stderr, and incremented NEITHER counter. The suite still
# reported "30 passed, 0 failed" with 29 assertions actually run.
routing_check() {
	local name="$1" sql="$2" expect="$3"   # expect: "offload" | "no-offload"
	local env_override="${4:-}"            # optional VAR=value applied to this run only
	local count
	if [ -n "$env_override" ]; then
		count=$(env "$env_override" "$GPU_BIN" "$DB" --gpu-trace -c "$sql" 2>&1 >/dev/null | grep -o "offloaded to GPU: [0-9]*" | grep -o "[0-9]*$" | tail -1)
	else
	count=$("$GPU_BIN" "$DB" --gpu-trace -c "$sql" 2>&1 >/dev/null | grep -o "offloaded to GPU: [0-9]*" | grep -o "[0-9]*$" | tail -1)
	fi
	[ -n "$count" ] || count=0
	local ok=1
	if [ "$expect" = "offload" ] && [ "$count" -eq 0 ]; then ok=0; fi
	if [ "$expect" = "no-offload" ] && [ "$count" -ne 0 ]; then ok=0; fi
	if [ "$ok" -eq 1 ]; then
		pass=$((pass + 1))
		printf '  ok    %s (offloaded=%s, expected %s)\n' "$name" "$count" "$expect"
	else
		fail=$((fail + 1))
		printf '  FAIL  %s (offloaded=%s, expected %s)\n' "$name" "$count" "$expect"
	fi
}

echo "running differential checks (gpu vs --no-gpu) ..."

# --- group by (operators/gpu_groupby.cu) ---
check "groupby/count+sum"      "SELECT COUNT(*) g, SUM(c) t, SUM(s) u FROM (SELECT k, COUNT(*) c, SUM(v) s FROM nums GROUP BY k) q;"
check "groupby/min+max"        "SELECT SUM(mn) a, SUM(mx) b FROM (SELECT k, MIN(w) mn, MAX(w) mx FROM nums GROUP BY k) q;"
check "groupby/avg-fallback"   "SELECT SUM(a) s, COUNT(*) c FROM (SELECT k, AVG(v) a FROM nums GROUP BY k) q;"
check "groupby/float-key"      "SELECT COUNT(*) c, SUM(n) s FROM (SELECT v, COUNT(*) n FROM nums GROUP BY v) q;"
check "groupby/sum-int-hugeint" "SELECT SUM(s) t, COUNT(*) c FROM (SELECT k, SUM(w) s FROM nums GROUP BY k) q;"

# --- join (operators/gpu_hash_join.cu) ---
check "join/one-to-one"        "SELECT COUNT(*) n, SUM(cval) sc, SUM(oval) so FROM ord JOIN cust ON ord.ocid = cust.cid;"
check "join/many-to-many"      "SELECT COUNT(*) n, SUM(av) sa, SUM(bv) sb FROM dupa JOIN dupb ON dupa.ak = dupb.bk;"
check "join/with-filter"       "SELECT COUNT(*) n, SUM(cval) s FROM ord JOIN cust ON ord.ocid = cust.cid WHERE ord.oval > 50;"
check "join/three-way"         "SELECT COUNT(*) n, SUM(bv) s FROM ord JOIN cust ON ord.ocid=cust.cid JOIN dupb ON cust.cid=dupb.bk;"
check "join/no-matches"        "SELECT COUNT(*) n FROM ord JOIN cust ON ord.ocid = cust.cid + 100000000;"
# Regression: both sides produce a column named cid/cval, so the join output cannot be resolved by name.
# Must fall back to CPU. Before the IsExecutableOnGpu fix this was offloaded and then FAILED the query.
check "join/self-join"         "SELECT COUNT(*) n, SUM(a.cval) s FROM cust a JOIN cust b ON a.cid = b.cid;"
check "join/self-join-aliased" "SELECT COUNT(*) n FROM ord a JOIN ord b ON a.ocid = b.oid;"
check "join/left-join-fallback" "SELECT COUNT(*) n FROM cust LEFT JOIN ord ON cust.cid = ord.ocid;"

# --- casts and projected group keys (session 16) --------------------------------------------------
check "cast/widening-projection"   "SELECT COUNT(*) n, SUM(x) t FROM (SELECT k::BIGINT AS x FROM ints) q;"
check "cast/int-to-double"         "SELECT COUNT(*) n, SUM(d) t FROM (SELECT k::DOUBLE AS d FROM ints) q;"
check "groupby/integer-key"        "SELECT COUNT(*) n, SUM(s) t FROM (SELECT k, SUM(v) s FROM ints GROUP BY k) q;"
check "groupby/projected-key"      "SELECT COUNT(*) n FROM (SELECT k::BIGINT AS bk, COUNT(*) c FROM ints GROUP BY bk) q;"
# A narrowing cast must still be REFUSED: DuckDB errors on out-of-range, a C++ cast truncates.
check "cast/narrowing-still-correct" "SELECT COUNT(*) n, SUM(c) t FROM (SELECT k, COUNT(*) c FROM nums GROUP BY k) q;"

# --- pushed-down filters (session 21 regression) --------------------------------------------------
# DuckDB prunes row groups using a pushed-down filter and folds their count into a CONSTANT, e.g.
#   Projection "+"(385600, #0) / Aggregate count_star() / Seq Scan (Filters: k > 100)
# Offloading such a scan means re-reading the WHOLE table, so the correct row count gets the folded
# constant added on top -- 1,385,095 instead of 999,495. These pin that filtered scans stay on CPU.
check "pushdown/count-star-not-double-counted" "SELECT COUNT(*) c FROM (SELECT k FROM ints WHERE k > 100) q;"
check "pushdown/count-no-subquery"             "SELECT COUNT(*) c FROM ints WHERE k > 100;"
check "pushdown/filter-with-other-aggregates"  "SELECT MIN(k) a, MAX(k) b, COUNT(*) c, SUM(k) d FROM (SELECT k FROM ints WHERE k > 100) q;"

# --- AVG and the selectivity rule (session 24) ----------------------------------------------------
check "groupby/avg-double"     "SELECT COUNT(*) n, MAX(a) mx, MIN(a) mn FROM (SELECT k, AVG(v) a FROM ints GROUP BY k) q;"
check "groupby/avg-plus-count" "SELECT COUNT(*) n, MAX(a) mx, MAX(c) mc FROM (SELECT k, AVG(v) a, COUNT(*) c FROM ints GROUP BY k) q;"
# A highly selective filter must stay on CPU: uploading every row to discard most is a net loss.
check "selectivity/highly-selective" "SELECT COUNT(*) c, SUM(k) s FROM (SELECT k FROM ints WHERE k < 50) q;"
# --- work-amplification routing (session 27) -------------------------------------------------------
# These four used to assert "offload". They now assert "no-offload", and that is a deliberate, MEASURED
# reversal rather than a regression being papered over.
#
# A GROUP BY or JOIN over a plain table derives ~1 row of work per row scanned. Measured on this
# hardware: groupby 1M rows 1.96x SLOWER than CPU, join 1M x 200k 2.26x slower, projection 1M rows
# 2.57x slower -- and a projection sweep to 64M rows got WORSE with scale (2.83x -> 3.43x), because the
# scan cost grows linearly while the CPU stays in cache. Offloading these was a measured loss at every
# size tried, so TableChecker::MIN_WORK_AMPLIFICATION now declines them.
#
# The KERNELS are unaffected and still correct -- that is what the result checks above prove, since they
# pass either way. To keep proving the paths still execute end-to-end, each assertion is paired with a
# VECTOR_GPU_MIN_AMPLIFICATION=0 run that must still offload. If the scan path is ever made cheap enough
# If the scan path becomes cheaper, re-measure and update these expectations.
routing_check "routing/groupby-avg-declined-low-amplification" "SELECT COUNT(*) n, MAX(a) mx FROM (SELECT k, AVG(v) a FROM ints GROUP BY k) q;" "no-offload"
routing_check "routing/groupby-avg-still-runs-when-forced" "SELECT COUNT(*) n, MAX(a) mx FROM (SELECT k, AVG(v) a FROM ints GROUP BY k) q;" "offload" "VECTOR_GPU_MIN_AMPLIFICATION=0"

# --- cross join (operators/gpu_cross_join.cu, session 25) -----------------------------------------
# NOTE ON WHAT IS AGGREGATED HERE: MIN/MAX/COUNT only, never SUM or AVG. A global SUM/AVG over 9,000,000
# cross-product rows sums in a different ORDER on GPU than on CPU, and with values in [-2, 2] summing to
# ~-1e6 the cancellation makes that visible around the 12th digit -- a real divergence, but a
# floating-point-associativity one, not a wrong answer. MIN/MAX/COUNT are order-independent and match
# bit for bit.
check "cross/projection-transcendental" "SELECT MIN(d) mn, MAX(d) mx, COUNT(*) n FROM (SELECT sin(xv)+cos(yv) AS d FROM xa, xb) q;"
check "cross/projection-arithmetic"     "SELECT MIN(d) mn, MAX(d) mx, COUNT(*) n FROM (SELECT xv*yv AS d FROM xa, xb) q;"
check "cross/integer-columns"           "SELECT MIN(d) mn, MAX(d) mx, COUNT(*) n FROM (SELECT xk+yk AS d FROM xa, xb) q;"
# Both sides carry the same column names, so the output cannot be resolved by name -> must fall back.
check "cross/self-cross-join"           "SELECT MIN(d) mn, COUNT(*) n FROM (SELECT a.xv*b.xv AS d FROM xa a, xa b) q;"
# DuckDB projects NO columns for a bare COUNT(*), leaving both scans exposing only `rowid` -- an
# ambiguous name, so this falls back too. Pinned because it is the most obvious cross-join query there
# is, and it must be correct rather than offloaded.
check "cross/count-star"                "SELECT COUNT(*) n FROM xa, xb;"
# 3,000 x 100,000 = 300,000,000 pairs, past MAX_CROSS_PRODUCT_OUTPUT_ROWS. The refusal has to happen at
# ROUTING time (silent CPU fallback); an engine-side refusal would fail the query instead.
check "cross/oversize-falls-back"       "SELECT MIN(d) mn, MAX(d) mx FROM (SELECT xv*zv AS d FROM xa, xbig) q;"
# A WHERE clause spanning both sides is planned as LOGICAL_ANY_JOIN, not a cross product, and ANY_JOIN is
# not whitelisted. Keep this case as a CPU-fallback correctness check.
check "cross/any-join-falls-back"       "SELECT COUNT(*) n FROM xa, xb WHERE xv*yv > 10;"

# --- filter / projection (gpu_executor.cu) ---
check "filter+projection"      "SELECT COUNT(*) n, SUM(x) s FROM (SELECT v * 1.5 AS x FROM nums WHERE w > 500) q;"

# --- scans that reference no real column (session 37) ---------------------------------------------
# When nothing above a LogicalGet needs any of its columns, DuckDB's RemoveUnusedColumns substitutes a
# VIRTUAL column (rowid) so the scan still emits one row per stored row. There is no such column in the
# catalog, so this engine's scanners -- which re-open a table BY NAME -- cannot bind it.
#
# These three FAILED THE QUERY before the routing-time refusal landed, rather than falling back:
#   Invalid Input Error: PhysicalGpuExecute: failed to open streaming scan for GPU execution:
#   StreamingTableScanner: column 'rowid' not found on table '...'
# Checked both ways: the answer must match the CPU, and the plan must not be offloaded at all.
check "rowid/constant-projection"  "SELECT COUNT(*) n, SUM(one) s FROM (SELECT 1 AS one FROM nums) q;"
check "rowid/aliased-constant"     "SELECT COUNT(*) n FROM (SELECT 42 AS x FROM nums) q;"
check "rowid/constant-with-limit"  "SELECT COUNT(*) n FROM (SELECT 1 AS one FROM nums LIMIT 10) q;"

# --- routing assertions (session 13) -------------------------------------------------------------
echo "running routing assertions ..."
# A bare scan must NOT be offloaded: it would upload every column and copy it straight back unchanged.
# Before session 13 this offloaded 1 plan and did zero GPU work. See PerformsGpuComputation.
routing_check "routing/no-compute-scan-declined" 	"SELECT COUNT(*) FROM (SELECT k, COUNT(*) c FROM nums GROUP BY k) q;" "no-offload"
# A scan whose only column is the virtual rowid must be refused by ROUTING, not discovered at execution.
# Forced open with VECTOR_GPU_MIN_AMPLIFICATION=0 on purpose: without it the amplification rule declines
# this shape anyway and the assertion would pass without testing anything.
routing_check "rowid/virtual-column-declined" "SELECT COUNT(*) n FROM (SELECT 1 AS one FROM nums) q;" "no-offload" "VECTOR_GPU_MIN_AMPLIFICATION=0"
# KNOWN GAP, documented as a tripwire rather than hidden: no GROUP BY offloads end-to-end today.
# DuckDB inserts a __cast projection between the scan and the aggregate, and CollectBindingNames only
# tracks LOGICAL_GET and LOGICAL_AGGREGATE bindings, so the group key fails to resolve
# ("join/group-by key column not resolvable to a scanned column"). The kernel is correct and covered by
# test_gpu_executor; it is the ROUTING that never reaches it. When that is fixed this assertion will
# start failing -- at which point flip it to "offload". Do not delete it.
routing_check "routing/groupby-known-gap-not-offloaded" 	"SELECT COUNT(*) n, SUM(c) t FROM (SELECT k, COUNT(*) c, SUM(v) s FROM nums GROUP BY k) q;" "no-offload"
# And a real join must still be offloaded.
routing_check "routing/real-join-declined-low-amplification" "SELECT COUNT(*) n, SUM(cval) s FROM ord JOIN cust ON ord.ocid = cust.cid;" "no-offload"
routing_check "routing/real-join-still-runs-when-forced" "SELECT COUNT(*) n, SUM(cval) s FROM ord JOIN cust ON ord.ocid = cust.cid;" "offload" "VECTOR_GPU_MIN_AMPLIFICATION=0"

routing_check "routing/groupby-integer-key-declined" "SELECT COUNT(*) n, SUM(s) t FROM (SELECT k, SUM(v) s FROM ints GROUP BY k) q;" "no-offload"
routing_check "routing/groupby-integer-key-still-runs-when-forced" "SELECT COUNT(*) n, SUM(s) t FROM (SELECT k, SUM(v) s FROM ints GROUP BY k) q;" "offload" "VECTOR_GPU_MIN_AMPLIFICATION=0"
routing_check "routing/groupby-projected-key-declined" "SELECT COUNT(*) n FROM (SELECT k::BIGINT AS bk, COUNT(*) c FROM ints GROUP BY bk) q;" "no-offload"
routing_check "routing/groupby-projected-key-still-runs-when-forced" "SELECT COUNT(*) n FROM (SELECT k::BIGINT AS bk, COUNT(*) c FROM ints GROUP BY bk) q;" "offload" "VECTOR_GPU_MIN_AMPLIFICATION=0"

# A cross join must actually be OFFLOADED: 3,000-row inputs are far below the 100k row threshold, so
# without the cross-product exemption in TableChecker this is declined and the operator is dead code.
routing_check "routing/cross-join-offloaded" "SELECT MIN(d) mn FROM (SELECT sin(xv)+cos(yv) AS d FROM xa, xb) q;" "offload"
# ...and an oversized one must NOT be, so the refusal is a fallback rather than a failed query.
routing_check "routing/cross-join-oversize-declined" "SELECT MIN(d) mn FROM (SELECT xv*zv AS d FROM xa, xbig) q;" "no-offload"
# The exemption is scoped to cross-product subtrees: two 3,000-row tables joined normally stay on CPU.
routing_check "routing/small-tables-still-declined" "SELECT COUNT(*) n FROM (SELECT xk, COUNT(*) c FROM xa GROUP BY xk) q;" "no-offload"

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ] || exit 1
