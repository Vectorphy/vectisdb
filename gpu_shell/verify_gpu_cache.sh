#!/usr/bin/env bash
# verify_gpu_cache.sh — warm-query correctness test for the GPU-resident column cache.
#
# verify_gpu_vs_cpu.sh runs each query in its OWN process, so every query it checks is a COLD one: the
# cache is empty, populated, and thrown away with the process. That is structurally blind to the entire
# reason this cache exists. The bug it missed (session 31) was exactly this shape -- the first query was
# right, the SECOND one silently returned only the last streamed chunk of the table (1,191,168 rows of a
# 20,000,000-row table), because GpuColumnCache::Put was keyed only by (table, column) and each chunk
# overwrote the previous one.
#
# So every case here runs a SEQUENCE of statements through ONE persistent `--serve` process (cache alive
# across them) and diffs it, block for block, against the identical sequence run through `--no-gpu` on an
# identical copy of the database. Mutating statements are therefore checked as naturally as reads: both
# sides see the same writes in the same order.
#
# Two things are asserted per case, because either alone can pass vacuously:
#   1. every result block matches the CPU's -- correctness;
#   2. the log actually shows a cache hit, and the hit served the table's FULL row count -- otherwise a
#      case that never reached the cache would "pass" while testing nothing, and a truncating hit would
#      still be caught even if some future result diff were loosened.
#
# Usage:  bash gpu_shell/verify_gpu_cache.sh [path-to-duckdb_gpu[.exe]]
# See SETUP.md for build instructions.

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_BIN="${1:-$HERE/build/duckdb_gpu.exe}"

if [ ! -x "$GPU_BIN" ] && [ ! -f "$GPU_BIN" ]; then
	echo "error: duckdb_gpu not found at $GPU_BIN (see SETUP.md)" >&2
	exit 1
fi

# A bare projection over a few hundred thousand rows is one row of work per row moved, which the
# amplification heuristic correctly declines in production. Every case here needs it offloaded to test
# anything, so the heuristic is switched off for this suite only -- the same override bench_offload.sh uses.
export VECTOR_GPU_MIN_AMPLIFICATION=0
# Split streaming projections consume DuckDB's already-filtered input and cannot publish a complete table
# into GpuColumnCache. Keep this cache-specific suite on PhysicalGpuExecute, whose scans stage full columns.
export VECTOR_GPU_STREAM_PIPELINE=0

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Row counts chosen to force MULTIPLE streaming chunks, which is the whole point: PhysicalGpuExecute ramps
# batches 65536 -> 131072 -> 262144 -> ... so 300k rows scans as 3 chunks and 600k as 4. A table small
# enough to arrive in one chunk cannot reproduce the bug this file exists to catch.
ROWS_SMALL=300000
ROWS_BIG=600000

FIXTURE="$WORK/fixture.duckdb"
echo "building fixtures in $FIXTURE ..."
"$GPU_BIN" "$FIXTURE" --no-gpu -c "
CREATE TABLE warm    AS SELECT i::BIGINT AS id, (i % 97)::DOUBLE AS a, (i % 13)::DOUBLE AS b, (i % 251)::DOUBLE AS c FROM range($ROWS_SMALL) t(i);
CREATE TABLE warmbig AS SELECT i::BIGINT AS id, (i % 97)::DOUBLE AS a, (i % 13)::DOUBLE AS b, (i % 251)::DOUBLE AS c FROM range($ROWS_BIG) t(i);
CREATE TABLE warmnull AS SELECT i::BIGINT AS id, (i % 97)::DOUBLE AS a, CASE WHEN i % 7 = 0 THEN NULL ELSE (i % 13)::DOUBLE END AS b FROM range($ROWS_SMALL) t(i);
CREATE TABLE warmw   AS SELECT i::BIGINT AS id, (i % 97)::DOUBLE AS a, (i % 13)::DOUBLE AS b FROM range($ROWS_SMALL) t(i);
-- Same BARE table name as warm, different schema, different data. The cache keys entries by table
-- identity and the scanner re-opens the table by that identity, so anything that identifies a table by
-- its bare name alone serves one of these for the other.
-- (No backticks in here: this SQL sits inside a double-quoted shell string, where they would run as
-- command substitution.)
CREATE SCHEMA sch;
CREATE TABLE sch.warm AS SELECT i::BIGINT AS id, (i % 97 + 1000)::DOUBLE AS a, (i % 13)::DOUBLE AS b, (i % 251)::DOUBLE AS c FROM range($ROWS_SMALL) t(i);
" >/dev/null 2>&1 || { echo "fixture creation failed" >&2; exit 1; }

# A SEPARATE database file holding a table with the SAME bare name as the main fixture's `warm`. Attached
# at run time by the attach/ case below, to prove the cache key separates two catalogs and not just two
# schemas. Converted to a Windows path: this binary is a Windows build, and while MSYS rewrites paths on
# the COMMAND LINE it does not touch them inside a SQL string literal -- a bare /tmp/... there is read as
# a UNC network path and fails to open.
ATTACH_DB="$WORK/attached.duckdb"
"$GPU_BIN" "$ATTACH_DB" --no-gpu -c "
CREATE TABLE warm AS SELECT i::BIGINT AS id, (i % 97 + 5000)::DOUBLE AS a, (i % 13)::DOUBLE AS b, (i % 251)::DOUBLE AS c FROM range($ROWS_SMALL) t(i);
" >/dev/null 2>&1 || { echo "attach fixture creation failed" >&2; exit 1; }
ATTACH_DB_SQL="$(cygpath -m "$ATTACH_DB" 2>/dev/null || echo "$ATTACH_DB")"

pass=0
fail=0

#! Splits a --serve transcript into one file per statement block. `--DONE--` is the server's per-block
#! terminator (see main.cpp's serve loop), so block N is everything printed between terminator N-1 and N.
#! Blocks are pre-created empty: a statement that prints nothing (DDL/DML) must still compare as an empty
#! block rather than a missing file.
split_blocks() {
	local src="$1" count="$2" prefix="$3" i
	for ((i = 1; i <= count; i++)); do
		: >"$prefix.$i"
	done
	awk -v prefix="$prefix" 'BEGIN { n = 1 } $0 == "--DONE--" { n++; next } { print >> (prefix "." n) }' "$src"
}

#! Runs a statement sequence through one persistent process and splits the transcript.
#! $1 = database, $2 = output prefix, $3 = extra binary flag ("" or "--no-gpu"), rest = statements.
run_sequence() {
	local db="$1" prefix="$2" flag="$3"
	shift 3
	local input="" stmt
	for stmt in "$@"; do
		input+="$stmt"$'\n'"--END--"$'\n'
	done
	if [ -n "$flag" ]; then
		printf '%s' "$input" | "$GPU_BIN" "$db" "$flag" -json --serve >"$prefix.raw" 2>"$prefix.err"
	else
		printf '%s' "$input" | "$GPU_BIN" "$db" -json --serve >"$prefix.raw" 2>"$prefix.err"
	fi
	split_blocks "$prefix.raw" "$#" "$prefix"
}

#! The suite's one assertion.
#!   $1 name
#!   $2 expected number of cache hits, exactly -- or "-" to leave it unasserted
#!   $3 space-separated row counts a hit is ALLOWED to have served (more than one when the sequence
#!      writes to the table, since the table legitimately has a different size afterwards)
#!   $4 "VAR=value" applied to the GPU run only, or "-" for none
#!   $5.. the statements
#!
#! Runs the statements through a warm GPU process and through a --no-gpu one, on independent copies of the
#! fixture, and requires every result block to agree. The row-count assertion is what turns "a cache hit
#! happened" into "a cache hit served the whole table" -- the precise distinction the session-31 bug got
#! wrong, and one a result diff alone would only catch by luck if the truncated prefix happened to look
#! like a plausible answer. The hit-count assertion is what stops a case that silently stopped reaching
#! the cache from passing while testing nothing.
check_case() {
	local name="$1" expect_hits="$2" allowed_rows="$3" env_override="$4"
	shift 4
	local case_dir="$WORK/case"
	rm -rf "$case_dir"
	mkdir -p "$case_dir/log"
	cp "$FIXTURE" "$case_dir/gpu.duckdb"
	cp "$FIXTURE" "$case_dir/cpu.duckdb"

	# Exported rather than passed through env(1): run_sequence is a shell function, and env(1) can only
	# launch a program. Unset again right after, so a per-case override cannot leak into the next case.
	export VECTOR_GPU_LOG=1 VECTOR_GPU_LOG_LEVEL=info VECTOR_GPU_LOG_PATH="$case_dir/log"
	if [ "$env_override" != "-" ]; then
		export "${env_override?}"
	fi
	run_sequence "$case_dir/gpu.duckdb" "$case_dir/gpu" "" "$@"
	unset VECTOR_GPU_LOG VECTOR_GPU_LOG_LEVEL VECTOR_GPU_LOG_PATH
	if [ "$env_override" != "-" ]; then
		unset "${env_override%%=*}"
	fi
	run_sequence "$case_dir/cpu.duckdb" "$case_dir/cpu" "--no-gpu" "$@"

	local i mismatched=""
	for ((i = 1; i <= $#; i++)); do
		if ! cmp -s "$case_dir/gpu.$i" "$case_dir/cpu.$i"; then
			mismatched="$mismatched $i"
		fi
	done

	# One number per cache hit: the rows that hit actually served.
	local served
	served=$(cat "$case_dir"/log/*.jsonl 2>/dev/null |
		grep -o '"evt":"cache_full_hit"[^}]*' | grep -o '"rows":[0-9]*' | grep -o '[0-9]*$')
	local hits=0 truncated="" r allowed
	for r in $served; do
		hits=$((hits + 1))
		local ok=0
		for allowed in $allowed_rows; do
			[ "$r" = "$allowed" ] && ok=1
		done
		[ "$ok" -eq 0 ] && truncated="$truncated $r"
	done

	local problems=""
	[ -n "$mismatched" ] && problems="$problems result block(s)$mismatched differ from CPU;"
	if [ "$expect_hits" != "-" ] && [ "$hits" -ne "$expect_hits" ]; then
		problems="$problems $hits cache hit(s), expected exactly $expect_hits;"
	fi
	[ -n "$truncated" ] && problems="$problems cache hit served$truncated rows, expected one of [$allowed_rows];"

	if [ -z "$problems" ]; then
		pass=$((pass + 1))
		printf '  ok    %-34s (%d cache hits)\n' "$name" "$hits"
	else
		fail=$((fail + 1))
		printf '  FAIL  %-34s%s\n' "$name" "$problems"
		for i in $mismatched; do
			printf '        block %s: gpu %s bytes, cpu %s bytes\n' "$i" \
				"$(wc -c <"$case_dir/gpu.$i")" "$(wc -c <"$case_dir/cpu.$i")"
		done
	fi
}

PROJ="SELECT a * b + sqrt(c) AS result FROM warm;"
PROJ_BIG="SELECT a * b + sqrt(c) AS result FROM warmbig;"

echo "running warm-query checks (gpu --serve vs --no-gpu --serve) ..."

# THE session-31 regression test. Query 1 populates the cache chunk by chunk; queries 2 and 3 must be
# served from it and must still be the whole table. Before the staging rewrite, query 2 returned only the
# final chunk and query 3 repeated that answer.
check_case "repeat/projection x3" 2 "$ROWS_SMALL" - "$PROJ" "$PROJ" "$PROJ"

# Same, on a table that scans as 4 chunks instead of 3 -- so a fix that happened to work for one specific
# chunk count does not pass by coincidence.
check_case "repeat/projection-4chunk x3" 2 "$ROWS_BIG" - "$PROJ_BIG" "$PROJ_BIG" "$PROJ_BIG"

# FILTER above the SCAN: still row-independent, so still chunked and still cache-eligible, but the rows
# that survive each chunk vary -- a cached chunk served at the wrong offset shows up here as a wrong count.
# The predicate spans two columns deliberately: a single-column comparison is pushed down into the GET as
# a table filter, which leaves no LOGICAL_FILTER in the plan at all and tests nothing here.
FILT="SELECT a * b AS result FROM warm WHERE a + b > 5.0;"
check_case "repeat/filter+projection x3" 2 "$ROWS_SMALL" - "$FILT" "$FILT" "$FILT"

# A nullable column. Validity is uploaded per chunk (a chunk with no NULLs carries none at all), so the
# cache must keep each chunk's validity with that chunk rather than assume one bitmap per column.
NULLQ="SELECT a + b AS result FROM warmnull;"
check_case "repeat/nullable-column x3" 2 "$ROWS_SMALL" - "$NULLQ" "$NULLQ" "$NULLQ"

# Second query needs a SUBSET of the cached columns: must hit, and must not mix up which column is which.
check_case "repeat/column-subset" 1 "$ROWS_SMALL" - "$PROJ" "SELECT a * b AS result FROM warm;"

# Second query needs a column the first did not cache: must MISS and re-scan. Exactly zero hits is the
# assertion -- serving this from a cache that never held column c would be the failure.
check_case "repeat/column-superset" 0 "$ROWS_SMALL" - "SELECT a * b AS result FROM warm;" "$PROJ"

# WHOLE-INPUT operators reading the cache (session 37). A GROUP BY / global aggregate cannot be replayed
# chunk by chunk -- it sorts across its entire input, so per-chunk execution would answer a different
# question -- which is why it needed its own cache mode (serve_whole_scan_from_cache: gather every cached
# chunk into one contiguous device buffer, device-to-device). Until that landed, these plans re-scanned
# and re-uploaded the whole table on EVERY run, measured at 0.04x of the CPU on 70M rows.
#
# Statement 1 populates from a row-independent scan; statements 2 and 3 are the aggregate, and both must
# be served from the cache over the table's FULL row count. Correctness is what the row-count assertion
# is really guarding: a gather that dropped or repeated a chunk would still produce a plausible SUM.
AGG="SELECT SUM(a) AS s FROM warm;"
check_case "whole-input/aggregate-hits-cache" 2 "$ROWS_SMALL" - "$PROJ" "$AGG" "$AGG"

# The aggregate alone must MISS every time and still be correct: this path reads the cache but does not
# populate it (only the streaming path produces bounded-size chunks). Asserted rather than left implicit,
# so the day that gap is closed this case fails and gets updated deliberately.
check_case "whole-input/aggregate-alone-misses" 0 "$ROWS_SMALL" - "$AGG" "$AGG"

# Writes must invalidate. The read AFTER each write has to reflect it -- the cache's whole correctness
# argument (push-based invalidation from the optimizer hook) fails silently if it does not. Each sequence
# is read, read(hit), write, read(repopulate), read(hit): two hits, and for DELETE/INSERT the second one
# legitimately serves the table's NEW size, which is why more than one row count is allowed.
WPROJ="SELECT a * b AS result FROM warmw;"
check_case "invalidate/update" 2 "$ROWS_SMALL" - \
	"$WPROJ" "$WPROJ" "UPDATE warmw SET a = a + 1000 WHERE id % 3 = 0;" "$WPROJ" "$WPROJ"
check_case "invalidate/delete" 2 "$ROWS_SMALL 240000" - \
	"$WPROJ" "$WPROJ" "DELETE FROM warmw WHERE id % 5 = 0;" "$WPROJ" "$WPROJ"
check_case "invalidate/insert" 2 "$ROWS_SMALL 327273" - \
	"$WPROJ" "$WPROJ" "INSERT INTO warmw SELECT id + 1000000, a, b FROM warmw WHERE id % 11 = 0;" \
	"$WPROJ" "$WPROJ"

# An upstream LIMIT stops DuckDB pulling before the scan finishes, leaving a half-staged column set.
# Publishing that would be the session-31 bug with a different cause, so the staging must be ABORTED.
#
# Three things had to line up for this to be a real test rather than a vacuous one, hence the exact shape:
#   - the limit is large (200k of the ~295k rows that pass the filter) so the plan still clears the
#     100k-row offload threshold and genuinely runs on the GPU; a small LIMIT is declined at routing;
#   - it is large enough to span more than one streamed chunk, so the scan really is cut off mid-table;
#   - the filter is present because `SELECT ... FROM warm LIMIT n` without one is a pre-existing routing
#     bug -- DuckDB's plan for that shape carries a `rowid` column the translator passes straight to the
#     scanner, which fails the query outright instead of falling back (reproduces on the pre-cache
#     baseline binary too, so it is not this branch's regression).
# Exactly one hit: the second query has to re-scan (which is what proves nothing was published), and only
# the third can hit.
check_case "abort/early-termination" 1 "$ROWS_SMALL" - \
	"SELECT a * b + sqrt(c) AS result FROM warm WHERE a + b > 5.0 LIMIT 200000;" "$PROJ" "$PROJ"

# Budget too small to hold even one chunk: every StageChunk is declined and the buffers fall back to the
# plan's DeviceArena instead. Answers must stay correct across all three runs -- an ownership mistake on
# that path is a double free, which shows up here as a crashed process and empty result blocks. Hit count
# is left unasserted: it depends on how much VRAM was free when the budget was measured.
check_case "budget/below-one-chunk" - "$ROWS_SMALL" VECTOR_GPU_CACHE_FRACTION=0.00002 \
	"$PROJ" "$PROJ" "$PROJ"

# Two tables competing for a budget that (on a small card) holds only one, so caching warmbig evicts warm
# and the third query has to re-scan it. Exercises LRU eviction of COMMITTED entries, which nothing above
# reaches. Both hit counts and row counts are left loose on purpose: whether eviction actually happens
# depends on the card's free VRAM, and the invariant being tested -- freeing an evicted column must not
# disturb any answer -- holds either way.
check_case "evict/two-tables" - "$ROWS_SMALL $ROWS_BIG" VECTOR_GPU_CACHE_FRACTION=0.004 \
	"$PROJ" "$PROJ_BIG" "$PROJ" "$PROJ_BIG"

# Two tables sharing a bare name in different schemas. Both the cache key and the scanner's catalog lookup
# have to carry the full catalog.schema.table identity; if either falls back to the bare name, one table's
# rows get served for the other's query. Interleaved, because a stale entry left by the previous statement
# is the most likely way for this to fail.
SPROJ="SELECT a * b + sqrt(c) AS result FROM sch.warm;"
check_case "schema/same-name-two-schemas" - "$ROWS_SMALL" - 	"$PROJ" "$SPROJ" "$PROJ" "$SPROJ"

# The search path decides which table a BARE name means. A cache keyed by the bare name -- or a scanner
# that re-resolves the bare name against the CURRENT search path rather than the plan's own identity --
# answers the post-USE queries from the pre-USE table. Both of those were real: the GPU returned
# main.warm's rows here where DuckDB returns sch.warm's.
WPROJ_BARE="SELECT a * b + sqrt(c) AS result FROM warm;"
check_case "schema/search-path-change" - "$ROWS_SMALL" - 	"$PROJ" "$PROJ" "USE sch;" "$WPROJ_BARE" "$WPROJ_BARE"

# Two ATTACHed databases, each with a table called `warm`. Same test as the schema cases one level up: the
# cache key has to separate them by CATALOG too, not just by schema and name. Reads are interleaved so a
# key collision shows up as the previous statement's table being served.
check_case "attach/same-name-two-catalogs" - "$ROWS_SMALL" - 	"ATTACH '$ATTACH_DB_SQL' AS att;" "$PROJ" "SELECT a * b + sqrt(c) AS result FROM att.warm;" 	"$PROJ" "SELECT a * b + sqrt(c) AS result FROM att.warm;"

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ] || exit 1
