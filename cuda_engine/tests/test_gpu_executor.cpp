#include "test_framework.hpp"
#include "gpu_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace vector_gpu;

// Proves GpuEngine::ExecutePlan genuinely EXECUTES a plan on the GPU — the thing that used to be a
// stub returning success=false ("kernel dispatch not wired up"). Every case below runs real NVRTC
// kernels and Thrust primitives on the device and checks the results against a CPU reference.
// Requires a real GPU.

namespace {

//! Builds a host-side GpuColumn over `values`, keeping the backing buffer alive in `owned`.
template <typename T>
GpuColumn MakeColumn(const std::string &name, GpuValueType type, const std::vector<T> &values,
                     std::vector<std::vector<uint8_t>> &owned) {
	owned.emplace_back(values.size() * sizeof(T));
	if (!values.empty()) {
		std::memcpy(owned.back().data(), values.data(), values.size() * sizeof(T));
	}
	GpuColumn column;
	column.name = name;
	column.type = type;
	column.data = owned.back().empty() ? nullptr : owned.back().data();
	column.row_count = values.size();
	return column;
}

std::shared_ptr<GpuPlanNode> MakeScan(const std::vector<std::string> &columns) {
	auto scan = std::make_shared<GpuPlanNode>();
	scan->op_type = GpuOpType::SCAN;
	scan->table = {"memory", "main", "t"};
	scan->column_names = columns;
	return scan;
}

//! NOTE ON THE KERNEL CONVENTION: expression bodies read their INPUTS at [row] and write their OUTPUT at
//! [idx]. Without a selection vector the generated kernel sets row = idx, so the two are the same; with
//! one (late materialization, produced by FILTER) row is selection[idx], i.e. the original base row,
//! while idx is the dense output slot. Bodies here mirror what expression_translator.cpp emits -- using
//! [idx] for an input silently reads the wrong rows once a filter is upstream.
GpuExpr MakeExpr(const std::string &body, const std::vector<std::string> &inputs, GpuValueType out_type) {
	GpuExpr expr;
	expr.generated_cuda_source = body;
	expr.input_columns = inputs;
	expr.output_type = out_type;
	return expr;
}

template <typename T>
std::vector<T> ReadColumn(const GpuColumn &column) {
	std::vector<T> values(column.row_count);
	if (column.row_count > 0) {
		std::memcpy(values.data(), column.data, column.row_count * sizeof(T));
	}
	return values;
}

//! Unpacks a GpuColumn's validity bitmap into one bool per row (matching the struct's own documented
//! convention: LSB-first, byte = row/8, bit set = valid). A null `validity` means every row is valid.
std::vector<bool> ReadValidity(const GpuColumn &column) {
	std::vector<bool> valid(column.row_count, true);
	if (column.validity == nullptr) {
		return valid;
	}
	for (uint64_t r = 0; r < column.row_count; r++) {
		valid[r] = (column.validity[r / 8] & (uint8_t(1) << (r % 8))) != 0;
	}
	return valid;
}

//! Sets bit `row` clear (NULL) in a freshly-allocated all-valid bitmap.
std::vector<uint8_t> AllValidExcept(size_t n, const std::vector<size_t> &null_rows) {
	std::vector<uint8_t> validity((n + 7) / 8, 0xFF);
	for (auto row : null_rows) {
		validity[row / 8] &= ~(uint8_t(1) << (row % 8));
	}
	return validity;
}

GpuAggregate MakeAggregate(GpuAggregateKind kind, const std::string &input_column, GpuValueType output_type) {
	GpuAggregate aggregate;
	aggregate.kind = kind;
	aggregate.input_column = input_column;
	aggregate.output_type = output_type;
	return aggregate;
}

std::shared_ptr<GpuPlanNode> MakeGroupBy(const std::vector<std::string> &group_keys,
                                         const std::vector<GpuAggregate> &aggregates,
                                         std::shared_ptr<GpuPlanNode> child) {
	auto node = std::make_shared<GpuPlanNode>();
	node->op_type = GpuOpType::GROUP_BY_AGGREGATE;
	node->group_keys = group_keys;
	node->aggregates = aggregates;
	node->children.push_back(std::move(child));
	return node;
}

//! Group output order is an implementation detail (it follows the sorted key permutation), so every
//! group-by assertion below checks the group SET, never a positional order. This also catches a
//! duplicated group, which a broken boundary computation would produce.
template <typename K>
bool KeysAreUnique(const std::vector<K> &keys) {
	std::set<K> distinct(keys.begin(), keys.end());
	return distinct.size() == keys.size();
}

} // namespace

static void TestProjectionRunsOnGpu() {
	// SELECT revenue * 1.18 FROM t  -> one fused NVRTC kernel over 50k rows.
	const size_t n = 50000;
	std::vector<double> revenue(n);
	for (size_t i = 0; i < n; i++) {
		revenue[i] = static_cast<double>(i % 997) + 0.5;
	}
	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("revenue", GpuValueType::FLOAT64, revenue, owned)};

	auto projection = std::make_shared<GpuPlanNode>();
	projection->op_type = GpuOpType::PROJECTION;
	projection->expressions.push_back(MakeExpr(
	    "((double*)out)[idx] = (((const double*)inputs[0])[row]) * 1.18;", {"revenue"}, GpuValueType::FLOAT64));
	projection->children.push_back(MakeScan({"revenue"}));

	auto result = GpuEngine::Instance().ExecutePlan(*projection, inputs);
	CHECK(result.success == true); // was ALWAYS false before the executor existed
	CHECK(result.error_message.empty());
	CHECK(result.columns.size() == 1);
	if (result.columns.size() != 1 || !result.success) {
		return;
	}
	CHECK(result.columns[0].row_count == n);
	auto got = ReadColumn<double>(result.columns[0]);
	bool all_match = got.size() == n;
	for (size_t i = 0; i < got.size() && all_match; i++) {
		all_match = std::fabs(got[i] - revenue[i] * 1.18) < 1e-9;
	}
	CHECK(all_match);
}

static void TestFilterCompactsOnGpu() {
	// SELECT id FROM t WHERE id > 100  -> predicate kernel + Thrust compaction, all on device.
	const size_t n = 20000;
	std::vector<int32_t> id(n);
	for (size_t i = 0; i < n; i++) {
		id[i] = static_cast<int32_t>(i % 500);
	}
	std::vector<int32_t> expected;
	for (auto v : id) {
		if (v > 100) {
			expected.push_back(v);
		}
	}

	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("id", GpuValueType::INT32, id, owned)};

	auto filter = std::make_shared<GpuPlanNode>();
	filter->op_type = GpuOpType::FILTER;
	filter->expressions.push_back(MakeExpr("((bool*)out)[idx] = (((const int*)inputs[0])[row]) > 100;", {"id"},
	                                       GpuValueType::BOOLEAN));
	filter->children.push_back(MakeScan({"id"}));

	auto result = GpuEngine::Instance().ExecutePlan(*filter, inputs);
	CHECK(result.success == true);
	CHECK(result.columns.size() == 1);
	if (!result.success || result.columns.size() != 1) {
		return;
	}
	CHECK(result.columns[0].row_count == expected.size());
	auto got = ReadColumn<int32_t>(result.columns[0]);
	CHECK(got == expected);
}

static void TestFilterThenProjectionPipeline() {
	// SELECT qty * 2 FROM t WHERE qty > 3 -> two operators, intermediate stays in device memory.
	const size_t n = 12000;
	std::vector<int32_t> qty(n);
	for (size_t i = 0; i < n; i++) {
		qty[i] = static_cast<int32_t>(i % 10);
	}
	std::vector<int32_t> expected;
	for (auto v : qty) {
		if (v > 3) {
			expected.push_back(v * 2);
		}
	}

	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("qty", GpuValueType::INT32, qty, owned)};

	auto filter = std::make_shared<GpuPlanNode>();
	filter->op_type = GpuOpType::FILTER;
	filter->expressions.push_back(
	    MakeExpr("((bool*)out)[idx] = (((const int*)inputs[0])[row]) > 3;", {"qty"}, GpuValueType::BOOLEAN));
	filter->children.push_back(MakeScan({"qty"}));

	auto projection = std::make_shared<GpuPlanNode>();
	projection->op_type = GpuOpType::PROJECTION;
	projection->expressions.push_back(
	    MakeExpr("((int*)out)[idx] = (((const int*)inputs[0])[row]) * 2;", {"qty"}, GpuValueType::INT32));
	projection->children.push_back(filter);

	auto result = GpuEngine::Instance().ExecutePlan(*projection, inputs);
	CHECK(result.success == true);
	if (!result.success || result.columns.size() != 1) {
		CHECK(false);
		return;
	}
	auto got = ReadColumn<int32_t>(result.columns[0]);
	CHECK(got == expected);
}

static void TestMultiColumnFilterGathersEveryColumn() {
	// Two carried columns must stay row-aligned through compaction.
	const size_t n = 5000;
	std::vector<int32_t> a(n);
	std::vector<double> b(n);
	for (size_t i = 0; i < n; i++) {
		a[i] = static_cast<int32_t>(i % 7);
		b[i] = static_cast<double>(i);
	}
	std::vector<int32_t> exp_a;
	std::vector<double> exp_b;
	for (size_t i = 0; i < n; i++) {
		if (a[i] > 4) {
			exp_a.push_back(a[i]);
			exp_b.push_back(b[i]);
		}
	}

	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("a", GpuValueType::INT32, a, owned),
	                               MakeColumn("b", GpuValueType::FLOAT64, b, owned)};

	auto filter = std::make_shared<GpuPlanNode>();
	filter->op_type = GpuOpType::FILTER;
	filter->expressions.push_back(
	    MakeExpr("((bool*)out)[idx] = (((const int*)inputs[0])[row]) > 4;", {"a"}, GpuValueType::BOOLEAN));
	filter->children.push_back(MakeScan({"a", "b"}));

	auto result = GpuEngine::Instance().ExecutePlan(*filter, inputs);
	CHECK(result.success == true);
	if (!result.success || result.columns.size() != 2) {
		CHECK(false);
		return;
	}
	CHECK(ReadColumn<int32_t>(result.columns[0]) == exp_a);
	CHECK(ReadColumn<double>(result.columns[1]) == exp_b);
}

static void TestFilterMatchingNothingProducesEmptyResult() {
	const size_t n = 4096;
	std::vector<int32_t> id(n, 1);
	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("id", GpuValueType::INT32, id, owned)};

	auto filter = std::make_shared<GpuPlanNode>();
	filter->op_type = GpuOpType::FILTER;
	filter->expressions.push_back(MakeExpr("((bool*)out)[idx] = (((const int*)inputs[0])[row]) > 999;", {"id"},
	                                       GpuValueType::BOOLEAN));
	filter->children.push_back(MakeScan({"id"}));

	auto result = GpuEngine::Instance().ExecutePlan(*filter, inputs);
	CHECK(result.success == true); // an empty result is success, not failure
	CHECK(result.columns.size() == 1);
	if (result.columns.size() == 1) {
		CHECK(result.columns[0].row_count == 0);
	}
}

static void TestNullableColumnPassesThroughBareScan() {
	// A bare SCAN is row-independent (see IsRowIndependent), so session 30 made ExecuteScan unpack a
	// nullable column's validity bitmap and carry it rather than refuse the plan outright. Only
	// GROUP_BY_AGGREGATE/HASH_JOIN/CROSS_PRODUCT still refuse -- see
	// TestNullableColumnRefusedForGroupByAggregate below, which is what this test used to assert for
	// EVERY operator before that change landed.
	const size_t n = 256;
	std::vector<int32_t> id(n);
	for (size_t i = 0; i < n; i++) {
		id[i] = static_cast<int32_t>(i);
	}
	std::vector<std::vector<uint8_t>> owned;
	auto column = MakeColumn("id", GpuValueType::INT32, id, owned);
	auto validity = AllValidExcept(n, {0}); // row 0 NULL
	column.validity = validity.data();
	std::vector<GpuColumn> inputs {column};

	auto scan = MakeScan({"id"});
	auto result = GpuEngine::Instance().ExecutePlan(*scan, inputs);
	CHECK(result.success == true);
	CHECK(result.error_message.empty());
	CHECK(result.columns.size() == 1);
	if (!result.success || result.columns.size() != 1) {
		return;
	}
	CHECK(result.columns[0].row_count == n);
	auto valid = ReadValidity(result.columns[0]);
	bool valid_matches = valid.size() == n;
	for (size_t i = 0; i < n && valid_matches; i++) {
		valid_matches = valid[i] == (i != 0);
	}
	CHECK(valid_matches);
	// A SCAN does not touch data, only carries validity alongside it -- every row's value round-trips
	// regardless of nullability.
	CHECK(ReadColumn<int32_t>(result.columns[0]) == id);
}

static void TestNullableColumnPropagatesThroughProjection() {
	// SELECT v + 1 FROM t -- an output row is NULL iff any referenced input was NULL at that row
	// (EvaluateExpression's `any_nullable` path; ordinary SQL propagation for a scalar expression).
	const size_t n = 128;
	std::vector<int32_t> v(n);
	for (size_t i = 0; i < n; i++) {
		v[i] = static_cast<int32_t>(i);
	}
	std::vector<std::vector<uint8_t>> owned;
	auto column = MakeColumn("v", GpuValueType::INT32, v, owned);
	auto validity = AllValidExcept(n, {3, 100}); // two NULLs, not adjacent, not at a byte boundary
	column.validity = validity.data();
	std::vector<GpuColumn> inputs {column};

	auto projection = std::make_shared<GpuPlanNode>();
	projection->op_type = GpuOpType::PROJECTION;
	projection->expressions.push_back(
	    MakeExpr("((int*)out)[idx] = (((const int*)inputs[0])[row]) + 1;", {"v"}, GpuValueType::INT32));
	projection->children.push_back(MakeScan({"v"}));

	auto result = GpuEngine::Instance().ExecutePlan(*projection, inputs);
	CHECK(result.success == true);
	CHECK(result.error_message.empty());
	CHECK(result.columns.size() == 1);
	if (!result.success || result.columns.size() != 1) {
		return;
	}
	CHECK(result.columns[0].row_count == n);
	auto valid = ReadValidity(result.columns[0]);
	bool valid_matches = valid.size() == n;
	for (size_t i = 0; i < n && valid_matches; i++) {
		valid_matches = valid[i] == (i != 3 && i != 100);
	}
	CHECK(valid_matches);
	// Every VALID row's computed value must still be correct -- nullability must not have disturbed the
	// arithmetic for rows it does not apply to. NULL rows are excluded: the kernel still writes some
	// value there (the CUDA convention here, matching a SQL NULL's underlying storage), but nothing
	// promises what it is.
	auto got = ReadColumn<int32_t>(result.columns[0]);
	bool values_match = got.size() == n;
	for (size_t i = 0; i < n && values_match; i++) {
		if (valid[i]) {
			values_match = got[i] == v[i] + 1;
		}
	}
	CHECK(values_match);
}

static void TestNullableColumnExcludedByFilterPredicate() {
	// SELECT id FROM t WHERE v > 5 -- SQL's three-valued logic treats a NULL predicate as neither true
	// nor false, so the row is excluded exactly like a FALSE predicate. ExecuteFilter folds this in by
	// ANDing the predicate's own mask with its nullability (AndMasks(mask, mask_column.valid, rows))
	// rather than exposing a nullable mask to the compaction step.
	const size_t n = 64;
	const size_t null_row = 10; // v[10] = 10, which would otherwise pass v > 5
	std::vector<int32_t> id(n), v(n);
	for (size_t i = 0; i < n; i++) {
		id[i] = static_cast<int32_t>(i);
		v[i] = static_cast<int32_t>(i);
	}
	std::vector<std::vector<uint8_t>> owned;
	auto id_col = MakeColumn("id", GpuValueType::INT32, id, owned);
	auto v_col = MakeColumn("v", GpuValueType::INT32, v, owned);
	auto validity = AllValidExcept(n, {null_row});
	v_col.validity = validity.data();
	std::vector<GpuColumn> inputs {id_col, v_col};

	auto filter = std::make_shared<GpuPlanNode>();
	filter->op_type = GpuOpType::FILTER;
	filter->expressions.push_back(
	    MakeExpr("((bool*)out)[idx] = (((const int*)inputs[0])[row]) > 5;", {"v"}, GpuValueType::BOOLEAN));
	filter->children.push_back(MakeScan({"id", "v"}));

	auto result = GpuEngine::Instance().ExecutePlan(*filter, inputs);
	CHECK(result.success == true);
	if (!result.success || result.columns.size() != 2) {
		CHECK(false);
		return;
	}
	std::vector<int32_t> expected;
	for (size_t i = 0; i < n; i++) {
		if (i != null_row && v[i] > 5) {
			expected.push_back(id[i]);
		}
	}
	CHECK(ReadColumn<int32_t>(result.columns[0]) == expected);
}

static void TestMalformedPlanFailsCleanly() {
	// A structurally invalid node (a join with one child instead of two) must fail cleanly with a
	// message rather than crash, read past its children, or return garbage.
	auto join = std::make_shared<GpuPlanNode>();
	join->op_type = GpuOpType::HASH_JOIN;
	join->children.push_back(MakeScan({}));
	auto result = GpuEngine::Instance().ExecutePlan(*join, {});
	CHECK(result.success == false);
	CHECK(!result.error_message.empty());
}

static void TestRepeatedExecutionHitsKernelCache() {
	// The second run of an identical expression must reuse the cached module (and still be correct) —
	// this is what keeps per-query NVRTC compilation off the hot path.
	const size_t n = 8192;
	std::vector<double> v(n);
	for (size_t i = 0; i < n; i++) {
		v[i] = static_cast<double>(i);
	}
	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("v", GpuValueType::FLOAT64, v, owned)};

	auto projection = std::make_shared<GpuPlanNode>();
	projection->op_type = GpuOpType::PROJECTION;
	projection->expressions.push_back(MakeExpr(
	    "((double*)out)[idx] = (((const double*)inputs[0])[row]) + 7.0;", {"v"}, GpuValueType::FLOAT64));
	projection->children.push_back(MakeScan({"v"}));

	for (int run = 0; run < 3; run++) {
		auto result = GpuEngine::Instance().ExecutePlan(*projection, inputs);
		CHECK(result.success == true);
		if (result.success && result.columns.size() == 1) {
			auto got = ReadColumn<double>(result.columns[0]);
			CHECK(got.size() == n);
			CHECK(!got.empty() && std::fabs(got[0] - 7.0) < 1e-9);
			CHECK(!got.empty() && std::fabs(got[n - 1] - (static_cast<double>(n - 1) + 7.0)) < 1e-9);
		}
	}
}

// --- GROUP BY ... AGGREGATE (operators/gpu_groupby.cu) -------------------------------------------
// Every case below ran as "not implemented — libcudf unavailable" before this operator existed.
// Aggregate values are chosen to be exactly representable so a parallel reduction's summation order
// cannot change the expected answer; the float-tolerance comparisons are belt-and-braces.

static void TestGroupBySumOnGpu() {
	// SELECT k, SUM(v) FROM t GROUP BY k
	const size_t n = 100000;
	std::vector<int32_t> k(n);
	std::vector<double> v(n);
	for (size_t i = 0; i < n; i++) {
		k[i] = static_cast<int32_t>(i % 1000);
		v[i] = static_cast<double>(i % 64);
	}
	std::map<int32_t, double> expected;
	for (size_t i = 0; i < n; i++) {
		expected[k[i]] += v[i];
	}

	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("k", GpuValueType::INT32, k, owned),
	                               MakeColumn("v", GpuValueType::FLOAT64, v, owned)};
	auto node = MakeGroupBy({"k"}, {MakeAggregate(GpuAggregateKind::SUM, "v", GpuValueType::FLOAT64)},
	                        MakeScan({"k", "v"}));

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == true);
	CHECK(result.error_message.empty());
	CHECK(result.columns.size() == 2);
	if (!result.success || result.columns.size() != 2) {
		return;
	}
	auto keys = ReadColumn<int32_t>(result.columns[0]);
	auto sums = ReadColumn<double>(result.columns[1]);
	CHECK(keys.size() == expected.size());
	CHECK(KeysAreUnique(keys));
	bool all_match = keys.size() == expected.size() && sums.size() == keys.size();
	for (size_t i = 0; i < keys.size() && all_match; i++) {
		auto it = expected.find(keys[i]);
		all_match = it != expected.end() && std::fabs(it->second - sums[i]) < 1e-9;
	}
	CHECK(all_match);
}

static void TestGroupByCountStarOnGpu() {
	// SELECT k, COUNT(*) FROM t GROUP BY k -- uneven group sizes so a wrong boundary shows up.
	const size_t n = 60000;
	std::vector<int64_t> k(n);
	for (size_t i = 0; i < n; i++) {
		k[i] = static_cast<int64_t>((i * i) % 251);
	}
	std::map<int64_t, int64_t> expected;
	for (size_t i = 0; i < n; i++) {
		expected[k[i]]++;
	}

	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("k", GpuValueType::INT64, k, owned)};
	auto node = MakeGroupBy({"k"}, {MakeAggregate(GpuAggregateKind::COUNT_STAR, "", GpuValueType::INT64)},
	                        MakeScan({"k"}));

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == true);
	if (!result.success || result.columns.size() != 2) {
		CHECK(false);
		return;
	}
	auto keys = ReadColumn<int64_t>(result.columns[0]);
	auto counts = ReadColumn<int64_t>(result.columns[1]);
	CHECK(keys.size() == expected.size());
	CHECK(KeysAreUnique(keys));
	int64_t total = 0;
	bool all_match = keys.size() == expected.size();
	for (size_t i = 0; i < keys.size() && all_match; i++) {
		auto it = expected.find(keys[i]);
		all_match = it != expected.end() && it->second == counts[i];
		total += counts[i];
	}
	CHECK(all_match);
	CHECK(total == static_cast<int64_t>(n)); // every input row landed in exactly one group
}

static void TestGroupByMinMaxOnGpu() {
	// SELECT k, MIN(v), MAX(v) FROM t GROUP BY k -- negative values, two aggregates in one node.
	const size_t n = 50000;
	std::vector<int32_t> k(n);
	std::vector<int64_t> v(n);
	for (size_t i = 0; i < n; i++) {
		k[i] = static_cast<int32_t>(i % 97);
		v[i] = static_cast<int64_t>((i * 7919) % 100000) - 50000;
	}
	std::map<int32_t, std::pair<int64_t, int64_t>> expected;
	for (size_t i = 0; i < n; i++) {
		auto it = expected.find(k[i]);
		if (it == expected.end()) {
			expected[k[i]] = {v[i], v[i]};
		} else {
			it->second.first = v[i] < it->second.first ? v[i] : it->second.first;
			it->second.second = v[i] > it->second.second ? v[i] : it->second.second;
		}
	}

	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("k", GpuValueType::INT32, k, owned),
	                               MakeColumn("v", GpuValueType::INT64, v, owned)};
	auto node = MakeGroupBy({"k"},
	                        {MakeAggregate(GpuAggregateKind::MIN, "v", GpuValueType::INT64),
	                         MakeAggregate(GpuAggregateKind::MAX, "v", GpuValueType::INT64)},
	                        MakeScan({"k", "v"}));

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == true);
	if (!result.success || result.columns.size() != 3) {
		CHECK(false);
		return;
	}
	auto keys = ReadColumn<int32_t>(result.columns[0]);
	auto mins = ReadColumn<int64_t>(result.columns[1]);
	auto maxes = ReadColumn<int64_t>(result.columns[2]);
	CHECK(keys.size() == expected.size());
	bool all_match = keys.size() == expected.size();
	for (size_t i = 0; i < keys.size() && all_match; i++) {
		auto it = expected.find(keys[i]);
		all_match = it != expected.end() && it->second.first == mins[i] && it->second.second == maxes[i];
	}
	CHECK(all_match);
}

static void TestGroupByMultipleKeysOnGpu() {
	// GROUP BY k1, k2 -- the multi-pass stable sort must group by the whole key tuple, not just one
	// column. Negative keys included: they group by raw bit pattern, which must still be exact.
	const size_t n = 80000;
	std::vector<int32_t> k1(n);
	std::vector<int64_t> k2(n);
	std::vector<double> v(n);
	for (size_t i = 0; i < n; i++) {
		k1[i] = static_cast<int32_t>(i % 13) - 6;
		k2[i] = static_cast<int64_t>(i % 7) - 3;
		v[i] = static_cast<double>(i % 32);
	}
	std::map<std::pair<int32_t, int64_t>, double> expected;
	for (size_t i = 0; i < n; i++) {
		expected[{k1[i], k2[i]}] += v[i];
	}

	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("k1", GpuValueType::INT32, k1, owned),
	                               MakeColumn("k2", GpuValueType::INT64, k2, owned),
	                               MakeColumn("v", GpuValueType::FLOAT64, v, owned)};
	auto node = MakeGroupBy({"k1", "k2"}, {MakeAggregate(GpuAggregateKind::SUM, "v", GpuValueType::FLOAT64)},
	                        MakeScan({"k1", "k2", "v"}));

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == true);
	if (!result.success || result.columns.size() != 3) {
		CHECK(false);
		return;
	}
	auto out_k1 = ReadColumn<int32_t>(result.columns[0]);
	auto out_k2 = ReadColumn<int64_t>(result.columns[1]);
	auto sums = ReadColumn<double>(result.columns[2]);
	CHECK(out_k1.size() == expected.size()); // 13 * 7 = 91 distinct pairs
	std::set<std::pair<int32_t, int64_t>> seen;
	bool all_match = out_k1.size() == expected.size();
	for (size_t i = 0; i < out_k1.size() && all_match; i++) {
		std::pair<int32_t, int64_t> key {out_k1[i], out_k2[i]};
		auto it = expected.find(key);
		all_match = it != expected.end() && std::fabs(it->second - sums[i]) < 1e-9 && seen.insert(key).second;
	}
	CHECK(all_match);
}

static void TestGroupByGlobalAggregateOnGpu() {
	// SELECT COUNT(*), SUM(v) FROM t -- no group keys at all is one group over every row.
	const size_t n = 30000;
	std::vector<double> v(n);
	double expected_sum = 0;
	for (size_t i = 0; i < n; i++) {
		v[i] = static_cast<double>(i % 128);
		expected_sum += v[i];
	}

	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("v", GpuValueType::FLOAT64, v, owned)};
	auto node = MakeGroupBy({},
	                        {MakeAggregate(GpuAggregateKind::COUNT_STAR, "", GpuValueType::INT64),
	                         MakeAggregate(GpuAggregateKind::SUM, "v", GpuValueType::FLOAT64)},
	                        MakeScan({"v"}));

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == true);
	if (!result.success || result.columns.size() != 2) {
		CHECK(false);
		return;
	}
	auto counts = ReadColumn<int64_t>(result.columns[0]);
	auto sums = ReadColumn<double>(result.columns[1]);
	CHECK(counts.size() == 1);
	CHECK(sums.size() == 1);
	CHECK(!counts.empty() && counts[0] == static_cast<int64_t>(n));
	CHECK(!sums.empty() && std::fabs(sums[0] - expected_sum) < 1e-6);
}

static void TestGroupByAfterFilterOnGpu() {
	// SELECT k, COUNT(*) FROM t WHERE v > 10 GROUP BY k -- the filtered intermediate never leaves the
	// device between the two operators.
	const size_t n = 40000;
	std::vector<int32_t> k(n);
	std::vector<int32_t> v(n);
	for (size_t i = 0; i < n; i++) {
		k[i] = static_cast<int32_t>(i % 50);
		v[i] = static_cast<int32_t>(i % 30);
	}
	std::map<int32_t, int64_t> expected;
	for (size_t i = 0; i < n; i++) {
		if (v[i] > 10) {
			expected[k[i]]++;
		}
	}

	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("k", GpuValueType::INT32, k, owned),
	                               MakeColumn("v", GpuValueType::INT32, v, owned)};
	auto filter = std::make_shared<GpuPlanNode>();
	filter->op_type = GpuOpType::FILTER;
	filter->expressions.push_back(
	    MakeExpr("((bool*)out)[idx] = (((const int*)inputs[1])[row]) > 10;", {"k", "v"}, GpuValueType::BOOLEAN));
	filter->children.push_back(MakeScan({"k", "v"}));

	auto node = MakeGroupBy({"k"}, {MakeAggregate(GpuAggregateKind::COUNT_STAR, "", GpuValueType::INT64)}, filter);

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == true);
	if (!result.success || result.columns.size() != 2) {
		CHECK(false);
		return;
	}
	auto keys = ReadColumn<int32_t>(result.columns[0]);
	auto counts = ReadColumn<int64_t>(result.columns[1]);
	CHECK(keys.size() == expected.size());
	bool all_match = keys.size() == expected.size();
	for (size_t i = 0; i < keys.size() && all_match; i++) {
		auto it = expected.find(keys[i]);
		all_match = it != expected.end() && it->second == counts[i];
	}
	CHECK(all_match);
}

static void TestGroupBySumOfFloatPromotesToDouble() {
	// DuckDB's SUM(REAL) returns DOUBLE, so the reduction must accumulate in double, not float.
	const size_t n = 20000;
	std::vector<int32_t> k(n);
	std::vector<float> v(n);
	std::map<int32_t, double> expected;
	for (size_t i = 0; i < n; i++) {
		k[i] = static_cast<int32_t>(i % 4);
		v[i] = static_cast<float>(i % 256);
		expected[k[i]] += static_cast<double>(v[i]);
	}

	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("k", GpuValueType::INT32, k, owned),
	                               MakeColumn("v", GpuValueType::FLOAT32, v, owned)};
	auto node = MakeGroupBy({"k"}, {MakeAggregate(GpuAggregateKind::SUM, "v", GpuValueType::FLOAT64)},
	                        MakeScan({"k", "v"}));

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == true);
	if (!result.success || result.columns.size() != 2) {
		CHECK(false);
		return;
	}
	auto keys = ReadColumn<int32_t>(result.columns[0]);
	auto sums = ReadColumn<double>(result.columns[1]);
	CHECK(keys.size() == 4);
	bool all_match = keys.size() == expected.size();
	for (size_t i = 0; i < keys.size() && all_match; i++) {
		auto it = expected.find(keys[i]);
		all_match = it != expected.end() && std::fabs(it->second - sums[i]) < 1e-6;
	}
	CHECK(all_match);
}

static void TestGroupByHighCardinalityOnGpu() {
	// Many small groups: exercises the sort/scan path at a size where a single-block assumption or a
	// 16-bit group counter would break. 200k groups of 2 rows each.
	const size_t groups = 200000;
	const size_t n = groups * 2;
	std::vector<int64_t> k(n);
	std::vector<int64_t> v(n);
	for (size_t i = 0; i < n; i++) {
		k[i] = static_cast<int64_t>(i % groups);
		v[i] = static_cast<int64_t>(i);
	}

	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("k", GpuValueType::INT64, k, owned),
	                               MakeColumn("v", GpuValueType::INT64, v, owned)};
	auto node = MakeGroupBy({"k"},
	                        {MakeAggregate(GpuAggregateKind::COUNT_STAR, "", GpuValueType::INT64),
	                         MakeAggregate(GpuAggregateKind::MAX, "v", GpuValueType::INT64)},
	                        MakeScan({"k", "v"}));

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == true);
	if (!result.success || result.columns.size() != 3) {
		CHECK(false);
		return;
	}
	auto keys = ReadColumn<int64_t>(result.columns[0]);
	auto counts = ReadColumn<int64_t>(result.columns[1]);
	auto maxes = ReadColumn<int64_t>(result.columns[2]);
	CHECK(keys.size() == groups);
	CHECK(KeysAreUnique(keys));
	bool all_match = keys.size() == groups;
	for (size_t i = 0; i < keys.size() && all_match; i++) {
		// Row i and row i+groups share key i; the later row always holds the larger v.
		all_match = counts[i] == 2 && maxes[i] == keys[i] + static_cast<int64_t>(groups);
	}
	CHECK(all_match);
}

static void TestGroupByEmptyInputProducesNoGroups() {
	// A filter matching nothing leaves the group-by with zero rows -> zero groups, which is a successful
	// empty result, not a failure.
	const size_t n = 8192;
	std::vector<int32_t> k(n, 3);
	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("k", GpuValueType::INT32, k, owned)};

	auto filter = std::make_shared<GpuPlanNode>();
	filter->op_type = GpuOpType::FILTER;
	filter->expressions.push_back(
	    MakeExpr("((bool*)out)[idx] = (((const int*)inputs[0])[row]) > 999;", {"k"}, GpuValueType::BOOLEAN));
	filter->children.push_back(MakeScan({"k"}));

	auto node = MakeGroupBy({"k"}, {MakeAggregate(GpuAggregateKind::COUNT_STAR, "", GpuValueType::INT64)}, filter);

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == true);
	CHECK(result.columns.size() == 2);
	if (result.columns.size() == 2) {
		CHECK(result.columns[0].row_count == 0);
		CHECK(result.columns[1].row_count == 0);
	}
}

static void TestGroupByFloatKeyRefusedCleanly() {
	// Grouping floats by raw bit pattern would split -0.0 from +0.0 and never match NaN to itself, so a
	// float key must be refused (-> CPU fallback) rather than answered differently from DuckDB.
	const size_t n = 1024;
	std::vector<double> k(n, 1.5);
	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("k", GpuValueType::FLOAT64, k, owned)};
	auto node = MakeGroupBy({"k"}, {MakeAggregate(GpuAggregateKind::COUNT_STAR, "", GpuValueType::INT64)},
	                        MakeScan({"k"}));

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == false);
	CHECK(result.error_message.find("integer/boolean") != std::string::npos);
}

static void TestNullableColumnRefusedForGroupByAggregate() {
	// Unlike the row-independent operators above (see TestNullableColumnPassesThroughBareScan and
	// friends), GROUP_BY_AGGREGATE's Thrust-based sort/reduce has no null semantics -- ExecuteGpuPlan
	// sets ctx.nulls_supported = IsRowIndependent(plan), which is false here, so ExecuteScan's original
	// refusal is still the live path for this operator. This is the case the old, single
	// TestNullableColumnIsRefusedCleanly was actually protecting: it built a bare SCAN, which used to
	// hit this same refusal unconditionally (before session 30 made the refusal conditional on operator
	// shape) and now does not.
	const size_t n = 256;
	std::vector<int32_t> k(n, 5);
	std::vector<std::vector<uint8_t>> owned;
	auto column = MakeColumn("k", GpuValueType::INT32, k, owned);
	auto validity = AllValidExcept(n, {0});
	column.validity = validity.data();
	std::vector<GpuColumn> inputs {column};

	auto node = MakeGroupBy({"k"}, {MakeAggregate(GpuAggregateKind::COUNT_STAR, "", GpuValueType::INT64)},
	                        MakeScan({"k"}));
	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == false);
	CHECK(result.error_message.find("NULL") != std::string::npos);
}

static void TestGroupByFloatMinMaxRefusedCleanly() {
	// DuckDB orders NaN above everything; thrust::maximum does not. Refuse instead of diverging.
	const size_t n = 1024;
	std::vector<int32_t> k(n, 1);
	std::vector<double> v(n, 2.0);
	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("k", GpuValueType::INT32, k, owned),
	                               MakeColumn("v", GpuValueType::FLOAT64, v, owned)};
	auto node = MakeGroupBy({"k"}, {MakeAggregate(GpuAggregateKind::MAX, "v", GpuValueType::FLOAT64)},
	                        MakeScan({"k", "v"}));

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == false);
	CHECK(!result.error_message.empty());
}

static void TestGroupByGlobalAggregateOverEmptyInputRefusedCleanly() {
	// SELECT SUM(v) FROM t with no rows is one row of NULL in DuckDB; this path carries no validity
	// bitmap, so it must refuse rather than answer 0.
	const size_t n = 4096;
	std::vector<double> v(n, 1.0);
	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("v", GpuValueType::FLOAT64, v, owned)};

	auto filter = std::make_shared<GpuPlanNode>();
	filter->op_type = GpuOpType::FILTER;
	filter->expressions.push_back(
	    MakeExpr("((bool*)out)[idx] = (((const double*)inputs[0])[row]) > 99.0;", {"v"}, GpuValueType::BOOLEAN));
	filter->children.push_back(MakeScan({"v"}));

	auto node = MakeGroupBy({}, {MakeAggregate(GpuAggregateKind::SUM, "v", GpuValueType::FLOAT64)}, filter);

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == false);
	CHECK(result.error_message.find("NULL") != std::string::npos);
}

// --- HASH_JOIN (operators/gpu_hash_join.cu) ------------------------------------------------------
// INNER equi-join by sort + vectorized binary search. Results are compared as a multiset of rows,
// because join output order is an implementation detail (and DuckDB does not promise one either).

namespace {

std::shared_ptr<GpuPlanNode> MakeJoin(const std::string &left_key, const std::string &right_key,
                                      const std::vector<std::string> &output_columns,
                                      std::shared_ptr<GpuPlanNode> left, std::shared_ptr<GpuPlanNode> right) {
	auto node = std::make_shared<GpuPlanNode>();
	node->op_type = GpuOpType::HASH_JOIN;
	node->build_keys = {left_key};
	node->probe_keys = {right_key};
	node->output_columns = output_columns;
	node->children.push_back(std::move(left));
	node->children.push_back(std::move(right));
	return node;
}

//! Sorted (col0, col1) pairs, so two join results can be compared regardless of row order.
std::vector<std::pair<int32_t, int32_t>> SortedPairs(const std::vector<int32_t> &a, const std::vector<int32_t> &b) {
	std::vector<std::pair<int32_t, int32_t>> pairs;
	pairs.reserve(a.size());
	for (size_t i = 0; i < a.size() && i < b.size(); i++) {
		pairs.emplace_back(a[i], b[i]);
	}
	std::sort(pairs.begin(), pairs.end());
	return pairs;
}

} // namespace

static void TestInnerJoinOnGpu() {
	// SELECT l.id, l.lv, r.rv FROM l JOIN r ON l.id = r.id -- one-to-one, with left rows that match
	// nothing (ids 5000..9999 exist only on the left).
	const size_t left_n = 10000;
	const size_t right_n = 5000;
	std::vector<int32_t> lid(left_n), lv(left_n);
	for (size_t i = 0; i < left_n; i++) {
		lid[i] = static_cast<int32_t>(i);
		lv[i] = static_cast<int32_t>(i * 2);
	}
	std::vector<int32_t> rid(right_n), rv(right_n);
	for (size_t i = 0; i < right_n; i++) {
		rid[i] = static_cast<int32_t>(i);
		rv[i] = static_cast<int32_t>(i * 3);
	}
	std::vector<std::pair<int32_t, int32_t>> expected;
	for (size_t i = 0; i < right_n; i++) {
		expected.emplace_back(static_cast<int32_t>(i * 2), static_cast<int32_t>(i * 3));
	}
	std::sort(expected.begin(), expected.end());

	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("lid", GpuValueType::INT32, lid, owned),
	                               MakeColumn("lv", GpuValueType::INT32, lv, owned),
	                               MakeColumn("rid", GpuValueType::INT32, rid, owned),
	                               MakeColumn("rv", GpuValueType::INT32, rv, owned)};
	auto node = MakeJoin("lid", "rid", {"lv", "rv"}, MakeScan({"lid", "lv"}), MakeScan({"rid", "rv"}));

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == true);
	CHECK(result.error_message.empty());
	CHECK(result.columns.size() == 2);
	if (!result.success || result.columns.size() != 2) {
		return;
	}
	CHECK(result.columns[0].row_count == right_n);
	CHECK(SortedPairs(ReadColumn<int32_t>(result.columns[0]), ReadColumn<int32_t>(result.columns[1])) == expected);
}

static void TestJoinManyToManyOnGpu() {
	// Duplicate keys on BOTH sides: every left row must pair with every matching right row.
	// 3 left rows and 4 right rows per key over 50 keys -> 600 output rows.
	const size_t keys = 50;
	std::vector<int32_t> lid, lv, rid, rv;
	for (size_t k = 0; k < keys; k++) {
		for (int i = 0; i < 3; i++) {
			lid.push_back(static_cast<int32_t>(k));
			lv.push_back(static_cast<int32_t>(k * 10 + i));
		}
		for (int j = 0; j < 4; j++) {
			rid.push_back(static_cast<int32_t>(k));
			rv.push_back(static_cast<int32_t>(k * 100 + j));
		}
	}
	std::vector<std::pair<int32_t, int32_t>> expected;
	for (size_t k = 0; k < keys; k++) {
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				expected.emplace_back(static_cast<int32_t>(k * 10 + i), static_cast<int32_t>(k * 100 + j));
			}
		}
	}
	std::sort(expected.begin(), expected.end());

	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("lid", GpuValueType::INT32, lid, owned),
	                               MakeColumn("lv", GpuValueType::INT32, lv, owned),
	                               MakeColumn("rid", GpuValueType::INT32, rid, owned),
	                               MakeColumn("rv", GpuValueType::INT32, rv, owned)};
	auto node = MakeJoin("lid", "rid", {"lv", "rv"}, MakeScan({"lid", "lv"}), MakeScan({"rid", "rv"}));

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == true);
	if (!result.success || result.columns.size() != 2) {
		CHECK(false);
		return;
	}
	CHECK(result.columns[0].row_count == keys * 12);
	CHECK(SortedPairs(ReadColumn<int32_t>(result.columns[0]), ReadColumn<int32_t>(result.columns[1])) == expected);
}

static void TestJoinWithNegativeKeysOnGpu() {
	// Keys are matched by widened bit pattern; negative INT64 keys must still match exactly.
	const size_t n = 2000;
	std::vector<int64_t> lid(n), rid(n);
	std::vector<int32_t> lv(n), rv(n);
	for (size_t i = 0; i < n; i++) {
		lid[i] = static_cast<int64_t>(i) - static_cast<int64_t>(n); // all negative
		lv[i] = static_cast<int32_t>(i);
		rid[i] = lid[i];
		rv[i] = static_cast<int32_t>(i) + 1000;
	}
	std::vector<std::pair<int32_t, int32_t>> expected;
	for (size_t i = 0; i < n; i++) {
		expected.emplace_back(static_cast<int32_t>(i), static_cast<int32_t>(i) + 1000);
	}
	std::sort(expected.begin(), expected.end());

	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("lid", GpuValueType::INT64, lid, owned),
	                               MakeColumn("lv", GpuValueType::INT32, lv, owned),
	                               MakeColumn("rid", GpuValueType::INT64, rid, owned),
	                               MakeColumn("rv", GpuValueType::INT32, rv, owned)};
	auto node = MakeJoin("lid", "rid", {"lv", "rv"}, MakeScan({"lid", "lv"}), MakeScan({"rid", "rv"}));

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == true);
	if (!result.success || result.columns.size() != 2) {
		CHECK(false);
		return;
	}
	CHECK(result.columns[0].row_count == n);
	CHECK(SortedPairs(ReadColumn<int32_t>(result.columns[0]), ReadColumn<int32_t>(result.columns[1])) == expected);
}

static void TestJoinOutputColumnOrderIsRespected() {
	// output_columns drives both WHICH columns are emitted and in what ORDER -- this is what carries
	// DuckDB's left/right projection maps, so a subset in a non-obvious order must come back exactly.
	const size_t n = 1000;
	std::vector<int32_t> lid(n), lv(n), rid(n), rv(n);
	for (size_t i = 0; i < n; i++) {
		lid[i] = static_cast<int32_t>(i);
		lv[i] = static_cast<int32_t>(i) + 7;
		rid[i] = static_cast<int32_t>(i);
		rv[i] = static_cast<int32_t>(i) + 9;
	}
	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("lid", GpuValueType::INT32, lid, owned),
	                               MakeColumn("lv", GpuValueType::INT32, lv, owned),
	                               MakeColumn("rid", GpuValueType::INT32, rid, owned),
	                               MakeColumn("rv", GpuValueType::INT32, rv, owned)};
	// Right column first, then a left column; `lid`/`rid` deliberately omitted.
	auto node = MakeJoin("lid", "rid", {"rv", "lv"}, MakeScan({"lid", "lv"}), MakeScan({"rid", "rv"}));

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == true);
	if (!result.success || result.columns.size() != 2) {
		CHECK(false);
		return;
	}
	CHECK(result.columns[0].name == "rv");
	CHECK(result.columns[1].name == "lv");
	auto first = ReadColumn<int32_t>(result.columns[0]);
	auto second = ReadColumn<int32_t>(result.columns[1]);
	bool all_match = first.size() == n && second.size() == n;
	for (size_t i = 0; i < first.size() && all_match; i++) {
		all_match = first[i] == second[i] + 2; // (i+9) vs (i+7)
	}
	CHECK(all_match);
}

static void TestJoinAfterFilterOnGpu() {
	// FILTER -> JOIN: the filtered left side never leaves the device between operators.
	const size_t n = 4000;
	std::vector<int32_t> lid(n), lv(n), rid(n), rv(n);
	for (size_t i = 0; i < n; i++) {
		lid[i] = static_cast<int32_t>(i);
		lv[i] = static_cast<int32_t>(i % 100);
		rid[i] = static_cast<int32_t>(i);
		rv[i] = static_cast<int32_t>(i);
	}
	size_t expected_rows = 0;
	for (size_t i = 0; i < n; i++) {
		if (lv[i] > 50) {
			expected_rows++;
		}
	}

	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("lid", GpuValueType::INT32, lid, owned),
	                               MakeColumn("lv", GpuValueType::INT32, lv, owned),
	                               MakeColumn("rid", GpuValueType::INT32, rid, owned),
	                               MakeColumn("rv", GpuValueType::INT32, rv, owned)};
	auto filter = std::make_shared<GpuPlanNode>();
	filter->op_type = GpuOpType::FILTER;
	filter->expressions.push_back(MakeExpr("((bool*)out)[idx] = (((const int*)inputs[0])[row]) > 50;", {"lv"},
	                                       GpuValueType::BOOLEAN));
	filter->children.push_back(MakeScan({"lid", "lv"}));

	auto node = MakeJoin("lid", "rid", {"lv", "rv"}, filter, MakeScan({"rid", "rv"}));

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == true);
	if (!result.success || result.columns.size() != 2) {
		CHECK(false);
		return;
	}
	CHECK(result.columns[0].row_count == expected_rows);
}

static void TestJoinMatchingNothingProducesEmptyResult() {
	const size_t n = 2048;
	std::vector<int32_t> lid(n, 1), lv(n, 5), rid(n, 999), rv(n, 7);
	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("lid", GpuValueType::INT32, lid, owned),
	                               MakeColumn("lv", GpuValueType::INT32, lv, owned),
	                               MakeColumn("rid", GpuValueType::INT32, rid, owned),
	                               MakeColumn("rv", GpuValueType::INT32, rv, owned)};
	auto node = MakeJoin("lid", "rid", {"lv", "rv"}, MakeScan({"lid", "lv"}), MakeScan({"rid", "rv"}));

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == true); // no matches is a successful empty result, not a failure
	CHECK(result.columns.size() == 2);
	if (result.columns.size() == 2) {
		CHECK(result.columns[0].row_count == 0);
		CHECK(result.columns[1].row_count == 0);
	}
}

static void TestJoinMismatchedKeyTypesRefusedCleanly() {
	// INT32 -1 and INT64 -1 widen to different uint64 values, so a mixed-width join would silently drop
	// negative matches. Refuse instead.
	const size_t n = 512;
	std::vector<int32_t> lid(n, 3);
	std::vector<int64_t> rid(n, 3);
	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("lid", GpuValueType::INT32, lid, owned),
	                               MakeColumn("rid", GpuValueType::INT64, rid, owned)};
	auto node = MakeJoin("lid", "rid", {"lid", "rid"}, MakeScan({"lid"}), MakeScan({"rid"}));

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == false);
	CHECK(result.error_message.find("same type") != std::string::npos);
}

static void TestJoinFloatKeyRefusedCleanly() {
	const size_t n = 512;
	std::vector<double> lid(n, 1.5), rid(n, 1.5);
	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("lid", GpuValueType::FLOAT64, lid, owned),
	                               MakeColumn("rid", GpuValueType::FLOAT64, rid, owned)};
	auto node = MakeJoin("lid", "rid", {"lid", "rid"}, MakeScan({"lid"}), MakeScan({"rid"}));

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == false);
	CHECK(result.error_message.find("integer/boolean") != std::string::npos);
}

static void TestJoinAmbiguousOutputColumnRefusedCleanly() {
	// Both sides produce a column called "id"; selecting it by name could silently take the wrong side.
	const size_t n = 512;
	std::vector<int32_t> a(n, 1), b(n, 1);
	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("id", GpuValueType::INT32, a, owned),
	                               MakeColumn("id", GpuValueType::INT32, b, owned)};
	auto node = MakeJoin("id", "id", {"id"}, MakeScan({"id"}), MakeScan({"id"}));

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == false);
	CHECK(result.error_message.find("ambiguous") != std::string::npos);
}

static void TestJoinCompositeKeyRefusedCleanly() {
	const size_t n = 512;
	std::vector<int32_t> a(n, 1);
	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("l1", GpuValueType::INT32, a, owned),
	                               MakeColumn("r1", GpuValueType::INT32, a, owned)};
	auto node = MakeJoin("l1", "r1", {"l1"}, MakeScan({"l1"}), MakeScan({"r1"}));
	node->build_keys.push_back("l1"); // a second key pair the engine must refuse
	node->probe_keys.push_back("r1");

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == false);
	CHECK(result.error_message.find("exactly one equality key") != std::string::npos);
}

// --- CROSS_PRODUCT (operators/gpu_cross_join.cu) --------------------------------------------------
// Unlike the join above, row ORDER is asserted positionally here: the operator promises left-major
// output (output row r pairs left row r/m with right row r%m), which is the order DuckDB's own
// PhysicalCrossProduct emits. Losing that would not be caught by a multiset comparison.

namespace {

std::shared_ptr<GpuPlanNode> MakeCrossProduct(const std::vector<std::string> &output_columns,
                                              std::shared_ptr<GpuPlanNode> left,
                                              std::shared_ptr<GpuPlanNode> right) {
	auto node = std::make_shared<GpuPlanNode>();
	node->op_type = GpuOpType::CROSS_PRODUCT;
	node->output_columns = output_columns;
	node->children.push_back(std::move(left));
	node->children.push_back(std::move(right));
	return node;
}

} // namespace

static void TestCrossProductOnGpu() {
	// 3 left rows x 4 right rows -> 12 pairs, left-major.
	std::vector<int32_t> lv {10, 20, 30};
	std::vector<int32_t> rv {1, 2, 3, 4};
	std::vector<int32_t> expected_left {10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30};
	std::vector<int32_t> expected_right {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};

	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("lv", GpuValueType::INT32, lv, owned),
	                               MakeColumn("rv", GpuValueType::INT32, rv, owned)};
	auto node = MakeCrossProduct({"lv", "rv"}, MakeScan({"lv"}), MakeScan({"rv"}));

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == true);
	CHECK(result.error_message.empty());
	CHECK(result.columns.size() == 2);
	if (!result.success || result.columns.size() != 2) {
		return;
	}
	CHECK(result.columns[0].row_count == 12);
	CHECK(ReadColumn<int32_t>(result.columns[0]) == expected_left);
	CHECK(ReadColumn<int32_t>(result.columns[1]) == expected_right);
}

static void TestCrossProductMixedWidthsAndColumnOrder() {
	// Output order is whatever output_columns says, not left-then-right, and the expansion must work for
	// columns of different widths (INT32 and FLOAT64) on the same side.
	std::vector<int32_t> lk {7, 8};
	std::vector<double> lval {1.5, 2.5};
	std::vector<int64_t> rk {100, 200, 300};

	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("lk", GpuValueType::INT32, lk, owned),
	                               MakeColumn("lval", GpuValueType::FLOAT64, lval, owned),
	                               MakeColumn("rk", GpuValueType::INT64, rk, owned)};
	auto node = MakeCrossProduct({"rk", "lval", "lk"}, MakeScan({"lk", "lval"}), MakeScan({"rk"}));

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == true);
	CHECK(result.columns.size() == 3);
	if (!result.success || result.columns.size() != 3) {
		return;
	}
	CHECK(result.columns[0].row_count == 6);
	CHECK((ReadColumn<int64_t>(result.columns[0]) == std::vector<int64_t> {100, 200, 300, 100, 200, 300}));
	CHECK((ReadColumn<double>(result.columns[1]) == std::vector<double> {1.5, 1.5, 1.5, 2.5, 2.5, 2.5}));
	CHECK((ReadColumn<int32_t>(result.columns[2]) == std::vector<int32_t> {7, 7, 7, 8, 8, 8}));
}

static void TestProjectionOverCrossProductOnGpu() {
	// The shape the operator exists for: a few thousand input rows driving millions of pairwise
	// evaluations that never leave the device until the projection has run.
	const size_t left_n = 500;
	const size_t right_n = 400;
	std::vector<double> lval(left_n), rval(right_n);
	for (size_t i = 0; i < left_n; i++) {
		lval[i] = static_cast<double>(i) * 0.25;
	}
	for (size_t j = 0; j < right_n; j++) {
		rval[j] = static_cast<double>(j) * 0.5 + 1.0;
	}

	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("lval", GpuValueType::FLOAT64, lval, owned),
	                               MakeColumn("rval", GpuValueType::FLOAT64, rval, owned)};
	auto cross = MakeCrossProduct({"lval", "rval"}, MakeScan({"lval"}), MakeScan({"rval"}));
	auto projection = std::make_shared<GpuPlanNode>();
	projection->op_type = GpuOpType::PROJECTION;
	projection->expressions.push_back(
	    MakeExpr("((double*)out)[idx] = (((const double*)inputs[0])[row]) * (((const double*)inputs[1])[row]);",
	             {"lval", "rval"}, GpuValueType::FLOAT64));
	projection->children.push_back(cross);

	auto result = GpuEngine::Instance().ExecutePlan(*projection, inputs);
	CHECK(result.success == true);
	CHECK(result.columns.size() == 1);
	if (!result.success || result.columns.size() != 1) {
		return;
	}
	CHECK(result.columns[0].row_count == left_n * right_n);
	auto got = ReadColumn<double>(result.columns[0]);
	bool all_match = got.size() == left_n * right_n;
	for (size_t i = 0; i < left_n && all_match; i++) {
		for (size_t j = 0; j < right_n && all_match; j++) {
			all_match = std::fabs(got[i * right_n + j] - lval[i] * rval[j]) < 1e-9;
		}
	}
	CHECK(all_match);
}

static void TestCrossProductEmptySideProducesNoRows() {
	std::vector<int32_t> lv {1, 2, 3};
	std::vector<int32_t> rv;
	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("lv", GpuValueType::INT32, lv, owned),
	                               MakeColumn("rv", GpuValueType::INT32, rv, owned)};
	auto node = MakeCrossProduct({"lv", "rv"}, MakeScan({"lv"}), MakeScan({"rv"}));

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == true);
	CHECK(result.columns.size() == 2);
	if (!result.success || result.columns.size() != 2) {
		return;
	}
	// Zero rows, but still correctly TYPED columns -- the result is packed against the operator's types.
	CHECK(result.columns[0].row_count == 0);
	CHECK(result.columns[1].row_count == 0);
	CHECK(result.columns[0].type == GpuValueType::INT32);
	CHECK(result.columns[1].type == GpuValueType::INT32);
}

static void TestCrossProductOversizeRefusedCleanly() {
	// 70000 x 70000 = 4.9e9 pairs, past the engine's 2^32 cap. The inputs are half a megabyte, which is
	// the whole hazard: a trivially small query can ask for more memory than exists. Must refuse rather
	// than attempt the allocation.
	const size_t n = 70000;
	std::vector<int32_t> lv(n, 1), rv(n, 2);
	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("lv", GpuValueType::INT32, lv, owned),
	                               MakeColumn("rv", GpuValueType::INT32, rv, owned)};
	auto node = MakeCrossProduct({"lv", "rv"}, MakeScan({"lv"}), MakeScan({"rv"}));

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == false);
	CHECK(result.error_message.find("too large") != std::string::npos);
}

static void TestCrossProductAmbiguousOutputColumnRefusedCleanly() {
	// Both sides produce "id"; selecting it by name could silently take the wrong side (a self cross
	// join). Same rule, and the same shared ResolveSideColumn, as the join.
	std::vector<int32_t> a {1, 2}, b {3, 4};
	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("id", GpuValueType::INT32, a, owned),
	                               MakeColumn("id", GpuValueType::INT32, b, owned)};
	auto node = MakeCrossProduct({"id"}, MakeScan({"id"}), MakeScan({"id"}));

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == false);
	CHECK(result.error_message.find("ambiguous") != std::string::npos);
}

// Tests for the PROJECTION → GROUP_BY plan shape that the translator emits when a group key is a
// computed expression (e.g. DuckDB's Perfect Hash Group By inserting CAST(k AS SMALLINT)).
// The plan is hand-built at the GpuPlanNode level — no translator is involved — so these tests
// verify the ENGINE correctly handles the "compute key in projection, group by its output name"
// pattern independently of the translator changes in gpu_offload_extension.cpp.

static void TestGroupByWithProjectedSmallintKey() {
	// Simulates: SELECT CAST(k AS SMALLINT), COUNT(*) FROM t GROUP BY CAST(k AS SMALLINT)
	// where the translator inserts a PROJECTION child that emits the cast result as "col0",
	// and the GROUP_BY key is "col0". Here we hand-build the plan with a raw INT16 pass-through
	// so the engine can actually run it (NVRTC compile of `(short)(...)` is covered elsewhere;
	// this test focuses on the GROUP_BY correctly reading its key from the projection's output).
	const size_t n = 30000;
	std::vector<int16_t> k(n);
	std::map<int16_t, int64_t> expected;
	for (size_t i = 0; i < n; i++) {
		k[i] = static_cast<int16_t>(i % 7); // 7 distinct groups: 0..6
		expected[k[i]]++;
	}

	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("k", GpuValueType::INT16, k, owned)};

	// PROJECTION child: emits col0 = k (INT16 pass-through), simulating CAST(k AS SMALLINT) output.
	auto proj = std::make_shared<GpuPlanNode>();
	proj->op_type = GpuOpType::PROJECTION;
	proj->expressions.push_back(MakeExpr(
	    "((short*)out)[idx] = ((const short*)inputs[0])[row];", {"k"}, GpuValueType::INT16));
	proj->children.push_back(MakeScan({"k"}));

	// GROUP_BY on top: groups by "col0" (the projection output), counts rows.
	auto node = MakeGroupBy({"col0"}, {MakeAggregate(GpuAggregateKind::COUNT_STAR, "", GpuValueType::INT64)},
	                        proj);

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == true);
	if (!result.success || result.columns.size() != 2) {
		CHECK(false);
		return;
	}
	// col0 is INT16; agg_0 is INT64 (COUNT(*)).
	CHECK(result.columns[0].type == GpuValueType::INT16);
	CHECK(result.columns[1].type == GpuValueType::INT64);
	auto keys   = ReadColumn<int16_t>(result.columns[0]);
	auto counts = ReadColumn<int64_t>(result.columns[1]);
	CHECK(keys.size() == 7);
	CHECK(KeysAreUnique(keys));
	bool all_match = keys.size() == expected.size();
	for (size_t i = 0; i < keys.size() && all_match; i++) {
		auto it = expected.find(keys[i]);
		all_match = it != expected.end() && it->second == counts[i];
	}
	CHECK(all_match);
}

static void TestGroupByWithProjectedComputedKey() {
	// Simulates: SELECT k % 10, SUM(v) FROM t GROUP BY k % 10
	// The translator emits a PROJECTION that evaluates `k % 10` as col0 and passes through v as
	// col1; the GROUP_BY groups by col0 and runs SUM over col1. This test hand-builds that shape
	// and verifies the engine reads the pre-computed key from col0 and the aggregate input from col1.
	const size_t n = 50000;
	std::vector<int32_t> k(n), v(n);
	std::map<int32_t, int64_t> expected; // key -> sum(v)
	for (size_t i = 0; i < n; i++) {
		k[i] = static_cast<int32_t>(i);
		v[i] = static_cast<int32_t>(i % 100);
		expected[k[i] % 10] += v[i];
	}

	std::vector<std::vector<uint8_t>> owned;
	std::vector<GpuColumn> inputs {MakeColumn("k", GpuValueType::INT32, k, owned),
	                               MakeColumn("v", GpuValueType::INT32, v, owned)};

	// PROJECTION: col0 = k % 10 (INT32 modulo), col1 = v (pass-through INT32).
	auto proj = std::make_shared<GpuPlanNode>();
	proj->op_type = GpuOpType::PROJECTION;
	proj->expressions.push_back(MakeExpr(
	    "((int*)out)[idx] = (((const int*)inputs[0])[row]) % 10;", {"k"}, GpuValueType::INT32));
	proj->expressions.push_back(MakeExpr(
	    "((double*)out)[idx] = (double)(((const int*)inputs[0])[row]);", {"v"}, GpuValueType::FLOAT64));
	proj->children.push_back(MakeScan({"k", "v"}));

	// GROUP_BY: group by col0 (the modulo result); SUM col1 (the v pass-through) into FLOAT64
	// matching DuckDB's SUM(INTEGER) -> HUGEINT promotion, but using FLOAT64 here since the engine
	// only runs SUM in float mode (per gpu_groupby.cu's restriction). The SUM aggregate type must
	// be FLOAT64 to pass engine validation.
	auto node = MakeGroupBy({"col0"}, {MakeAggregate(GpuAggregateKind::SUM, "col1", GpuValueType::FLOAT64)},
	                        proj);

	auto result = GpuEngine::Instance().ExecutePlan(*node, inputs);
	CHECK(result.success == true);
	if (!result.success || result.columns.size() != 2) {
		CHECK(false);
		return;
	}
	CHECK(result.columns[0].type == GpuValueType::INT32);
	CHECK(result.columns[1].type == GpuValueType::FLOAT64);
	auto keys = ReadColumn<int32_t>(result.columns[0]);
	auto sums  = ReadColumn<double>(result.columns[1]);
	// Exactly 10 groups: k%10 in [0,9]
	CHECK(keys.size() == 10);
	CHECK(KeysAreUnique(keys));
	bool all_match = keys.size() == expected.size();
	for (size_t i = 0; i < keys.size() && all_match; i++) {
		auto it = expected.find(keys[i]);
		all_match = it != expected.end() && std::fabs(static_cast<double>(it->second) - sums[i]) < 0.5;
	}
	CHECK(all_match);
}

int main() {
	RUN_TEST(TestProjectionRunsOnGpu);
	RUN_TEST(TestFilterCompactsOnGpu);
	RUN_TEST(TestFilterThenProjectionPipeline);
	RUN_TEST(TestMultiColumnFilterGathersEveryColumn);
	RUN_TEST(TestFilterMatchingNothingProducesEmptyResult);
	RUN_TEST(TestNullableColumnPassesThroughBareScan);
	RUN_TEST(TestNullableColumnPropagatesThroughProjection);
	RUN_TEST(TestNullableColumnExcludedByFilterPredicate);
	RUN_TEST(TestMalformedPlanFailsCleanly);
	RUN_TEST(TestRepeatedExecutionHitsKernelCache);
	RUN_TEST(TestGroupBySumOnGpu);
	RUN_TEST(TestGroupByCountStarOnGpu);
	RUN_TEST(TestGroupByMinMaxOnGpu);
	RUN_TEST(TestGroupByMultipleKeysOnGpu);
	RUN_TEST(TestGroupByGlobalAggregateOnGpu);
	RUN_TEST(TestGroupByAfterFilterOnGpu);
	RUN_TEST(TestGroupBySumOfFloatPromotesToDouble);
	RUN_TEST(TestGroupByHighCardinalityOnGpu);
	RUN_TEST(TestGroupByEmptyInputProducesNoGroups);
	RUN_TEST(TestNullableColumnRefusedForGroupByAggregate);
	RUN_TEST(TestGroupByFloatKeyRefusedCleanly);
	RUN_TEST(TestGroupByFloatMinMaxRefusedCleanly);
	RUN_TEST(TestGroupByGlobalAggregateOverEmptyInputRefusedCleanly);
	RUN_TEST(TestInnerJoinOnGpu);
	RUN_TEST(TestJoinManyToManyOnGpu);
	RUN_TEST(TestJoinWithNegativeKeysOnGpu);
	RUN_TEST(TestJoinOutputColumnOrderIsRespected);
	RUN_TEST(TestJoinAfterFilterOnGpu);
	RUN_TEST(TestJoinMatchingNothingProducesEmptyResult);
	RUN_TEST(TestJoinMismatchedKeyTypesRefusedCleanly);
	RUN_TEST(TestJoinFloatKeyRefusedCleanly);
	RUN_TEST(TestJoinAmbiguousOutputColumnRefusedCleanly);
	RUN_TEST(TestJoinCompositeKeyRefusedCleanly);
	RUN_TEST(TestCrossProductOnGpu);
	RUN_TEST(TestCrossProductMixedWidthsAndColumnOrder);
	RUN_TEST(TestProjectionOverCrossProductOnGpu);
	RUN_TEST(TestCrossProductEmptySideProducesNoRows);
	RUN_TEST(TestCrossProductOversizeRefusedCleanly);
	RUN_TEST(TestCrossProductAmbiguousOutputColumnRefusedCleanly);
	RUN_TEST(TestGroupByWithProjectedSmallintKey);
	RUN_TEST(TestGroupByWithProjectedComputedKey);
	TEST_MAIN_EPILOGUE();
}
