#include "expression_translator.hpp"
#include "code_generator.hpp"

#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <atomic>
#include <set>
#include <sstream>

namespace vector_gpu {

using namespace duckdb;

namespace {

//! Per-call translation state: accumulates the ordered, de-duplicated list of input columns referenced
//! so far, so the same bound column referenced twice in one expression reuses the same inputs[] slot.
struct TranslationState {
	const ColumnBindingNames &bindings;
	std::vector<std::string> input_columns;
	std::map<std::pair<idx_t, idx_t>, size_t> slot_by_binding;
};

GpuValueType MapLogicalType(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::DECIMAL:
		switch (type.InternalType()) {
		case PhysicalType::INT16:
			return GpuValueType::INT16;
		case PhysicalType::INT32:
			return GpuValueType::INT32;
		case PhysicalType::INT64:
			return GpuValueType::INT64;
		case PhysicalType::INT128:
			return GpuValueType::HUGEINT;
		default:
			throw std::runtime_error("unsupported decimal width");
		}
	case LogicalTypeId::HUGEINT:
		return GpuValueType::HUGEINT;
	case LogicalTypeId::SMALLINT:
		return GpuValueType::INT16;
	case LogicalTypeId::INTEGER:
		return GpuValueType::INT32;
	case LogicalTypeId::BIGINT:
		return GpuValueType::INT64;
	case LogicalTypeId::FLOAT:
		return GpuValueType::FLOAT32;
	case LogicalTypeId::DOUBLE:
		return GpuValueType::FLOAT64;
	case LogicalTypeId::BOOLEAN:
		return GpuValueType::BOOLEAN;
	default:
		throw GpuUnsupportedExpression("unsupported type for GPU expression: " + type.ToString());
	}
}

//! True when the GPU's C-style cast is guaranteed to AGREE with DuckDB's CAST for every input value.
//!
//! Agreement, not losslessness, is the property that matters. An earlier version required the cast to
//! be lossless, which refused every mixed int/float expression (`bigint_col * 1.5`, `sum(i) / 2.0`,
//! and everything DuckDB implicitly widens) -- the single biggest reason the engine "refused
//! everything". Integer->floating is lossy past 2^24 / 2^53, but BOTH sides perform the same IEEE-754
//! round-to-nearest-even conversion, so they lose precision identically and produce identical bits.
//!
//! Still excluded, because these genuinely disagree:
//!   - integer narrowing (BIGINT->INTEGER): DuckDB raises a conversion error when the value does not
//!     fit; a C++ static_cast silently truncates the low bits.
//!   - floating->integer: DuckDB rounds and range-checks; C++ truncates toward zero and is undefined
//!     out of range.
//!   - DOUBLE->FLOAT: DuckDB range-checks and raises on overflow; the C++ conversion yields infinity.
bool IsLosslessGpuCast(GpuValueType from, GpuValueType to) {
	if (from == to) {
		return true;
	}
	switch (from) {
	case GpuValueType::BOOLEAN:
		return to == GpuValueType::INT32 || to == GpuValueType::INT64 || to == GpuValueType::FLOAT32 ||
		       to == GpuValueType::FLOAT64;
	case GpuValueType::INT16:
		// Widening integer and integer->floating
		return to == GpuValueType::INT32 || to == GpuValueType::INT64 || to == GpuValueType::FLOAT32 ||
		       to == GpuValueType::FLOAT64;
	case GpuValueType::INT32:
		return to == GpuValueType::INT64 || to == GpuValueType::FLOAT32 || to == GpuValueType::FLOAT64;
	case GpuValueType::INT64:
		// Widening integer->floating. INT64->INT32 is absent on purpose (narrowing).
		return to == GpuValueType::FLOAT32 || to == GpuValueType::FLOAT64;
	case GpuValueType::FLOAT32:
		return to == GpuValueType::FLOAT64;
	default:
		return false;
	}
}

//! Maps a DuckDB scalar math function to its CUDA equivalent.
//!
//! Adding a function here means adding its weight to MathFunctionWeight in expression_cost.cpp too:
//! anything this table can lower but that one does not price is counted at the unknown weight, which
//! under-counts a transcendental 30-fold and quietly stops the routing layer from offloading it.
//!
//! DOUBLE-only on purpose. Every function here returns DOUBLE in DuckDB, and DuckDB inserts the casts to
//! get there, so emitting a double-typed call matches what the CPU actually computes.
//!
//! NAMING TRAPS, verified against the real duckdb.exe rather than assumed:
//!   - DuckDB's `log(x)` is LOG BASE 10, not the natural log. CUDA's `log()` is natural. Mapping `log`
//!     to `log` would silently return wrong numbers for every row.
//!   - DuckDB's natural log is `ln`, which maps to CUDA `log`.
//!
//! ACCURACY: CUDA's libdevice and the host libm are both correctly rounded for sqrt, and within ~1-2 ULP
//! for the transcendentals -- they are NOT guaranteed bit-identical. See docs/CHANGELOG.md for the
//! measured divergence on real data. --use_fast_math stays off (session-7 KI-6), so this is ordinary
//! libm-vs-libdevice variation, not fast-math error.
const char *CudaMathFunction(const std::string &name, size_t arity) {
	if (arity == 1) {
		if (name == "sqrt") return "sqrt";
		if (name == "exp") return "exp";
		if (name == "ln") return "log";     // DuckDB ln  -> natural log
		if (name == "log10") return "log10";
		if (name == "log") return "log10";  // DuckDB log -> BASE 10
		if (name == "log2") return "log2";
		if (name == "sin") return "sin";
		if (name == "cos") return "cos";
		if (name == "tan") return "tan";
		if (name == "asin") return "asin";
		if (name == "acos") return "acos";
		if (name == "atan") return "atan";
		if (name == "floor") return "floor";
		if (name == "ceil" || name == "ceiling") return "ceil";
		if (name == "radians") return "__vgpu_radians";
		if (name == "degrees") return "__vgpu_degrees";
	}
	if (arity == 2) {
		if (name == "pow" || name == "power") return "pow";
		if (name == "atan2") return "atan2";
	}
	return nullptr;
}

const char *CudaMathFunctionFp32(const std::string &name, size_t arity, bool fast_math) {
	if (arity == 1) {
		if (name == "sqrt") return "sqrtf";
		if (name == "exp") return fast_math ? "__expf" : "expf";
		if (name == "ln") return fast_math ? "__logf" : "logf";
		if (name == "log10" || name == "log") return fast_math ? "__log10f" : "log10f";
		if (name == "log2") return fast_math ? "__log2f" : "log2f";
		if (name == "sin") return fast_math ? "__sinf" : "sinf";
		if (name == "cos") return fast_math ? "__cosf" : "cosf";
		if (name == "tan") return fast_math ? "__tanf" : "tanf";
		if (name == "asin") return "asinf";
		if (name == "acos") return "acosf";
		if (name == "atan") return "atanf";
		if (name == "floor") return "floorf";
		if (name == "ceil" || name == "ceiling") return "ceilf";
		if (name == "radians") return "__vgpu_radians";
		if (name == "degrees") return "__vgpu_degrees";
	}
	if (arity == 2) {
		if (name == "pow" || name == "power") return "powf";
		if (name == "atan2") return "atan2f";
	}
	return nullptr;
}

const char *CudaTypeName(GpuValueType type) {
	switch (type) {
	case GpuValueType::INT16:
		return "short";
	case GpuValueType::INT32:
		return "int";
	case GpuValueType::INT64:
		return "long long";
	case GpuValueType::FLOAT32:
		return "float";
	case GpuValueType::FLOAT64:
		return "double";
	case GpuValueType::BOOLEAN:
		return "bool";
	case GpuValueType::DICTIONARY_STRING:
		throw GpuUnsupportedExpression("dictionary-encoded strings are not usable directly in a fused "
		                               "arithmetic/comparison expression");
	}
	throw GpuUnsupportedExpression("unknown GpuValueType");
}

//! Formats a bound constant as a CUDA literal matching `type`. Deliberately does NOT use Value::ToString()
//! -- that's meant for human-readable SQL formatting (locale-sensitive, may add type suffixes) and isn't
//! guaranteed to round-trip as valid, precise CUDA C++ literal syntax. std::ostringstream with explicit
//! precision is used instead so doubles survive the round trip exactly.
//! Formats a floating constant as a VALID CUDA literal, with `suffix` ("f" for float, "" for double).
//!
//! The decimal point is the whole point. `operator<<` prints 2.0f as "2", so appending the float suffix
//! produced `2f` -- which is not a C++ float literal at all, it parses as a user-defined literal and
//! fails NVRTC with "user-defined literal operator not found". That made `SELECT f32 * 2.0 FROM t`
//! ERROR the query instead of running, and unlike a routing decline an NVRTC failure happens at
//! execution time, so there is no fallback: the query dies. Any whole-numbered FLOAT constant hit it.
//! For double it was merely sloppy (`2` is an int literal that converts), but the same fix applies.
//!
//! Non-finite values are refused rather than emitted: "inf"/"nan" are not literals either, and CUDA's
//! INFINITY/NAN macros would silently change overflow semantics versus DuckDB.
std::string FormatFloatingLiteral(double value, int precision, const char *suffix) {
	if (std::isnan(value) || std::isinf(value)) {
		throw GpuUnsupportedExpression("non-finite floating constants are not representable as a CUDA literal");
	}
	std::ostringstream oss;
	oss << std::setprecision(precision) << value;
	auto text = oss.str();
	// Force a decimal point unless the formatter already produced one (or an exponent, which also makes
	// it a floating literal).
	if (text.find('.') == std::string::npos && text.find('e') == std::string::npos &&
	    text.find('E') == std::string::npos) {
		text += ".0";
	}
	return text + suffix;
}

std::string FormatConstantLiteral(const Value &value, GpuValueType type) {
	if (value.IsNull()) {
		// NULL handling (a proper null mask on the output) is not implemented in this first pass -- a
		// constant NULL folded into arithmetic would silently produce a wrong (non-NULL) result if we let
		// it through, which is worse than refusing to translate at all.
		throw GpuUnsupportedExpression("NULL constants are not yet supported in GPU expressions");
	}
	std::ostringstream oss;
	switch (type) {
	case GpuValueType::INT16:
		oss << value.GetValue<int16_t>();
		return oss.str();
	case GpuValueType::INT32:
		oss << value.GetValue<int32_t>();
		return oss.str();
	case GpuValueType::INT64:
		oss << value.GetValue<int64_t>() << "LL";
		return oss.str();
	case GpuValueType::FLOAT32:
		return FormatFloatingLiteral(static_cast<double>(value.GetValue<float>()), 9, "f");
	case GpuValueType::FLOAT64:
		return FormatFloatingLiteral(value.GetValue<double>(), 17, "");
	case GpuValueType::BOOLEAN:
		return value.GetValue<bool>() ? "true" : "false";
	case GpuValueType::DICTIONARY_STRING:
		throw GpuUnsupportedExpression("string constants are not usable directly in a fused "
		                               "arithmetic/comparison expression");
	}
	throw GpuUnsupportedExpression("unknown GpuValueType for constant");
}

const char *ComparisonOperator(ExpressionType type) {
	switch (type) {
	case ExpressionType::COMPARE_EQUAL:
		return "==";
	case ExpressionType::COMPARE_NOTEQUAL:
		return "!=";
	case ExpressionType::COMPARE_LESSTHAN:
		return "<";
	case ExpressionType::COMPARE_GREATERTHAN:
		return ">";
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		return "<=";
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return ">=";
	default:
		throw GpuUnsupportedExpression("unsupported comparison operator: " + ExpressionTypeToString(type));
	}
}

//! nullptr if `name` isn't one of the arithmetic operators this pass supports.
const char *ArithmeticOperator(const std::string &name) {
	if (name == "+")
		return "+";
	if (name == "-")
		return "-";
	if (name == "*")
		return "*";
	if (name == "/")
		return "/";
	return nullptr;
}

std::string TranslateRecursive(const Expression &expr, TranslationState &state) {
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::BOUND_COLUMN_REF: {
		auto &colref = expr.Cast<BoundColumnRefExpression>();
		auto &binding = colref.Binding();
		std::pair<idx_t, idx_t> key {binding.table_index.index, binding.column_index.GetIndex()};

		auto existing_slot = state.slot_by_binding.find(key);
		size_t slot;
		if (existing_slot != state.slot_by_binding.end()) {
			slot = existing_slot->second;
		} else {
			auto name_it = state.bindings.find(key);
			if (name_it == state.bindings.end()) {
				throw GpuUnsupportedExpression("column reference not resolvable to a scanned column "
				                               "(likely a correlated/outer reference)");
			}
			slot = state.input_columns.size();
			state.input_columns.push_back(name_it->second);
			state.slot_by_binding[key] = slot;
		}

		auto gpu_type = MapLogicalType(colref.GetReturnType());
		std::ostringstream oss;
		// [row], not [idx]: under late materialization `row` is selection[idx], so the expression reads
		// the ORIGINAL row while the result is written to dense output slot idx. With no selection vector
		// the generated kernel sets row = idx, so this is identical for the dense case.
		oss << "((const " << CudaTypeName(gpu_type) << "*)inputs[" << slot << "])[row]";
		return oss.str();
	}
	case ExpressionClass::BOUND_CONSTANT: {
		auto &constant = expr.Cast<BoundConstantExpression>();
		auto gpu_type = MapLogicalType(constant.GetReturnType());
		return FormatConstantLiteral(constant.GetValue(), gpu_type);
	}
	case ExpressionClass::BOUND_FUNCTION: {
		auto &func_expr = expr.Cast<BoundFunctionExpression>();
		if (BoundComparisonExpression::IsComparison(expr)) {
			auto &left = BoundComparisonExpression::Left(func_expr);
			auto &right = BoundComparisonExpression::Right(func_expr);
			auto left_str = TranslateRecursive(left, state);
			auto right_str = TranslateRecursive(right, state);
			auto op = ComparisonOperator(expr.GetExpressionType());
			return "(" + left_str + " " + op + " " + right_str + ")";
		}

		auto &children = func_expr.GetChildren();
		auto function_name = func_expr.Function().GetName().GetIdentifierName();

		if (function_name == "__cast") {
			// DuckDB binds CAST as a function call. Only LOSSLESS conversions are translated: the GPU must
			// produce bit-identical results to DuckDB's own cast, and two families do not.
			//   - Narrowing integer (BIGINT->INTEGER): DuckDB raises a conversion error when a value does
			//     not fit; a C++ static_cast silently truncates the low bits. DuckDB's optimizer inserts
			//     exactly this cast to narrow group keys when statistics prove the range, so refusing it
			//     costs us those plans -- but accepting it would turn an explicit out-of-range
			//     CAST(big AS INTEGER) into a wrong answer instead of an error. See docs/TODO.md.
			//   - Floating -> integer: DuckDB rounds and range-checks; C++ truncates toward zero and is
			//     undefined out of range.
			if (children.size() != 1) {
				throw GpuUnsupportedExpression("cast with unexpected arity");
			}
			auto source_type = MapLogicalType(children[0]->GetReturnType());
			auto target_type = MapLogicalType(expr.GetReturnType());
			if (!IsLosslessGpuCast(source_type, target_type)) {
				throw GpuUnsupportedExpression(std::string("cast from ") + CudaTypeName(source_type) + " to " +
				                               CudaTypeName(target_type) +
				                               " is not lossless, so GPU and CPU results could differ");
			}
			auto operand_str = TranslateRecursive(*children[0], state);
			if (source_type == target_type) {
				return operand_str;
			}
			return std::string("((") + CudaTypeName(target_type) + ")(" + operand_str + "))";
		}

		// Scalar math functions. These are the reason a GPU is worth using at all for an
		// arithmetic-bound query: one fused kernel evaluates the whole expression per row.
		if (auto cuda_name = CudaMathFunction(function_name, children.size())) {
			// VECTOR_GPU_STRICT_FP=1 refuses the functions that are not bit-identical to the CPU, so a
			// caller that needs exact CPU parity can have it. sqrt/abs/floor/ceil are correctly rounded on
			// both sides and stay allowed; the transcendentals differ by ~1 ULP (measured: max absolute
			// difference 1.1e-16 on values of order 1-100). That is the floor for libdevice vs host libm,
			// not an error introduced here -- and it is a different order of magnitude from the session-7
			// fast-math problem, which flushed denormals to zero.
			static const bool strict_fp = [] {
				const char *v = std::getenv("VECTOR_GPU_STRICT_FP");
				return v != nullptr && std::strcmp(v, "0") != 0;
			}();
			// Measured bit-exact against DuckDB on 300k rows (0 differing) once --fmad=false landed. abs,
			// radians and degrees are plain arithmetic; sqrt/floor/ceil are correctly rounded on both sides.
			static const std::set<std::string> exact_functions {"sqrt",    "floor",   "ceil",
			                                                   "ceiling", "abs",     "radians",
			                                                   "degrees"};
			if (strict_fp && exact_functions.find(function_name) == exact_functions.end()) {
				throw GpuUnsupportedExpression("VECTOR_GPU_STRICT_FP is set and '" + function_name +
				                               "' is not bit-identical to the CPU implementation");
			}
			if (MapLogicalType(expr.GetReturnType()) != GpuValueType::FLOAT64) {
				throw GpuUnsupportedExpression("GPU math function '" + function_name +
				                               "' is supported only where it returns DOUBLE");
			}
			bool relax = IsGpuFp32RelaxationEnabled();
			bool fast_m = IsGpuFastMathEnabled();
			if (relax) {
				if (auto fp32_name = CudaMathFunctionFp32(function_name, children.size(), fast_m)) {
					std::string call = "(double)(" + std::string(fp32_name) + "(";
					for (size_t i = 0; i < children.size(); i++) {
						if (i > 0) {
							call += ", ";
						}
						call += "(float)(" + TranslateRecursive(*children[i], state) + ")";
					}
					return call + "))";
				}
			}
			std::string call = std::string(cuda_name) + "(";
			for (size_t i = 0; i < children.size(); i++) {
				if (i > 0) {
					call += ", ";
				}
				call += "(double)(" + TranslateRecursive(*children[i], state) + ")";
			}
			return call + ")";
		}
		// ABS is special: DuckDB keeps the input type, so an integer abs must stay integral rather than
		// round-tripping through double.
		if (function_name == "abs" && children.size() == 1) {
			auto operand = TranslateRecursive(*children[0], state);
			auto result_type = MapLogicalType(expr.GetReturnType());
			if (result_type == GpuValueType::FLOAT64 || result_type == GpuValueType::FLOAT32) {
				return "fabs((double)(" + operand + "))";
			}
			if (result_type == GpuValueType::INT32 || result_type == GpuValueType::INT64) {
				// Evaluates the operand twice; it is a pure column/arithmetic expression, so this is a
				// cost question rather than a correctness one.
				return "((" + operand + ") < 0 ? -(" + operand + ") : (" + operand + "))";
			}
			throw GpuUnsupportedExpression("GPU abs supports numeric types only");
		}

		if (function_name == "%") {
			if (children.size() != 2) throw GpuUnsupportedExpression("modulo with unexpected arity");
			auto left_type = MapLogicalType(children[0]->GetReturnType());
			auto left_str = TranslateRecursive(*children[0], state);
			auto right_str = TranslateRecursive(*children[1], state);
			if (left_type == GpuValueType::FLOAT64 || left_type == GpuValueType::FLOAT32) {
				return "fmod((double)(" + left_str + "), (double)(" + right_str + "))";
			} else {
				return "(" + left_str + " % " + right_str + ")";
			}
		}

		auto op = ArithmeticOperator(function_name);
		if (!op) {
			throw GpuUnsupportedExpression("unsupported function for GPU expression: " + function_name);
		}
		if (children.size() == 2) {
			auto left_str = TranslateRecursive(*children[0], state);
			auto right_str = TranslateRecursive(*children[1], state);
			return "(" + left_str + " " + op + " " + right_str + ")";
		}
		if (children.size() == 1 && op[0] == '-') {
			// Unary minus: DuckDB binds it as a 1-argument call to the "-" function.
			auto operand_str = TranslateRecursive(*children[0], state);
			return "(-" + operand_str + ")";
		}
		throw GpuUnsupportedExpression("unsupported arity for GPU arithmetic function: " + function_name);
	}
	default:
		throw GpuUnsupportedExpression("unsupported expression class: " +
		                               std::to_string(static_cast<int>(expr.GetExpressionClass())));
	}
}

} // namespace

static std::atomic<int> g_fp32_relaxation_override {-1};

bool IsGpuFp32RelaxationEnabled() {
	int v = g_fp32_relaxation_override.load();
	if (v != -1) {
		return v != 0;
	}
	const char *env = std::getenv("VECTOR_GPU_FP32_RELAXATION");
	return env && (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 || std::strcmp(env, "TRUE") == 0);
}

void SetGpuFp32RelaxationEnabled(bool enabled) {
	g_fp32_relaxation_override.store(enabled ? 1 : 0);
}

TranslatedExpression TranslateExpression(const Expression &expr, const ColumnBindingNames &bindings) {
	TranslationState state {bindings, {}, {}};
	auto cuda_expr = TranslateRecursive(expr, state);
	TranslatedExpression result;
	result.cuda_expression = std::move(cuda_expr);
	result.input_columns = std::move(state.input_columns);
	result.output_type = MapLogicalType(expr.GetReturnType());
	return result;
}

std::string WrapAsOutputStatement(const TranslatedExpression &translated) {
	std::ostringstream oss;
	oss << "((" << CudaTypeName(translated.output_type) << "*)out)[idx] = " << translated.cuda_expression << ";";
	return oss.str();
}

} // namespace vector_gpu
