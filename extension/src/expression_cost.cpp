#include "expression_cost.hpp"

#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"

#include <string>

namespace vector_gpu {

using namespace duckdb;

namespace {

//! Per-call weight of a scalar function, mirroring CudaMathFunction in expression_translator.cpp.
//!
//! KEEP THE TWO LISTS IN LOCKSTEP. A function the translator can lower but this table does not know
//! scores UNKNOWN (1.0), which under-counts a transcendental by 30x and quietly makes the query look
//! CPU-shaped -- a routing regression with no symptom other than "the GPU stopped being used". The
//! arity split is the translator's too: `log` is one function name with different meanings, and `pow`
//! only exists at arity 2.
double MathFunctionWeight(const std::string &name, size_t arity) {
	if (arity == 1) {
		if (name == "sqrt") {
			return ExpressionCostVisitor::SQRT;
		}
		if (name == "exp" || name == "ln" || name == "log" || name == "log2" || name == "log10" ||
		    name == "sin" || name == "cos" || name == "tan" || name == "asin" || name == "acos" ||
		    name == "atan") {
			return ExpressionCostVisitor::TRANSCENDENTAL;
		}
		if (name == "floor" || name == "ceil" || name == "ceiling" || name == "abs" || name == "radians" ||
		    name == "degrees") {
			return ExpressionCostVisitor::ROUNDING;
		}
	}
	if (arity == 2) {
		// pow is a full exp/log pair internally, atan2 a transcendental with a quadrant fixup.
		if (name == "pow" || name == "power" || name == "atan2") {
			return ExpressionCostVisitor::TRANSCENDENTAL;
		}
	}
	return 0.0; // not a math function -- the caller falls through to arithmetic/unknown handling
}

double ArithmeticWeight(const std::string &name) {
	if (name == "+" || name == "-") {
		return ExpressionCostVisitor::ADD_SUB;
	}
	if (name == "*" || name == "/") {
		return ExpressionCostVisitor::MUL_DIV;
	}
	return 0.0;
}

} // namespace

double ExpressionCostVisitor::VisitChildren(const Expression &expr) {
	double total = 0.0;
	ExpressionIterator::EnumerateChildren(expr, [&](const Expression &child) { total += Visit(child); });
	return total;
}

double ExpressionCostVisitor::VisitFunction(const Expression &expr) {
	// Comparisons bind as functions in this DuckDB version, so they arrive here rather than as their own
	// expression class. One predicate evaluation, at add/subtract cost.
	if (BoundComparisonExpression::IsComparison(expr)) {
		return ADD_SUB + VisitChildren(expr);
	}

	auto &func_expr = expr.Cast<BoundFunctionExpression>();
	auto function_name = func_expr.Function().GetName().GetIdentifierName();
	const auto arity = func_expr.GetChildren().size();
	const auto children_cost = VisitChildren(expr);

	if (function_name == "__cast") {
		// DuckDB binds CAST as a function call. A cast to the type the operand already has is a no-op the
		// translator drops entirely, so it must not be charged for -- DuckDB's own type resolution
		// inserts plenty of these, and charging them would score plans by how many casts the binder
		// happened to add.
		if (arity == 1 && func_expr.GetChildren()[0]->GetReturnType() == expr.GetReturnType()) {
			return children_cost;
		}
		return CAST + children_cost;
	}

	const auto math_weight = MathFunctionWeight(function_name, arity);
	if (math_weight != 0.0) {
		return math_weight + children_cost;
	}
	// Unary minus binds as a 1-argument "-": one negation, same weight as a subtraction.
	const auto arithmetic_weight = ArithmeticWeight(function_name);
	if (arithmetic_weight != 0.0) {
		return arithmetic_weight + children_cost;
	}
	return UNKNOWN + children_cost;
}

double ExpressionCostVisitor::Visit(const Expression &expr) {
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::BOUND_COLUMN_REF:
	case ExpressionClass::BOUND_CONSTANT:
		// Loads, not arithmetic. The bytes a column reference moves are exactly the denominator this
		// count is divided by (GpuPlanShape::scanned_bytes), so charging for them here would put the same
		// quantity on both sides of the ratio.
		return 0.0;
	case ExpressionClass::BOUND_FUNCTION:
		return VisitFunction(expr);
	default:
		// Unreachable for anything that actually routes -- expression_translator.cpp refuses every other
		// expression class, so a plan containing one never reaches the cost model. Handled anyway, at the
		// unknown weight, so that widening the translator later cannot silently produce a NaN or a zero
		// where a cost belongs.
		return UNKNOWN + VisitChildren(expr);
	}
}

double EstimateExpressionOpsPerRow(const Expression &expr) {
	ExpressionCostVisitor visitor;
	return visitor.Visit(expr);
}

} // namespace vector_gpu
