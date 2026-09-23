#pragma once

#include "duckdb/planner/expression.hpp"

namespace vector_gpu {

//! Counts the weighted arithmetic one row costs, by walking a BOUND expression tree.
//!
//! WHY THIS EXISTS. Routing used to measure work by ROW COUNT (GpuPlanShape::amplification): rows out
//! over rows scanned. That metric cannot distinguish `a + b` from `sin(a)+cos(b)+exp(c/100)` -- both
//! produce one row per row scanned, so both amplify by exactly 1.0 -- yet session 32 measured the first
//! at 0.75-0.85x of DuckDB's CPU path and the second at 1.14-1.79x. Every mathematically dense query in
//! the benchmark set had to be routed by hand with VECTOR_GPU_MIN_AMPLIFICATION=0. This visitor supplies
//! the missing numerator.
//!
//! WHY IT LIVES ON THE EXTENSION SIDE AND NOT IN gpu_cost_model.cpp, where the rest of the cost model
//! is. cuda_engine is a standalone library with no DuckDB dependency (see the note at the top of
//! gpu_engine.hpp) -- it does not include duckdb.hpp and does not link duckdb_static, and its unit tests
//! build against neither. A bound expression tree only exists on this side of that boundary; by the time
//! a plan reaches the engine, every expression has been lowered to a CUDA source STRING, and counting
//! `sin(` occurrences in generated text is not a cost model. So the COUNTING happens here, the count
//! rides across on GpuExpr::estimated_ops_per_row, and the THRESHOLDING stays in gpu_cost_model.cpp with
//! the rest of the routing model.
//!
//! WEIGHTS ARE RELATIVE THROUGHPUT, NOT CYCLES. They rank operations against each other on the target
//! class of hardware (an FP64 multiply is worth about two adds; a libdevice transcendental costs an
//! order of magnitude more than either, since it is range reduction plus a polynomial rather than one
//! instruction). Absolute accuracy is neither achievable nor needed -- the number is divided by the
//! scanned row count and compared against a threshold, so only the ratios matter.
class ExpressionCostVisitor {
public:
	//! Addition, subtraction, unary minus, comparisons.
	static constexpr double ADD_SUB = 1.0;
	//! Multiplication and division. Division is materially worse than multiplication on real hardware;
	//! they share a weight because the routing decision has never turned on telling them apart.
	static constexpr double MUL_DIV = 2.0;
	//! sin, cos, tan, asin, acos, atan, atan2, exp, ln, log, log2, log10, pow.
	static constexpr double TRANSCENDENTAL = 30.0;
	//! sqrt sits between the two: hardware-assisted and far cheaper than a libdevice call, but not free.
	//!
	//! CALIBRATED, not guessed. At TRANSCENDENTAL's weight, `a*b + sqrt(c)` scores 34 and `a*b, a+c,
	//! sqrt(c), a-b` scores 35, so both would clear kOffloadOpsPerRow -- and both are measured LOSSES
	//! (0.88x and 0.61x, session 32). At this weight they score 7 and 8, below the threshold, which is
	//! where the measurements put them.
	static constexpr double SQRT = 4.0;
	//! floor, ceil, abs, radians, degrees -- a comparison, a mask or a single multiply.
	static constexpr double ROUNDING = 1.0;
	//! A numeric conversion instruction. Only the lossless widenings reach the GPU at all (see
	//! IsLosslessGpuCast in expression_translator.cpp); an identity cast costs nothing.
	static constexpr double CAST = 1.0;
	//! Anything this visitor does not recognise. Deliberately the CHEAPEST non-zero weight rather than a
	//! conservative-looking large one: a large weight would be conservative in the wrong direction, since
	//! high cost is what talks routing INTO offloading. An unknown function is not evidence of density.
	static constexpr double UNKNOWN = 1.0;

	//! Weighted operations `expr` performs for one row. Column references and constants are 0 -- they are
	//! loads, and the bytes they move are already the denominator this number is divided by.
	double Visit(const duckdb::Expression &expr);

private:
	double VisitFunction(const duckdb::Expression &expr);
	//! Summed cost of every child of `expr`, so an operator's own weight is added to its operands'.
	double VisitChildren(const duckdb::Expression &expr);
};

//! Convenience wrapper: the visitor is stateless, so one call needs no instance of its own.
double EstimateExpressionOpsPerRow(const duckdb::Expression &expr);

} // namespace vector_gpu
