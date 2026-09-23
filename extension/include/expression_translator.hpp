#pragma once

#include "duckdb/planner/expression.hpp"
#include "gpu_engine.hpp" // vector_gpu::GpuValueType

#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vector_gpu {

//! Thrown when an expression (or sub-expression) can't be translated to a GPU-fusable CUDA body --
//! aggregates, casts, string/UDF calls, and anything else outside the supported arithmetic/comparison
//! subset (see docs/KNOWN_ISSUES.md). This is a normal, EXPECTED control-flow signal, not a bug report:
//! the optimizer callback in gpu_offload_extension.cpp catches it and silently falls back to CPU
//! execution, exactly like a Table Checker rejection.
struct GpuUnsupportedExpression : std::runtime_error {
	explicit GpuUnsupportedExpression(const std::string &msg) : std::runtime_error(msg) {
	}
};

//! Controls precision relaxation: evaluating intermediate transcendentals as float on GPU.
bool IsGpuFp32RelaxationEnabled();
void SetGpuFp32RelaxationEnabled(bool enabled);

//! Maps a bound column, identified by (table_index, column_index), to the physical column name it will
//! be materialized from. Built by the caller (gpu_offload_extension.cpp) by walking the LogicalGet(s)
//! that feed the expression being translated -- this file has no opinion on where columns come from,
//! only on how to turn references to them into CUDA source.
using ColumnBindingNames = std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, std::string>;

//! Result of translating one scalar expression tree.
struct TranslatedExpression {
	//! A CUDA C++ expression (not a full statement) evaluating to the expression's value for the current
	//! thread's row, e.g. "(((const double*)inputs[0])[idx]) * (((const double*)inputs[1])[idx])".
	std::string cuda_expression;
	//! Column name feeding inputs[i], in the order first referenced. Same column referenced twice reuses
	//! the same slot rather than appearing twice.
	std::vector<std::string> input_columns;
	//! The CUDA/GPU type of the expression's result (drives both the literal formatting used while
	//! building `cuda_expression` and the output buffer type the caller must allocate).
	GpuValueType output_type;
};

//! Translates a scalar DuckDB Expression tree into a GPU-fusable CUDA expression. Supports: column
//! references (resolved against `bindings`), numeric/boolean constants, comparisons (<, <=, >, >=, =,
//! <>), and +,-,*,/ arithmetic (binary, plus unary minus). Throws GpuUnsupportedExpression for anything
//! else -- casts, aggregates, conjunctions (AND/OR), CASE, string functions, and general UDFs are all
//! explicitly out of scope for this first pass; see docs/KNOWN_ISSUES.md.
TranslatedExpression TranslateExpression(const duckdb::Expression &expr, const ColumnBindingNames &bindings);

//! Wraps a translated expression as a full CUDA statement writing its result to `out[idx]` -- ready to
//! drop straight into GpuExpr::generated_cuda_source (which CodeGenerator::WrapGridStrideLoop expects to
//! already be a complete statement, not a bare value).
std::string WrapAsOutputStatement(const TranslatedExpression &translated);

} // namespace vector_gpu
