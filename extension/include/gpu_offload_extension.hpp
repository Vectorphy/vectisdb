#pragma once

#include "duckdb/main/config.hpp"

#include <cstdint>

namespace vector_gpu {

//! Registers the GPU offload OptimizerExtension (Table Checker + LogicalOperator -> GpuPlanNode
//! translation) on `config`. Factored out of the DUCKDB_CPP_EXTENSION_ENTRY entrypoint so it can also be
//! called directly by tests that link statically against DuckDB instead of loading a `.duckdb_extension`
//! at runtime (see extension/tests/test_translator_integration.cpp), and by the statically-linked
//! duckdb_gpu shell (vector-gpu-engine/gpu_shell).
void RegisterGpuOffloadOptimizer(duckdb::DBConfig &config);

//! Enables counting of plans the optimizer actually rewrote for GPU execution. Off by default so the
//! counter costs nothing in normal use. Exists so "did this query really run on the GPU?" is an
//! observed fact rather than an inference from the routing rules — see duckdb_gpu --gpu-trace.
void SetGpuOffloadTrace(bool enabled);

//! Number of plans rewritten to a GpuExecuteOperator since process start (0 unless tracing is on).
uint64_t GetGpuOffloadCount();

//! Cumulative wall-clock, in microseconds, spent in each phase of GPU materialization. Only accumulated
//! while tracing is on, so it costs nothing in normal use. Exists because "the GPU path is slow" is not
//! an actionable statement until you know WHICH phase is slow — the answer (the host-side scan, not the
//! device compute) is what drives the P1 data-path work. See duckdb_gpu --gpu-trace.
struct GpuPhaseTimings {
	//! DuckDB storage scan + host-side GpuColumn conversion (table_scanner.cpp / vector_converter.cpp).
	uint64_t scan_us = 0;
	//! GpuEngine::ExecutePlan: H2D upload, kernels, D2H copy-back.
	uint64_t execute_us = 0;
	//! GPU result -> ColumnDataCollection (PackGpuColumnsIntoCollection).
	uint64_t pack_us = 0;
	//! Rows handed to the engine, summed across input columns' row counts.
	uint64_t input_rows = 0;
	//! The optimizer callback itself: TableChecker routing decision + LogicalOperator -> GpuPlanNode
	//! translation, for every query the extension sees (including ones it declines to offload).
	uint64_t optimize_us = 0;
};

GpuPhaseTimings GetGpuPhaseTimings();

//! Internal: records one materialization's phase timings. A no-op unless tracing is on. Called by
//! PhysicalGpuExecute::MaterializeOnce; declared here only because the counters live alongside the
//! offload counter in gpu_offload_extension.cpp.
void RecordGpuPhaseTimings(uint64_t scan_us, uint64_t execute_us, uint64_t pack_us, uint64_t input_rows);

} // namespace vector_gpu
