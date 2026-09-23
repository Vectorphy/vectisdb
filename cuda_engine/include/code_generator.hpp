#pragma once

#include "gpu_engine.hpp"
#include "kernel_cache.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace vector_gpu {

//! Controls NVRTC compilation flags: --use_fast_math vs strict --fmad=false IEEE compliance.
bool IsGpuFastMathEnabled();
void SetGpuFastMathEnabled(bool enabled);

//! Fuses a GpuExpr's scalar computation into a single CUDA __global__ function, compiles it via NVRTC
//! (or fetches it from KernelCache on a hash hit), and returns a launchable handle.
//! Mirrors docs/ARCHITECTURE.md's "Mechanism of Operator Fusion" section: all inputs for one thread are
//! loaded into registers once, the entire expression evaluates in-register/L1, and the result is written
//! back to global memory exactly once.
//!
//! IMPORTANT (see extension/src/gpu_offload_extension.cpp TODO): the AST traversal that turns a DuckDB
//! Expression tree into `GpuExpr::generated_cuda_source` is NOT implemented here. This class takes an
//! already-generated per-thread statement body (e.g. "((double*)out)[idx] = a * b;" with `a`/`b` already
//! loaded from `inputs[]`) and handles the NVRTC/driver-API mechanics: wrapping, compiling, caching,
//! launching. That split is deliberate — it lets this half be built and tested against the real CUDA
//! Toolkit independently of DuckDB's Expression AST shape.
class CodeGenerator {
public:
	//! Registers DestroyCompiledKernel as `cache`'s HandleDeleter, making the cache own inserted
	//! kernels (unloading modules on eviction/replacement/destruction — session-7 fix). Defined in
	//! code_generator.cpp because only it knows the concrete type behind CompiledKernelHandle.
	explicit CodeGenerator(KernelCache &cache);

	//! Releases the persistent input-pointer scratch buffer (see `input_ptrs_device_`).
	~CodeGenerator();

	CodeGenerator(const CodeGenerator &) = delete;
	CodeGenerator &operator=(const CodeGenerator &) = delete;

	//! Returns a compiled, launchable kernel for `expr`. Compiles via NVRTC on cache miss.
	//! The cache key is derived internally as HashSource(expr.generated_cuda_source) — the
	//! caller-supplied expr.source_hash is NOT trusted as a key (session-7 fix: a shared/empty hash
	//! across different expressions used to execute the wrong cached kernel).
	//! The returned handle is BORROWED from the cache: valid until the next expression is compiled
	//! through this cache, so launch it promptly and don't store it across calls.
	//! Thread-safe: the get-compile-insert sequence is serialized per generator.
	//! Throws std::runtime_error with the NVRTC/driver error log on compilation or module-load failure —
	//! a fused-kernel compile failure is a bug in the code generator or an unsupported expression that
	//! should have been rejected earlier by TableChecker, not a silent-fallback condition.
	CompiledKernelHandle CompileOrFetch(const GpuExpr &expr);

	//! Launches a previously compiled kernel over `n` elements. `inputs` are device pointers (already on
	//! GPU), `output` is a device pointer sized for `n` elements of the kernel's output type. Throws
	//! std::invalid_argument if n == 0 (nothing to launch is a caller bug, not a valid empty-result case —
	//! callers should special-case empty input upstream).
	//!
	//! ASYNCHRONOUS as of the P1 pass: the kernel is launched on the legacy default stream and this call
	//! does NOT block for it. It used to end in cudaDeviceSynchronize() purely so it could cudaFree its
	//! own scratch buffer; that buffer is now persistent (reused and grown across launches), which removes
	//! both the free and the reason to wait. Every caller's subsequent work — Thrust `thrust::device`
	//! algorithms, further kernels, and the final D2H copy — also runs on the legacy default stream and is
	//! therefore ordered after this kernel, so results are never read early.
	//!
	//! Consequence for error reporting: launch-configuration errors are still detected here, but a fault
	//! DURING execution now surfaces at the caller's next synchronization point rather than inside this
	//! call, so its message may name that later operation. Callers that need a precise attribution should
	//! synchronize themselves right after launching.
	//! `selection`, when non-null, is a LATE-MATERIALIZATION selection vector: `n` is the number of
	//! surviving rows, and thread i reads its inputs at row `selection[i]` while writing output slot i.
	//! Pass nullptr for a dense pass over rows [0, n). See WrapGridStrideLoop for the generated shape.
	//!
	//! `stream`: an opaque cudaStream_t (or nullptr for the legacy default stream), typed as void* so this
	//! header stays free of a CUDA Runtime include -- see gpu_engine.hpp's header comment on the same
	//! deliberate split. The kernel launch, and this call's own upload of `inputs`/`input_valid` into their
	//! device-side scratch arrays, are ordered on this stream; nothing here calls cudaStreamSynchronize, so
	//! the caller decides when (or whether) to wait.
	//!
	//! `input_valid[i]`, when this vector is non-empty, is a device pointer to a dense bool array (one byte
	//! per BASE row, non-zero = valid) for inputs[i], or nullptr if that particular column has no nulls.
	//! Pass an empty vector when no referenced input can be null -- the generated kernel skips all validity
	//! work in that case. `out_valid`, when non-null, receives one byte per OUTPUT row (dense, indexed by
	//! idx like `out` itself): the AND of every referenced input's validity bit at that row, i.e. ordinary
	//! SQL NULL propagation for a scalar expression. Pass nullptr when the caller does not need the
	//! expression's nullability (e.g. a FILTER predicate's own boolean mask, which is combined into
	//! selectivity by the caller rather than exposed as a nullable column).
	void Launch(CompiledKernelHandle handle, const std::vector<const void *> &inputs, void *output, size_t n,
	            const uint32_t *selection = nullptr, void *stream = nullptr,
	            const std::vector<const bool *> &input_valid = {}, bool *out_valid = nullptr) const;

	//! Same launch, but the caller has ALREADY placed the input-pointer array (and, if it passes one, the
	//! input-validity pointer array) in device memory and owns it. `inputs_device` is a DEVICE pointer to
	//! `input_count` device pointers; `input_valid_device` is either nullptr (the expression references
	//! nothing nullable) or a DEVICE pointer to `input_count` entries, each a dense per-BASE-row bool array
	//! or nullptr. Everything else means exactly what it does in Launch above.
	//!
	//! WHY THIS EXISTS, and when to prefer it: Launch() above stages its pointer arrays through scratch
	//! buffers SHARED by every caller of this generator (input_ptrs_device_/input_valid_device_), and that
	//! is only safe because every launch in the legacy executor is issued on ONE process-wide stream, so
	//! the next launch's overwrite of the scratch is ordered after the previous kernel's reads. The moment
	//! a second stream launches through the same generator, that ordering is gone: launch B's
	//! cudaMemcpyAsync into the shared scratch can land while launch A's kernel, on another stream, has
	//! not yet read it -- A then dereferences B's pointers and silently computes over the wrong columns.
	//! GpuStreamPipeline (gpu_stream_pipeline.hpp) runs exactly that way, so it pre-uploads one pointer
	//! array per pipeline stage ONCE (its device buffers are fixed for the pipeline's lifetime) and
	//! launches through here instead. This overload touches no shared mutable state at all, so it needs no
	//! lock and has no cross-stream hazard -- and it also drops two small H2D copies off every launch.
	void LaunchPreloaded(CompiledKernelHandle handle, const void *const *inputs_device, int input_count,
	                     void *output, size_t n, const uint32_t *selection = nullptr, void *stream = nullptr,
	                     const bool *const *input_valid_device = nullptr, bool *out_valid = nullptr) const;

private:
	//! Wraps the fused expression body in a standard CUDA grid-stride loop skeleton so the workload is
	//! evenly balanced across all available SMs regardless of hardware generation.
	std::string WrapGridStrideLoop(const std::string &expression_body) const;

	//! Invokes nvrtcCreateProgram / nvrtcCompileProgram targeting the active device's compute
	//! capability (deliberately WITHOUT --use_fast_math — see session-7 KI-6: fast-math silently
	//! diverged FLOAT32 results from the CPU engine), then cuModuleLoadDataEx + cuModuleGetFunction to
	//! obtain a launchable CUfunction. Returns a heap-allocated `CompiledKernel*` (see
	//! code_generator.cpp) cast to CompiledKernelHandle — ownership transfers to the caller (normally
	//! immediately to KernelCache, which frees it via the registered deleter on eviction/replacement).
	//! Leak-safe on failure: the nvrtcProgram, the CompiledKernel, and a loaded-but-unusable CUmodule
	//! are all released on every throw path.
	//! `out_ptx`, when non-null, receives the generated PTX on success — CompileOrFetch persists it to
	//! the on-disk PTX cache so later PROCESSES skip NVRTC entirely (the in-memory KernelCache only
	//! helps within one process).
	CompiledKernelHandle CompileWithNvrtc(const std::string &cuda_source, std::string *out_ptx = nullptr) const;

	//! Persistent device-side scratch for the array of input pointers a kernel dereferences. Reused and
	//! grown across launches instead of being cudaMalloc'd and cudaFree'd every single launch — those two
	//! calls are device-wide serializing points, so paying them per kernel was pure latency. Guarded by
	//! `launch_mutex_`; mutable because Launch is const.
	mutable void *input_ptrs_device_ = nullptr;
	mutable size_t input_ptrs_capacity_ = 0;
	//! Same persistent-scratch treatment as input_ptrs_device_ above, for `input_valid`'s device pointer
	//! array. Separate allocation because it is grown/read independently -- most launches pass no
	//! validity pointers at all (input_valid.empty()), so this buffer is only ever touched by expressions
	//! that reference a nullable column.
	mutable void *input_valid_device_ = nullptr;
	mutable size_t input_valid_capacity_ = 0;
	//! Serializes the reuse of the shared scratch buffers above across concurrent Launch calls. Safe to
	//! hold for the whole call (not just the buffer-growth section): only one GPU chunk is ever in flight
	//! at a time in this engine's execution model (see gpu_executor_internal.hpp's ExecContext::stream
	//! comment), so this has never been a throughput bottleneck in practice.
	mutable std::mutex launch_mutex_;

	KernelCache &cache_;
	//! Serializes CompileOrFetch's get-compile-insert sequence (session-7 fix): without it, two threads
	//! compiling the same expression raced, and the loser's Insert freed the winner's still-in-use
	//! handle now that the cache owns handles.
	std::mutex compile_mutex_;
};

} // namespace vector_gpu
