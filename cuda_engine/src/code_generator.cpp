#include <cstring>
#include "code_generator.hpp"
#include "gpu_logger.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace vector_gpu {

namespace {

#define NVRTC_CHECK(call)                                                                                            \
	do {                                                                                                              \
		nvrtcResult _res = (call);                                                                                   \
		if (_res != NVRTC_SUCCESS) {                                                                                 \
			throw std::runtime_error(std::string("NVRTC error: ") + nvrtcGetErrorString(_res) + " at " #call);       \
		}                                                                                                             \
	} while (0)

#define CU_CHECK(call)                                                                                                \
	do {                                                                                                              \
		CUresult _res = (call);                                                                                      \
		if (_res != CUDA_SUCCESS) {                                                                                  \
			const char *_msg = nullptr;                                                                              \
			cuGetErrorString(_res, &_msg);                                                                           \
			throw std::runtime_error(std::string("CUDA driver error: ") + (_msg ? _msg : "unknown") + " at " #call); \
		}                                                                                                             \
	} while (0)

//! Runtime/driver API interop: the CUDA Runtime API (used elsewhere for cudaMalloc/cudaMemcpy, e.g.
//! rmm_pool.cpp) and the driver API (needed here for cuModuleLoadDataEx/cuLaunchKernel — NVRTC only
//! produces PTX, and loading+launching that is driver-API-only) must share one primary context per
//! device, or driver-API calls will operate against a different context than the runtime's allocations.
//! This retains (or reuses, if already retained by the runtime) that shared primary context exactly once
//! per process.
CUcontext GetSharedPrimaryContext() {
	static std::once_flag init_flag;
	static CUcontext context = nullptr;
	std::call_once(init_flag, [] {
		CU_CHECK(cuInit(0));
		CUdevice device;
		CU_CHECK(cuDeviceGet(&device, 0));
		CU_CHECK(cuDevicePrimaryCtxRetain(&context, device));
	});
	CU_CHECK(cuCtxSetCurrent(context));
	return context;
}

std::string GetComputeCapabilityArch() {
	int major = 0, minor = 0;
	auto status_major = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
	auto status_minor = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
	if (status_major != cudaSuccess || status_minor != cudaSuccess) {
		throw std::runtime_error("code_generator: failed to query device compute capability");
	}
	std::ostringstream oss;
	oss << "--gpu-architecture=compute_" << major << minor;
	return oss.str();
}

//! Heap-allocated so a stable void* handle can be stored in KernelCache. Owned by the cache once
//! inserted: DestroyCompiledKernel below is registered as the cache's HandleDeleter, so eviction /
//! replacement / cache destruction unloads the module and frees this struct (session-7 fix — the old
//! TODO here said "until module unload is wired there"; it now is).
struct CompiledKernel {
	CUmodule module = nullptr;
	CUfunction function = nullptr;
};

//! KernelCache::HandleDeleter for cache-owned kernels. Best-effort on the unload: by deletion time the
//! context may already be torn down (process exit), and a deleter must not throw.
void DestroyCompiledKernel(CompiledKernelHandle handle) {
	auto *kernel = static_cast<CompiledKernel *>(handle);
	if (kernel == nullptr) {
		return;
	}
	if (kernel->module != nullptr) {
		cuModuleUnload(kernel->module);
	}
	delete kernel;
}

//! RAII so the nvrtcProgram is destroyed on every exit path — the old code leaked it whenever
//! nvrtcGetPTXSize/nvrtcGetPTX threw (session-7 fix).
struct NvrtcProgramGuard {
	nvrtcProgram prog = nullptr;
	~NvrtcProgramGuard() {
		if (prog != nullptr) {
			nvrtcDestroyProgram(&prog);
		}
	}
};

constexpr int kBlockSize = 256;
constexpr const char *kKernelName = "fused_kernel";

//! On-disk PTX cache, so a FRESH PROCESS skips NVRTC for any kernel some earlier process already
//! compiled. The in-memory KernelCache only helps within one process; the measured cost of the first
//! NVRTC compile in a cold process is hundreds of ms (it dominates gpu-trace's first-run `optimize`
//! phase), and loading cached PTX via cuModuleLoadDataEx takes single-digit ms.
//!
//! Directory: VECTOR_GPU_KERNEL_CACHE_DIR (set empty to disable), else ~/.cache/vector_gpu (POSIX) /
//! %LOCALAPPDATA%\vector_gpu (Windows); unresolvable -> disabled. Filenames carry the device compute
//! capability and NVRTC version, and the key hashes the WRAPPED source, so a stale entry can never be
//! loaded after a toolkit upgrade, a different GPU, or a change to WrapGridStrideLoop / the compile
//! options. A corrupt file is not trusted either: cuModuleLoadDataEx failing just falls back to NVRTC.
const std::string &DiskCacheDir() {
	static const std::string dir = [] {
		std::string root;
		if (const char *env = std::getenv("VECTOR_GPU_KERNEL_CACHE_DIR")) {
			root = env; // empty string = explicitly disabled
		} else {
#ifdef _WIN32
			if (const char *base = std::getenv("LOCALAPPDATA")) {
				root = std::string(base) + "\\vector_gpu";
			}
#else
			if (const char *home = std::getenv("HOME")) {
				root = std::string(home) + "/.cache/vector_gpu";
			}
#endif
		}
		if (root.empty()) {
			return std::string();
		}
		std::error_code ec;
		std::filesystem::create_directories(root, ec);
		return ec ? std::string() : root;
	}();
	return dir;
}

std::string DiskCachePath(const std::string &wrapped_hash) {
	auto &dir = DiskCacheDir();
	if (dir.empty()) {
		return std::string();
	}
	int major = 0, minor = 0;
	if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) != cudaSuccess ||
	    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0) != cudaSuccess) {
		return std::string();
	}
	int nvrtc_major = 0, nvrtc_minor = 0;
	if (nvrtcVersion(&nvrtc_major, &nvrtc_minor) != NVRTC_SUCCESS) {
		return std::string();
	}
	std::ostringstream oss;
	oss << dir << "/cc" << major << minor << "_nvrtc" << nvrtc_major << "." << nvrtc_minor << "_" << wrapped_hash
	    << ".ptx";
	return oss.str();
}

bool TryReadDiskCache(const std::string &path, std::string &out_ptx) {
	if (path.empty()) {
		return false;
	}
	std::ifstream in(path, std::ios::binary);
	if (!in) {
		return false;
	}
	std::ostringstream buffer;
	buffer << in.rdbuf();
	out_ptx = buffer.str();
	return !out_ptx.empty();
}

void WriteDiskCache(const std::string &path, const std::string &ptx) {
	if (path.empty() || ptx.empty()) {
		return;
	}
	// Write-then-rename so a concurrent reader never sees a half-written file; if two processes race the
	// rename, either winner's content is correct (same key -> same PTX). Best-effort throughout: a full
	// disk or permission problem must never fail the query that just compiled successfully.
	auto tmp = path + ".tmp";
	{
		std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
		if (!out) {
			return;
		}
		out.write(ptx.data(), static_cast<std::streamsize>(ptx.size()));
		if (!out) {
			std::error_code ec;
			std::filesystem::remove(tmp, ec);
			return;
		}
	}
	std::error_code ec;
	std::filesystem::rename(tmp, path, ec);
	if (ec) {
		std::filesystem::remove(tmp, ec);
	}
}

//! Loads PTX (from NVRTC or the disk cache) into a launchable CompiledKernel. Session-7 fix preserved:
//! the heap CompiledKernel (and, once loaded, its CUmodule) never leak when a driver call throws —
//! unique_ptr owns the struct until success, and a failed cuModuleGetFunction unloads the just-loaded
//! module before the throw propagates.
CompiledKernelHandle LoadPtxModule(const std::string &ptx) {
	GetSharedPrimaryContext();
	auto kernel = std::make_unique<CompiledKernel>();
	CU_CHECK(cuModuleLoadDataEx(&kernel->module, ptx.c_str(), 0, nullptr, nullptr));
	try {
		CU_CHECK(cuModuleGetFunction(&kernel->function, kernel->module, kKernelName));
	} catch (...) {
		cuModuleUnload(kernel->module);
		throw;
	}
	return static_cast<CompiledKernelHandle>(kernel.release());
}

//! The actual cuLaunchKernel call, shared by CodeGenerator::Launch and ::LaunchPreloaded. Both build the
//! same seven kernel arguments; they differ only in WHERE the two pointer arrays came from (this
//! generator's shared scratch, or the caller's own device memory). Factored out so the argument order and
//! the grid sizing have exactly one definition -- a silent mismatch between two copies of this would
//! misbind kernel parameters rather than fail to compile.
void LaunchFused(CUfunction function, const void *const *inputs_device, int input_count, void *output, size_t n,
                 const uint32_t *selection, CUstream stream, const bool *const *input_valid_device,
                 bool *out_valid) {
	unsigned long long n64 = static_cast<unsigned long long>(n);
	auto inputs_arg = inputs_device;
	auto selection_arg = selection;
	auto input_valid_arg = input_valid_device;
	auto count_arg = input_count;
	void *kernel_args[] = {&inputs_arg, &output, &n64, &selection_arg, &input_valid_arg, &count_arg, &out_valid};

	unsigned int grid_size = static_cast<unsigned int>((n + kBlockSize - 1) / kBlockSize);
	VGPU_LOG(LogLevel::DEBUG, "launch",
	         "\"n\":" + std::to_string(n) + ",\"grid\":" + std::to_string(grid_size) +
	             ",\"block\":" + std::to_string(kBlockSize) + ",\"inputs\":" + std::to_string(input_count));
	auto launch_status = cuLaunchKernel(function, grid_size, 1, 1, kBlockSize, 1, 1, 0, stream, kernel_args, nullptr);
	if (launch_status != CUDA_SUCCESS) {
		const char *msg = nullptr;
		cuGetErrorString(launch_status, &msg);
		throw std::runtime_error(std::string("CodeGenerator::Launch: cuLaunchKernel failed: ") +
		                         (msg ? msg : "unknown"));
	}

	// No cudaStreamSynchronize here any more: nothing needs freeing, and the caller's next same-stream
	// operation is already ordered after this kernel. cudaPeekAtLastError still catches an invalid launch
	// configuration immediately (it does not wait for execution), so the cheap, precisely-attributable
	// class of failure is still reported from here — see the header for what moves to the caller's sync.
	auto launch_error = cudaPeekAtLastError();
	if (launch_error != cudaSuccess) {
		throw std::runtime_error(std::string("CodeGenerator::Launch: kernel launch failed: ") +
		                         cudaGetErrorString(launch_error));
	}
}

} // namespace

CodeGenerator::CodeGenerator(KernelCache &cache) : cache_(cache) {
	// Make the cache the owner of every handle this generator inserts (session-7 fix: eviction /
	// replacement / cache destruction now unload the CUmodule instead of leaking it). Idempotent —
	// multiple generators sharing one cache all register the same function.
	cache_.SetHandleDeleter(&DestroyCompiledKernel);
}

std::string CodeGenerator::WrapGridStrideLoop(const std::string &expression_body) const {
	// Standard grid-stride skeleton. `expression_body` is expected to already be a valid CUDA statement
	// operating on `idx` that reads `inputs[]` arrays and writes `out[idx]`.
	std::ostringstream oss;
	// Two indices, deliberately distinct:
	//   idx = output slot (always dense, 0..n)
	//   row = input row to read (idx when dense; selection[idx] under late materialization)
	// expression_translator emits input reads as [row] and the output write as [idx], so the SAME
	// generated body works with and without a selection vector -- a null `sel` collapses row to idx.
	// Small preamble for functions DuckDB has but CUDA does not provide as builtins.
	oss << "__device__ __forceinline__ double __vgpu_radians(double d) { return d * 0.017453292519943295; }\n"
	       "__device__ __forceinline__ double __vgpu_degrees(double r) { return r * 57.29577951308232; }\n";
	// input_valid/input_count/out_valid: NULL propagation for scalar expressions. A row is valid in the
	// output iff every referenced input is valid at that row (ordinary SQL semantics for +,-,*,/ and
	// comparisons -- any NULL operand makes the whole expression NULL). `input_valid` may itself be null
	// (no referenced column can ever be null) or contain a mix of null/non-null per-column pointers (only
	// SOME referenced columns are nullable); either way costs nothing when out_valid is also null, which
	// is the common case (FILTER predicates check validity themselves -- see gpu_executor.cu -- rather
	// than exposing it here).
	oss << "extern \"C\" __global__ void " << kKernelName
	    << "(const void **inputs, void *out, unsigned long long n, const unsigned int *sel,\n"
	       "     const bool **input_valid, int input_count, bool *out_valid) {\n"
	       "  for (unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; idx < n;\n"
	       "       idx += (unsigned long long)blockDim.x * gridDim.x) {\n"
	       "    const unsigned long long row = (sel == 0) ? idx : (unsigned long long)sel[idx];\n"
	       "    (void)row;\n"
	       "    if (out_valid != 0) {\n"
	       "      bool __vgpu_valid = true;\n"
	       "      if (input_valid != 0) {\n"
	       "        for (int __vgpu_i = 0; __vgpu_i < input_count; __vgpu_i++) {\n"
	       "          if (input_valid[__vgpu_i] != 0 && !input_valid[__vgpu_i][row]) { __vgpu_valid = false; }\n"
	       "        }\n"
	       "      }\n"
	       "      out_valid[idx] = __vgpu_valid;\n"
	       "    }\n"
	       "    "
	    << expression_body
	    << "\n"
	       "  }\n"
	       "}\n";
	return oss.str();
}

static std::atomic<int> g_fast_math_override {-1};

bool IsGpuFastMathEnabled() {
	int v = g_fast_math_override.load();
	if (v != -1) {
		return v != 0;
	}
	const char *env = std::getenv("VECTOR_GPU_FAST_MATH");
	return env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0 || strcmp(env, "TRUE") == 0);
}

void SetGpuFastMathEnabled(bool enabled) {
	g_fast_math_override.store(enabled ? 1 : 0);
}

CompiledKernelHandle CodeGenerator::CompileWithNvrtc(const std::string &cuda_source, std::string *out_ptx) const {
	GetSharedPrimaryContext();

	NvrtcProgramGuard guard;
	NVRTC_CHECK(nvrtcCreateProgram(&guard.prog, cuda_source.c_str(), "fused_kernel.cu", 0, nullptr, nullptr));

	// --use_fast_math can be toggled via VECTOR_GPU_FAST_MATH=1 or SetGpuFastMathEnabled.
	// When off (default), strict --fmad=false is used to match CPU rounding exactly.
	bool fast_math = IsGpuFastMathEnabled();
	auto arch_flag = GetComputeCapabilityArch();
	auto arch_option = arch_flag;
	std::vector<const char *> options = {arch_option.c_str()};
	if (fast_math) {
		options.push_back("--use_fast_math");
	} else {
		options.push_back("--fmad=false");
	}

	nvrtcResult compile_result = nvrtcCompileProgram(guard.prog, static_cast<int>(options.size()), options.data());

	size_t log_size = 0;
	nvrtcGetProgramLogSize(guard.prog, &log_size);
	std::string log;
	if (log_size > 1) {
		log.resize(log_size);
		nvrtcGetProgramLog(guard.prog, &log[0]);
	}

	if (compile_result != NVRTC_SUCCESS) {
		throw std::runtime_error("NVRTC compilation failed:\n" + log +
		                         "\n--- generated source ---\n" + cuda_source);
	}

	size_t ptx_size = 0;
	NVRTC_CHECK(nvrtcGetPTXSize(guard.prog, &ptx_size));
	std::string ptx(ptx_size, '\0');
	NVRTC_CHECK(nvrtcGetPTX(guard.prog, &ptx[0]));

	auto handle = LoadPtxModule(ptx);
	if (out_ptx != nullptr) {
		*out_ptx = std::move(ptx);
	}
	return handle;
}

CompiledKernelHandle CodeGenerator::CompileOrFetch(const GpuExpr &expr) {
	// Session-7 fix: the cache key is derived HERE from the source that is actually compiled, rather
	// than trusting the caller-supplied expr.source_hash — two different expressions carrying the same
	// (e.g. default-constructed empty) hash used to silently execute whichever kernel was cached first.
	// Well-behaved callers (the extension's translator, the unit tests) already set source_hash to
	// HashSource(generated_cuda_source), so their behavior and cache hit rates are unchanged.
	bool fast_math = IsGpuFastMathEnabled();
	std::string key_input = expr.generated_cuda_source + (fast_math ? " [opt:fast_math]" : " [opt:strict]");
	auto key = KernelCache::HashSource(key_input);

	// Session-7 fix: serialize the whole get-compile-insert sequence. Unsynchronized, two threads
	// compiling the same expression raced: the second Insert freed the first thread's handle (now that
	// the cache owns handles) while that thread was still about to launch it.
	std::lock_guard<std::mutex> guard(compile_mutex_);
	auto compile_start = std::chrono::steady_clock::now();
	if (auto cached = cache_.Get(key)) {
		VGPU_LOG(LogLevel::DEBUG, "compile", "\"hit\":true,\"key\":" + GpuLogger::Quote(key.substr(0, 16)));
		return cached;
	}
	auto wrapped = WrapGridStrideLoop(expr.generated_cuda_source);

	// Disk-cache probe before NVRTC. Keyed by the hash of the WRAPPED source (not `key`, which hashes
	// only the expression body) so any change to the wrapper skeleton or its semantics invalidates the
	// cached PTX; the filename additionally carries compute capability and NVRTC version. A hit loads in
	// single-digit ms where NVRTC costs hundreds; a failed load (corrupt/stale file) falls back to NVRTC.
	auto disk_path = DiskCachePath(KernelCache::HashSource(wrapped + (fast_math ? " [opt:fast_math]" : " [opt:strict]")));
	CompiledKernelHandle handle = nullptr;
	std::string disk = "miss";
	if (disk_path.empty()) {
		disk = "off";
	} else {
		std::string cached_ptx;
		if (TryReadDiskCache(disk_path, cached_ptx)) {
			try {
				handle = LoadPtxModule(cached_ptx);
				disk = "hit";
			} catch (const std::exception &) {
				handle = nullptr; // stale or corrupt PTX — recompile below and overwrite it
			}
		}
	}
	if (handle == nullptr) {
		std::string ptx;
		handle = CompileWithNvrtc(wrapped, &ptx);
		WriteDiskCache(disk_path, ptx);
	}
	VGPU_LOG(LogLevel::INFO, "compile",
	         "\"hit\":false,\"disk\":" + GpuLogger::Quote(disk) + ",\"key\":" + GpuLogger::Quote(key.substr(0, 16)) +
	             ",\"us\":" +
	             std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(
	                                std::chrono::steady_clock::now() - compile_start)
	                                .count()));
	cache_.Insert(key, handle);
	// Fix-audit finding: do NOT return `handle` directly. Insert takes ownership, and a zero-capacity
	// cache (documented as valid) DECLINES the insert and frees the handle immediately — returning the
	// raw pointer would hand the caller a just-deleted CompiledKernel whose CUmodule is unloaded
	// (use-after-free on the next Launch). Re-fetching gives the canonical stored handle; inserting key
	// K can only ever evict a DIFFERENT key, so a capacity >= 1 cache always returns K here.
	auto stored = cache_.Get(key);
	if (stored == nullptr) {
		throw std::logic_error("CodeGenerator::CompileOrFetch: the kernel cache declined to store the "
		                       "compiled kernel (zero-capacity cache?) — CodeGenerator requires a cache "
		                       "with capacity >= 1");
	}
	return stored;
}

void CodeGenerator::Launch(CompiledKernelHandle handle, const std::vector<const void *> &inputs, void *output,
                           size_t n, const uint32_t *selection, void *stream,
                           const std::vector<const bool *> &input_valid, bool *out_valid) const {
	if (n == 0) {
		throw std::invalid_argument("CodeGenerator::Launch: n must be > 0 (caller should special-case "
		                            "empty input rather than launching a zero-element grid)");
	}
	// Session-7 fix: a null handle used to be dereferenced below (while n was carefully validated).
	if (handle == nullptr) {
		throw std::invalid_argument("CodeGenerator::Launch: handle must not be null — pass the value returned "
		                            "by CompileOrFetch");
	}
	if (!input_valid.empty() && input_valid.size() != inputs.size()) {
		throw std::invalid_argument("CodeGenerator::Launch: input_valid must be empty or exactly "
		                            "inputs.size() entries (one slot per referenced column, nullptr where "
		                            "that column has no nulls)");
	}
	// Runtime/driver API mixing: every driver-API call below (cuLaunchKernel) must run against the same
	// context the runtime API uses, or the kernel launches into a different context than the one holding
	// the caller's device pointers. GetSharedPrimaryContext() is what guarantees that.
	GetSharedPrimaryContext();
	auto *kernel = static_cast<CompiledKernel *>(handle);

	std::lock_guard<std::mutex> guard(launch_mutex_);

	// Device-side array of input pointers: `inputs` itself is a host std::vector, but the kernel
	// dereferences `inputs[i]` on the GPU, so the pointer array must also live in device memory.
	//
	// P1: this buffer is persistent and grown on demand rather than cudaMalloc'd/cudaFree'd per launch.
	// Reusing it is safe because the copy below, the kernel, and every caller operation all run on the
	// legacy default stream, so the next launch's overwrite is ordered after the previous kernel's reads.
	auto bytes = inputs.empty() ? sizeof(void *) : inputs.size() * sizeof(void *);
	if (bytes > input_ptrs_capacity_) {
		void *grown = nullptr;
		auto malloc_status = cudaMalloc(&grown, bytes);
		if (malloc_status != cudaSuccess) {
			throw std::runtime_error(
			    std::string("CodeGenerator::Launch: cudaMalloc for input pointer array failed: ") +
			    cudaGetErrorString(malloc_status));
		}
		// Only reached when a wider expression than any seen before is launched, so the old buffer is no
		// longer needed. Freeing it here (not per launch) is still ordered after prior default-stream work.
		cudaFree(input_ptrs_device_);
		input_ptrs_device_ = grown;
		input_ptrs_capacity_ = bytes;
	}

	auto cu_stream = reinterpret_cast<CUstream>(stream); // cudaStream_t and CUstream are the same type
	                                                      // (documented Runtime/driver interop), so a
	                                                      // caller holding a cudaStream_t can pass it here
	                                                      // as an opaque void* without this header (or its
	                                                      // caller) needing a CUDA Runtime include.

	if (!inputs.empty()) {
		// Session-7 fix: this return code was ignored — a failed upload would have launched the kernel
		// over an uninitialized device pointer array.
		// Async + stream-ordered: the previous synchronous cudaMemcpy blocked the issuing CPU thread for
		// the transfer's duration even though it is a handful of pointers (a few hundred bytes at most).
		// On `stream`, it queues behind (and orders correctly after) any H2D the caller already issued on
		// the same stream for the column data these pointers reference.
		auto copy_status = cudaMemcpyAsync(input_ptrs_device_, inputs.data(), bytes, cudaMemcpyHostToDevice, cu_stream);
		if (copy_status != cudaSuccess) {
			throw std::runtime_error(
			    std::string("CodeGenerator::Launch: cudaMemcpyAsync of input pointer array failed: ") +
			    cudaGetErrorString(copy_status));
		}
	}

	const bool **input_valid_arg = nullptr;
	if (!input_valid.empty()) {
		auto valid_bytes = input_valid.size() * sizeof(const bool *);
		if (valid_bytes > input_valid_capacity_) {
			void *grown = nullptr;
			auto malloc_status = cudaMalloc(&grown, valid_bytes);
			if (malloc_status != cudaSuccess) {
				throw std::runtime_error(
				    std::string("CodeGenerator::Launch: cudaMalloc for input validity array failed: ") +
				    cudaGetErrorString(malloc_status));
			}
			cudaFree(input_valid_device_);
			input_valid_device_ = grown;
			input_valid_capacity_ = valid_bytes;
		}
		auto copy_status = cudaMemcpyAsync(input_valid_device_, input_valid.data(), valid_bytes,
		                                   cudaMemcpyHostToDevice, cu_stream);
		if (copy_status != cudaSuccess) {
			throw std::runtime_error(
			    std::string("CodeGenerator::Launch: cudaMemcpyAsync of input validity array failed: ") +
			    cudaGetErrorString(copy_status));
		}
		input_valid_arg = static_cast<const bool **>(input_valid_device_);
	}

	// `selection` is a plain device pointer (or null); the kernel branches on it per row.
	LaunchFused(kernel->function, static_cast<const void *const *>(input_ptrs_device_),
	            static_cast<int>(inputs.size()), output, n, selection, cu_stream, input_valid_arg, out_valid);
}

void CodeGenerator::LaunchPreloaded(CompiledKernelHandle handle, const void *const *inputs_device, int input_count,
                                    void *output, size_t n, const uint32_t *selection, void *stream,
                                    const bool *const *input_valid_device, bool *out_valid) const {
	if (n == 0) {
		throw std::invalid_argument("CodeGenerator::LaunchPreloaded: n must be > 0 (caller should special-case "
		                            "empty input rather than launching a zero-element grid)");
	}
	if (handle == nullptr) {
		throw std::invalid_argument("CodeGenerator::LaunchPreloaded: handle must not be null — pass the value "
		                            "returned by CompileOrFetch");
	}
	if (input_count < 0 || (input_count > 0 && inputs_device == nullptr)) {
		throw std::invalid_argument("CodeGenerator::LaunchPreloaded: input_count must be >= 0 and inputs_device "
		                            "must be non-null whenever it is > 0");
	}
	// Same runtime/driver context requirement as Launch: cuLaunchKernel must run against the context that
	// owns the caller's device pointers.
	GetSharedPrimaryContext();
	auto *kernel = static_cast<CompiledKernel *>(handle);
	// No launch_mutex_ here, deliberately: this overload reads no member state, so there is nothing for a
	// concurrent launch on another stream to collide with. That absence IS the reason this overload exists
	// -- see the header.
	LaunchFused(kernel->function, inputs_device, input_count, output, n, selection,
	            reinterpret_cast<CUstream>(stream), input_valid_device, out_valid);
}

CodeGenerator::~CodeGenerator() {
	// May run at process teardown after the CUDA context is already gone; a failure here has nothing to
	// report to and nothing to clean up beyond what context destruction already reclaims.
	cudaFree(input_ptrs_device_);
	cudaFree(input_valid_device_);
}

} // namespace vector_gpu
