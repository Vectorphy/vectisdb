#include "test_framework.hpp"
#include "code_generator.hpp"
#include "kernel_cache.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace vector_gpu;

namespace {

void CheckCuda(cudaError_t status, const char *what) {
	if (status != cudaSuccess) {
		std::cerr << "CUDA error in " << what << ": " << cudaGetErrorString(status) << "\n";
		std::exit(1);
	}
}

//! Fused expression matching docs/ARCHITECTURE.md's worked example: out[idx] = a*b - c, reading three
//! double columns from inputs[0..2] exactly once per thread (registers, not repeated global loads).
GpuExpr MakeMulSubExpr() {
	GpuExpr expr;
	expr.generated_cuda_source = "double a = ((const double*)inputs[0])[idx];\n"
	                             "    double b = ((const double*)inputs[1])[idx];\n"
	                             "    double c = ((const double*)inputs[2])[idx];\n"
	                             "    ((double*)out)[idx] = a * b - c;";
	expr.source_hash = KernelCache::HashSource(expr.generated_cuda_source);
	return expr;
}

//! Deliberately invalid CUDA source (unmatched syntax) to test that compile failures surface as
//! exceptions rather than silently producing a null/garbage kernel.
GpuExpr MakeBrokenExpr() {
	GpuExpr expr;
	expr.generated_cuda_source = "this is not valid CUDA C++ at all {{{ )(";
	expr.source_hash = KernelCache::HashSource(expr.generated_cuda_source);
	return expr;
}

std::vector<double> RunMulSub(CodeGenerator &gen, CompiledKernelHandle kernel, const std::vector<double> &a,
                              const std::vector<double> &b, const std::vector<double> &c) {
	auto n = a.size();
	double *da = nullptr, *db = nullptr, *dc = nullptr, *dout = nullptr;
	CheckCuda(cudaMalloc(&da, n * sizeof(double)), "malloc a");
	CheckCuda(cudaMalloc(&db, n * sizeof(double)), "malloc b");
	CheckCuda(cudaMalloc(&dc, n * sizeof(double)), "malloc c");
	CheckCuda(cudaMalloc(&dout, n * sizeof(double)), "malloc out");
	CheckCuda(cudaMemcpy(da, a.data(), n * sizeof(double), cudaMemcpyHostToDevice), "copy a");
	CheckCuda(cudaMemcpy(db, b.data(), n * sizeof(double), cudaMemcpyHostToDevice), "copy b");
	CheckCuda(cudaMemcpy(dc, c.data(), n * sizeof(double), cudaMemcpyHostToDevice), "copy c");

	gen.Launch(kernel, {da, db, dc}, dout, n);

	std::vector<double> result(n);
	CheckCuda(cudaMemcpy(result.data(), dout, n * sizeof(double), cudaMemcpyDeviceToHost), "copy out");
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
	cudaFree(dout);
	return result;
}

} // namespace

static void TestCompileAndRunSimpleArray() {
	KernelCache cache;
	CodeGenerator gen(cache);
	auto expr = MakeMulSubExpr();
	auto kernel = gen.CompileOrFetch(expr);
	CHECK(kernel != nullptr);

	std::vector<double> a = {2, 3, 4, 5};
	std::vector<double> b = {10, 10, 10, 10};
	std::vector<double> c = {1, 2, 3, 4};
	auto result = RunMulSub(gen, kernel, a, b, c);
	CHECK(result.size() == 4);
	for (size_t i = 0; i < a.size(); i++) {
		double expected = a[i] * b[i] - c[i];
		CHECK(std::abs(result[i] - expected) < 1e-9);
	}
}

static void TestSingleElement() {
	KernelCache cache;
	CodeGenerator gen(cache);
	auto kernel = gen.CompileOrFetch(MakeMulSubExpr());
	auto result = RunMulSub(gen, kernel, {7.0}, {6.0}, {2.0});
	CHECK(result.size() == 1);
	CHECK(std::abs(result[0] - 40.0) < 1e-9); // 7*6 - 2 = 40
}

static void TestCacheHitReturnsSameHandleAndSkipsRecompile() {
	KernelCache cache;
	CodeGenerator gen(cache);
	auto expr = MakeMulSubExpr();
	auto handle1 = gen.CompileOrFetch(expr);
	auto handle2 = gen.CompileOrFetch(expr); // same source_hash -> must be a cache hit
	CHECK(handle1 == handle2);
	// And the cached kernel must still execute correctly (proves the cache didn't return a stale/invalid
	// handle from some prior interior state).
	auto result = RunMulSub(gen, handle2, {3.0}, {3.0}, {1.0});
	CHECK(std::abs(result[0] - 8.0) < 1e-9); // 3*3 - 1 = 8
}

static void TestCompileFailureThrows() {
	KernelCache cache;
	CodeGenerator gen(cache);
	CHECK_THROWS(gen.CompileOrFetch(MakeBrokenExpr()));
}

static void TestLaunchWithZeroElementsThrows() {
	KernelCache cache;
	CodeGenerator gen(cache);
	auto kernel = gen.CompileOrFetch(MakeMulSubExpr());
	double dummy_out;
	CHECK_THROWS(gen.Launch(kernel, {}, &dummy_out, 0));
}

static void TestDivisionProducesInfNotCrash() {
	// Edge case from the architecture doc's example expression (division by LOG(volume)) — a
	// division-by-zero on GPU floating point produces IEEE inf, not a trap/crash, and the fused kernel
	// must propagate that correctly rather than silently clamping it.
	GpuExpr expr;
	expr.generated_cuda_source = "double a = ((const double*)inputs[0])[idx];\n"
	                             "    double b = ((const double*)inputs[1])[idx];\n"
	                             "    ((double*)out)[idx] = a / b;";
	expr.source_hash = KernelCache::HashSource(expr.generated_cuda_source);

	KernelCache cache;
	CodeGenerator gen(cache);
	auto kernel = gen.CompileOrFetch(expr);

	double *da = nullptr, *db = nullptr, *dout = nullptr;
	std::vector<double> a = {10.0};
	std::vector<double> b = {0.0};
	CheckCuda(cudaMalloc(&da, sizeof(double)), "malloc a");
	CheckCuda(cudaMalloc(&db, sizeof(double)), "malloc b");
	CheckCuda(cudaMalloc(&dout, sizeof(double)), "malloc out");
	CheckCuda(cudaMemcpy(da, a.data(), sizeof(double), cudaMemcpyHostToDevice), "copy a");
	CheckCuda(cudaMemcpy(db, b.data(), sizeof(double), cudaMemcpyHostToDevice), "copy b");

	gen.Launch(kernel, {da, db}, dout, 1);

	double result = 0;
	CheckCuda(cudaMemcpy(&result, dout, sizeof(double), cudaMemcpyDeviceToHost), "copy out");
	CHECK(std::isinf(result));

	cudaFree(da);
	cudaFree(db);
	cudaFree(dout);
}

static void TestLargeArrayMultiBlockDispatch() {
	// Large enough to require many grid-stride iterations across multiple thread blocks — the most
	// likely place for an indexing bug in WrapGridStrideLoop to show up.
	KernelCache cache;
	CodeGenerator gen(cache);
	auto kernel = gen.CompileOrFetch(MakeMulSubExpr());

	const size_t n = 500'003;
	std::vector<double> a(n), b(n), c(n);
	for (size_t i = 0; i < n; i++) {
		a[i] = static_cast<double>(i % 97);
		b[i] = static_cast<double>((i * 3) % 89);
		c[i] = static_cast<double>(i % 13);
	}
	auto result = RunMulSub(gen, kernel, a, b, c);
	CHECK(result.size() == n);
	for (size_t i = 0; i < n; i += 4999) { // spot-check across the range rather than all 500k
		double expected = a[i] * b[i] - c[i];
		CHECK(std::abs(result[i] - expected) < 1e-9);
	}
}

static void TestFloat32DenormalDivisionMatchesCpu() {
	// Session-7 KI-6 regression: --use_fast_math flushed FLOAT32 denormal results to zero (proven on
	// hardware: 1e-38f / 1e5f -> 0.0 on GPU vs 9.95e-44 on CPU) and used approximate division, so an
	// offloaded SQL REAL expression diverged bitwise from DuckDB's CPU engine. With the flag removed,
	// the GPU result must match strict host IEEE division bit-for-bit.
	GpuExpr expr;
	expr.generated_cuda_source =
	    "((float*)out)[idx] = ((const float*)inputs[0])[idx] / ((const float*)inputs[1])[idx];";
	expr.source_hash = KernelCache::HashSource(expr.generated_cuda_source);
	expr.output_type = GpuValueType::FLOAT32;

	KernelCache cache;
	CodeGenerator gen(cache);
	auto kernel = gen.CompileOrFetch(expr);

	std::vector<float> a = {1.0f, 2.0f, 7.0f, 1e-38f, 3.14159265f, 1e30f, 1.0f, 123456.789f};
	std::vector<float> b = {3.0f, 7.0f, 13.0f, 1e5f, 2.71828182f, 3e-8f, 49.0f, 0.0001f};
	const size_t n = a.size();

	float *da = nullptr, *db = nullptr, *dout = nullptr;
	CheckCuda(cudaMalloc(&da, n * sizeof(float)), "malloc f32 a");
	CheckCuda(cudaMalloc(&db, n * sizeof(float)), "malloc f32 b");
	CheckCuda(cudaMalloc(&dout, n * sizeof(float)), "malloc f32 out");
	CheckCuda(cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice), "copy f32 a");
	CheckCuda(cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice), "copy f32 b");

	gen.Launch(kernel, {da, db}, dout, n);

	std::vector<float> gpu(n);
	CheckCuda(cudaMemcpy(gpu.data(), dout, n * sizeof(float), cudaMemcpyDeviceToHost), "copy f32 out");

	for (size_t i = 0; i < n; i++) {
		volatile float num = a[i], den = b[i]; // volatile blocks host-side constant folding
		float cpu = num / den;
		uint32_t cpu_bits = 0, gpu_bits = 0;
		std::memcpy(&cpu_bits, &cpu, sizeof(float));
		std::memcpy(&gpu_bits, &gpu[i], sizeof(float));
		CHECK(cpu_bits == gpu_bits); // bitwise parity, including the denormal at i == 3
	}

	cudaFree(da);
	cudaFree(db);
	cudaFree(dout);
}

static void TestNullHandleLaunchThrows() {
	// Session-7 regression: Launch validated n but dereferenced a null handle.
	KernelCache cache;
	CodeGenerator gen(cache);
	double dummy_out = 0;
	CHECK_THROWS(gen.Launch(nullptr, {}, &dummy_out, 1));
}

static void TestMismatchedSourceHashCannotServeWrongKernel() {
	// Session-7 regression: the cache key used to be the caller-supplied expr.source_hash, so two
	// different expressions carrying the same hash silently executed whichever compiled first. The key
	// is now derived from the source actually compiled — a bogus shared source_hash must not collide.
	GpuExpr add_expr;
	add_expr.generated_cuda_source = "double a = ((const double*)inputs[0])[idx];\n"
	                                 "    double b = ((const double*)inputs[1])[idx];\n"
	                                 "    double c = ((const double*)inputs[2])[idx];\n"
	                                 "    ((double*)out)[idx] = a + b + c;";
	add_expr.source_hash = "same-bogus-hash-for-both"; // deliberately shared, deliberately wrong

	GpuExpr mul_expr;
	mul_expr.generated_cuda_source = "double a = ((const double*)inputs[0])[idx];\n"
	                                 "    double b = ((const double*)inputs[1])[idx];\n"
	                                 "    double c = ((const double*)inputs[2])[idx];\n"
	                                 "    ((double*)out)[idx] = a * b * c;";
	mul_expr.source_hash = "same-bogus-hash-for-both";

	KernelCache cache;
	CodeGenerator gen(cache);
	auto add_kernel = gen.CompileOrFetch(add_expr);
	auto mul_kernel = gen.CompileOrFetch(mul_expr); // was a wrong-kernel cache hit before the fix
	CHECK(add_kernel != mul_kernel);

	auto add_result = RunMulSub(gen, add_kernel, {2.0}, {3.0}, {4.0}); // helper just launches 3-in-1-out
	auto mul_result = RunMulSub(gen, mul_kernel, {2.0}, {3.0}, {4.0});
	CHECK(std::abs(add_result[0] - 9.0) < 1e-9);  // 2+3+4
	CHECK(std::abs(mul_result[0] - 24.0) < 1e-9); // 2*3*4 — was 9.0 (the add kernel) before the fix
}

int main() {
	RUN_TEST(TestCompileAndRunSimpleArray);
	RUN_TEST(TestSingleElement);
	RUN_TEST(TestCacheHitReturnsSameHandleAndSkipsRecompile);
	RUN_TEST(TestCompileFailureThrows);
	RUN_TEST(TestLaunchWithZeroElementsThrows);
	RUN_TEST(TestDivisionProducesInfNotCrash);
	RUN_TEST(TestLargeArrayMultiBlockDispatch);
	RUN_TEST(TestFloat32DenormalDivisionMatchesCpu);
	RUN_TEST(TestNullHandleLaunchThrows);
	RUN_TEST(TestMismatchedSourceHashCannotServeWrongKernel);
	TEST_MAIN_EPILOGUE();
}
