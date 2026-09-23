#include "test_framework.hpp"
#include "code_generator.hpp"
#include "gpu_engine.hpp"
#include "kernel_cache.hpp"

#include <string>

using namespace vector_gpu;

// Pins the session-7 gpu_engine fixes (fix-audit finding: they previously had no coverage in any
// suite) plus the fix-audit use-after-free finding in CodeGenerator + zero-capacity cache.
// Requires a real GPU: ExecutePlan initializes the RmmPool on a live CUDA device.

static void TestUnknownOpTypeFailsWithExplanatoryMessage() {
	// Session-7 fix: an out-of-range op_type used to fall through the switch silently, returning
	// success=false with an EMPTY error message. GpuOpType is uint8_t-backed, so casting an
	// out-of-range value is well-defined and models a corrupted/version-skewed plan.
	GpuPlanNode plan;
	plan.op_type = static_cast<GpuOpType>(255);
	auto result = GpuEngine::Instance().ExecutePlan(plan, {});
	CHECK(result.success == false);
	CHECK(!result.error_message.empty()); // was empty before the fix
	CHECK(result.error_message.find("unknown GpuOpType") != std::string::npos);
}

static void TestDefaultConstructedPlanIsADefinedEmptyScan() {
	// GpuPlanNode::op_type is default-initialized to SCAN, so a default-constructed node takes a
	// DEFINED path rather than dispatching on an uninitialized byte. With the real executor in place
	// that path now succeeds trivially: a scan of zero columns yields zero result columns.
	// (Before the executor existed this returned success=false "not implemented yet" — the assertion
	// changed because the behavior it pinned was the stub, not a contract.)
	GpuPlanNode plan; // op_type deliberately untouched
	auto result = GpuEngine::Instance().ExecutePlan(plan, {});
	CHECK(result.success == true);
	CHECK(result.columns.empty());
	CHECK(result.error_message.empty());
}

static void TestExecutableOpsSucceedAndUnexecutableOnesFailCleanly() {
	// SCAN/FILTER/PROJECTION are genuinely executed by gpu_executor.cu; HASH_JOIN and
	// GROUP_BY_AGGREGATE have no GPU kernel and must fail with an explanatory message (the optimizer
	// separately declines to offload plans containing them, so they run on CPU rather than failing).
	for (auto op : {GpuOpType::SCAN, GpuOpType::FILTER, GpuOpType::PROJECTION}) {
		GpuPlanNode plan;
		plan.op_type = op;
		if (op != GpuOpType::SCAN) {
			plan.children.push_back(std::make_shared<GpuPlanNode>()); // an empty SCAN child
		}
		auto result = GpuEngine::Instance().ExecutePlan(plan, {});
		CHECK(result.success == true);
		CHECK(result.error_message.empty());
	}
	for (auto op : {GpuOpType::HASH_JOIN, GpuOpType::GROUP_BY_AGGREGATE}) {
		GpuPlanNode plan;
		plan.op_type = op;
		auto result = GpuEngine::Instance().ExecutePlan(plan, {});
		CHECK(result.success == false);
		CHECK(!result.error_message.empty());
	}
}

static void TestZeroCapacityCacheCompileFailsCleanlyNotUseAfterFree() {
	// Fix-audit regression: with a zero-capacity cache, Insert (as the owner) frees the declined
	// handle — CompileOrFetch used to then RETURN that freed pointer, a deterministic use-after-free
	// on the next Launch. It must now fail cleanly instead.
	KernelCache cache(0);
	CodeGenerator gen(cache);
	GpuExpr expr;
	expr.generated_cuda_source = "((double*)out)[idx] = ((const double*)inputs[0])[idx] * 2.0;";
	expr.source_hash = KernelCache::HashSource(expr.generated_cuda_source);
	CHECK_THROWS(gen.CompileOrFetch(expr)); // clean failure, not a dangling handle
}

int main() {
	RUN_TEST(TestUnknownOpTypeFailsWithExplanatoryMessage);
	RUN_TEST(TestDefaultConstructedPlanIsADefinedEmptyScan);
	RUN_TEST(TestExecutableOpsSucceedAndUnexecutableOnesFailCleanly);
	RUN_TEST(TestZeroCapacityCacheCompileFailsCleanlyNotUseAfterFree);
	TEST_MAIN_EPILOGUE();
}
