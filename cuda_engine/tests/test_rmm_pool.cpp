#include "test_framework.hpp"
#include "rmm_pool.hpp"

using namespace vector_gpu;

// These tests share the RmmPool singleton, so each test resets it first — order-independent by
// construction rather than by accident.

static void TestInvalidFractionRejected() {
	auto &pool = RmmPool::Instance();
	pool.ResetForTesting();
	CHECK_THROWS(pool.EnsureInitialized(0.0));
	CHECK_THROWS(pool.EnsureInitialized(-0.5));
	CHECK_THROWS(pool.EnsureInitialized(1.5));
	pool.ResetForTesting();
}

static void TestBasicAllocateFree() {
	auto &pool = RmmPool::Instance();
	pool.ResetForTesting();
	pool.EnsureInitializedExactBytesForTesting(1024 * 1024); // 1 MiB test arena

	CHECK(pool.AllocatedBytes() == 0);
	auto *p = pool.Allocate(4096);
	CHECK(p != nullptr);
	CHECK(pool.AllocatedBytes() >= 4096); // >= because of 256-byte alignment rounding
	pool.Free(p, 4096);
	CHECK(pool.AllocatedBytes() == 0);
	pool.ResetForTesting();
}

static void TestZeroByteAllocateIsNullNotError() {
	auto &pool = RmmPool::Instance();
	pool.ResetForTesting();
	pool.EnsureInitializedExactBytesForTesting(1024 * 1024);
	CHECK(pool.Allocate(0) == nullptr);
	pool.ResetForTesting();
}

static void TestAllocateBeforeInitializedReturnsNull() {
	auto &pool = RmmPool::Instance();
	pool.ResetForTesting(); // uninitialized state
	CHECK(pool.Allocate(1024) == nullptr);
	CHECK(pool.HasHeadroomFor(1) == false);
}

static void TestOverAllocationFails() {
	auto &pool = RmmPool::Instance();
	pool.ResetForTesting();
	pool.EnsureInitializedExactBytesForTesting(4096);
	// Requesting more than the entire pool must fail cleanly (nullptr), not crash or wrap around.
	CHECK(pool.Allocate(1024 * 1024) == nullptr);
	pool.ResetForTesting();
}

static void TestFragmentationAndCoalescing() {
	auto &pool = RmmPool::Instance();
	pool.ResetForTesting();
	// 3 * 4096-aligned chunks fit exactly, nothing left over, so we can reason about exact headroom.
	pool.EnsureInitializedExactBytesForTesting(3 * 4096);

	auto *a = pool.Allocate(4096);
	auto *b = pool.Allocate(4096);
	auto *c = pool.Allocate(4096);
	CHECK(a && b && c);
	CHECK(pool.Allocate(1) == nullptr); // pool fully committed

	pool.Free(b, 4096); // free the middle block: isolated, non-adjacent-mergeable free block
	CHECK(pool.HasHeadroomFor(4096));
	CHECK(!pool.HasHeadroomFor(8192)); // 8 KiB doesn't fit in a single isolated 4 KiB free block

	pool.Free(a, 4096); // now adjacent to b's freed block -> must coalesce into one 8 KiB block
	CHECK(pool.HasHeadroomFor(8192));

	pool.Free(c, 4096);
	CHECK(pool.AllocatedBytes() == 0);
	pool.ResetForTesting();
}

static void TestFreeOfForeignPointerIsNoop() {
	auto &pool = RmmPool::Instance();
	pool.ResetForTesting();
	pool.EnsureInitializedExactBytesForTesting(4096);
	auto *a = pool.Allocate(4096);
	CHECK(a != nullptr);

	int not_ours = 0;
	// Freeing a pointer this pool never handed out must not corrupt internal state — verified by
	// confirming AllocatedBytes() is unaffected and a subsequent legitimate Free still works cleanly.
	pool.Free(&not_ours, 4);
	CHECK(pool.AllocatedBytes() >= 4096);

	pool.Free(a, 4096);
	CHECK(pool.AllocatedBytes() == 0);
	pool.ResetForTesting();
}

static void TestFreeNullptrIsNoop() {
	auto &pool = RmmPool::Instance();
	pool.ResetForTesting();
	pool.EnsureInitializedExactBytesForTesting(4096);
	pool.Free(nullptr, 100); // must not crash
	CHECK(pool.AllocatedBytes() == 0);
	pool.ResetForTesting();
}

static void TestSubByteAlignmentAllocationRoundsUp() {
	// Edge case: a 1-byte request must still round up to the full 256-byte alignment quantum internally
	// (so AllocatedBytes() reflects real arena consumption), and the caller-visible pointer must still be
	// usable (non-null, distinct across successive allocations).
	auto &pool = RmmPool::Instance();
	pool.ResetForTesting();
	pool.EnsureInitializedExactBytesForTesting(4096);
	auto *a = pool.Allocate(1);
	CHECK(a != nullptr);
	CHECK(pool.AllocatedBytes() == 256); // rounded up from 1 byte to the 256-byte alignment quantum
	auto *b = pool.Allocate(1);
	CHECK(b != nullptr);
	CHECK(a != b);
	CHECK(pool.AllocatedBytes() == 512);
	pool.Free(a, 1);
	pool.Free(b, 1);
	CHECK(pool.AllocatedBytes() == 0);
	pool.ResetForTesting();
}

static void TestDoubleInitializeIsNoop() {
	auto &pool = RmmPool::Instance();
	pool.ResetForTesting();
	pool.EnsureInitializedExactBytesForTesting(4096);
	auto total_first = pool.TotalPoolBytes();
	pool.EnsureInitializedExactBytesForTesting(999999); // must be ignored — already initialized
	CHECK(pool.TotalPoolBytes() == total_first);
	pool.ResetForTesting();
}

static void TestHeadroomRejectsSaturatedEstimate() {
	// Session-7 KI-1 regression: AlignUp wraps to 0 for the top 255 size_t values, which used to make
	// HasHeadroomFor approve the largest requests representable — including the exact value
	// TableChecker's SaturatingMultiply produces for an overflowing cardinality estimate.
	auto &pool = RmmPool::Instance();
	pool.ResetForTesting();
	pool.EnsureInitializedExactBytesForTesting(1024 * 1024); // 1 MiB arena
	const size_t kMax = static_cast<size_t>(-1);
	CHECK(pool.HasHeadroomFor(kMax) == false);       // was true before the fix
	CHECK(pool.HasHeadroomFor(kMax - 100) == false); // inside the wrap band — was true
	CHECK(pool.HasHeadroomFor(kMax - 255) == false); // just outside the wrap band, still absurdly big
	CHECK(pool.HasHeadroomFor(2 * 1024 * 1024) == false); // honest over-request
	CHECK(pool.HasHeadroomFor(4096) == true);             // sanity: normal requests still fit
	pool.ResetForTesting();
}

static void TestAllocateInWrapBandReturnsNull() {
	// Session-7 KI-2 regression: Allocate in the wrap band used to return a real arena pointer while
	// consuming zero pool space — every subsequent allocation aliased it.
	auto &pool = RmmPool::Instance();
	pool.ResetForTesting();
	pool.EnsureInitializedExactBytesForTesting(1024 * 1024);
	const size_t kMax = static_cast<size_t>(-1);
	CHECK(pool.Allocate(kMax) == nullptr);
	CHECK(pool.Allocate(kMax - 254) == nullptr);
	CHECK(pool.AllocatedBytes() == 0); // and nothing was consumed by the rejected requests
	auto *honest = pool.Allocate(4096);
	CHECK(honest != nullptr); // pool still fully usable afterwards
	pool.Free(honest, 4096);
	pool.ResetForTesting();
}

static void TestDoubleFreeIsNoop() {
	// Session-7 KI-3 regression: double-free used to insert a duplicate free block — AllocatedBytes
	// underflowed to 0 with a block still live, and the next two allocations aliased one address.
	auto &pool = RmmPool::Instance();
	pool.ResetForTesting();
	pool.EnsureInitializedExactBytesForTesting(64 * 1024);
	auto *a = pool.Allocate(4096);
	auto *b = pool.Allocate(4096);
	CHECK(a && b);
	pool.Free(a, 4096);
	pool.Free(a, 4096); // double free — must be a no-op
	CHECK(pool.AllocatedBytes() == 4096); // b (and only b) is still live; was 0 before the fix
	auto *c = pool.Allocate(4096);
	auto *d = pool.Allocate(4096);
	CHECK(c && d);
	CHECK(c != d);            // was c == d before the fix
	CHECK(c != b && d != b);  // and neither may alias the still-live b
	pool.ResetForTesting();
}

static void TestFreeWithWrongSizeIsSafe() {
	// Session-7 KI-4 regression: Free trusted the caller's byte count, so freeing a with a larger size
	// put live b inside a free block — the next Allocate handed out memory overlapping b.
	auto &pool = RmmPool::Instance();
	pool.ResetForTesting();
	pool.EnsureInitializedExactBytesForTesting(64 * 1024);
	auto *a = pool.Allocate(4096);
	auto *b = pool.Allocate(4096); // adjacent to a; stays LIVE
	CHECK(a && b);
	pool.Free(a, 8192); // wrong size (real size 4096) — the tracked size must win
	CHECK(pool.AllocatedBytes() == 4096); // only b remains live, and exactly b's 4096
	auto *c = pool.Allocate(8192);
	if (c != nullptr) {
		// Whatever the allocator returned must not overlap live b.
		auto *cb = static_cast<uint8_t *>(c);
		auto *bb = static_cast<uint8_t *>(b);
		bool overlaps = cb <= bb && bb < cb + 8192;
		CHECK(!overlaps); // was an overlap before the fix
		pool.Free(c, 8192);
	}
	pool.Free(b, 4096);
	CHECK(pool.AllocatedBytes() == 0);
	pool.ResetForTesting();
}

int main() {
	RUN_TEST(TestInvalidFractionRejected);
	RUN_TEST(TestBasicAllocateFree);
	RUN_TEST(TestZeroByteAllocateIsNullNotError);
	RUN_TEST(TestAllocateBeforeInitializedReturnsNull);
	RUN_TEST(TestOverAllocationFails);
	RUN_TEST(TestFragmentationAndCoalescing);
	RUN_TEST(TestFreeOfForeignPointerIsNoop);
	RUN_TEST(TestFreeNullptrIsNoop);
	RUN_TEST(TestSubByteAlignmentAllocationRoundsUp);
	RUN_TEST(TestDoubleInitializeIsNoop);
	RUN_TEST(TestHeadroomRejectsSaturatedEstimate);
	RUN_TEST(TestAllocateInWrapBandReturnsNull);
	RUN_TEST(TestDoubleFreeIsNoop);
	RUN_TEST(TestFreeWithWrongSizeIsSafe);
	TEST_MAIN_EPILOGUE();
}
