#include "test_framework.hpp"
#include "gpu_memory_pool.hpp"

#include <chrono>
#include <cstring>
#include <future>
#include <set>
#include <vector>

using namespace vector_gpu;

// GpuMemoryPool: fixed cyclic ring of pinned host memory. Unlike test_rmm_pool.cpp's target (which
// varies its budget size per test via EnsureInitializedExactBytesForTesting), this pool's slot count and
// slot size are compile-time constants -- nothing to vary between tests -- so every test here shares the
// SAME real cudaHostAlloc'd slots and resets only the cyclic cursor between cases.
//
// Written against kSlotCount rather than a hardcoded 3: the constant is sized against the deepest
// pipeline that draws from the ring (see gpu_memory_pool.hpp), so it is expected to change when that
// does, and a test that bakes the number in fails for the wrong reason when it does.

static void TestSlotsAreDistinctAndFullCapacity() {
	auto &pool = GpuMemoryPool::Instance();
	pool.ResetCursorForTesting();

	std::vector<GpuMemoryPoolSlot> slots;
	std::set<void *> pointers;
	for (size_t i = 0; i < GpuMemoryPool::kSlotCount; i++) {
		slots.push_back(pool.AcquireNext());
		CHECK(slots.back().valid());
		CHECK(slots.back().capacity() == GpuMemoryPool::kSlotBytes);
		pointers.insert(slots.back().data());
	}
	// Genuinely distinct allocations, not the same buffer handed out repeatedly.
	CHECK(pointers.size() == GpuMemoryPool::kSlotCount);

	// The memory is real and writable across its FULL advertised capacity, not just some prefix -- touch
	// the first and last byte of each slot.
	for (auto &slot : slots) {
		auto *bytes = static_cast<uint8_t *>(slot.data());
		bytes[0] = 0xAB;
		bytes[slot.capacity() - 1] = 0xCD;
		CHECK(bytes[0] == 0xAB);
		CHECK(bytes[slot.capacity() - 1] == 0xCD);
	}
}

static void TestCyclicOrderRepeatsAfterFullRotation() {
	auto &pool = GpuMemoryPool::Instance();
	pool.ResetCursorForTesting();

	std::vector<void *> first_round;
	{
		std::vector<GpuMemoryPoolSlot> slots;
		for (size_t i = 0; i < GpuMemoryPool::kSlotCount; i++) {
			slots.push_back(pool.AcquireNext());
			first_round.push_back(slots.back().data());
		}
		// Every slot is released here, at end of scope; reset the cursor and go around again -- a
		// genuinely cyclic ring hands back the identical sequence of physical slots every full rotation.
	}

	pool.ResetCursorForTesting();
	bool same_rotation = true;
	{
		std::vector<GpuMemoryPoolSlot> slots;
		for (size_t i = 0; i < GpuMemoryPool::kSlotCount; i++) {
			slots.push_back(pool.AcquireNext());
			same_rotation = same_rotation && slots.back().data() == first_round[i];
		}
	}
	CHECK(same_rotation);
}

static void TestMoveTransfersOwnershipAndInvalidatesSource() {
	auto &pool = GpuMemoryPool::Instance();
	pool.ResetCursorForTesting();

	auto original = pool.AcquireNext();
	auto *original_data = original.data();
	CHECK(original.valid());

	GpuMemoryPoolSlot moved(std::move(original));
	CHECK(moved.valid());
	CHECK(moved.data() == original_data);
	CHECK(!original.valid()); // moved-from: must not be dereferenced, and valid() is how a caller checks

	GpuMemoryPoolSlot move_assigned;
	move_assigned = std::move(moved);
	CHECK(move_assigned.valid());
	CHECK(move_assigned.data() == original_data);
	CHECK(!moved.valid());
}

static void TestReleasedSlotCanBeReacquiredWithoutBlocking() {
	// RAII release: letting a slot fall out of scope must return it to the pool immediately, with no
	// separate cleanup call -- this IS the "cleans up safely" property GpuBatchAccumulator relies on.
	auto &pool = GpuMemoryPool::Instance();
	pool.ResetCursorForTesting();
	{
		auto held = pool.AcquireNext(); // slot 0, released when this block ends
		(void)held;
	}
	pool.ResetCursorForTesting(); // safe: nothing is checked out at this point
	auto start = std::chrono::steady_clock::now();
	auto reacquired = pool.AcquireNext(); // must be slot 0 again, and must not block
	auto elapsed = std::chrono::steady_clock::now() - start;
	CHECK(reacquired.valid());
	CHECK(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() < 100);
}

static void TestAcquireBlocksUntilHeldSlotIsReleased() {
	// The whole point of a FIXED ring: AcquireNext() for the slot the cursor is about to hand out must
	// wait for whoever currently holds it, not grow a new one (unlike PinnedBufferPool). Proven with a
	// real background thread and a timing bound, not just inspected by reading the source.
	auto &pool = GpuMemoryPool::Instance();
	pool.ResetCursorForTesting();

	auto held = pool.AcquireNext(); // checks out slot 0; cursor now points at slot 1

	// Walk the cursor through every OTHER slot so the next AcquireNext() call wraps back around to slot 0
	// -- the one still held above -- forcing it to block rather than simply handing out an available one.
	for (size_t i = 1; i < GpuMemoryPool::kSlotCount; i++) {
		auto passing = pool.AcquireNext();
		(void)passing; // released immediately; only the cursor advance matters here
	}

	auto blocked = std::async(std::launch::async, [&pool] { return pool.AcquireNext(); });

	// Give the background call every chance to (wrongly) return early before checking it hasn't.
	auto still_blocked = blocked.wait_for(std::chrono::milliseconds(200)) != std::future_status::ready;
	CHECK(still_blocked);

	held = GpuMemoryPoolSlot(); // release slot 0 -- this must be what unblocks the waiter
	auto unblocked_slot = blocked.get();
	CHECK(unblocked_slot.valid());
}

int main() {
	RUN_TEST(TestSlotsAreDistinctAndFullCapacity);
	RUN_TEST(TestCyclicOrderRepeatsAfterFullRotation);
	RUN_TEST(TestMoveTransfersOwnershipAndInvalidatesSource);
	RUN_TEST(TestReleasedSlotCanBeReacquiredWithoutBlocking);
	RUN_TEST(TestAcquireBlocksUntilHeldSlotIsReleased);
	TEST_MAIN_EPILOGUE();
}
