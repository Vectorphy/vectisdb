#include "test_framework.hpp"
#include "kernel_cache.hpp"

#include <vector>

using namespace vector_gpu;

static void TestMissThenHit() {
	KernelCache cache(4);
	CHECK(cache.Get("a") == nullptr);
	int dummy_a = 1;
	cache.Insert("a", &dummy_a);
	CHECK(cache.Get("a") == &dummy_a);
}

static void TestOverwriteSameKey() {
	KernelCache cache(4);
	int dummy_a1 = 1, dummy_a2 = 2;
	cache.Insert("a", &dummy_a1);
	cache.Insert("a", &dummy_a2); // re-inserting the same key must replace, not duplicate
	CHECK(cache.Get("a") == &dummy_a2);
}

static void TestLruEvictionOrder() {
	// capacity 2: insert a, b -> both present. Access a (making b the LRU). Insert c -> b should evict,
	// a and c should remain. This is the core correctness property of an LRU cache and the most likely
	// place for an off-by-one/iterator bug.
	KernelCache cache(2);
	int a = 1, b = 2, c = 3;
	cache.Insert("a", &a);
	cache.Insert("b", &b);
	CHECK(cache.Get("a") == &a); // touch a -> a becomes MRU, b becomes LRU
	cache.Insert("c", &c);       // should evict b, not a
	CHECK(cache.Get("a") == &a);
	CHECK(cache.Get("b") == nullptr); // evicted
	CHECK(cache.Get("c") == &c);
}

static void TestEvictionAtExactCapacity() {
	// Edge case: capacity of 1 — every insert after the first must evict the previous entry.
	KernelCache cache(1);
	int a = 1, b = 2;
	cache.Insert("a", &a);
	CHECK(cache.Get("a") == &a);
	cache.Insert("b", &b);
	CHECK(cache.Get("a") == nullptr);
	CHECK(cache.Get("b") == &b);
}

static void TestHashSourceDeterministic() {
	// Same source string must always hash the same way (cache correctness depends on this), and
	// different source strings should (almost always) hash differently.
	auto h1 = KernelCache::HashSource("out[idx] = a + b;");
	auto h2 = KernelCache::HashSource("out[idx] = a + b;");
	auto h3 = KernelCache::HashSource("out[idx] = a - b;");
	CHECK(h1 == h2);
	CHECK(h1 != h3);
}

static void TestHashSourceEmptyString() {
	// Edge case: empty expression source must not crash and must still be a valid, stable key.
	auto h1 = KernelCache::HashSource("");
	auto h2 = KernelCache::HashSource("");
	CHECK(h1 == h2);
	CHECK(!h1.empty());
}

static void TestHashSourceKnownVectors() {
	// KernelCache::HashSource is now real SHA-256 (Windows BCrypt) -- verify against the standard,
	// widely-published SHA-256 test vectors for "" and "abc" rather than just checking internal
	// consistency. This is what actually proves it's SHA-256 and not just "some 64-hex-char hash".
	CHECK(KernelCache::HashSource("") ==
	      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
	CHECK(KernelCache::HashSource("abc") ==
	      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

static void TestHashSourceLength() {
	// SHA-256 always produces exactly 32 bytes -> 64 hex characters, regardless of input length.
	CHECK(KernelCache::HashSource("x").size() == 64);
	CHECK(KernelCache::HashSource(std::string(10000, 'z')).size() == 64);
}

static void TestGetOnEmptyCache() {
	KernelCache cache(4);
	CHECK(cache.Get("anything") == nullptr);
}

static void TestZeroCapacityCacheNeverStores() {
	// Regression test: max_entries_ == 0 used to call lru_order_.back() on an empty list inside Insert
	// (undefined behavior — the eviction branch triggered on the very first Insert since size() (0) >=
	// max_entries_ (0), but there was nothing to evict). A zero-capacity cache is a degenerate but valid
	// configuration and must simply never retain anything, not crash.
	KernelCache cache(0);
	int dummy = 1;
	cache.Insert("a", &dummy);
	CHECK(cache.Get("a") == nullptr);
	// Repeated inserts into a zero-capacity cache must also stay safe (would have hit the same UB path
	// on every call, not just the first).
	cache.Insert("b", &dummy);
	cache.Insert("c", &dummy);
	CHECK(cache.Get("b") == nullptr);
	CHECK(cache.Get("c") == nullptr);
}

// Session-7 ownership regressions: with a HandleDeleter registered, the cache must free evicted,
// replaced, declined (zero-capacity), and destruction-time handles — and must not free anything else.
// HandleDeleter is a plain function pointer, so the recorder is file-static state reset per test.
static std::vector<void *> g_deleted_handles;
static void RecordingDeleter(CompiledKernelHandle handle) {
	g_deleted_handles.push_back(handle);
}

static void TestDeleterCalledOnEviction() {
	g_deleted_handles.clear();
	KernelCache cache(1);
	cache.SetHandleDeleter(&RecordingDeleter);
	int a = 1, b = 2;
	cache.Insert("a", &a);
	cache.Insert("b", &b); // evicts a — the deleter must receive exactly a, and only a
	CHECK(g_deleted_handles.size() == 1);
	CHECK(!g_deleted_handles.empty() && g_deleted_handles[0] == &a);
	CHECK(cache.Get("b") == &b); // survivor untouched
}

static void TestDeleterCalledOnReplacement() {
	g_deleted_handles.clear();
	KernelCache cache(4);
	cache.SetHandleDeleter(&RecordingDeleter);
	int a1 = 1, a2 = 2;
	cache.Insert("a", &a1);
	cache.Insert("a", &a2); // replaces a1 — was silently dropped (leaked) before the fix
	CHECK(g_deleted_handles.size() == 1);
	CHECK(!g_deleted_handles.empty() && g_deleted_handles[0] == &a1);
	CHECK(cache.Get("a") == &a2);
	// Re-inserting the SAME handle under the same key is an LRU refresh, not a free.
	cache.Insert("a", &a2);
	CHECK(g_deleted_handles.size() == 1); // unchanged
	CHECK(cache.Get("a") == &a2);
}

static void TestDeleterCalledOnDestruction() {
	g_deleted_handles.clear();
	int a = 1, b = 2;
	{
		KernelCache cache(4);
		cache.SetHandleDeleter(&RecordingDeleter);
		cache.Insert("a", &a);
		cache.Insert("b", &b);
	} // destructor must free both — cache teardown used to leak every entry
	CHECK(g_deleted_handles.size() == 2);
	bool saw_a = false, saw_b = false;
	for (auto *h : g_deleted_handles) {
		saw_a = saw_a || h == &a;
		saw_b = saw_b || h == &b;
	}
	CHECK(saw_a && saw_b);
}

static void TestZeroCapacityCacheDeletesDeclinedHandle() {
	// A zero-capacity cache declines every Insert — but with an owner registered it must free the
	// declined handle rather than leak it (before the fix: one leaked CUmodule per compile).
	g_deleted_handles.clear();
	KernelCache cache(0);
	cache.SetHandleDeleter(&RecordingDeleter);
	int a = 1;
	cache.Insert("a", &a);
	CHECK(g_deleted_handles.size() == 1);
	CHECK(!g_deleted_handles.empty() && g_deleted_handles[0] == &a);
	CHECK(cache.Get("a") == nullptr);
}

static void TestNoDeleterMeansNoOwnership() {
	// Backwards compatibility: with no deleter registered (this suite's other tests, standalone use),
	// eviction/destruction must not try to free anything — dummy stack pointers stay untouched.
	KernelCache cache(1);
	int a = 1, b = 2;
	cache.Insert("a", &a);
	cache.Insert("b", &b); // evicts a with no deleter — must simply drop it
	CHECK(cache.Get("b") == &b);
}

int main() {
	RUN_TEST(TestMissThenHit);
	RUN_TEST(TestOverwriteSameKey);
	RUN_TEST(TestLruEvictionOrder);
	RUN_TEST(TestEvictionAtExactCapacity);
	RUN_TEST(TestHashSourceDeterministic);
	RUN_TEST(TestHashSourceEmptyString);
	RUN_TEST(TestHashSourceKnownVectors);
	RUN_TEST(TestHashSourceLength);
	RUN_TEST(TestGetOnEmptyCache);
	RUN_TEST(TestZeroCapacityCacheNeverStores);
	RUN_TEST(TestDeleterCalledOnEviction);
	RUN_TEST(TestDeleterCalledOnReplacement);
	RUN_TEST(TestDeleterCalledOnDestruction);
	RUN_TEST(TestZeroCapacityCacheDeletesDeclinedHandle);
	RUN_TEST(TestNoDeleterMeansNoOwnership);
	TEST_MAIN_EPILOGUE();
}
