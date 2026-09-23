#pragma once

// Minimal single-header test harness. No external test framework dependency (no network fetch needed to
// build/run these) — this is intentionally small: enough to report pass/fail per-check with a message and
// return a nonzero process exit code if anything failed, which is all a CI/CMake `ctest` run needs.

#include <exception>
#include <iostream>
#include <sstream>
#include <string>

namespace vector_gpu_test {

inline int &FailureCount() {
	static int count = 0;
	return count;
}
inline int &CheckCount() {
	static int count = 0;
	return count;
}

inline void ReportCheck(bool passed, const std::string &expr, const char *file, int line) {
	CheckCount()++;
	if (!passed) {
		FailureCount()++;
		std::cerr << "[FAIL] " << file << ":" << line << "  CHECK(" << expr << ")\n";
	}
}

} // namespace vector_gpu_test

#define CHECK(expr) vector_gpu_test::ReportCheck((expr), #expr, __FILE__, __LINE__)

#define CHECK_THROWS(stmt)                                                                                            \
	do {                                                                                                              \
		bool _threw = false;                                                                                         \
		try {                                                                                                        \
			stmt;                                                                                                    \
		} catch (const std::exception &) {                                                                           \
			_threw = true;                                                                                           \
		}                                                                                                             \
		vector_gpu_test::ReportCheck(_threw, "throws: " #stmt, __FILE__, __LINE__);                                  \
	} while (0)

// Session-7 fix: an exception escaping a test used to reach std::terminate — every remaining test was
// silently dropped and no summary printed. Now it's counted as one failed check and the run continues.
#define RUN_TEST(fn)                                                                                                 \
	do {                                                                                                              \
		std::cout << "--- running " #fn " ---\n";                                                                    \
		try {                                                                                                        \
			fn();                                                                                                    \
		} catch (const std::exception &e) {                                                                          \
			vector_gpu_test::ReportCheck(false, std::string("unhandled exception in " #fn ": ") + e.what(),          \
			                             __FILE__, __LINE__);                                                        \
		} catch (...) {                                                                                              \
			vector_gpu_test::ReportCheck(false, "unhandled non-std exception in " #fn, __FILE__, __LINE__);          \
		}                                                                                                             \
	} while (0)

#define TEST_MAIN_EPILOGUE()                                                                                         \
	do {                                                                                                              \
		std::cout << vector_gpu_test::CheckCount() - vector_gpu_test::FailureCount() << "/"                          \
		          << vector_gpu_test::CheckCount() << " checks passed\n";                                            \
		return vector_gpu_test::FailureCount() == 0 ? 0 : 1;                                                         \
	} while (0)
