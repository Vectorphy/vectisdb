#include "test_framework.hpp"
#include "gpu_logger.hpp"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace vector_gpu;

// Covers the bounded activity logger. No GPU required — this is pure host-side I/O.

namespace {

std::filesystem::path TestDir() {
	auto dir = std::filesystem::temp_directory_path() / "vector_gpu_logger_test";
	std::filesystem::create_directories(dir);
	return dir;
}

//! Reads both log files, oldest-file-first, and returns every line.
std::vector<std::string> ReadAllLines(const std::filesystem::path &dir) {
	std::vector<std::string> lines;
	for (int i = 0; i < 2; i++) {
		std::ifstream in(dir / ("gpu_log." + std::to_string(i) + ".jsonl"));
		std::string line;
		while (std::getline(in, line)) {
			if (!line.empty()) {
				lines.push_back(line);
			}
		}
	}
	return lines;
}

//! Every record must be one complete JSON object on its own line. This is what catches torn/interleaved
//! writes from concurrent threads — a half-written record shows up as unbalanced braces.
bool LooksLikeCompleteJsonObject(const std::string &line) {
	if (line.size() < 2 || line.front() != '{' || line.back() != '}') {
		return false;
	}
	int depth = 0;
	bool in_string = false;
	bool escaped = false;
	for (char c : line) {
		if (in_string) {
			if (escaped) {
				escaped = false;
			} else if (c == '\\') {
				escaped = true;
			} else if (c == '"') {
				in_string = false;
			}
			continue;
		}
		if (c == '"') {
			in_string = true;
		} else if (c == '{') {
			depth++;
		} else if (c == '}') {
			depth--;
			if (depth < 0) {
				return false;
			}
		}
	}
	return depth == 0 && !in_string;
}

} // namespace

static void TestDisabledByDefaultWritesNothing() {
	auto dir = TestDir();
	GpuLogger::Instance().ConfigureForTesting(false, LogLevel::DEBUG, 1024 * 1024, dir.string());
	// Remove any file a previous test left behind so "wrote nothing" is unambiguous.
	std::filesystem::remove(dir / "gpu_log.0.jsonl");
	std::filesystem::remove(dir / "gpu_log.1.jsonl");

	for (int i = 0; i < 100; i++) {
		VGPU_LOG(LogLevel::ERROR, "should_not_appear", "");
	}
	GpuLogger::Instance().FlushForTesting();

	CHECK(!std::filesystem::exists(dir / "gpu_log.0.jsonl"));
	CHECK(ReadAllLines(dir).empty());
}

static void TestRecordsRoundTripAsJsonLines() {
	auto dir = TestDir();
	GpuLogger::Instance().ConfigureForTesting(true, LogLevel::DEBUG, 1024 * 1024, dir.string());

	VGPU_LOG(LogLevel::INFO, "h2d", "\"column\":\"revenue\",\"bytes\":4096,\"us\":12");
	VGPU_LOG(LogLevel::INFO, "decline", std::string("\"reason\":") + GpuLogger::Quote("ORDER BY is not offloadable"));
	GpuLogger::Instance().FlushForTesting();

	auto lines = ReadAllLines(dir);
	CHECK(lines.size() == 2);
	bool all_json = !lines.empty();
	for (auto &line : lines) {
		all_json = all_json && LooksLikeCompleteJsonObject(line);
	}
	CHECK(all_json);
	CHECK(!lines.empty() && lines[0].find("\"evt\":\"h2d\"") != std::string::npos);
	CHECK(!lines.empty() && lines[0].find("\"ts_us\":") != std::string::npos);
	CHECK(!lines.empty() && lines[0].find("\"thread\":") != std::string::npos);
	CHECK(lines.size() > 1 && lines[1].find("ORDER BY is not offloadable") != std::string::npos);
}

static void TestLevelFiltering() {
	auto dir = TestDir();
	GpuLogger::Instance().ConfigureForTesting(true, LogLevel::ERROR, 1024 * 1024, dir.string());

	VGPU_LOG(LogLevel::ERROR, "kept", "");
	VGPU_LOG(LogLevel::INFO, "dropped", "");
	VGPU_LOG(LogLevel::DEBUG, "dropped", "");
	GpuLogger::Instance().FlushForTesting();

	auto lines = ReadAllLines(dir);
	CHECK(lines.size() == 1);
	CHECK(!lines.empty() && lines[0].find("\"evt\":\"kept\"") != std::string::npos);
}

static void TestQuoteEscapesJson() {
	auto quoted = GpuLogger::Quote("a\"b\\c\nd");
	CHECK(quoted == "\"a\\\"b\\\\c\\nd\"");
}

static void TestConcurrentWritersProduceWholeLines() {
	auto dir = TestDir();
	// Cap high enough that no rotation happens, so every record must survive.
	GpuLogger::Instance().ConfigureForTesting(true, LogLevel::DEBUG, 32 * 1024 * 1024, dir.string());

	const int threads = 8;
	const int per_thread = 500;
	std::vector<std::thread> workers;
	for (int t = 0; t < threads; t++) {
		workers.emplace_back([t, per_thread] {
			for (int i = 0; i < per_thread; i++) {
				VGPU_LOG(LogLevel::INFO, "launch",
				         "\"worker\":" + std::to_string(t) + ",\"i\":" + std::to_string(i) +
				             ",\"pad\":\"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\"");
			}
		});
	}
	for (auto &worker : workers) {
		worker.join();
	}
	GpuLogger::Instance().FlushForTesting();

	auto lines = ReadAllLines(dir);
	CHECK(lines.size() == static_cast<size_t>(threads * per_thread));
	bool all_whole = !lines.empty();
	for (auto &line : lines) {
		all_whole = all_whole && LooksLikeCompleteJsonObject(line);
	}
	CHECK(all_whole); // a torn/interleaved write would fail brace balance
}

static void TestCapEvictsOldestAndKeepsNewest() {
	auto dir = TestDir();
	// Small cap so the test runs fast; the policy is identical at 50 MiB.
	const size_t cap = 256 * 1024;
	GpuLogger::Instance().ConfigureForTesting(true, LogLevel::DEBUG, cap, dir.string());

	const int records = 4000; // ~100 bytes each => ~400 KiB, comfortably over the cap
	for (int i = 0; i < records; i++) {
		VGPU_LOG(LogLevel::INFO, "seq",
		         "\"n\":" + std::to_string(i) + ",\"pad\":\"yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy\"");
	}
	GpuLogger::Instance().FlushForTesting();

	// 1. Total on disk stays within the cap.
	auto on_disk = GpuLogger::Instance().OnDiskBytesForTesting();
	CHECK(on_disk <= cap);
	CHECK(on_disk > 0);

	auto lines = ReadAllLines(dir);
	CHECK(!lines.empty());

	// 2. The NEWEST record survived.
	std::string newest = "\"n\":" + std::to_string(records - 1) + ",";
	bool found_newest = false;
	for (auto &line : lines) {
		if (line.find(newest) != std::string::npos) {
			found_newest = true;
		}
	}
	CHECK(found_newest);

	// 3. The OLDEST records were discarded — this is the eviction policy under test.
	bool found_oldest = false;
	for (auto &line : lines) {
		if (line.find("\"n\":0,") != std::string::npos) {
			found_oldest = true;
		}
	}
	CHECK(!found_oldest);

	// 4. And we did not throw everything away: a meaningful tail is retained.
	CHECK(lines.size() > 100);
}

int main() {
	RUN_TEST(TestDisabledByDefaultWritesNothing);
	RUN_TEST(TestRecordsRoundTripAsJsonLines);
	RUN_TEST(TestLevelFiltering);
	RUN_TEST(TestQuoteEscapesJson);
	RUN_TEST(TestConcurrentWritersProduceWholeLines);
	RUN_TEST(TestCapEvictsOldestAndKeepsNewest);
	TEST_MAIN_EPILOGUE();
}
