#include "gpu_logger.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <sstream>
#include <thread>

namespace vector_gpu {

namespace {

const char *LevelName(LogLevel level) {
	switch (level) {
	case LogLevel::ERROR:
		return "error";
	case LogLevel::DEBUG:
		return "debug";
	default:
		return "info";
	}
}

//! Env var read helper. Returns nullptr when unset OR empty, so `VECTOR_GPU_LOG=` behaves like unset.
const char *Env(const char *name) {
	const char *value = std::getenv(name);
	return (value != nullptr && value[0] != '\0') ? value : nullptr;
}

uint64_t NowMicros() {
	using namespace std::chrono;
	return static_cast<uint64_t>(duration_cast<microseconds>(system_clock::now().time_since_epoch()).count());
}

uint64_t ThisThreadId() {
	return static_cast<uint64_t>(std::hash<std::thread::id> {}(std::this_thread::get_id()));
}

std::string JoinPath(const std::string &directory, const char *name) {
	if (directory.empty()) {
		return name;
	}
	auto last = directory.back();
	if (last == '/' || last == '\\') {
		return directory + name;
	}
	return directory + "/" + name;
}

} // namespace

GpuLogger::GpuLogger() {
	EnsureConfigured();
}

GpuLogger &GpuLogger::Instance() {
	static GpuLogger instance; // function-local static: construction is thread-safe and happens once
	return instance;
}

GpuLogger::~GpuLogger() {
	if (file_ != nullptr) {
		std::fclose(static_cast<std::FILE *>(file_));
		file_ = nullptr;
	}
}

std::string GpuLogger::Quote(const std::string &value) {
	std::string out;
	out.reserve(value.size() + 2);
	out.push_back('"');
	for (unsigned char c : value) {
		switch (c) {
		case '"':
			out += "\\\"";
			break;
		case '\\':
			out += "\\\\";
			break;
		case '\n':
			out += "\\n";
			break;
		case '\r':
			out += "\\r";
			break;
		case '\t':
			out += "\\t";
			break;
		default:
			if (c < 0x20) {
				char buf[8];
				std::snprintf(buf, sizeof(buf), "\\u%04x", c);
				out += buf;
			} else {
				out.push_back(static_cast<char>(c));
			}
		}
	}
	out.push_back('"');
	return out;
}

void GpuLogger::EnsureConfigured() {
	std::lock_guard<std::mutex> guard(mutex_);
	if (configured_) {
		return;
	}
	configured_ = true; // set first: a failure below must not retry-storm on every call

	const char *enable = Env("VECTOR_GPU_LOG");
	bool on = enable != nullptr && std::strcmp(enable, "0") != 0;
	if (!on) {
		return; // stays disabled; no file is created, so logging off leaves no trace at all
	}

	if (const char *level = Env("VECTOR_GPU_LOG_LEVEL")) {
		if (std::strcmp(level, "error") == 0) {
			level_.store(static_cast<int>(LogLevel::ERROR), std::memory_order_relaxed);
		} else if (std::strcmp(level, "debug") == 0) {
			level_.store(static_cast<int>(LogLevel::DEBUG), std::memory_order_relaxed);
		} else {
			level_.store(static_cast<int>(LogLevel::INFO), std::memory_order_relaxed);
		}
	}

	size_t cap_mib = 50;
	if (const char *cap = Env("VECTOR_GPU_LOG_CAP_MIB")) {
		auto parsed = std::strtoull(cap, nullptr, 10);
		if (parsed > 0) {
			cap_mib = static_cast<size_t>(parsed);
		}
	}
	half_cap_bytes_ = (cap_mib * 1024u * 1024u) / 2u;
	if (half_cap_bytes_ == 0) {
		half_cap_bytes_ = 1; // a degenerate cap must still rotate rather than divide by zero
	}

	if (const char *path = Env("VECTOR_GPU_LOG_PATH")) {
		directory_ = path;
	}

	OpenActiveLocked(true);
	if (file_ == nullptr) {
		return; // could not open (bad path / permissions): stay disabled rather than fail queries
	}
	enabled_.store(true, std::memory_order_relaxed);
}

void GpuLogger::OpenActiveLocked(bool truncate) {
	if (file_ != nullptr) {
		std::fclose(static_cast<std::FILE *>(file_));
		file_ = nullptr;
	}
	auto name = std::string("gpu_log.") + std::to_string(active_file_) + ".jsonl";
	auto full = JoinPath(directory_, name.c_str());
	file_ = std::fopen(full.c_str(), truncate ? "wb" : "ab");
	active_bytes_ = 0;
}

void GpuLogger::RotateIfNeededLocked(size_t incoming_bytes) {
	if (file_ == nullptr) {
		return;
	}
	if (active_bytes_ + incoming_bytes <= half_cap_bytes_) {
		return;
	}
	// Roll over to the other file, truncating it. This is what bounds total on-disk size and what
	// discards the OLDEST records: the file being truncated is the older of the two by construction.
	active_file_ = 1 - active_file_;
	OpenActiveLocked(true);
}

void GpuLogger::Log(LogLevel level, const char *event, const std::string &fields) noexcept {
	try {
		EnsureConfigured();
		if (!Enabled(level)) {
			return;
		}
		std::ostringstream line;
		line << "{\"ts_us\":" << NowMicros() << ",\"thread\":" << ThisThreadId() << ",\"lvl\":\""
		     << LevelName(level) << "\",\"evt\":\"" << (event != nullptr ? event : "?") << "\"";
		if (!fields.empty()) {
			line << "," << fields;
		}
		line << "}\n";
		auto text = line.str();

		std::lock_guard<std::mutex> guard(mutex_);
		if (file_ == nullptr) {
			return;
		}
		RotateIfNeededLocked(text.size());
		if (file_ == nullptr) {
			return;
		}
		// One fwrite per record under the mutex: that is what makes concurrent writers produce whole,
		// non-interleaved lines rather than torn JSON.
		std::fwrite(text.data(), 1, text.size(), static_cast<std::FILE *>(file_));
		// Flush every record. This logger's primary use is diagnosing hangs and timeouts, and those end
		// with the process being KILLED — buffered records would be lost exactly when they matter most.
		// (Observed: a 60s-timeout kill produced an empty log before this.) Logging is off by default, so
		// the cost is only paid when someone is deliberately debugging.
		std::fflush(static_cast<std::FILE *>(file_));
		active_bytes_ += text.size();
	} catch (...) {
		// A logger must never take down a query.
	}
}

void GpuLogger::ConfigureForTesting(bool enabled, LogLevel level, size_t cap_bytes, const std::string &directory) {
	std::lock_guard<std::mutex> guard(mutex_);
	configured_ = true;
	level_.store(static_cast<int>(level), std::memory_order_relaxed);
	half_cap_bytes_ = cap_bytes / 2u;
	if (half_cap_bytes_ == 0) {
		half_cap_bytes_ = 1;
	}
	directory_ = directory;
	active_file_ = 0;
	if (!enabled) {
		enabled_.store(false, std::memory_order_relaxed);
		if (file_ != nullptr) {
			std::fclose(static_cast<std::FILE *>(file_));
			file_ = nullptr;
		}
		return;
	}
	OpenActiveLocked(true);
	// Truncate the other file too, so a test starts from a known-clean pair.
	auto other = std::string("gpu_log.") + std::to_string(1 - active_file_) + ".jsonl";
	if (auto *f = std::fopen(JoinPath(directory_, other.c_str()).c_str(), "wb")) {
		std::fclose(f);
	}
	enabled_.store(file_ != nullptr, std::memory_order_relaxed);
}

void GpuLogger::FlushForTesting() {
	std::lock_guard<std::mutex> guard(mutex_);
	if (file_ != nullptr) {
		std::fflush(static_cast<std::FILE *>(file_));
	}
}

size_t GpuLogger::OnDiskBytesForTesting() const {
	std::lock_guard<std::mutex> guard(mutex_);
	size_t total = 0;
	for (int i = 0; i < 2; i++) {
		auto name = std::string("gpu_log.") + std::to_string(i) + ".jsonl";
		if (auto *f = std::fopen(JoinPath(directory_, name.c_str()).c_str(), "rb")) {
			std::fseek(f, 0, SEEK_END);
			auto size = std::ftell(f);
			if (size > 0) {
				total += static_cast<size_t>(size);
			}
			std::fclose(f);
		}
	}
	return total;
}

} // namespace vector_gpu
