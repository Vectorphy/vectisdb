#include "kernel_cache.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace vector_gpu {

namespace {
//! Portable SHA-256 (public-domain algorithm, no OS or third-party crypto dependency) -- replaces the
//! earlier Windows-CNG/BCrypt implementation so the kernel cache builds identically on Linux/WSL2 as on
//! Windows. Replaces the still-earlier std::hash-based placeholder (see docs/KNOWN_ISSUES.md): that was
//! deterministic but not collision-resistant, which matters here because the hash IS the kernel-cache
//! key -- a collision would silently serve the wrong compiled kernel for a different expression.
class Sha256 {
public:
	Sha256() {
		state_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
		          0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
	}

	void Update(const uint8_t *data, size_t len) {
		total_len_ += len;
		while (len > 0) {
			size_t take = std::min(len, size_t(64) - buffer_len_);
			std::memcpy(buffer_.data() + buffer_len_, data, take);
			buffer_len_ += take;
			data += take;
			len -= take;
			if (buffer_len_ == 64) {
				Transform(buffer_.data());
				buffer_len_ = 0;
			}
		}
	}

	std::array<uint8_t, 32> Finish() {
		uint64_t bit_len = total_len_ * 8;
		uint8_t pad = 0x80;
		Update(&pad, 1);
		uint8_t zero = 0x00;
		while (buffer_len_ != 56) {
			Update(&zero, 1);
		}
		for (int i = 7; i >= 0; i--) {
			uint8_t byte = static_cast<uint8_t>(bit_len >> (i * 8));
			// Bypass Update()'s total_len_ accounting: the length field is metadata about the
			// message, not message content.
			buffer_[buffer_len_++] = byte;
			if (buffer_len_ == 64) {
				Transform(buffer_.data());
				buffer_len_ = 0;
			}
		}
		std::array<uint8_t, 32> out;
		for (int i = 0; i < 8; i++) {
			out[i * 4 + 0] = static_cast<uint8_t>(state_[i] >> 24);
			out[i * 4 + 1] = static_cast<uint8_t>(state_[i] >> 16);
			out[i * 4 + 2] = static_cast<uint8_t>(state_[i] >> 8);
			out[i * 4 + 3] = static_cast<uint8_t>(state_[i]);
		}
		return out;
	}

private:
	static uint32_t Rotr(uint32_t x, uint32_t n) {
		return (x >> n) | (x << (32 - n));
	}

	void Transform(const uint8_t *chunk) {
		static const uint32_t k[64] = {
		    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
		    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
		    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
		    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
		    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
		    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
		    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
		    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

		uint32_t w[64];
		for (int i = 0; i < 16; i++) {
			w[i] = (uint32_t(chunk[i * 4]) << 24) | (uint32_t(chunk[i * 4 + 1]) << 16) |
			       (uint32_t(chunk[i * 4 + 2]) << 8) | uint32_t(chunk[i * 4 + 3]);
		}
		for (int i = 16; i < 64; i++) {
			uint32_t s0 = Rotr(w[i - 15], 7) ^ Rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
			uint32_t s1 = Rotr(w[i - 2], 17) ^ Rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
			w[i] = w[i - 16] + s0 + w[i - 7] + s1;
		}

		uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
		uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];

		for (int i = 0; i < 64; i++) {
			uint32_t s1 = Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25);
			uint32_t ch = (e & f) ^ (~e & g);
			uint32_t temp1 = h + s1 + ch + k[i] + w[i];
			uint32_t s0 = Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22);
			uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
			uint32_t temp2 = s0 + maj;

			h = g;
			g = f;
			f = e;
			e = d + temp1;
			d = c;
			c = b;
			b = a;
			a = temp1 + temp2;
		}

		state_[0] += a;
		state_[1] += b;
		state_[2] += c;
		state_[3] += d;
		state_[4] += e;
		state_[5] += f;
		state_[6] += g;
		state_[7] += h;
	}

	std::array<uint32_t, 8> state_;
	std::array<uint8_t, 64> buffer_ {};
	size_t buffer_len_ = 0;
	uint64_t total_len_ = 0;
};

std::string Sha256Hex(const std::string &data) {
	Sha256 sha;
	if (!data.empty()) {
		sha.Update(reinterpret_cast<const uint8_t *>(data.data()), data.size());
	}
	auto digest = sha.Finish();

	std::ostringstream oss;
	for (auto byte : digest) {
		oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
	}
	return oss.str();
}
} // namespace

KernelCache::~KernelCache() {
	// No lock: by the time the destructor runs, concurrent access is already a caller bug. Free every
	// owned handle so cache destruction doesn't leak modules (session-7 fix).
	for (auto &entry : entries_) {
		DeleteHandleLocked(entry.second.first);
	}
}

void KernelCache::SetHandleDeleter(HandleDeleter deleter) {
	std::lock_guard<std::mutex> guard(mutex_);
	if (deleter_ != nullptr && deleter_ != deleter) {
		throw std::logic_error("KernelCache::SetHandleDeleter: a different deleter is already registered");
	}
	deleter_ = deleter;
}

void KernelCache::DeleteHandleLocked(CompiledKernelHandle handle) {
	if (deleter_ != nullptr && handle != nullptr) {
		deleter_(handle);
	}
}

CompiledKernelHandle KernelCache::Get(const std::string &source_hash) {
	std::lock_guard<std::mutex> guard(mutex_);
	auto it = entries_.find(source_hash);
	if (it == entries_.end()) {
		return nullptr;
	}
	// Move to front (most recently used).
	lru_order_.erase(it->second.second);
	lru_order_.push_front(source_hash);
	it->second.second = lru_order_.begin();
	return it->second.first;
}

void KernelCache::Insert(const std::string &source_hash, CompiledKernelHandle handle) {
	std::lock_guard<std::mutex> guard(mutex_);
	if (max_entries_ == 0) {
		// Degenerate but valid configuration: a cache that holds nothing. Every Insert is immediately a
		// no-op miss; without this early return, entries_.size() (0) >= max_entries_ (0) would be true
		// and the eviction branch below would call lru_order_.back() on an empty list (undefined
		// behavior) since nothing was ever inserted to evict.
		// Session-7 fix: with an owner registered, "a cache that holds nothing" must still not leak the
		// kernel it declines to store — free it, otherwise every compile of even the same expression
		// leaks a CUmodule.
		DeleteHandleLocked(handle);
		return;
	}
	auto existing = entries_.find(source_hash);
	if (existing != entries_.end()) {
		lru_order_.erase(existing->second.second);
		// Session-7 fix: replacing an entry frees the replaced handle (previously dropped on the floor —
		// a CUmodule/VRAM leak per replacement). Re-inserting the identical handle is just an LRU refresh.
		if (existing->second.first != handle) {
			DeleteHandleLocked(existing->second.first);
		}
	} else if (entries_.size() >= max_entries_) {
		auto &lru_key = lru_order_.back();
		// Session-7 fix (was a TODO whose precondition had long been met): eviction frees the evicted
		// kernel via the registered deleter — cuModuleUnload + heap free — closing the VRAM leak.
		auto evicted = entries_.find(lru_key);
		if (evicted != entries_.end()) {
			DeleteHandleLocked(evicted->second.first);
			entries_.erase(evicted);
		}
		lru_order_.pop_back();
	}
	lru_order_.push_front(source_hash);
	entries_[source_hash] = {handle, lru_order_.begin()};
}

std::string KernelCache::HashSource(const std::string &cuda_source) {
	return Sha256Hex(cuda_source);
}

} // namespace vector_gpu
