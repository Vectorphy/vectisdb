#pragma once

#include <cstddef>
#include <cstdint>

namespace vector_gpu {

//! Real, testable primitive behind GpuOpType::FILTER for the simple-predicate case (see
//! docs/ARCHITECTURE.md — complex/nested predicates go through the NVRTC code generator instead; this is
//! the libcudf-style "just call the library op" path, implemented here with Thrust since Thrust ships
//! with the CUDA Toolkit and needs no RAPIDS/libcudf, which aren't available on native Windows — see
//! docs/KNOWN_ISSUES.md).
//!
//! Compacts `input` (device pointer, `n` elements) down to only the elements > `threshold`, writing them
//! into `output` (device pointer, must be allocated for at least `n` elements by the caller — worst case
//! is no elements filtered out) and returning the resulting count via `out_count` (host pointer — this
//! function writes the count synchronously before returning).
//!
//! Edge cases handled explicitly:
//!   - n == 0: writes *out_count = 0 and returns without touching input/output pointers (which may be
//!     nullptr in this case — that's valid, not an error).
//!   - all elements pass / none pass: both are normal Thrust compaction outcomes, no special-casing
//!     needed, exercised explicitly in unit tests.
//!   - a stale CUDA error latched by an earlier, unrelated call is cleared on entry rather than being
//!     misattributed to this call (session-7 KI-5 — verified on hardware that a latched error made a
//!     valid call fail spuriously).
//! `input` and `output` must not overlap (Thrust/CUB requirement); exact aliasing (input == output)
//! throws std::invalid_argument, partial overlap is undetectable and remains the caller's contract.
//! Throws std::runtime_error (with the CUDA error string) if a CUDA API call fails — including
//! device-OOM during Thrust's temporary allocation, which is translated from std::bad_alloc so the
//! documented exception type holds.
void FilterGreaterThanInt32(const int32_t *input, size_t n, int32_t threshold, int32_t *output, size_t *out_count);

} // namespace vector_gpu
