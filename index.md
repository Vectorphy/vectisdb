# VectisDB internal benchmark notes

These pages record internal performance tests of a VectisDB prototype against a DuckDB CPU baseline. They are a snapshot of exploratory work, prepared on 28 September 2026 from the supplied test outputs.

## Project status

VectisDB is a rapid prototype used to test a technical approach. It is not production software, and it should not be treated as a finished or supported product.

The measurements here are internal test results. They have not been independently reproduced or reviewed, and they are not suitable for external comparative-performance claims.

## Reading the results

- [Results summary](results.md) lists the TPC-H scale-factor results and the aggregation used for each figure.
- [Methodology and limits](methodology.md) describes the recorded platform, test states, profile selection, and limitations.

## Snapshot

The supplied TPC-H results cover scale factors 1 through 64, 22 queries per scale factor, and two cache states. At scale factor 64, the best recorded GPU time was lower than the CPU baseline for all 22 queries in both cached and cold-cache runs. At scale factor 1, it was lower for 21 of 22 cached queries and 17 of 22 cold-cache queries.

Those counts describe this particular test matrix only. They do not predict workload performance on another system.

## Recorded platform

The source reports identify an NVIDIA GeForce GTX 1650 running CUDA 13.3 with an x86_64 host. The CPU reference is standard DuckDB CPU execution. The GPU side is the VectisDB prototype.

## Branch contents

This branch intentionally contains documentation only. It does not include VectisDB source code, benchmark scripts, SQL, binaries, datasets, or raw result files.
