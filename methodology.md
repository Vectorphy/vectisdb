# Methodology and limits

## Scope of the recorded tests

The supplied outputs contain TPC-H query timings at scale factors 1, 2, 4, 8, 16, 32, and 64. Each scale factor includes 22 TPC-H queries in cached and cold-cache states. The source reports identify the CPU reference as standard DuckDB CPU execution and the GPU system as the VectisDB prototype.

The recorded platform is an NVIDIA GeForce GTX 1650 with CUDA 13.3 on an x86_64 host. Hardware, driver, operating-system settings, compiler settings, and data placement can materially affect results. This page records the information supplied with the results; it does not fill in missing environment detail.

## Cache states

The source reports describe cached runs as hot in-memory execution. Cold-cache runs are described as execution after operating-system page-cache clearing. These states are intentionally different measurements. Neither one should be read as a general substitute for a workload's own cache, concurrency, or storage conditions.

## GPU configurations

Each query was recorded across eight GPU variants. The matrix changes stream-pipeline state, floating-point mode, and fast-math mode:

- Pipeline on or off
- FP64 strict or FP32 relaxed precision
- Fast math on or off

The results summary selects the fastest recorded GPU value in that matrix for each query. This is a best-observed result, not a single fixed profile. A fair deployment comparison would choose and hold one configuration in advance, then reproduce both CPU and GPU runs under the same stated conditions.

## Limits on interpretation

- These are internal tests of a rapid prototype, not production benchmarks.
- The runs use one recorded GPU and host class. They do not establish performance on other machines.
- The best-GPU column selects from eight variants for each query. It should not be used as a fixed-configuration result.
- The branch excludes source code, test harnesses, SQL, datasets, binaries, and raw result files. It is a documentation site, not a reproducibility package.
- No independent reproduction, statistical analysis, or external review is represented here.

## Source handling

This documentation was prepared from the supplied internal CSV and Markdown reports. The original benchmark material remains outside this branch by design. Before using these numbers in a public comparison, retain the raw captures, document the full environment, pre-register the selected configuration, and reproduce the tests on the intended target system.
