# Release notes

## v1.0.0 developer release

- Establishes VectisDB as a separately versioned project.
- Pins the DuckDB core dependency to `v1.6.0-dev11577`, commit `af1b4a9bd2`.
- Adds a CUDA backend build, a DuckDB-linked shell build, and GPU differential and cache verification scripts.
- Adds local Git checks, GitHub CI, and a tag-triggered source package workflow.

This is a developer release. The CUDA backend must be built for the target system, and the GPU runtime
checks require a compatible NVIDIA GPU and driver.
