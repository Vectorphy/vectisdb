# Changelog

## v1.0.0 (developer release)

This release establishes VectisDB as a separately versioned project.

### Added

- CUDA backend build
- DuckDB-linked shell build
- GPU differential and cache verification scripts
- Local Git checks, GitHub CI, and a tag-triggered source package workflow

### Changed

- The DuckDB core dependency is pinned to v1.6.0-dev11577, commit af1b4a9bd2

### Notes

This is a developer release. Build the CUDA backend for your target system. The GPU runtime checks need a compatible NVIDIA GPU and driver.
