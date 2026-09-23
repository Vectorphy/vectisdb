# VectisDB

VectisDB is an experimental DuckDB GPU execution project. Version **1.0.0** is a developer release.
It contains a CUDA execution library, DuckDB optimizer and operator integration, and an optional shell
that links the integration into DuckDB.

## DuckDB version

The DuckDB core build used for this release reports **v1.6.0-dev11577** and is pinned to commit
`af1b4a9bd2`. This is DuckDB's development version, not VectisDB's version. The label
`gpu-v0.2.0-20-g4583f50964` refers to the GPU project revision. The `duckdb_gpu` executable is produced
by the build and is not included in the repository.

## Repository layout

- `cuda_engine/` contains the CUDA, NVRTC, memory, cache, and execution code.
- `extension/` contains the DuckDB optimizer and physical operator integration.
- `gpu_shell/` builds an optional DuckDB-linked command-line executable and holds GPU integration checks.
- `SETUP.md` records the pinned DuckDB source and build steps.
- `SYSTEM_OPTIMIZATION_GUIDE.md` documents the runtime controls that exist in this version.

The backend can be built on its own. Building the shell requires a DuckDB core build from the pinned
commit, with the `core_functions` and `parquet` extensions linked statically. See [SETUP.md](SETUP.md).

## Requirements

- CMake 3.20 or newer
- A C++17 compiler supported by the installed CUDA Toolkit
- CUDA Toolkit with NVCC, NVRTC, and a compatible NVIDIA driver
- An NVIDIA GPU to run device tests or GPU integration checks

The GitHub Actions workflow compiles the CUDA backend and runs its device-independent checks. GPU
execution tests need a machine with an NVIDIA GPU and are not run on GitHub-hosted runners.

## Git checks

The repository includes a local pre-commit hook and a GitHub Actions workflow. Enable the local hook
once after cloning:

```sh
git config core.hooksPath .githooks
```

The hook checks staged whitespace. GitHub Actions also checks shell syntax and builds the CUDA backend.
Pushing a `v*` tag starts the developer-release workflow, which creates a GitHub prerelease
and attaches a source ZIP with a SHA-256 checksum. The CUDA backend is distributed as source because it
must be compiled for the target CUDA architecture.
