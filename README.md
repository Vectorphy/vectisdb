# VectisDB

VectisDB is an experimental DuckDB GPU execution project. Version 1.0.0 is a developer release.
It contains a CUDA execution library, DuckDB optimizer and operator integration, and an optional shell
that links the integration into DuckDB. Most of the code was written by AI tools; see the
[AI policy](AI_POLICY.md).

## DuckDB version

The DuckDB core build used for this release reports v1.6.0-dev11577 and is pinned to commit
`af1b4a9bd2`. That is DuckDB's development version, not VectisDB's. The label
`gpu-v0.2.0-20-g4583f50964` is the GPU project revision.

The build produces the `duckdb_gpu` executable. It is not in the repository.

## Repository layout

- `cuda_engine/` contains the CUDA, NVRTC, memory, cache, and execution code.
- `extension/` holds the DuckDB optimizer and physical operator integration.
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

For WSL with CUDA Toolkit 12.4, use GCC 13 for both the DuckDB core build and VectisDB. NVCC 12.4
supports GCC through 13.2 and rejects GCC 15; see NVIDIA's [CUDA 12.4 host compiler
matrix](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-installation-guide-linux/#host-compiler-support-policy).
The WSL commands in [SETUP.md](SETUP.md) use GCC 13 and separate shell and test build directories.

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

## Project files

- [Contributing](CONTRIBUTING.md) explains how to report a bug and prepare a change.
- [Code of Conduct](CODE_OF_CONDUCT.md) sets expectations for project discussions.
- [Security policy](SECURITY.md) describes how to report a vulnerability privately.
- [AI policy](AI_POLICY.md) explains how the code was built with AI tools and what that means for contributors.
- [License](LICENSE) has the VectisDB license and third-party notices.

Use the issue forms under `.github/ISSUE_TEMPLATE/` for bugs and feature requests. Pull requests
use `.github/PULL_REQUEST_TEMPLATE.md`.

## Citation

If you use or build on VectisDB, cite the project and credit Vectorphy. GitHub reads the citation
metadata in [`CITATION.cff`](CITATION.cff) and offers it through the repository's **Cite this
repository** control.
