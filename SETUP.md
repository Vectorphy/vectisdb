# Developer setup

This guide builds the VectisDB CUDA backend and, optionally, the DuckDB-linked `duckdb_gpu` shell.
Commands assume a Bash-compatible shell for DuckDB's Makefile and the integration scripts.

## DuckDB source pin

VectisDB 1.0.0 was developed against DuckDB **v1.6.0-dev11577**, commit `af1b4a9bd2`. Use that exact
commit when building the DuckDB core used by `gpu_shell`; the extension sources and DuckDB headers must
come from matching revisions.

The GPU label `gpu-v0.2.0-20-g4583f50964` is a VectisDB/GPU-project label. It is not a DuckDB version.
The DuckDB version comes from the core build. The `duckdb_gpu` executable is generated during setup;
there is no prebuilt executable in this repository.

Clone and check out the pinned DuckDB source if you do not already have a matching build:

```sh
git clone https://github.com/duckdb/duckdb.git ../duckdb-core
git -C ../duckdb-core checkout af1b4a9bd2
git -C ../duckdb-core rev-parse --short=10 HEAD
```

The final command should print `af1b4a9bd2`. Build DuckDB with the static extensions VectisDB uses:

```sh
cd ../duckdb-core
CORE_EXTENSIONS="core_functions;parquet" make reldebug
```

On PowerShell, set the environment variable first:

```powershell
$env:CORE_EXTENSIONS = "core_functions;parquet"
make reldebug
```

Return to the VectisDB repository root before following the VectisDB build commands.
Use the DuckDB source directory as `DUCKDB_SOURCE` and the resulting build directory as `BUILD_CORE`,
commonly `../duckdb-core` and `../duckdb-core/build/reldebug` from the VectisDB root.

## Build the CUDA backend

From the VectisDB repository root, configure for your GPU architecture. The default is compute
capability 7.5; change `75` to a supported architecture for your target GPU.

```sh
cmake -S . -B build -G Ninja \
  -DVECTISDB_BUILD_GPU_SHELL=OFF \
  -DVECTISDB_BUILD_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build --parallel
```

To run only the checks that do not need an NVIDIA device:

```sh
ctest --test-dir build --output-on-failure \
  -R '^(test_kernel_cache|test_gpu_logger|test_plan_shape)$'
```

The remaining CUDA tests require a working driver and GPU. Run them on a GPU-equipped machine with:

```sh
ctest --test-dir build --output-on-failure
```

## Build the DuckDB shell

Pass the DuckDB build directory that contains `src/libduckdb_static.a` on Linux or
`src/duckdb_static.lib` on Windows:

```sh
cmake -S . -B build/shell -G Ninja \
  -DVECTISDB_BUILD_GPU_SHELL=ON \
  -DVECTISDB_BUILD_TESTS=OFF \
  -DDUCKDB_SOURCE=/absolute/path/to/duckdb-core \
  -DBUILD_CORE=/absolute/path/to/duckdb-core/build/reldebug \
  -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/shell --target duckdb_gpu --parallel
```

The generated executable is under `build/shell/gpu_shell/`. On Windows, build from a Visual Studio
Developer PowerShell or Command Prompt so NVCC can find the MSVC host compiler.

Check the DuckDB version reported by the linked core:

```sh
build/shell/gpu_shell/duckdb_gpu --gpu-probe
```

The probe opens the linked core and prints its `version()` result. Compare it with
`v1.6.0-dev11577` before using the shell.

## GPU integration checks

These scripts need the generated shell, Bash, and a working NVIDIA GPU. Pass the executable path because
the repository's default build directory may differ:

```sh
bash gpu_shell/verify_gpu_vs_cpu.sh build/shell/gpu_shell/duckdb_gpu
bash gpu_shell/verify_gpu_cache.sh build/shell/gpu_shell/duckdb_gpu
```

The first compares GPU-enabled query results with CPU-only results. The second checks cache behavior
across multiple queries in one persistent process. Both create temporary databases and remove them on
exit.

## Local Git hook

Enable the repository's pre-commit checks once per clone:

```sh
git config core.hooksPath .githooks
```

The hook runs `git diff --cached --check`. GitHub Actions also validates the verification scripts with
`bash -n`.
