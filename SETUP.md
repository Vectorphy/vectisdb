# Developer setup

This guide builds the VectisDB CUDA backend, the optional DuckDB-linked `duckdb_gpu` shell, and the
test targets. Commands are written for Bash in Linux or WSL.

## Version and toolchain

VectisDB 1.0.0 was developed against DuckDB **v1.6.0-dev11577**, commit `af1b4a9bd2`. Build DuckDB
and VectisDB from matching source revisions.

In WSL with CUDA Toolkit 12.4, use GCC 13 for both builds. NVCC 12.4 supports GCC through 13.2 and
rejects GCC 15. A mixed compiler/runtime setup can also make the final shell link fail. See NVIDIA's
[CUDA 12.4 host compiler support table](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-installation-guide-linux/#host-compiler-support-policy).

Check the tools before building:

```sh
nvcc --version
/usr/bin/g++-13 --version
cmake --version
ninja --version
```

Install GCC 13 and G++ 13 from your WSL distribution if `/usr/bin/g++-13` is missing. Keep the build
inside WSL and use Linux paths and Linux DuckDB libraries; do not mix them with a Windows-native build.

## Build the DuckDB core

Clone DuckDB if needed, then check out the pinned commit:

```sh
git clone https://github.com/duckdb/duckdb.git /path/to/duckdb-core
git -C /path/to/duckdb-core checkout af1b4a9bd2
git -C /path/to/duckdb-core rev-parse --short=10 HEAD
```

The final command should print `af1b4a9bd2`. Set the paths for your checkout and build DuckDB with
the static extensions used by the VectisDB shell:

```sh
export DUCKDB_SOURCE="/path/to/duckdb-core"
export BUILD_CORE="$DUCKDB_SOURCE/build/reldebug"

cd "$DUCKDB_SOURCE"
CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 \
  CORE_EXTENSIONS="core_functions;parquet" make reldebug
```

Confirm the static core library was produced:

```sh
test -f "$BUILD_CORE/src/libduckdb_static.a" && echo "DuckDB library found"
```

If changing compiler versions in an existing DuckDB build, use a fresh build directory or clear its
generated `build/reldebug` output before rebuilding. CMake caches compiler and linker paths.

## Build the DuckDB shell

From the VectisDB repository root, configure a separate shell build with GCC 13 as both the C++ and
CUDA host compiler. Change `75` to your GPU's CUDA architecture if needed.

```sh
cmake -S . -B build/shell-gcc13 -G Ninja \
  -DVECTISDB_BUILD_GPU_SHELL=ON \
  -DVECTISDB_BUILD_TESTS=OFF \
  -DDUCKDB_SOURCE="$DUCKDB_SOURCE" \
  -DBUILD_CORE="$BUILD_CORE" \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_ARCHITECTURES=75

cmake --build build/shell-gcc13 --target duckdb_gpu --parallel
```

Check the shell's linked DuckDB version and optimizer registration:

```sh
./build/shell-gcc13/gpu_shell/duckdb_gpu --gpu-probe
```

The probe should report DuckDB `v1.6.0-dev11577` and `gpu optimizer: registered`. It checks the linked
core and optimizer registration; use the GPU integration scripts below to exercise query execution.

## Build and run tests

The shell build above sets `VECTISDB_BUILD_TESTS=OFF`, so CTest will report no tests for that build
directory. Configure a separate test build with tests enabled:

```sh
cmake -S . -B build/tests-gcc13 -G Ninja \
  -DVECTISDB_BUILD_GPU_SHELL=OFF \
  -DVECTISDB_BUILD_TESTS=ON \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_ARCHITECTURES=75

cmake --build build/tests-gcc13 --parallel
ctest --test-dir build/tests-gcc13 --output-on-failure
```

The full CUDA test suite needs a working NVIDIA driver and GPU. To run only the device-independent
tests:

```sh
ctest --test-dir build/tests-gcc13 --output-on-failure \
  -R '^(test_kernel_cache|test_gpu_logger|test_plan_shape)$'
```

## GPU integration checks

These scripts compare GPU and CPU results and exercise the resident cache in one persistent process.
They require the shell built above and a working GPU in WSL:

```sh
bash gpu_shell/verify_gpu_vs_cpu.sh build/shell-gcc13/gpu_shell/duckdb_gpu
bash gpu_shell/verify_gpu_cache.sh build/shell-gcc13/gpu_shell/duckdb_gpu
```

## Windows-native builds

For a Windows-native shell, build DuckDB and VectisDB with the Windows toolchain and use the DuckDB
build directory containing `src/duckdb_static.lib`. Run CMake from a Visual Studio Developer PowerShell
or Command Prompt so NVCC can find MSVC. In PowerShell, set the extension list before running
`make reldebug` in DuckDB's source directory:

```powershell
$env:CORE_EXTENSIONS = "core_functions;parquet"
make reldebug
```

From the VectisDB root, configure and build with the matching source and build paths:

```powershell
cmake -S . -B build/shell -G Ninja `
  -DVECTISDB_BUILD_GPU_SHELL=ON `
  -DVECTISDB_BUILD_TESTS=OFF `
  -DDUCKDB_SOURCE=C:/path/to/duckdb-core `
  -DBUILD_CORE=C:/path/to/duckdb-core/build/reldebug `
  -DCMAKE_CUDA_ARCHITECTURES=75

cmake --build build/shell --target duckdb_gpu --parallel
.\build\shell\gpu_shell\duckdb_gpu.exe --gpu-probe
```

WSL instructions and `.a` libraries in this guide apply to Linux builds only.

## Local Git hook

Enable the repository's pre-commit check once per clone:

```sh
git config core.hooksPath .githooks
```

The hook runs `git diff --cached --check`. GitHub Actions also checks the verification scripts with
`bash -n`.
