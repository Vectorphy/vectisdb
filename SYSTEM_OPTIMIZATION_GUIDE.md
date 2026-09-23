# Hardware and runtime notes

VectisDB builds CUDA code for a selected GPU architecture. Runtime tuning changes the amount of device
memory available to its pools, execution batch sizes, and floating-point behavior. Check the effect on
your workload before keeping a change.

## CUDA architecture

The backend defaults to CUDA architecture `75` in `cuda_engine/CMakeLists.txt`. Set the target when you
configure the build:

```sh
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
```

Choose a value supported by your CUDA Toolkit and deployment GPUs. Reconfigure when changing the target.

## Device memory budgets

`VECTOR_GPU_VRAM_FRACTION` controls the memory budget recorded by the GPU memory pool. Its default is
`0.85` of currently free device memory. Values must be greater than zero and at most one. The pool
records a budget during initialization and allocates its arena only when requested.

The resident column cache has a separate budget. Set `VECTOR_GPU_CACHE_FRACTION` to use a fraction of
free device memory, or set `VECTOR_GPU_CACHE_MB` to an explicit MiB budget. The absolute MiB setting
takes precedence when both are valid.

Set environment variables before starting `duckdb_gpu`; these values are read when the relevant pool is
first initialized.

## Batch size and streaming

Streaming projections use pinned host buffers and several CUDA streams. The engine computes a batch
size that fits its buffer slots. `VECTOR_GPU_STREAM_BATCH_ROWS` can override that size; use a positive
multiple of DuckDB's standard vector size. Very wide rows reduce the number of rows that fit in a slot.

Streaming is enabled by default. `VECTOR_GPU_STREAM_PIPELINE=0` selects the non-streaming GPU execution
path. A process permits one active streaming query at a time because the pinned ring buffers are shared.

## Floating-point behavior

`VECTOR_GPU_FAST_MATH=1` enables NVRTC fast math. `VECTOR_GPU_FP32_RELAXATION=1` opts into reduced
precision for supported math expressions. `VECTOR_GPU_STRICT_FP=1` declines expressions whose math
functions are not bit-identical to DuckDB's CPU implementation. These options can change which
expressions are routed to the GPU or the numerical result; compare output against CPU execution for the
queries that matter to you.

## Offload decisions

The optimizer applies a 100,000-row work threshold and an expression-density model before sending work
to the GPU. `VECTOR_GPU_MIN_AMPLIFICATION` selects the legacy row-amplification rule instead. The
verification scripts set it for specific coverage cases; leave it unset for normal operation unless you
are measuring that alternate rule.

To inspect routing decisions, run the shell with `--gpu-trace`. To compare results, use the two checks
in `gpu_shell/` described in [SETUP.md](SETUP.md). They need a real GPU.
