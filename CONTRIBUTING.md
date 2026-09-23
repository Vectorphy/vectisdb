# Contributing

VectisDB is an experimental DuckDB GPU execution extension. Bug reports are easiest to investigate
when they include the environment and the exact command or query that failed.

## Before changing code

For a larger change, open an issue first so we can agree on the problem and scope. Bug reports and
feature requests can use the forms on the Issues page. Do not include credentials, private data, or
unpublished vulnerability details in an issue.

Read [SETUP.md](SETUP.md) before building. VectisDB 1.0.0 uses DuckDB `v1.6.0-dev11577` at commit
`af1b4a9bd2`; the shell integration expects matching DuckDB sources and headers.

## Preparing a change

Work on a branch based on `main`. Keep the patch focused, and update the documentation when behavior
or setup steps change. Avoid committing generated build output or files under `dist/`.

Run checks that match the change. For a local review, start with:

```sh
git diff --check
bash -n gpu_shell/verify_gpu_cache.sh
bash -n gpu_shell/verify_gpu_vs_cpu.sh
```

For C++ or build-system changes, configure and build the test-enabled directory described in
[SETUP.md](SETUP.md), then run CTest from that same directory. The shell build has tests disabled and
will show `No tests were found` if you run CTest there. GPU execution and the two shell verification
scripts need a compatible NVIDIA GPU, driver, and the pinned DuckDB shell. In the pull request, say
which commands you ran and include the CUDA version and GPU model when you report GPU results.

## Pull requests

Describe the problem, the change, and how you checked it. Link the related issue if there is one. If a
check needs hardware you do not have, say so instead of implying it passed. The repository's pull
request template includes a short checklist.
