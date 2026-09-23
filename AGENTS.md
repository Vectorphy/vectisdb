# Repository guidance for coding agents

Read [README.md](README.md) for the project layout and [SETUP.md](SETUP.md) before changing build
commands. VectisDB 1.0.0 is tied to DuckDB `v1.6.0-dev11577`, commit `af1b4a9bd2`; keep the extension,
headers, and shell build on that revision unless the change explicitly updates the dependency pin.

Keep changes scoped to this C++/CUDA project. Do not add GUI files or commit build output. `dist/` is
for release packages and is ignored by Git. The shell is optional and depends on an external DuckDB
build.

For code changes, use the relevant checks in [CONTRIBUTING.md](CONTRIBUTING.md). GPU checks require a
compatible NVIDIA device; do not report them as passing unless they were run on one.
