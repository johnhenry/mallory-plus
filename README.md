# Math Plus

A JS/TypeScript-native numeric computation runtime — a NumPy + PyTorch + pandas + SciPy equivalent for Node/Deno/browser, built on Rust→WASM kernels, optional WebGPU acceleration, and Apache Arrow for tabular data.

Part of the **math** family: the high-performance sibling of [`@johnhenry/math`](https://github.com/johnhenry/math) (education/CAS-oriented scalar math), reusing its scalar types (`ComplexNumber`, `Rational`, `Decimal`) at tensor API edges and bridging its `Symbolic` CAS into the tensor compiler. Data pipelines build on [`@johnhenry/iteration`](https://github.com/johnhenry/math) (a pull-based async iterator/transducer toolkit living in the same monorepo).

**Status:** actively published. Most packages below are shipping to npm and JSR under the `@johnhenry` scope; see each package's own `CHANGELOG.md` for its release history. See [docs/PLAN.md](./docs/PLAN.md) for the original implementation plan (with later-renamed identifiers annotated, not rewritten) and [docs/perplexity-conversation.md](./docs/perplexity-conversation.md) for the source design conversation.

## Packages (npm + JSR, `@johnhenry/math-plus-*`)

| Package | Role |
|---|---|
| `@johnhenry/math-plus-tensor-core` | Typed n-D arrays: dtypes, strides/views, broadcasting, `.npy` I/O |
| `@johnhenry/math-plus-tensor-wasm` | Rust→WASM kernels (SIMD, arena allocator, `...Into` zero-alloc ops) |
| `@johnhenry/math-plus-tensor-autograd` | Reverse-mode tape, `nn.*`, `optim.AdamW` |
| `@johnhenry/math-plus-tensor-compile` | Expression IR + elementwise fusion (opt-in) |
| `@johnhenry/math-plus-tensor-webgpu` | GPU matmul/attention kernels |
| `@johnhenry/math-plus-frame-arrow` | Immutable Arrow-backed `Frame`/`Series` with lazy expression API |
| `@johnhenry/math-plus-frame-parquet` | Parquet scan/write with projection/predicate pushdown |
| `@johnhenry/math-plus-scalar-types` | Thin re-export of `@johnhenry/math` scalars + tensor-boundary converters |
| `@johnhenry/math-plus-telemetry` | Shared event schema + sink registry, zero-cost no-op default |
| `@johnhenry/math-plus-fft` | `ComplexTensor` + `fft`/`ifft`/`rfft`/`irfft`/`fft2`/`fftn` |
| `@johnhenry/math-plus-signal` | `convolve`/`stft`/`findPeaks`/`sosFilter`/`butter`/`resamplePoly` (SciPy-equivalent slice) |
| `@johnhenry/math-plus-image` | resize/normalize tensor ops for practical ML/media pipelines |
| `@johnhenry/math-plus-data` | Async dataset pipelines: chunk/batch/shuffle/epochs/mapConcurrent/prefetch |
| `@johnhenry/math-plus-mcp` | MCP server exposing symbolic + guarded numeric tools to agents |
| `@johnhenry/math-plus-adapter-math` | Bridge to `@johnhenry/math` (Matrix/Vector ↔ Tensor, Symbolic → IR, Graph → CSR) |
| `@johnhenry/math-plus-adapter-onnx` | ONNX Runtime Web wrapper (Tensor marshalling) |
| `@johnhenry/math-plus-unit` | Unit/dimension scalar type with dimensional-analysis-checked arithmetic |
| `johnhenry-math-plus-interop` (PyPI) | Python-side Arrow IPC/Parquet/npy helpers |
