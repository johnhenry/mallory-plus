# Mallory Plus

A JS/TypeScript-native numeric computation runtime — a NumPy + PyTorch + pandas + SciPy equivalent for Node/Deno/browser, built on Rust→WASM kernels, optional WebGPU acceleration, and Apache Arrow for tabular data.

Part of the **Mallory** family: the high-performance sibling of [`mallory-math`](https://github.com/johnhenry/mallory) (education/CAS-oriented scalar math), reusing its scalar types (`ComplexNumber`, `Rational`, `Decimal`) at tensor API edges and bridging its `Symbolic` CAS into the tensor compiler. Data pipelines build on [`async-itertools`](https://github.com/johnhenry/async-itertools).

**Status:** planning. See [docs/PLAN.md](./docs/PLAN.md) for the full implementation plan and [docs/perplexity-conversation.md](./docs/perplexity-conversation.md) for the source design conversation.

## Planned packages (unscoped npm, `mallory-*`)

| Package | Role |
|---|---|
| `mallory-runtime` | Umbrella — single import surface with named namespaces |
| `mallory-tensor-core` | Typed n-D arrays: dtypes, strides/views, broadcasting, `.npy` I/O |
| `mallory-tensor-wasm` | Rust→WASM kernels (SIMD, arena allocator, `...Into` zero-alloc ops) |
| `mallory-tensor-autograd` | Reverse-mode tape, `nn.*`, `optim.AdamW` |
| `mallory-tensor-compile` | Expression IR + elementwise fusion (opt-in) |
| `mallory-tensor-webgpu` | GPU matmul/attention kernels (v2) |
| `mallory-frame-arrow` | Immutable Arrow-backed `Frame`/`Series` with lazy expression API |
| `mallory-frame-parquet` | Parquet scan/write with projection/predicate pushdown |
| `mallory-scalar-types` | Thin re-export of mallory-math scalars + tensor-boundary converters |
| `mallory-adapter-*` | Compatibility bridges (math ⁽ᵐᵃˡˡᵒʳʸ⁾, onnx, ml-matrix, mathjs, tfjs, danfo, stdlib, arrow) |
| `mallory-interop` (PyPI) | Python-side Arrow IPC/Parquet/npy helpers |
