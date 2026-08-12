# Roadmap

The live status of this project is the [GitHub issue tracker](https://github.com/johnhenry/mallory-plus/issues) —
this file is a snapshot, not a substitute for it. For the full original design record (why each
package exists, the source design conversation, non-goals, risk register), see
[`docs/PLAN.md`](docs/PLAN.md).

## Status (2026-08-11)

**v1 (tensor foundation + parallel dataframe start): fully shipped.**

| Package | What it is |
|---|---|
| `mallory-tensor-core` | Typed n-D arrays: dtypes, shapes, broadcasting, views, reductions, `.npy` |
| `mallory-tensor-wasm` | Rust/WASM kernels under tensor-core's hot paths (flat extern-C ABI, no wasm-bindgen), incl. a measured SIMD128 fast path (~2.6-3x for contiguous elementwise add/mul, feature-detected, separate `.wasm` artifact) |
| `mallory-tensor-autograd` | Reverse-mode tape, `nn.Linear/Embedding/LayerNorm`, `optim.SGD/AdamW` |
| `mallory-scalar-types` | Re-exports mallory-math's `ComplexNumber`/`Rational`/`Decimal` at the tensor boundary |
| `mallory-unit` | Dimensioned quantities (`Unit.of(55, "cm").to("m")`), dimensional-analysis-checked arithmetic |
| `mallory-adapter-onnx` | `onnx.load()`/`model.run()` over ONNX Runtime Web, verified against real ONNX models |
| `mallory-frame-arrow` | Immutable, expression-oriented `Frame`/`Series` on Apache Arrow, real column pruning + predicate pushdown |
| `mallory-adapter-math` | Bridge to `mallory-math` (the pure-TS science/CAS sibling library): Matrix/Vector↔Tensor, Symbolic Expr→IR, DualNumber gradient oracle, reference-speed `linalg` |

**v2 (compilation, GPU, scientific core, dataframe I/O): in progress.**

| Package | Status |
|---|---|
| `mallory-tensor-compile` | ✅ Shipped — elementwise expression IR + fusion, the shared lowering target `compileExpr` also targets |
| `mallory-frame-parquet` | 🔄 In progress — hyparquet-based read/write with genuine row-group pushdown |
| `mallory-tensor-webgpu` | Not started |
| `data` namespace (async loaders on `mallory-iteration`) | Blocked — waiting on `mallory-iteration`'s npm publish (it lives in the sibling [`mallory`](https://github.com/johnhenry/mallory) monorepo) |
| `fft`/`signal`/`image`/`trainer`/checkpoint format | Not started |
| Native (WASM) linalg kernels | Not started — the reference-speed path in `adapter-math` covers this today |

**v3 (full dataframe system, Python interop, scientific breadth): `interop-python` shipped, everything else not started** as planned.

| Package | Status |
|---|---|
| `mallory-interop` (PyPI, `packages/interop-python`) | ✅ Shipped — `read_ipc`/`write_ipc`/`read_parquet`/`write_parquet`/`.npy`/`.npz` helpers, bidirectional JS↔Python conformance proven both ways with committed fixtures |
| Window ops, full groupby/join maturity | Not started |
| Sparse solvers, GPU kernel DSL maturity, Danfo/pandas parity | Not started — intentionally deferred until v1/v2 prerequisites land |

## Open decisions

- **Danfo.js-like ergonomics for `frame-arrow`** — unresolved; `frame-arrow` shipped its own expression-oriented API rather than mimicking Danfo. Whether `adapter-danfo` is worth building, or Danfo idioms should fold into `frame-arrow` directly, is still open.
- **Rust `Complex`/`Fraction` scalars** (issue #27) — kept open by choice, though the recorded reasoning leans toward "not needed" (mallory-math's boxed scalars already fill the role).

## Cross-repo dependencies

- [`johnhenry/mallory`](https://github.com/johnhenry/mallory) — `mallory-math` (pure-TS science/CAS library) and `mallory-iteration` (the future `data` namespace's foundation). [Issue #13](https://github.com/johnhenry/mallory/issues/13) tracks JSR-publishing `mallory-math`/`mallory-iteration` to unblock this repo's own dual npm+JSR distribution.
- [`johnhenry/mallory-graph`](https://github.com/johnhenry/mallory-graph) — an unrelated app whose `CellGraph` reactive store was studied as design prior art for `frame-arrow`'s lazy planner (verdict: reference only, see `docs/spikes/cellgraph-study.md`). 5 sharp edges found during that spike were reported upstream as issues #12–#16; fixes in progress.
