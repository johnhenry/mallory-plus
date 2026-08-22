# Roadmap

The live status of this project is the [GitHub issue tracker](https://github.com/johnhenry/math-plus/issues) —
this file is a snapshot, not a substitute for it. For the full original design record (why each
package exists, the source design conversation, non-goals, risk register), see
[`docs/PLAN.md`](docs/PLAN.md).

## Status (2026-08-13)

**v1 (tensor foundation + parallel dataframe start): fully shipped.**

| Package | What it is |
|---|---|
| `@johnhenry/math-plus-tensor-core` | Typed n-D arrays: dtypes, shapes, broadcasting, views, reductions, `.npy` |
| `@johnhenry/math-plus-tensor-wasm` | Rust/WASM kernels under tensor-core's hot paths (flat extern-C ABI, no wasm-bindgen), incl. a measured SIMD128 fast path (~2.6-4.4x for contiguous elementwise add/mul, feature-detected, separate `.wasm` artifact) and a native `solve()` kernel (LU with partial pivoting) |
| `@johnhenry/math-plus-tensor-autograd` | Reverse-mode tape, `nn.Linear/Embedding/LayerNorm`, `optim.SGD/AdamW`, a `trainer.configure/fit` facade, and a checkpoint format (`Module.stateDict()/loadStateDict()` + `io.writeCheckpoint/loadCheckpoint`) |
| `@johnhenry/math-plus-scalar-types` | Re-exports @johnhenry/math's `ComplexNumber`/`Rational`/`Decimal`/`Interval`/`Quaternion` at the tensor boundary |
| `@johnhenry/math-plus-unit` | Dimensioned quantities (`Unit.of(55, "cm").to("m")`), dimensional-analysis-checked arithmetic |
| `@johnhenry/math-plus-adapter-onnx` | `onnx.load()`/`model.run()` over ONNX Runtime Web, verified against real ONNX models |
| `@johnhenry/math-plus-frame-arrow` | Immutable, expression-oriented `Frame`/`Series` on Apache Arrow, real column pruning + predicate pushdown, plus a `"lazySource"`/`collectAsync()` extension point for genuinely-lazy I/O-backed sources |
| `@johnhenry/math-plus-adapter-math` | Bridge to `@johnhenry/math` (the pure-TS science/CAS sibling library): Matrix/Vector↔Tensor, Symbolic Expr→IR, DualNumber gradient oracle, reference-speed `linalg`, `Graph`↔CSR sparse-matrix bridge, `fft`/`ifft`/`fftPadded`/`convolve`, `SpecialFunctions`/`Distributions`/`HypothesisTests` + a `Statistics.ts` subset |
| `@johnhenry/math-plus-fft` | `ComplexTensor` (split real/imag storage, boxed `ComplexNumber` at edges) + a fresh tensor-shaped `fft`/`ifft`/`fftPadded`/`rfft`/`irfft` (Cooley-Tukey, reference-speed) |
| `@johnhenry/math-plus-image` | `resize` (nearest/bilinear) and `normalize` on `[H,W,C]`/`[N,H,W,C]` tensors |
| `@johnhenry/math-plus-signal` | `convolve`/`sosFilter`/`butter`/`stft`/`istft`/`findPeaks`/`resamplePoly` — verified against a `scipy.signal` subprocess oracle where it matters most (`butter` end-to-end filtering behavior especially) |

**v2 (compilation, GPU, scientific core, dataframe I/O): mostly shipped.**

| Package | Status |
|---|---|
| `@johnhenry/math-plus-tensor-compile` | ✅ Shipped — elementwise expression IR + fusion, the shared lowering target `compileExpr` also targets |
| `@johnhenry/math-plus-frame-parquet` | ✅ Shipped — hyparquet-based read/write with genuine row-group pushdown, LIST/STRUCT (single-level nested) type support, and a genuinely-lazy `scanParquetLazy` on top of frame-arrow's lazy-source extension point |
| `@johnhenry/math-plus-tensor-webgpu` | ✅ Shipped — GEMM/attention/elementwise-fusion WGSL kernels, verified against a real headless GPUAdapter; GEMM stays WASM-routed by default (measured, no crossover found on this machine's integrated-GPU + naive-kernel combination — see `docs/spikes/webgpu-baseline.md`); `scalar-types`' `Interval` used as an f32-vs-f64 rounding-error bound oracle |
| `fft`/`image`/`signal`/`trainer`/checkpoint format | ✅ Shipped (see `@johnhenry/math-plus-fft`/`@johnhenry/math-plus-image`/`@johnhenry/math-plus-signal`/`@johnhenry/math-plus-tensor-autograd` rows above) |
| `@johnhenry/math-plus-data` (the `data` namespace, async loaders on `@johnhenry/iteration`) | ✅ Shipped (issue #22, unblocked by `@johnhenry/iteration@0.0.0`'s npm publish 2026-08-13) — curated `Dataset` facade: `fromAsync`/`chunk` (← upstream `group`)/`batch`+`collate` (Tensor batches, incl. trainer-shaped `{x,y}`)/seeded `shuffle`/`epochs` with per-epoch reshuffle/`mapConcurrent`/`prefetch`/`fold`, `AbortSignal` cancellation end to end |
| Native (WASM) linalg kernels | `solve` ✅ shipped (the first candidate named in `docs/PLAN.md` §9); QR/SVD/eigen/Cholesky stay reference-speed in `adapter-math` for now |

**v3 (full dataframe system, Python interop, scientific breadth): `interop-python` shipped, everything else not started** as planned.

| Package | Status |
|---|---|
| `johnhenry-math-plus-interop` (PyPI, `packages/interop-python`) | ✅ Shipped — `read_ipc`/`write_ipc`/`read_parquet`/`write_parquet`/`.npy`/`.npz` helpers, bidirectional JS↔Python conformance proven both ways with committed fixtures |
| Window ops, full groupby/join maturity | Not started |
| Sparse solvers, GPU kernel DSL maturity, Danfo/pandas parity | Not started — intentionally deferred until v1/v2 prerequisites land |

## Open decisions

- **Rust `Complex`/`Fraction` scalars** (issue #27) — kept open as "someday" by choice (reaffirmed 2026-08-13), now with a concrete implementation roadmap on the issue: split-storage Complex kernels fit the existing WASM ABI cleanly (`fft_f64_split` is the natural first candidate, triggered by a measured `@johnhenry/math-plus-fft` bottleneck); Fraction/Rational can never cross the flat-numeric ABI and would be a separate crate if ever needed.

**Resolved:**
- Danfo.js-like ergonomics for `frame-arrow` — no new `adapter-danfo` package for now; `frame-arrow` keeps its own expression-oriented API (see `docs/PLAN.md` §9 item 2 for the full reasoning).
- Browser bundle-size budget (issue #24, closed 2026-08-13) — no numeric budget, final; the lazy-loading policy + per-package granularity is the size control (see `docs/PLAN.md` §9 item 7).

## Cross-repo dependencies

See [`docs/FAMILY.md`](docs/FAMILY.md) for the full picture of how the math family of repos relates (why they stay separate, what's actually connected today, and open interop opportunities).

- [`johnhenry/math`](https://github.com/johnhenry/math) — `@johnhenry/math` (pure-TS science/CAS library) and `@johnhenry/iteration` (the future `data` namespace's foundation). [Issue #13](https://github.com/johnhenry/math/issues/13) tracks JSR-publishing `@johnhenry/math`/`@johnhenry/iteration` to unblock this repo's own dual npm+JSR distribution.
- [`johnhenry/mallory`](https://github.com/johnhenry/mallory) — a reactive graphing-calculator app (own `CellGraph` store, unrelated to `@johnhenry/math`'s `Graph<T>` — see `docs/FAMILY.md`'s naming note) that depends on `@johnhenry/math` directly for its formula language. Its `CellGraph` was separately studied as design prior art for `frame-arrow`'s lazy planner (reference only, see `docs/spikes/cellgraph-study.md`); 5 sharp edges found during that spike were reported upstream as issues #12–#16, all fixed and closed.
