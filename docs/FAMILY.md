# The Mallory family

A lightweight index of every `johnhenry` repo carrying the Mallory name (or
closely related to one), what each is *for*, and how they actually connect
today — kept separate from any one repo's own build/release tooling on
purpose (see "Why these stay separate repos" below).

This is a snapshot, not a live registry — if a repo's shape has changed,
trust that repo's own README/PLAN over this file and fix this file to match.

## The repos

| Repo | What it is | Depends on |
|---|---|---|
| [`mallory`](https://github.com/johnhenry/mallory) | Pure-TS, zero-dependency science/CAS monorepo. Two packages: `mallory-math` (scalar types, symbolic algebra, linear algebra, graph theory, numerical methods, statistics — see below) and `mallory-iteration` (a pull-based async iterator/transducer toolkit). | Nothing — the dependency-free foundation of the family. |
| [`mallory-plus`](https://github.com/johnhenry/mallory-plus) | JS/TS-native numeric computation runtime — NumPy + PyTorch + Arrow equivalent. Rust→WASM tensor kernels, optional WebGPU acceleration, an Arrow dataframe layer, ONNX/Python interop. | `mallory-math` (scalar types + `adapter-math` bridge), `mallory-iteration` (planned, blocked on its npm publish). |
| [`mallory-graph`](https://github.com/johnhenry/mallory-graph) | An app: a GeoGebra/Observable-notebook-style reactive graphing calculator, built on its own `CellGraph` reactive dependency-graph store. | `mallory-math` directly, at runtime — see below. Not a library other repos depend on. |

**Naming note:** despite the name, `mallory-graph`'s "graph" is a *reactive
dependency graph* (`CellGraph` — cells with formulas that recompute when
their dependencies change), unrelated to `mallory-math`'s `Graph<T>` (a
classic graph-theory class: vertices/edges, BFS/DFS, Dijkstra, MST,
topological sort). The two "Graph" concepts share nothing but the word.

## Why these stay separate repos

Each repo has a genuinely different shape, and merging them would trade a
small discoverability win for real, ongoing friction:

- **`mallory`** is pure TS, zero dependencies, npm-only. Its whole appeal
  (to itself and to consumers) is being simple and stable.
- **`mallory-plus`** adds a Rust/Cargo/WASM/WebGPU build chain and a much
  larger, faster-moving surface. Folding it into `mallory` would put that
  toolchain in front of anyone touching the CAS library, for no benefit to
  either side.
- **`mallory-graph`** is a deployed app with its own release/hosting story,
  not a library — there's no "workspace" relationship to have with it.

The actual mechanism that connects them is dependencies (npm packages) plus
occasional cross-repo issues (e.g. `mallory-graph`'s `CellGraph` was studied
as design prior art for `mallory-plus`'s dataframe lazy planner — 5 bugs
found during that read were filed and fixed upstream as
[`mallory-graph#12`–`#16`](https://github.com/johnhenry/mallory-graph/issues)).
When a genuinely shared *piece of code* is needed (not just a similar idea),
the right move is extracting it into its own package at that point — the
same way `async-itertools` graduated into `mallory-iteration` inside the
`mallory` monorepo once `mallory-plus`'s `data` namespace concretely needed
it as a dependency. Extract on demand, not preemptively.

## What's actually connected right now

- `mallory-plus`'s `scalar-types` package re-exports `mallory-math`'s
  `ComplexNumber`/`Rational`/`Decimal`/`Interval`/`Quaternion` verbatim,
  plus adds `complexToParts`/`partsToComplex` conversion helpers (boxed
  `ComplexNumber[]` ↔ split `{real, imag}` `Float64Array`s) for tensor edges.
- `mallory-plus`'s `adapter-math` package bridges `mallory-math` into
  tensor/dataframe land: `Matrix`/`Vector` ↔ `Tensor` conversion,
  `compileExpr` (compiles `mallory-math`'s symbolic `Expr` AST into
  `tensor-compile`'s IR — all 39 named functions map through except
  `gcd`/`lcm`/`sum`/`product`, which throw a typed `UnsupportedExprError`
  since they're integer-domain/reduction ops with no elementwise-tensor
  meaning), `toCSR`/`toDense` (`Graph<T>` → sparse-matrix CSR format), a
  `DualNumber`-based autodiff test oracle (test-only, not a runtime
  dependency), `fft`/`ifft`/`fftPadded`/`convolve` (wraps `mallory-math`'s
  `FFT` class), and `SpecialFunctions`/`Distributions`/`HypothesisTests`
  (re-exported verbatim) plus a `Statistics.ts` subset wrapped for
  `Float64Array` input. Its `linalg` module (`solve`/`qr`/`svd`/
  `eigSymmetric`/etc.) is a thin Tensor-shaped wrapper that literally
  delegates to `mallory-math`'s own `MatrixMath` — not reimplemented.
- `mallory-plus`'s `tensor-webgpu` package uses `scalar-types`'s `Interval`
  as an f32-vs-f64 rounding-error bound oracle (`test/precision-oracle.test.ts`)
  — propagates a per-operation f32 ULP bound through `Interval`'s real
  arithmetic across a fused kernel chain, then asserts the real GPU f32
  result falls inside that interval.
- `mallory-graph` imports `mallory-math`'s `Expr`/`FuncName`/`CmpOp`/
  `BinaryFuncName` types directly (`src/lib/free-vars.ts`,
  `expr-to-latex.ts`) and pairs `mallory-math`'s `Structure<number>` with an
  explicit element enumeration in `src/lib/finite-structure.ts`.
  `mallory-math`'s symbolic `Expr` IS `mallory-graph`'s formula language —
  `CellGraph` itself is fully formula-agnostic (every cell is an opaque
  `() => T` closure); the app layer is what plugs `mallory-math`'s AST in.

## Interop opportunities

Surveyed 2026-08-12 by reading `mallory`'s `packages/math`/`packages/iteration`
source, `mallory-graph`'s `src/lib`, and `mallory-plus`'s existing bridge
code (`adapter-math`, `tensor-compile`'s IR, `frame-arrow`'s `Expr`), looking
for real, concrete gaps — not vague thematic similarity. Filed as issues the
same day; 6 of 7 actionable ones shipped 2026-08-12 (all except #38, a
deliberately-deferred future direction). #8 below is a documented non-gap,
never filed.

1. ✅ **Shipped** — `mallory-graph`'s cycle detection, investigated: `CellGraph`'s `this.stack.includes(id)` check looked like it could miss a cycle routed through a clean cell. Turned out, on careful empirical investigation (4 constructed repro scenarios, checked against the actual unmodified code before implementing anything), the existing code already catches every case via `propagateDirty()`'s unconditional cascade — the hypothesis was wrong. Reverted the speculative fix, added 4 regression tests locking in the correct existing behavior instead. — [`mallory-graph#18`](https://github.com/johnhenry/mallory-graph/issues/18)
2. ✅ **Shipped** — FFT bridge. `adapter-math` now exports `fft`/`ifft`/`fftPadded`/`convolve`, thin wrappers around `mallory-math`'s `FFT` class using a `{real, imag}` `Float64Array` split-storage convention. Reference-speed, no native kernel. — [`mallory-plus#33`](https://github.com/johnhenry/mallory-plus/issues/33)
3. ✅ **Shipped** — differential test cross-checking `tensor-compile`'s independently-written `erf` against `mallory-math`'s `SpecialFunctions.erf` (devDependency only, no runtime coupling). The two formulas agree within `1e-6`. — [`mallory-plus#34`](https://github.com/johnhenry/mallory-plus/issues/34)
4. ✅ **Shipped** — `SpecialFunctions`/`Distributions`/`HypothesisTests` re-exported verbatim from `adapter-math`; a `Statistics.ts` subset (`mean`/`variance`/`standardDeviation`/`median`/`percentile`/`correlation`/`linearRegression`) gets `Float64Array`-accepting wrappers via `Vector.fromArray`. — [`mallory-plus#35`](https://github.com/johnhenry/mallory-plus/issues/35)
5. ✅ **Shipped** — `Interval` re-exported from `scalar-types`; `tensor-webgpu` now has a real demonstration (`test/precision-oracle.test.ts`) propagating a per-step f32 rounding-error bound through `Interval`'s own arithmetic across the `add -> mul -> sigmoid` fusion chain, asserting the real GPU f32 result falls inside the bound — a stronger claim than the existing tolerance-based comparisons. — [`mallory-plus#36`](https://github.com/johnhenry/mallory-plus/issues/36)
6. ✅ **Shipped** — `Quaternion` re-exported from `scalar-types`, ready if a 3D/rotation `tensor-webgpu` use case ever shows up. — [`mallory-plus#37`](https://github.com/johnhenry/mallory-plus/issues/37)
7. **Still open, deliberately deferred.** `frame-arrow`'s `Expr`/`fn.*` namespace is currently too thin to be a second `Symbolic` compile target (`ScalarFnOp` has exactly one member, `"month"`, no transcendental functions). `mallory-math`'s `Symbolic` `Expr` AST already compiles to `tensor-compile`'s IR via `adapter-math`'s `compileExpr` — if/when `frame-arrow`'s `fn.*` namespace grows real math functions on its own merits, the same `UNARY_FUNC_MAP`-style translation could give computed dataframe columns a second `Symbolic`-Expr compile target largely for free. Revisit then, don't force `fn.*` growth for this reason alone. — [`mallory-plus#38`](https://github.com/johnhenry/mallory-plus/issues/38)
8. **`mallory-iteration` (pull-only) and `CellGraph` (push-only) are intentional duals, not a gap.** `mallory-iteration`'s entire toolkit (itertools, transducers, `AsyncChannel`) is pull-based — every consumer actively pulls from a source, nothing broadcasts. `CellGraph`'s dependency propagation is the structural opposite: a synchronous `Map<string, Set<Listener>>` push-callback model, no iterators involved at all. Worth documenting as a real architectural contrast within the family (and a reason NOT to force one to reuse the other's primitives) rather than something to reconcile. No issue filed — nothing to track.
