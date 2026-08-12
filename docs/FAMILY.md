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
  `ComplexNumber`/`Rational`/`Decimal` verbatim, plus adds
  `complexToParts`/`partsToComplex` conversion helpers (boxed
  `ComplexNumber[]` ↔ split `{real, imag}` `Float64Array`s) for tensor edges.
- `mallory-plus`'s `adapter-math` package bridges `mallory-math` into
  tensor/dataframe land in four ways: `Matrix`/`Vector` ↔ `Tensor`
  conversion, `compileExpr` (compiles `mallory-math`'s symbolic `Expr` AST
  into `tensor-compile`'s IR — all 39 named functions map through except
  `gcd`/`lcm`/`sum`/`product`, which throw a typed `UnsupportedExprError`
  since they're integer-domain/reduction ops with no elementwise-tensor
  meaning), `toCSR`/`toDense` (`Graph<T>` → sparse-matrix CSR format), and a
  `DualNumber`-based autodiff test oracle (test-only, not a runtime
  dependency). Its `linalg` module (`solve`/`qr`/`svd`/`eigSymmetric`/etc.)
  is a thin Tensor-shaped wrapper that literally delegates to
  `mallory-math`'s own `MatrixMath` — not reimplemented.
- `mallory-graph` imports `mallory-math`'s `Expr`/`FuncName`/`CmpOp`/
  `BinaryFuncName` types directly (`src/lib/free-vars.ts`,
  `expr-to-latex.ts`) and pairs `mallory-math`'s `Structure<number>` with an
  explicit element enumeration in `src/lib/finite-structure.ts`.
  `mallory-math`'s symbolic `Expr` IS `mallory-graph`'s formula language —
  `CellGraph` itself is fully formula-agnostic (every cell is an opaque
  `() => T` closure); the app layer is what plugs `mallory-math`'s AST in.

## Open interop opportunities

Surveyed 2026-08-12 by reading `mallory`'s `packages/math`/`packages/iteration`
source, `mallory-graph`'s `src/lib`, and `mallory-plus`'s existing bridge
code (`adapter-math`, `tensor-compile`'s IR, `frame-arrow`'s `Expr`), looking
for real, concrete gaps — not vague thematic similarity. Ranked roughly by
how actionable/valuable each looks; none of these are started.

1. **`mallory-graph`'s cycle detection is incomplete — `mallory-math`'s `Graph<T>` already has the fix.** `CellGraph`'s only cycle guard is `this.stack.includes(id)` inside `get()` — a per-evaluation check that can miss a cycle routed through a currently-clean (non-dirty) cell, since a clean cell short-circuits without re-walking its dependencies. `mallory-math`'s `Graph<T>` already has real `hasCycle()`/`topologicalSort()` over its adjacency structure; `CellGraph`'s private `Map<string, CellRecord>` (each record holding `dependencies`/`dependents` sets) is structurally an adjacency-list digraph already, just unexported and inlined. A full topological-sort-based cycle check run after each `propagateDirty` (or on demand) would close this gap. This is the one finding that looks like an actual latent correctness bug, not just an opportunity.
2. **No FFT anywhere in `mallory-plus`.** `mallory-math`'s `FFT.ts` (`FFT.fft`/`ifft`/`dft`/`convolve`) is real, tested, radix-2 Cooley-Tukey — and is the single most tensor-shaped file in `mallory-math` (array in, array out) with zero equivalent on the `mallory-plus` side. It operates on plain `(ComplexNumber|number)[]`, not typed arrays, so a bridge would need a `Float32Array`/`Float64Array` in/out wrapper (parallel to `scalar-types`'s existing `complexToParts`/`partsToComplex`) — straightforward, and would fill a real gap in the "practical ML/media compute" bundle `docs/PLAN.md` §5 already lists `fft` under (currently "not started").
3. **`SpecialFunctions.erf` vs `tensor-compile`'s own `erf`.** `tensor-compile`'s IR and `tensor-webgpu`'s WGSL fusion both implement `erf` independently (same Abramowitz-Stegun 7.1.26 approximation, deliberately not calling `mallory-math` — "tensor-compile stays dependency-free of `mallory-math`," which is the right call for a hot-path kernel). That's fine as a design choice, but there's no cross-check anywhere that the two independently-written formulas actually agree with `mallory-math`'s own `SpecialFunctions.erf`/`erfc`. A cheap, low-risk differential test (compare `tensor-compile`'s erf against `mallory-math`'s across a value range) would be a real correctness win without touching the runtime dependency boundary at all.
4. **`gamma`/`beta`/`lnGamma` are a real gap for a future SciPy-equivalent tier.** `docs/PLAN.md` §6.3 already earmarks a SciPy-equivalent layer (stats/special functions) as a later-priority tier; `mallory-math`'s `SpecialFunctions.ts` (`gamma`, `lnGamma`, `beta`, `regularizedGammaP/Q`, `regularizedIncompleteBeta`) and `Distributions.ts`/`Statistics.ts` (distributions, hypothesis tests, descriptive stats) already exist and are scalar/array-in — the same "thin Tensor-shaped wrapper delegating to `mallory-math`" pattern `adapter-math`'s `linalg.ts` already uses for matrix decompositions would apply directly here when that tier starts, rather than reimplementing from scratch.
5. **Interval arithmetic has no equivalent, and pairs naturally with the WebGPU precision story.** `mallory-math`'s `Interval.ts` (rigorous interval arithmetic: `add`/`multiply`/`sqrt`/`sin`/etc. over `[lo, hi]` bounds) has zero counterpart anywhere in `mallory-plus`. `docs/spikes/webgpu-baseline.md` already documents f32-GPU-vs-f64-CPU precision as a live concern for `tensor-webgpu`'s fusion kernels — `Interval` could serve as a genuine rounding-error bound oracle there, a complement (not a duplicate) to the existing bit-for-bit differential tests.
6. **Quaternion has no equivalent — small, ready-made if 3D/rotation use cases show up.** `mallory-math`'s `Quaternion.ts` (Hamilton product, `slerp`, `toRotationMatrix`, `rotateVector`) is a complete, self-contained value type with no array/batch operations — `tensor-webgpu` has no rotation/3D-transform primitive today. Lower priority than the above (no concrete consumer yet), but worth knowing it exists ready-made rather than reinventing it if a WebGPU graphics/game-engine-adjacent use case ever comes up.
7. **`frame-arrow`'s `Expr`/`fn.*` namespace is currently too thin to be a second `Symbolic` compile target, but the door is open.** `mallory-math`'s `Symbolic` `Expr` AST already compiles to `tensor-compile`'s IR via `adapter-math`'s `compileExpr`. `frame-arrow` has its own, structurally similar `Expr` AST (`ArithExpr`/`CompareExpr`/`LogicalExpr` + a `ScalarFnOp` namespace) — but `ScalarFnOp` currently has exactly one member (`"month"`), no transcendental functions at all. Not actionable today, but if/when `frame-arrow`'s `fn.*` namespace grows real math functions (a plausible direction on its own merits, independent of this), the same `UNARY_FUNC_MAP`-style translation `adapter-math/expr.ts` already established could give computed dataframe columns a second `Symbolic`-Expr compile target for free (e.g. auto-differentiating a computed column's formula). Flagging as a future direction, not a near-term task.
8. **`mallory-iteration` (pull-only) and `CellGraph` (push-only) are intentional duals, not a gap.** `mallory-iteration`'s entire toolkit (itertools, transducers, `AsyncChannel`) is pull-based — every consumer actively pulls from a source, nothing broadcasts. `CellGraph`'s dependency propagation is the structural opposite: a synchronous `Map<string, Set<Listener>>` push-callback model, no iterators involved at all. Worth documenting as a real architectural contrast within the family (and a reason NOT to force one to reuse the other's primitives) rather than something to reconcile.

None of the above are filed as issues yet — happy to file tracking issues (in the relevant repo) for whichever of these are worth pursuing, starting with #1 (the `mallory-graph` cycle-detection gap) since that one reads as an actual bug rather than a nice-to-have.
