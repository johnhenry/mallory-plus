---
title: "JS numeric/tensor/dataframe runtime design (Python NumPy/pandas/SciPy/PyTorch/TensorFlow equivalents for JS/Node/Deno)"
platform: Perplexity
source_url: https://www.perplexity.ai/search/db20e348-ea46-401c-8146-cf44e5ff1f58
extracted_at: 2026-08-10
extraction_method: agent-browser (DOM text dump; Perplexity is not supported by extract-chat-shares or Agent-Reach)
---

# Source conversation (raw extracted text)

> Extracted via headless Chrome DOM text dump (`document.body.innerText`) since Perplexity blocks direct WebFetch (403) and has no official share-snapshot API like ChatGPT/Claude.

## Turn 1: Python numeric libraries → JS equivalents (Node/Deno)

A lot of python numeric libraries are performant because they are written in low-level languages. Are there equivalents in JavaScript? Node? Deno?

Yes, but the landscape looks pretty different from Python's — JavaScript doesn't have a single dominant NumPy-equivalent backed by C, and both Node and Deno rely on similar underlying strategies to get performance.

**Why Python's model doesn't map directly:** NumPy's speed comes from C extensions calling into optimized BLAS/LAPACK routines, with Python acting as a thin orchestration layer over contiguous memory buffers. JS engines (V8) don't have an equivalent "compile once, ship a C extension" ecosystem.

**Pure-JS numeric libraries:** math.js (most comprehensive, not built for heavy numerical workloads), numeric.js/NumJS (dated), stdlib (actively maintained, modern, high-performance math/stats/array utilities for Node and browsers), simple-statistics/jStat/Sylvester (narrow stats/vector libs).

**Where "low-level" equivalents come in:** WebAssembly (closest parallel to Python C extensions — compile C/C++/Rust to WASM), Native Node addons (N-API, Node-only), GPU-backed libraries (WebGL/WebGPU), TensorFlow.js (uses WASM + WebGL/WebGPU backends).

**Node vs Deno:** Node has mature N-API/native bindings, larger numeric ecosystem (numjs, danfojs-node, tensorflow.js). Deno has first-class WASM support and npm compatibility but a thinner native-binding story and only a thin `@std/math` module.

**Practical takeaway:** pure-JS libraries work for moderate workloads; for heavy compute, compile numeric kernels to WASM (Rust/C++) and call from JS — the same trick Python plays with C extensions via a different toolchain.

## Turn 2: Is @std/math written in Rust? How does it compare to NumPy/pandas?

No — `@std/math` is not a Rust numeric backend. It's a small, portable TS/JS utility package (Deno stdlib family) providing scalar helpers like `clamp` and `modulo` — not array computing, tabular data, or BLAS/LAPACK integration. Works across Deno, Node, Bun, browsers, Cloudflare Workers.

**Comparison table:** `@std/math` = individual JS number values / scalar utility functions, no multidimensional typed arrays, no broadcasting, no native numeric kernels. NumPy = homogeneous N-dimensional ndarray, vectorized numerical computing, native C API kernels. pandas = labeled Series/DataFrame, tabular analysis, Cython/C++ optimized paths.

Deno the *runtime* contains Rust (its systems implementation), but importing `@std/math` does not route calls to Rust — the stdlib repo is overwhelmingly TS/JS.

**For a true native-numeric-extension analogue in JS/Deno/Node:** WebAssembly (Rust/C/C++), WebGPU (parallel workloads), Node N-API addons, or a purpose-built tensor library (TensorFlow.js).

## Turn 3: Can NumPy/pandas be recompiled with a JS front end?

Yes — but usually by compiling the *Python runtime and its native extensions* to WebAssembly (Pyodide: CPython compiled via Emscripten + WASM builds of NumPy/pandas/SciPy/Matplotlib/scikit-learn), not by reimplementing them as idiomatic JS libraries.

**Architecture:** TS/JS app ↔ JS↔Python proxy/conversion layer ↔ Pyodide (CPython on WASM) ↔ NumPy/pandas compiled for wasm32-emscripten.

**Where it works best:** interactive scientific/data exploration in-browser, client-side CSV/data transforms, reusing existing NumPy/pandas codebases in a web UI, offline-capable analytical tools/notebooks.

**Tradeoffs:** high Python API compatibility but medium/low JS-native ergonomics (proxies/conversions at the boundary), substantial startup/download cost, constrained threading, must have prebuilt WASM package or port one.

**Three sensible designs if you want a JS-facing API:**
1. Pyodide façade — narrow TS API wrapping one Python function call.
2. Move only hot kernels to Rust/WASM — clean JS API, no CPython/pandas shipped (best for a greenfield, performance-sensitive app).
3. Run normal Python on the server (Node/Deno backend calls a Python service) — best for large data / native BLAS / Parquet / Arrow / multiprocessing.

**Dividing line:** Pyodide is excellent for bringing *existing* NumPy/pandas code to a JS client; Rust/WASM or server-side Python is better for a *greenfield* performance-sensitive JS application.

## Turn 4: Full API design — could this be pandas-compatible / interoperate with the Python ecosystem?

Distinguish **pandas-compatible data interchange** (very achievable) from **pandas API compatibility** (multi-year project, compromises JS-first design — not recommended).

### Proposed architecture
```
Tensor runtime              Table runtime                 Interop
──────────────              ─────────────                 ───────
Rust/WASM CPU kernels       Arrow-backed column buffers   Arrow IPC / Parquet
WebGPU kernels              JS-native Frame API           .npy / .npz
Autograd + optimizer        Streaming scan/query APIs     ONNX / DLPack where viable
TypedArray/WASM memory      Expression + query planner    Python bridge helpers
```

Treat **tensors** (homogeneous, position-oriented) and **tables** (named, heterogeneous, nullable columns) as separate but interoperable domains.

### Core API surface (full TypeScript interfaces given in the source)
- `Tensor` interface: `shape`, `ndim`, `size`, `dtype`, `device`, `requiresGrad`, `grad`; methods `item()`, `toArray()`, `data()`, `reshape()`, `flatten()`, `squeeze()`, `unsqueeze()`, `transpose()`, `permute()`, `broadcastTo()`, `contiguous()`, `at()`, `slice()`, `select()`, `gather()`, `cast()`, `clone()`, `detach()`, `to(device)`, elementwise ops (`add/sub/mul/div/pow/neg/exp/log/sqrt/abs/clip`), reductions (`sum/mean/max/argmax`), `matmul`, `dot`, `softmax`, `relu`, `sigmoid`, `gelu`, `backward()`.
- `DType` union: bool, u8/i8, u16/i16, u32/i32, f16/bf16/f32/f64. `Device` = "wasm" | "webgpu".
- Constructors: `Tensor.scalar/from/fromTypedArray/zeros/ones/full/arange/linspace/eye/empty/concat/stack/where/einsum/loadNpy`.
- `random` namespace: `seed/uniform/normal/randint`.
- Functional ops mirror methods (`ops.add`, `ops.addInto`, `ops.matmulInto`, `ops.softmaxInto`) — the `Into` variants are the performance interface (buffer reuse, no allocation, explicit aliasing).
- Explicit rule: don't use JS `Proxy` for pseudo-Python indexing — bad for typed APIs, stack traces, profiling, WASM-backed storage.
- Autograd/NN: `grad.noGrad/enable/valueAndGrad/checkpoint`; `nn.Parameter`, `nn.Module` (abstract `forward`), `nn.Linear/Embedding/LayerNorm/RMSNorm/Sequential/Dropout`, `nn.mseLoss/crossEntropy`; `optim.AdamW` etc. Full training-loop example given.
- Linear algebra / stats / signal / image namespaces: `linalg.matmul/solve/cholesky/qr/svd/norm`, `stats.mean/variance/quantile`, `fft.fft/rfft/irfft`, `image.resize/normalize`.
- **v1 scope guidance:** prioritize elementwise ops+broadcasting, reductions/masking, matmul+dense linalg, random gen, softmax/normalization/activations/losses. Convolution/pooling only if targeting ML/image. FFT only with a concrete use case. Sparse tensors, complex dtypes, generalized ufuncs, full LAPACK parity = later projects.

### DataFrame layer (`Frame`/`Series`)
- Built on **Apache Arrow** (language-independent columnar format), not ad hoc array-of-objects.
- `Frame` interface: `schema/columns/length`, `select/drop/rename/withColumns`, `filter/sortBy/limit/slice`, `groupBy/join/concat`, `nullCount/fillNull/dropNull`, `toArrow/toRows/toTensor/toCSV/toIPC/toParquet`.
- `Series<T>` interface: `name/dtype/length`, `cast/isNull/fillNull/unique/valueCounts/toTensor`.
- Pandas → proposed-JS translation table provided (e.g. `pd.DataFrame(...)` → `Frame.from(...)`, `df[df.spend>20]` → `df.filter(col("spend").gt(20))`, `pd.merge(...)` → `a.join(b, {...})`).
- Recommend **immutable, expression-oriented Frame by default** — enables lazy query planning, column pruning, predicate pushdown, worker execution, backend substitution (closer to Polars/DuckDB design than pandas' eager model). `.collect()` materializes a lazy scan.

### Interoperability strategy
- **Tables:** Arrow IPC for streams/transport/worker messages/JS↔Python boundaries; Parquet for durable storage. "Feather V2" = Arrow IPC on disk (not a separate format to support).
- Pandas compatibility matrix:
  - Read pandas-written files: **yes** (Arrow IPC/Parquet/CSV/JSON/.npy).
  - Export data for pandas to consume: **yes** (Arrow IPC/Parquet → PyArrow → `to_pandas()`).
  - Import a pandas DataFrame into JS: **yes** (Python exports Arrow IPC/Parquet, JS reads it).
  - Same-process zero-copy Python↔JS/WASM browser transfer: **no** (different process/runtime/memory models).
  - Same-process zero-copy Python↔native Node addon: **potentially** (Arrow C Data Interface / PyCapsule).
  - Implement pandas.DataFrame semantics exactly: **technically possible, strategically poor**.
  - Run existing pandas Python code unchanged in this runtime: **no** — use CPython/Pyodide/server-side Python instead.
- **Do not** target Python's `__dataframe__()` DataFrame Interchange Protocol as the primary bridge — pandas deprecates it, recommends Arrow C Data Interface + PyCapsule, and drops the interchange fallback in pandas 4.0. Arrow C Data Interface is excellent for native same-process boundaries but doesn't cross browser/WASM↔CPython. **Arrow IPC is the practical, portable answer for browser/Node/Deno.**
- **Tensors:** support `.npy` (implement early — stores single array + dtype/shape metadata), `.npz` (after .npy, named tensor bundles), **ONNX** (model interchange/inference via ONNX Runtime Web, not the core tensor API — interoperate rather than rebuild), **DLPack** (native/GPU backend interop only, not a browser file format, no zero-copy across WASM↔CPython).

### Explicit non-goals for v1 (write these down before building)
- No full pandas source/API compatibility.
- No `eval()`/string-expression API.
- No object-dtype tensors.
- No Python pickle compatibility (unsafe, ecosystem-specific).
- No implicit CPU↔GPU or JS↔WASM copying.
- No magic lazy execution — eager is default, compilation is opt-in.
- No in-place autograd mutation until versioning/saved-tensor rules are solid.
- No arbitrary user-defined WASM kernels in the trusted process without a validated ABI, quotas, and differential tests.

### Suggested release sequence (packages)
1. `tensor-core` — storage handles, shapes, dtypes, constructors, broadcasting, elementwise ops, reductions, views, `.npy`.
2. `tensor-wasm` — Rust kernels, allocator, SIMD where measured, batched/fused CPU kernels.
3. `tensor-autograd` — reverse tape, `Parameter`, common gradients, SGD/AdamW.
4. `tensor-compile` — expression IR, elementwise fusion, temporary-memory planning.
5. `frame-arrow` — Arrow-backed Frame/Series/expressions, filter/project/groupby/join, Arrow IPC.
6. `frame-parquet` — scans, writes, projection/predicate pushdown.
7. `tensor-webgpu` — large matmuls, attention-adjacent ops, image/model kernels.
8. `interop-python` — Python helper package moving Arrow IPC/Parquet/.npy/.npz, optional native Arrow C Data Interface bridges.

## Turn 5: More examples — ONNX, TensorFlow, SciPy coverage

Clarified "onyx" = ONNX/ONNX Runtime. Established a boundary:
- NumPy/PyTorch/TensorFlow → map onto the Tensor runtime.
- SciPy → specialized algorithm modules built on tensors.
- pandas → Arrow-backed tabular dataframes.
- ONNX Runtime → imported-model inference, not the main tensor API.

### Proposed package/module map
```
@jh/runtime
├── tensor        # n-D homogeneous numeric arrays
├── ops           # elementwise, shape, indexing, reductions
├── linalg        # dense linear algebra
├── sparse        # sparse tensor/matrix representations
├── fft           # Fourier transforms
├── signal        # filters, convolution, resampling, STFT
├── stats         # distributions, descriptive statistics, sampling
├── optimize      # minimization, root finding, least squares
├── integrate     # quadrature, ODE solvers
├── interpolate   # splines, gridded/unstructured interpolation
├── special       # gamma, erf, Bessel, etc.
├── random        # seeded RNG + distributions
├── grad          # reverse-mode autodiff
├── nn            # modules, layers, losses, parameters
├── optim         # SGD, AdamW, schedules
├── data          # async datasets, batching, transforms
├── frame         # Arrow-backed Frame / Series
├── io            # Arrow IPC, Parquet, NPY, NPZ, CSV
└── onnx          # imported-model inference adapter
```
Recommendation: expose as named namespaces (`import { Tensor, ops, linalg, ... } from "@jh/runtime"`), not hundreds of methods hung off `Tensor`.

### NumPy translations
Full table: construct/convert/range/reshape/transpose/slice/mask/broadcast-math/reduce/conditional/concatenate/save/load, each with NumPy call ↔ proposed JS/TS call. Recommended tensor additions: comparison ops (`eq/ne/lt/lte/gt/gte`), logical ops, `any/all`, `std/variance` with `ddof`, `cumsum/cumprod`, `sort/argsort`, `topK`, `take/gather/scatter/mask`, `Tensor.where`. Keep **view** (`permute`, no copy) vs **contiguous** (packed, may copy) semantically explicit.

### PyTorch translations
Table: grad-enabled tensor, no_grad, backward, zero_grad, layers, Sequential, train/eval mode, optimizer, checkpoint save, device transfer. Full `MLP extends nn.Module` example. Recommend making async device moves (`await x.to("webgpu")`) explicit/visible rather than hiding CPU↔GPU copies behind sync calls (avoids unexplained UI stalls, keeps a browser runtime profileable).

### TensorFlow / TensorFlow.js translations
Table: `tf.tensor2d`→`Tensor.from`, `tf.variable`→`nn.parameter`, `tf.tidy`→`scope()`, `tf.grad`→`grad.of`, `tf.keras.Sequential`→`nn.Sequential`, `model.fit/fitDataset`→`trainer.fit`, `tf.function`→`compile`. Recommend supporting **both** functional gradient style (`grad.valueAndGrad`, good for optimizers/custom loops) and PyTorch-style tape/`.backward()` (friendlier for model authors). Memory-scoping API: prefer automatic backend-neutral scoping/fusion over TF.js's manual `tf.tidy()` ritual; make `scope()`/`keep()` optional infra, not mandatory.

### ONNX Runtime translations
Not a competing authoring API — a model interchange format + runtime. Table: `ort.Tensor`→adapter, `InferenceSession.create`→`onnx.load`, `session.run`→`model.run`, etc. **Three ONNX modes:** (1) use ONNX Runtime directly for an imported pretrained model — **start here**; (2) adapt one of your tensors to ORT tensor storage; (3) import/compile ONNX graphs into your own runtime — only with a focused model class and clear reason, since it risks reimplementing a huge operator compatibility matrix.

### SciPy translations (needs the most expansion — build as opt-in namespaces)
- **Dense linear algebra:** `linalg.solve/lu/qr/svd/eigh/matrixExp/norm`, prefer named result objects over Python tuple unpacking.
- **Sparse arrays/solvers:** dedicated CSR representation first (`sparse.csr({values, columnIndices, rowPointers, shape})`), then COO for construction/interchange, CSC only if justified. `sparse.linalg.cg` (conjugate gradient w/ tolerance/maxIterations/preconditioner), `sparse.linalg.solve`.
- **Optimization/root-finding:** `optimize.minimize({objective, initial, method, gradient: "autodiff", tolerance})` — autodiff-by-default gradient is an advantage over base SciPy; allow explicit analytical-gradient callbacks. `optimize.root`, `optimize.leastSquares`, `optimize.brent`, `optimize.assignment`.
- **Integration/ODEs:** `integrate.quad`, `integrate.solveIVP({tSpan, initial, derivative, method: "rk45", tEval})`. Deliberately omit legacy `odeint` — one modern `solveIVP` API.
- **Interpolation:** `interpolate.linear/cubicSpline/regularGrid/scattered`.
- **FFT/signal:** `fft.fft/rfft`, `signal.convolve/butter/sosFilter/stft/istft/findPeaks/resamplePoly`. Give explicit channel/batch-axis options (relevant to video work) rather than assuming 1-D audio.
- **Special functions/distributions:** `special.erf/gamma/logGamma/besselJ/softmax`, `stats.normal.pdf/cdf/sample`, `stats.describe/pearsonR/ttestIndependent`. Keep `special`/`stats.distributions` as optional packages (large algorithm/test surface).

### pandas/Arrow examples
Side-by-side pandas method-chain (`query`/`assign`/`groupby`/`agg`/`sort_values`) vs proposed `Frame` expression API (`filter(col(...).and(...))`/`withColumns`/`groupBy`/`aggregate`/`sortBy`). Round-trip example: JS writes Arrow IPC/Parquet → Python (`pyarrow`) reads it → `.to_pandas()`.

### What to build first (staged)
- **Stage 1:** credible tensor core — `tensor, ops, random, linalg.matmul/solve, autograd, nn.Linear/Embedding/LayerNorm, optim.AdamW, io.npy/npz, onnx adapter`.
- **Stage 2:** practical ML/media compute — `compile/fusion, webgpu, fft, signal, convolution, attention primitives, data loaders, checkpoint format`.
- **Stage 3:** scientific compute — `linalg QR/SVD/eigen, sparse CSR/COO, optimize, integrate, interpolate, special functions, statistics`.
- **Stage 4:** dataframe system — `Arrow Frame/Series, Arrow IPC, Parquet scan/write, expressions, group-by, joins, window operations`.
- Rationale: avoid becoming "a partial NumPy + partial PyTorch + partial SciPy + partial pandas with none of the foundations finished." First wedge = typed-array/WASM tensor runtime with eager autodiff, ONNX inference interop, and unusually good signal/video-oriented kernels.

## Turn 6: Should we study other JS libraries and accelerate versions of their functions?

Yes — as a **semantic/API audit**, not "port every JS math package to Rust/WASM." Use existing libraries as sources of JS ergonomics, edge-case behavior, test cases, module boundaries; selectively accelerate their computational primitives while improving storage/execution model underneath.

**Strongest candidates:** stdlib, math.js, TensorFlow.js, Danfo.js, ml-matrix, GPU.js. (numeric.js = historical API reference only.)

### What to borrow / not copy (per library)
| Library | Adopt | Avoid |
|---|---|---|
| **stdlib** | Granular modules, explicit dtypes/strides/offsets, low-level `Into` kernels, stats coverage | Fragmented function-per-package import ergonomics as primary user API |
| **math.js** | Scalar `Complex`, `Fraction`, `Unit` objects + friendly construction API | Heterogeneous dynamic matrix representation for high-throughput tensors |
| **TensorFlow.js** | Functional gradient APIs, training/data APIs, Layers-compatible ideas | Implicit tensor lifetime management, large global-style surface |
| **Danfo.js** | Familiar table operations, tensor/table conversions | TensorFlow-backed table arithmetic — use Arrow storage/query execution instead |
| **ml-matrix** | Discoverable Matrix/decomposition names, small-data convenience API | Nested JS arrays / object-heavy matrix storage in hot paths |
| **GPU.js** | "Write kernel-like JS, run on GPU if possible" mental model, opt-in kernel/compile UX | WebGL shader backend, unrestricted JS-to-kernel transpilation |
| **numeric.js** | Familiar function names | Old dense-array model, no modern typed/backend abstractions |

### The acceleration test
**Accelerate** a function when it's mostly: homogeneous numeric data (f32/f64/int/complex), bulk data-parallel work over typed buffers, a standard dense/sparse/FFT/statistical kernel, repeated enough that allocation/loop overhead/SIMD/threading/GPU matters, expressible without arbitrary per-element JS callbacks.

**Keep in TypeScript** when primarily: metadata transformation (shape validation, axis normalization, schema mgmt), expression-plan construction, unit conversion/parsing/formatting/errors/API adaptation, arbitrary-precision arithmetic, user callbacks/comparators/dynamic property access/object data.

### stdlib — the closest API to mine deeply
Has BLAS-like routines, ndarray strides/offsets. Proposed pattern: friendly public API (`a.matmul(b)`, `ops.gemmInto(output, {a,b,alpha,beta,transposeA,transposeB})`) over a low-level Rust/WASM BLAS-like internal ABI (`kernels.gemmF32({outOffset, aOffset, bOffset, m, n, k, strides, alpha, beta})`) — gives stdlib's efficiency without forcing users to reason about leading dimensions/offsets directly. Also worth matching: `stats.mean(x, {ignoreNaN:true})`, `stats.variance/quantile/kde2d/ttestIndependent`, `special.erf/logGamma/logSumExp/sigmoid`, `random.normal/poisson`.

### math.js — split scalar math from tensor math
Keep `Complex`/`Fraction`/`Unit` as **separate value domains** (`@jh/runtime/scalar`), not generic `Tensor<T>`. Table of value kinds → representation → "accelerated?" (JS number/bigint: no; Complex scalar: usually no; Fraction/BigNumber: generally no; Unit: conversion mostly TS; `Tensor<"f32">`: yes; `ComplexTensor`: yes; `Frame`: yes for scans/aggregates). **Do not** permit `Tensor<BigNumber>` or generic `Tensor<T>` — wrecks the memory model, prevents fixed-width SIMD/GPU kernels. A later `DecimalTensor` could exist but as its own fixed-width storage/operator family, not boxed `BigNumber` in a `Float32Array`-style tensor.

**Expression parsing** (math.js's parser/symbolic derivatives): useful for a REPL/notebook/calculator/graphing view/LLM-facing constrained compute DSL, but **not part of core tensor execution** — compile only into a validated tensor-expression IR (not arbitrary JS), then lower to fused WASM/WebGPU kernels. Notably maps well to "constrained/generated WASM" interest — the expression language is a safe source language you fully control the IR/WASM emitter for.

### TensorFlow.js — keep the useful JS conventions
Translation table (`tf.tensor2d`→`Tensor.from`, `tf.variable`→`nn.parameter`, `tf.tidy`→`scope`, `tf.grad`→`grad.of`, `tf.train.adam`→`optim.Adam`, `tf.data.generator`→`data.fromAsync`, `model.fitDataset`→`trainer.fit`). Improve on `tf.tidy`: offer 3 levels — (1) normal ergonomic code where the backend plans/fuses temp buffers automatically, (2) explicit `scope(() => { ...; return keep(hidden); })` owner scope for unusual allocation-heavy work, (3) explicit no-allocation production kernel via `...Into` ops. A real backend allocator + operation fusion should mean users only reach for `scope()` in low-level code or long-running loops.

### Danfo.js — borrow the surface, replace the foundation
Danfo: `DataFrame` ── TensorFlow.js tensor. Proposed: `Frame` ── Arrow columns ── explicit `Tensor` conversion (`frame.select("spend").toTensor({dtype:"f32", nulls:"error"}).reshape([-1,1])`). Worth matching from the general DataFrame ecosystem: `Frame.from/readCSV/readParquet/readIPC`, `select/withColumns/filter/groupBy().aggregate()/join/sortBy/pivot/melt/describe/toArrow/toTensor`. Avoid pandas/Danfo's overloaded indexing (`.loc`/`.iloc`) and in-place mutation; build immutable Arrow-backed Frame + lazy `scanParquet()` pipeline for column pruning/predicate pushdown/future worker-WASM query execution.

### ml-matrix / Numeric.js — compatibility adapters, not foundations
Support as migration helpers (`Tensor.fromMatrix`/`Tensor.toMatrix` for ml-matrix `Matrix`; `Tensor.from2D` for nested-array input). Don't copy their backing representation — nested JS arrays fine at input/output edges, unsuitable as core layout for SIMD/WASM/GPU execution.

### GPU.js — keep the idea, not the implementation
GPU.js compiles a constrained JS subset to shaders with JS fallback. Proposed modern WebGPU equivalent: deliberately narrower and typed — a `kernel({inputs, output, expression})` DSL with `TensorSpec.f32([...])` shapes, or explicit vetted WGSL for advanced users via `gpuKernel.define({name, inputs, output, wgsl})`. **Do not** transpile arbitrary JS callbacks into GPU code — use a typed, side-effect-free expression DSL or accept explicit WGSL, for tractable shape inference/lowering/fusion/safety/deterministic tests/error reporting.

### Recommended compatibility/adapter packages (core has zero dependency on these)
```
@jh/runtime-stdlib       # Optional adapters for typed arrays / selected functions
@jh/runtime-mathjs       # Scalar Complex, Unit, Fraction conversion bridge
@jh/runtime-tfjs         # Tensor import/export and model/data migration helpers
@jh/runtime-danfo        # Frame import/export, if the data layout allows it
@jh/runtime-ml-matrix    # Matrix <-> Tensor
@jh/runtime-onnx         # ONNX Runtime and imported-model bridge
@jh/runtime-arrow        # Arrow JS Table / RecordBatch <-> Frame
```

### Prioritized feature harvest
- **Build now:** stdlib-style strided/offset-aware CPU kernels behind friendly `Tensor` methods; math.js-inspired `Complex`/`Unit` scalar packages (separate from dense tensors); TensorFlow.js-style `grad.of`/`valueAndGrad`/async datasets/layers/training loops; Danfo-style DataFrame verbs implemented as Arrow `Frame` expressions; ml-matrix/Numeric-style dense decompositions (LU, QR, SVD, solve, inverse); GPU.js-inspired opt-in `compile()`/kernel workflow, lowered to WASM first, WebGPU second.
- **Build later:** arbitrary-precision complex/decimal/rational/unit-aware tensor storage; a symbolic math engine; general JS-callback-to-WASM/WGSL compilation; sparse matrices + iterative solvers; full SciPy-style optimize/ODE/interpolate/special-functions suites; full DataFrame compatibility adapters for every Danfo/pandas behavior.
- **Explicitly decline:** generic boxed-object tensors; eval-based expression execution; nested-array storage in kernels; a global mutable default backend; implicit device transfer; API-level dependence on TensorFlow.js/Danfo.js/math.js/stdlib.

**Bottom line from the source:** "The resulting project would not be 'accelerated math.js' or 'Danfo rewritten in Rust.' It would be a modern JS computation substrate that adopts the best parts of those APIs while making fixed-width buffers, explicit devices, WASM kernels, Arrow tables, and opt-in compilation the real foundation."

### Suggested (unexplored) follow-ups noted at end of source conversation
- Which ml-matrix decompositions should be prioritized first
- How to structure Arrow-backed DataFrames with a Danfo.js-like API
- Implementing Complex and Fraction types in Rust for WASM
