# Spike: WASM SIMD128 for the contiguous elementwise fast path (2026-08-12)

**Status: measured, shipped.** Per issue #13's own instruction ("do not merge SIMD without an
attached benchmark showing a real speedup — if unproven, ship scalar-only and say so"), this
document is that benchmark. `docs/spikes/wasm-baseline.md` measured the *scalar* `...Into` path's
1.78x win over pure JS at N=1e6 and explicitly deferred SIMD ("a 1.78x scalar win already exists
untapped behind the copy overhead — measure SIMD only after the Into path lands," issue #3). #3
shipped; this is that follow-up measurement.

## Method

`crates/tensor-wasm-kernels/src/lib.rs` gained a `simd` Cargo feature gating a new `simd` module:
`add_f32_contiguous_simd128`/`mul_f32_contiguous_simd128`, hand-written `std::arch::wasm32`
SIMD128 (`f32x4_add`/`f32x4_mul`, 4 lanes/instruction, scalar tail loop for `len % 4 != 0`).
Deliberately **contiguous-only, no offset/stride params** — SIMD loads need contiguous memory,
and there's no benefit to a strided variant; the existing general `add_f32_strided`/
`mul_f32_strided` kernels stay as the fallback for non-contiguous views.

Benchmarked three variants at N=1,000,000 (Node, `process.hrtime.bigint()`, 3-iteration warmup,
50-iteration mean, `/tmp/simd_bench.mjs` — not committed, throwaway measurement script):

1. **strided-scalar** — the kernel that ships today (`add_f32_strided`, called with `stride=1`).
2. **contiguous-scalar** — a new scalar-only kernel taking plain base pointers (no offset/stride
   params at all), isolating how much of any speedup is just from removing the runtime
   offset/stride multiply the compiler can't prove is trivial (a `stride=1` call to
   `add_f32_strided` still can't auto-vectorize, since `stride` is a runtime `isize` parameter).
3. **SIMD128** — `add_f32_contiguous_simd128`.

## Results (representative run; stable across repeats, see below)

| Variant | Time (ms/call) | vs. strided-scalar | vs. contiguous-scalar |
|---|---|---|---|
| strided-scalar (shipped today) | 1.29 ms | baseline | — |
| contiguous-scalar (no SIMD) | 0.95 ms | 1.36x | baseline |
| SIMD128 | 0.33 ms | **3.88x** | **2.84x** |

Repeated (3 runs): contiguous-scalar-vs-strided-scalar ranged **1.21x–1.42x**; SIMD-vs-contiguous-
scalar (SIMD's *own* marginal contribution, apples-to-apples) ranged **2.63x–3.03x**; total
SIMD-vs-strided-scalar (what actually ships today) ranged **3.17x–4.27x**. Correctness verified
every run: 0 mismatches between the SIMD output and the scalar `add_f32_strided` output over all
1,000,000 elements.

**Attribution matters here**: roughly 1.2–1.4x of the total win is just from having a dedicated
contiguous fast path at all (no runtime stride multiply) — SIMD's *own* contribution on top of
that is a separate, real 2.6–3x. Both numbers are reported so a future reader can tell how much
of the observed total speedup to credit to "SIMD" specifically vs. "a contiguous fast path,
which could in principle have been scalar."

## Decision: ship it

2.6–3x is well above any reasonable bar for "a real speedup" — this isn't a wash or a marginal
win. Shipped as a **separate `.wasm` build artifact** (`packages/tensor-wasm/wasm/
tensor_wasm_kernels_simd128.wasm`, `npm run build:wasm:simd`), never merged into the default
scalar build:

- **Why a separate artifact, not a single module with both paths**: a WASM module containing ANY
  v128 instruction fails `WebAssembly` validation **in its entirety** on a runtime without SIMD
  support — module loading is all-or-nothing (unlike native code's per-call feature detection),
  so there is no way to ship one `.wasm` file that both uses SIMD and is guaranteed loadable
  everywhere.
- **Feature detection**: `Kernels.load()` reads the SIMD module's bytes and calls
  `WebAssembly.validate()` on them directly (not a separate hand-crafted minimal probe module —
  simpler, and it's checking the exact bytes about to be instantiated) before attempting
  `WebAssembly.instantiate()`. Any failure at any step (unsupported runtime, the artifact wasn't
  built, a bad import) is caught and leaves the SIMD path unavailable — `addInto`/`mulInto` fall
  back to the always-present scalar/strided kernels transparently, never throwing.
- **Shared memory, not a second buffer**: the SIMD build is compiled with
  `RUSTFLAGS="-C link-args=--import-memory"`, so it *imports* `env.memory` instead of allocating
  its own. `Kernels.load()` instantiates it passing the scalar module's own `memory` export as
  that import — both modules genuinely share one linear memory / one `ArrayBuffer`, so the SIMD
  kernels operate on the *exact same* resident `WasmTensor` data with zero copying. Without this,
  the SIMD module would have its own separate memory and `WasmTensor` data (allocated via the
  scalar module's `alloc`) would be invisible to it — defeating the entire point.
- **Eligibility**: `addInto`/`mulInto` use the SIMD path only when it's available AND every
  operand (`a`, `b`, `out`) is contiguous (`stride === 1`, per `flatSpec`'s check) — a
  non-contiguous view (e.g. a strided slice) always falls back to the general strided kernel,
  which is unaffected and unchanged.

## Known cost (risk #7 in docs/PLAN.md, paid deliberately here)

Two implementations of each accelerated kernel (add, mul) that must agree bit-for-bit — verified
by a dedicated test (`addInto/mulInto: SIMD-accelerated result is bit-for-bit identical to the
scalar fallback`) comparing SIMD output against the scalar `add_f32_strided`/`mul_f32_strided`
kernels element-by-element, including a length not a multiple of 4 to exercise the SIMD kernel's
scalar tail loop. `gemm_f32` (matmul) does **not** get a SIMD variant in this pass — it's
compute-bound rather than memory-bandwidth-bound like elementwise add/mul, has a fundamentally
different (blocked/tiled) vectorization shape, and is explicitly out of scope for issue #13
(which named "the Into path," i.e. the elementwise kernels this measured).

## Reproduction

Scratch benchmark script was `/tmp/simd_bench.mjs` (not committed — throwaway, superseded by the
package's own committed tests, which cover the same ground with proper CI-safe thresholds). To
re-measure by hand: build both artifacts (`npm run build:wasm -w mallory-tensor-wasm`), then
compare `add_f32_strided`/`add_f32_contiguous_simd128` directly via `WebAssembly.instantiate()`
on the two `.wasm` files in `packages/tensor-wasm/wasm/`, matching this doc's three-variant
methodology.
