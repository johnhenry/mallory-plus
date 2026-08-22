# Deno FFI native-kernel baseline (issue #55, Phase 1)

**Date**: 2026-08-13 · **Machine**: trycooy (x86_64 NixOS, the same machine
as `wasm-baseline.md` / `wasm-simd.md`) · **Deno**: 2.6.10 · **Rust**:
workspace toolchain, `--release`.

The question this measurement answers (the issue's Phase-1 gate): is a
native Deno-FFI path over the *same Rust source* enough faster than the
WASM path to justify Phase 2 (per-platform binary distribution, panic-
boundary hardening, conditional exports)?

## Method

`packages/tensor-wasm/scripts/deno-ffi-bench.ts`, run as:

```bash
cargo build --release -p tensor-wasm-kernels   # cdylib — ZERO crate changes needed
deno run --allow-ffi --allow-read packages/tensor-wasm/scripts/deno-ffi-bench.ts
```

- Both sides call the **identical Rust functions** — the crate's
  `crate-type` already included `cdylib` (the flat extern-C ABI decision
  paying out exactly as designed), so `cargo build --release` produced the
  `.so` with no source or manifest changes at all.
- WASM timings are kernel-call-only on **resident** buffers (its best
  case); the copy-in cost the FFI path structurally avoids is measured
  separately, not folded in.
- Correctness cross-checked first: native and WASM results agree within
  f32 tolerance on every op/size before anything is timed.
- Deterministic inputs (seeded mulberry32); diagonally-dominant matrices
  for `solve`.

## Results

| op | size | native ms | wasm ms | native speedup |
|---|---|---:|---:|---:|
| addInto | N=10,000 | 0.0029 | 0.0134 | **4.65x** |
| addInto | N=1,000,000 | 0.8567 | 1.6074 | **1.88x** |
| addInto | N=4,000,000 | 3.5941 | 6.2791 | **1.75x** |
| gemm | 64×64 | 0.2083 | 0.2833 | 1.36x |
| gemm | 256×256 | 20.14 | 26.60 | 1.32x |
| gemm | 1024×1024 | 5130.6 | 6072.2 | 1.18x |
| solve | n=64 | 0.0627 | 0.1860 | **2.97x** |
| solve | n=256 | 2.0372 | 10.7320 | **5.27x** |
| copy-in (wasm residency tax) | N=1,000,000 | — | 0.4960 | — |

## Reading the numbers

- **`solve` is the headline: 3–5.3x.** Compute-bound with branchy pivoting
  and internal `Vec` allocations — native codegen and a real allocator beat
  the WASM equivalents decisively, and the gap *grows* with n.
- **Elementwise ~1.8x kernel-only at large N** (memory-bandwidth-bound;
  native autovectorization vs scalar WASM), and **4.65x at small N** where
  Deno's FFI fast path beats the WASM call overhead. End-to-end the gap is
  larger than kernel-only: the FFI path operates on host `Float32Array`s
  directly, so it never pays the ~0.5 ms/10⁶-element copy-in (×3 buffers
  for a binary op on non-resident data — for one-shot use, wasm's real
  cost at N=10⁶ is ~3.1 ms vs native 0.86 ms, ≈3.6x).
- **`gemm` is the weakest case (1.2–1.4x)** — both sides are the same naive
  triple loop and at 1024² both are cache-miss-bound. A native gemm only
  gets interesting with blocking/BLAS, which is out of scope for this
  crate's "reference kernels" role.

## Verdict: gate PASSES for Phase 2 — with a scope note

The crossover is real and material for exactly the ops where native was
predicted to win (compute-bound `solve`, call-overhead-dominated small ops,
copy-free residency). Phase 2 (error-code panic conversion, CI build
matrix, `deno` conditional exports, graceful dlopen→WASM fallback) is
justified **for the Deno distribution channel** — tracked on issue #55.

Scope note for expectations: this does NOT make WASM second-class. WASM
remains the only zero-install, platform-independent, browser-capable
artifact and the default everywhere; the native path is an opt-in
accelerator for Deno (and likely Bun, same dylib) where `--allow-ffi` is
acceptable.

One Phase-2 cost got cheaper during Phase 1: no crate/manifest changes
were needed at all for the native build, so "distribution matrix + JS
binding + panic hardening" is the entire remaining work.

## Phase 2 (shipped, same issue)

- **Panic boundary**: `alloc`'s invalid-layout `.expect()` is now a defined
  null return on both build targets (JS surfaces it as "allocation
  failed", never poisoning); `dealloc` on an invalid layout is a defined
  no-op. Remaining panics (genuine bugs) abort the process natively —
  guaranteed-defined on Rust ≥1.81, documented in the crate — vs trapping
  + poisoning (#46) in WASM. The trap-poisoning tests moved to an
  out-of-bounds-load trigger accordingly.
- **Binding**: `packages/tensor-wasm/src/native.ts` — `NativeKernels` over
  host `Float32Array`s (addInto/mulInto/matmulInto/solveInto, strided
  `MatrixRef` operands), `load()` returning `undefined` (never throwing)
  outside Deno / without `--allow-ffi` / with no binary; resolution:
  explicit path → `$MATH_PLUS_NATIVE_KERNELS_PATH` → bundled
  `native/<os>-<arch>/` → repo `target/release/`.
- **Conditional exports**: `"deno"` → `native-entry` (default entry's
  surface + the native API; WASM stays the default path even on Deno —
  native is opt-in); `./native` subpath for explicit access.
- **CI**: `.github/workflows/native-kernels.yml` — 5-platform cdylib
  matrix (linux/darwin × x86_64/aarch64 + windows), artifact names
  matching the binding's lookup, linux-x86_64 job runs the Deno
  verification script. npm platform-package publishing deliberately
  unwired pending NPM_TOKEN (#28).
- **Verification**: `scripts/deno-native-test.ts` — native vs WASM
  agreement on every op (incl. transposed-stride matmul), fallback
  contract, RangeError validation. Passing under Deno 2.6.10 locally.
