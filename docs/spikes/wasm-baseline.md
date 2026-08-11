# WASM pipeline baseline (2026-08-11)

Status check on the Rust→WASM seam after the Task 3 scaffold. **The toolchain is proven
end-to-end; the current kernel *wrapper* is a hello-world that violates non-goal 5 and is a
net performance regression.** Numbers below are the baseline M2 must beat.

## What's working

| Element | State |
|---|---|
| Toolchain | plain `cargo build --release --target wasm32-unknown-unknown` + `scripts/copy-wasm.mjs`. **No wasm-pack / wasm-bindgen** — unavailable on NixOS, and the flat extern-C ABI needs no JS glue |
| Linker | `lld` (via `nix-install` locally, `apt-get install lld` in CI) |
| Artifact | 19.4 kB, **0 imports** — self-contained, instantiable with an empty import object |
| Exports | `memory`, `alloc`, `dealloc`, `add_f32`, `__data_end`, `__heap_base` |
| Packaging | `npm pack` includes `wasm/` despite the root `.gitignore` (npm-packlist only reads ignore files inside the package dir) — verified |
| CI | green on ubuntu-latest, Node 22.x/24.x, first try |
| Tests | 2 JS seam tests + 1 Rust unit test |

## Measured: the wrapper, not the kernel, is the problem

`f32` elementwise add, N = 1,000,000, Node v26.5.0:

| Path | Time | vs pure JS |
|---|---|---|
| Pure JS loop over `Float32Array` | 1.539 ms | baseline |
| **WASM kernel with resident buffers** (`add_f32(ap,bp,op,N)`, no copies) | **0.864 ms** | **1.78× faster** |
| WASM via the current `Kernels.addF32()` wrapper | 4.046 ms | 2.27× *slower* |

The kernel itself is already a solid win — and that's a scalar loop with **no SIMD yet**. The
regression comes entirely from `addF32()` doing 3 `alloc`s + 2 copies in + 1 `slice()` out per
call (~3.2 ms of pure overhead). Scaling confirms it: the wrapper is 1.63× slower at N=1e3,
1.80× at 1e5, 2.27× at 1e6 — the penalty grows with data size, which is exactly backwards.

This is a live demonstration of the plan's **non-goal 5 ("no implicit CPU↔GPU or JS↔WASM
copying")** and the reason for the `...Into` performance interface: tensors must *live* in
linear memory and kernels must write in place, rather than marshalling per call.

## Consequences for M2 (`tensor-wasm`)

1. **Tensor storage must move into WASM linear memory.** `tensor-core` is currently pure
   JS/TypedArray and does not import `tensor-wasm` at all — the two packages are still
   disconnected. Wiring them is the M2 gate, and the storage handle is the design decision that
   makes or breaks the numbers above.
2. **Ship `...Into` variants first, not last.** `addInto/mulInto/matmulInto/softmaxInto` over
   resident buffers are what deliver the 1.78×; the ergonomic allocating form should be built
   *on top* of them (allocate-then-call), never the reverse.
3. **Strides/offsets belong in the ABI now.** `add_f32` is contiguous-only. Adding stride/offset
   params before more kernels exist avoids retrofitting every signature (and non-contiguous
   views are already first-class in `tensor-core`).
4. **SIMD is not the priority.** A 1.78× scalar win already exists untapped behind the copy
   overhead. Per the plan's "SIMD where measured", removing copies is the larger, cheaper win —
   measure SIMD only after the `Into` path lands.

## Latent risk: allocator alignment

`alloc()` is `Vec::<u8>::with_capacity` — **align 1 by contract**, though the current dlmalloc
returns 16-byte-aligned pointers in practice (probed across sizes 1–16; no misalignment
observed). JS `new Float32Array(buffer, ptr, n)` *throws* on a non-4-aligned `ptr`, so this is
correct today only by the allocator's grace. When the bump/arena allocator from the plan
replaces dlmalloc, either guarantee alignment in the ABI or take an explicit `align` parameter.

Not a live bug — a contract to pin down before the allocator changes.
