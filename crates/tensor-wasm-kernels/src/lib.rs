//! Flat-numeric kernel ABI for @johnhenry/math-plus-tensor-wasm.
//!
//! Exports use plain `extern "C"` with pointer/offset/stride params (no
//! wasm-bindgen object marshalling) — callers on the JS side own buffer
//! layout and pass offsets into linear memory. Offsets/strides are in
//! *elements*, not bytes (matches @johnhenry/math-plus-tensor-core's `Tensor.strides`
//! convention on the JS side).
//!
//! Alignment contract (issue #7): `alloc` takes an explicit `align` and
//! guarantees the returned pointer satisfies it, via `std::alloc` +
//! `Layout` rather than relying on the allocator's incidental behavior.

use std::alloc::{alloc as std_alloc, dealloc as std_dealloc, Layout};

/// Allocate `len` bytes aligned to `align` (must be a power of two; 4 or 8
/// for f32/f64 buffers). Ownership passes to the caller; free with
/// `dealloc(ptr, len, align)` using the SAME align used here.
///
/// Returns null for an invalid layout (non-power-of-two `align`, or a
/// rounded size overflowing `isize`) as well as for allocator exhaustion —
/// a DEFINED failure signal on both build targets, rather than a panic.
/// (Issue #55 Phase 2: this crate now also ships as a native cdylib called
/// over FFI. A panic escaping an `extern "C"` fn is a guaranteed process
/// abort on Rust >= 1.81 — never UB — but a null return is catchable by
/// the JS caller on both the WASM and native paths, so the one input-
/// validation panic this crate had is now an error value instead. Panics
/// from genuine bugs still abort natively / trap in WASM, where the JS
/// wrapper's trap poisoning (issue #46) takes over.)
#[no_mangle]
pub extern "C" fn alloc(len: usize, align: usize) -> *mut u8 {
    match Layout::from_size_align(len.max(1), align) {
        // SAFETY: layout has non-zero size (len.max(1)) and a validated alignment.
        Ok(layout) => unsafe { std_alloc(layout) },
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a buffer previously returned by `alloc(len, align)`.
///
/// An invalid layout is a no-op rather than a panic (same reasoning as
/// `alloc`'s null return): such a pointer cannot have come from `alloc`,
/// which never returns memory for an invalid layout, so there is nothing
/// this function could correctly free — leaking is the defined, safe
/// outcome on both build targets.
///
/// # Safety
/// `ptr` must come from `alloc(len, align)` with the SAME `len`/`align`, and
/// not have been freed already.
#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: *mut u8, len: usize, align: usize) {
    if let Ok(layout) = Layout::from_size_align(len.max(1), align) {
        std_dealloc(ptr, layout);
    }
}

/// out[outOffset + i*outStride] = a[aOffset + i*aStride] + b[bOffset + i*bStride]
/// for i in 0..len. Strided so the caller never needs to pack a non-contiguous
/// view into a temporary buffer first (issue #6).
///
/// # Safety
/// All three (pointer, offset, stride) triples must describe `len` valid,
/// non-overlapping-with-`out` f32 slots reachable from that pointer.
#[no_mangle]
pub unsafe extern "C" fn add_f32_strided(
    a_ptr: *const f32,
    a_offset: isize,
    a_stride: isize,
    b_ptr: *const f32,
    b_offset: isize,
    b_stride: isize,
    out_ptr: *mut f32,
    out_offset: isize,
    out_stride: isize,
    len: usize,
) {
    for i in 0..len as isize {
        let av = *a_ptr.offset(a_offset + i * a_stride);
        let bv = *b_ptr.offset(b_offset + i * b_stride);
        *out_ptr.offset(out_offset + i * out_stride) = av + bv;
    }
}

/// Strided elementwise multiply — same shape of contract as `add_f32_strided`.
///
/// # Safety
/// Same requirements as `add_f32_strided`.
#[no_mangle]
pub unsafe extern "C" fn mul_f32_strided(
    a_ptr: *const f32,
    a_offset: isize,
    a_stride: isize,
    b_ptr: *const f32,
    b_offset: isize,
    b_stride: isize,
    out_ptr: *mut f32,
    out_offset: isize,
    out_stride: isize,
    len: usize,
) {
    for i in 0..len as isize {
        let av = *a_ptr.offset(a_offset + i * a_stride);
        let bv = *b_ptr.offset(b_offset + i * b_stride);
        *out_ptr.offset(out_offset + i * out_stride) = av * bv;
    }
}

/// Strided elementwise subtract (`a - b`) — same shape of contract as
/// `add_f32_strided`. Added alongside `div_f32_strided` (issue #66) for
/// kernel parity: `add`/`mul` had WASM kernels, `sub`/`div` didn't, for no
/// principled reason.
///
/// # Safety
/// Same requirements as `add_f32_strided`.
#[no_mangle]
pub unsafe extern "C" fn sub_f32_strided(
    a_ptr: *const f32,
    a_offset: isize,
    a_stride: isize,
    b_ptr: *const f32,
    b_offset: isize,
    b_stride: isize,
    out_ptr: *mut f32,
    out_offset: isize,
    out_stride: isize,
    len: usize,
) {
    for i in 0..len as isize {
        let av = *a_ptr.offset(a_offset + i * a_stride);
        let bv = *b_ptr.offset(b_offset + i * b_stride);
        *out_ptr.offset(out_offset + i * out_stride) = av - bv;
    }
}

/// Strided elementwise divide (`a / b`) — same shape of contract as
/// `add_f32_strided`. No special-casing for division by zero: f32 division
/// follows IEEE 754 (±Infinity / NaN), matching JS `Number` division and
/// this repo's existing `Tensor.div` — never a panic/trap, so never
/// interacts with issue #46's poisoning.
///
/// # Safety
/// Same requirements as `add_f32_strided`.
#[no_mangle]
pub unsafe extern "C" fn div_f32_strided(
    a_ptr: *const f32,
    a_offset: isize,
    a_stride: isize,
    b_ptr: *const f32,
    b_offset: isize,
    b_stride: isize,
    out_ptr: *mut f32,
    out_offset: isize,
    out_stride: isize,
    len: usize,
) {
    for i in 0..len as isize {
        let av = *a_ptr.offset(a_offset + i * a_stride);
        let bv = *b_ptr.offset(b_offset + i * b_stride);
        *out_ptr.offset(out_offset + i * out_stride) = av / bv;
    }
}

/// GEMM: `out = alpha * A@B + beta * out`, A is (m x k), B is (k x n), out is
/// (m x n). Row/col strides let the caller pass a transposed or otherwise
/// non-contiguous operand without copying it first — this is the exact ABI
/// sketched in docs/PLAN.md §6.1 (`kernels.gemmF32({...})`).
///
/// # Safety
/// The three (pointer, offset, row_stride, col_stride) groups must describe
/// valid, in-bounds f32 storage for an (m x k), (k x n), and (m x n) matrix
/// respectively (offsets/strides in elements, not bytes).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gemm_f32(
    a_ptr: *const f32,
    a_offset: isize,
    a_row_stride: isize,
    a_col_stride: isize,
    b_ptr: *const f32,
    b_offset: isize,
    b_row_stride: isize,
    b_col_stride: isize,
    out_ptr: *mut f32,
    out_offset: isize,
    out_row_stride: isize,
    out_col_stride: isize,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    for i in 0..m as isize {
        for j in 0..n as isize {
            let mut acc = 0.0f32;
            for p in 0..k as isize {
                let av = *a_ptr.offset(a_offset + i * a_row_stride + p * a_col_stride);
                let bv = *b_ptr.offset(b_offset + p * b_row_stride + j * b_col_stride);
                acc += av * bv;
            }
            let out_idx = out_offset + i * out_row_stride + j * out_col_stride;
            let prev = if beta != 0.0 { *out_ptr.offset(out_idx) } else { 0.0 };
            *out_ptr.offset(out_idx) = alpha * acc + beta * prev;
        }
    }
}

/// Solve `A·x = b` for a square `n x n` system via LU decomposition with
/// partial pivoting (issue #39, the first native-kernel candidate named in
/// docs/PLAN.md §9 item 1) — same algorithm as @johnhenry/math's
/// `MatrixMath.lu`/`solve` (largest-absolute-value-in-column pivot
/// selection, in-place Doolittle elimination storing L's multipliers where
/// U's zeros go, forward-substitute `Ly=Pb` then back-substitute `Ux=y`),
/// so results agree with `adapter-math`'s existing reference-speed `solve`
/// (which delegates to that same @johnhenry/math algorithm) up to f32-vs-f64
/// precision — see `adapter-math/src/linalg.ts`'s own doc comment: this
/// native kernel is meant to sit ALONGSIDE that reference path as a faster
/// option, not replace it (the reference path stays the correctness
/// oracle).
///
/// `A` is read via row/col strides (never copied by the caller — the
/// kernel copies it into an owned scratch buffer internally, since partial
/// pivoting needs row swaps that are far simpler on a packed buffer than
/// via arbitrary strides). A near-zero pivot (a singular/near-singular
/// column) writes `0.0` into that row of `x` rather than dividing by
/// (near-)zero — matching `MatrixMath.solve`'s own documented behavior,
/// for output-compatibility with the existing reference oracle rather than
/// inventing new error-handling semantics here.
///
/// # Safety
/// `a_ptr` must describe a valid `n x n` f32 matrix reachable via the given
/// offset/strides; `b_ptr` must describe `n` valid f32 values reachable via
/// `b_offset`/`b_stride`; `out_ptr` must describe `n` valid, writable f32
/// slots reachable via `out_offset`/`out_stride`, non-overlapping with `a`/`b`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn solve_f32(
    a_ptr: *const f32,
    a_offset: isize,
    a_row_stride: isize,
    a_col_stride: isize,
    b_ptr: *const f32,
    b_offset: isize,
    b_stride: isize,
    out_ptr: *mut f32,
    out_offset: isize,
    out_stride: isize,
    n: usize,
) {
    // Copy A (n x n) into an owned, packed row-major scratch buffer.
    let mut a: Vec<f32> = Vec::with_capacity(n * n);
    for i in 0..n as isize {
        for j in 0..n as isize {
            a.push(*a_ptr.offset(a_offset + i * a_row_stride + j * a_col_stride));
        }
    }
    let mut b: Vec<f32> = Vec::with_capacity(n);
    for i in 0..n as isize {
        b.push(*b_ptr.offset(b_offset + i * b_stride));
    }

    // In-place LU with partial pivoting; `perm[i]` = original row now at position i.
    let mut perm: Vec<usize> = (0..n).collect();
    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_val = a[k * n + k].abs();
        for i in (k + 1)..n {
            let v = a[i * n + k].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = i;
            }
        }
        if pivot_row != k {
            for j in 0..n {
                a.swap(k * n + j, pivot_row * n + j);
            }
            perm.swap(k, pivot_row);
        }
        let pivot = a[k * n + k];
        if pivot.abs() < f32::EPSILON {
            continue; // singular column -- leave this row's multipliers at 0, matching MatrixMath.lu
        }
        for i in (k + 1)..n {
            let factor = a[i * n + k] / pivot;
            a[i * n + k] = factor; // store L's multiplier where U's zero would go (classic in-place LU)
            for j in (k + 1)..n {
                a[i * n + j] -= factor * a[k * n + j];
            }
        }
    }

    // Forward-substitute L*y = P*b (L has an implicit unit diagonal, not stored).
    let mut y = vec![0.0f32; n];
    for i in 0..n {
        let mut s = b[perm[i]];
        for j in 0..i {
            s -= a[i * n + j] * y[j];
        }
        y[i] = s;
    }

    // Back-substitute U*x = y.
    let mut x = vec![0.0f32; n];
    for ii in 0..n {
        let i = n - 1 - ii;
        let mut s = y[i];
        for j in (i + 1)..n {
            s -= a[i * n + j] * x[j];
        }
        let diag = a[i * n + i];
        x[i] = if diag.abs() < f32::EPSILON { 0.0 } else { s / diag };
    }

    for i in 0..n as isize {
        *out_ptr.offset(out_offset + i * out_stride) = x[i as usize];
    }
}

/// SIMD128 kernels (issue #13) — measured, then shipped: `docs/spikes/
/// wasm-simd.md` records a stable ~2.6-3x SIMD-only speedup over an
/// apples-to-apples contiguous-scalar baseline (~3.2-4.4x total vs. the
/// strided kernel these replace for the contiguous case), well above any
/// reasonable bar for "a real speedup," so this ships as a SEPARATE build
/// (Cargo `simd` feature) rather than staying scalar-only.
///
/// Built as a second .wasm artifact, never merged into the default build:
/// a wasm32 module containing ANY v128 instruction fails WebAssembly
/// validation in its ENTIRETY on a runtime without SIMD support — module
/// loading is all-or-nothing, unlike native code's per-call feature
/// detection, so there is no way to ship one module with both a SIMD path
/// and a guaranteed-always-loadable fallback. `@johnhenry/math-plus-tensor-wasm`'s
/// `Kernels.load()` feature-detects at runtime (`WebAssembly.validate()`)
/// and picks whichever of the two built .wasm files is appropriate,
/// falling back to the always-present scalar/strided kernels above for
/// every case this module doesn't cover (non-contiguous views, and any
/// runtime without SIMD support).
///
/// The `simd`-featured build (`npm run build:wasm:simd`) also passes
/// `RUSTFLAGS="-C link-args=--import-memory"` — WITHOUT it, this module
/// would allocate and export its OWN separate WASM linear memory, and
/// `WasmTensor` data (allocated via the always-loaded scalar module's
/// `alloc`) would live in a completely different buffer these SIMD
/// kernels can't see, defeating the entire point of a zero-copy fast
/// path. `--import-memory` makes this module IMPORT `env.memory` instead
/// of exporting its own; the JS loader instantiates it passing the
/// SCALAR module's `memory` export as that import, so both modules
/// genuinely share one linear memory / one `ArrayBuffer`.
#[cfg(feature = "simd")]
mod simd {
    use std::arch::wasm32::{f32x4_add, f32x4_mul, v128, v128_load, v128_store};

    /// Contiguous-only f32 elementwise add via WASM SIMD128 (4 lanes/store).
    /// Deliberately has NO offset/stride params (unlike `add_f32_strided`):
    /// SIMD loads need contiguous memory, and there is no benefit to a
    /// strided variant — the caller already applies any offset to the base
    /// pointers before calling. A scalar tail loop handles `len % 4 != 0`.
    ///
    /// # Safety
    /// `a_ptr`/`b_ptr`/`out_ptr` must each describe `len` valid, contiguous,
    /// non-overlapping-with-`out` f32 slots.
    #[no_mangle]
    #[target_feature(enable = "simd128")]
    pub unsafe extern "C" fn add_f32_contiguous_simd128(
        a_ptr: *const f32,
        b_ptr: *const f32,
        out_ptr: *mut f32,
        len: usize,
    ) {
        let chunks = len / 4;
        for i in 0..chunks {
            let idx = i * 4;
            let av = v128_load(a_ptr.add(idx) as *const v128);
            let bv = v128_load(b_ptr.add(idx) as *const v128);
            let sum = f32x4_add(av, bv);
            v128_store(out_ptr.add(idx) as *mut v128, sum);
        }
        for i in (chunks * 4)..len {
            *out_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
        }
    }

    /// Contiguous-only f32 elementwise multiply — same contract as
    /// `add_f32_contiguous_simd128`.
    ///
    /// # Safety
    /// Same requirements as `add_f32_contiguous_simd128`.
    #[no_mangle]
    #[target_feature(enable = "simd128")]
    pub unsafe extern "C" fn mul_f32_contiguous_simd128(
        a_ptr: *const f32,
        b_ptr: *const f32,
        out_ptr: *mut f32,
        len: usize,
    ) {
        let chunks = len / 4;
        for i in 0..chunks {
            let idx = i * 4;
            let av = v128_load(a_ptr.add(idx) as *const v128);
            let bv = v128_load(b_ptr.add(idx) as *const v128);
            let product = f32x4_mul(av, bv);
            v128_store(out_ptr.add(idx) as *mut v128, product);
        }
        for i in (chunks * 4)..len {
            *out_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_invalid_layout_returns_null_not_panic() {
        // Issue #55 Phase 2: defined failure signal on both build targets.
        assert!(alloc(16, 3).is_null()); // non-power-of-two align
        assert!(alloc(usize::MAX, 8).is_null()); // size overflows isize when rounded
        let ok = alloc(16, 4);
        assert!(!ok.is_null());
        unsafe { dealloc(ok, 16, 4) };
        // dealloc with an invalid layout: defined no-op (nothing to free --
        // alloc never returns memory for an invalid layout).
        unsafe { dealloc(std::ptr::null_mut(), 16, 3) };
    }

    #[test]
    fn add_f32_strided_contiguous() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [10.0f32, 20.0, 30.0];
        let mut out = [0.0f32; 3];
        unsafe {
            add_f32_strided(a.as_ptr(), 0, 1, b.as_ptr(), 0, 1, out.as_mut_ptr(), 0, 1, 3)
        };
        assert_eq!(out, [11.0, 22.0, 33.0]);
    }

    #[test]
    fn add_f32_strided_with_stride_and_offset() {
        // a = [x, 1, x, 2, x, 3] read every-other starting at offset 1.
        let a = [0.0f32, 1.0, 0.0, 2.0, 0.0, 3.0];
        let b = [10.0f32, 20.0, 30.0];
        let mut out = [0.0f32; 3];
        unsafe {
            add_f32_strided(a.as_ptr(), 1, 2, b.as_ptr(), 0, 1, out.as_mut_ptr(), 0, 1, 3)
        };
        assert_eq!(out, [11.0, 22.0, 33.0]);
    }

    #[test]
    fn mul_f32_strided_contiguous() {
        let a = [2.0f32, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0];
        let mut out = [0.0f32; 3];
        unsafe {
            mul_f32_strided(a.as_ptr(), 0, 1, b.as_ptr(), 0, 1, out.as_mut_ptr(), 0, 1, 3)
        };
        assert_eq!(out, [10.0, 18.0, 28.0]);
    }

    #[test]
    fn sub_f32_strided_contiguous() {
        let a = [10.0f32, 20.0, 30.0];
        let b = [1.0f32, 2.0, 3.0];
        let mut out = [0.0f32; 3];
        unsafe {
            sub_f32_strided(a.as_ptr(), 0, 1, b.as_ptr(), 0, 1, out.as_mut_ptr(), 0, 1, 3)
        };
        assert_eq!(out, [9.0, 18.0, 27.0]);
    }

    #[test]
    fn sub_f32_strided_with_stride_and_offset() {
        let a = [0.0f32, 11.0, 0.0, 22.0, 0.0, 33.0];
        let b = [1.0f32, 2.0, 3.0];
        let mut out = [0.0f32; 3];
        unsafe {
            sub_f32_strided(a.as_ptr(), 1, 2, b.as_ptr(), 0, 1, out.as_mut_ptr(), 0, 1, 3)
        };
        assert_eq!(out, [10.0, 20.0, 30.0]);
    }

    #[test]
    fn div_f32_strided_contiguous() {
        let a = [10.0f32, 20.0, 30.0];
        let b = [2.0f32, 4.0, 5.0];
        let mut out = [0.0f32; 3];
        unsafe {
            div_f32_strided(a.as_ptr(), 0, 1, b.as_ptr(), 0, 1, out.as_mut_ptr(), 0, 1, 3)
        };
        assert_eq!(out, [5.0, 5.0, 6.0]);
    }

    #[test]
    fn div_f32_strided_by_zero_yields_ieee754_infinity_not_a_panic() {
        let a = [1.0f32, -1.0, 0.0];
        let b = [0.0f32, 0.0, 0.0];
        let mut out = [0.0f32; 3];
        unsafe {
            div_f32_strided(a.as_ptr(), 0, 1, b.as_ptr(), 0, 1, out.as_mut_ptr(), 0, 1, 3)
        };
        assert_eq!(out[0], f32::INFINITY);
        assert_eq!(out[1], f32::NEG_INFINITY);
        assert!(out[2].is_nan());
    }

    #[test]
    fn gemm_f32_matches_hand_computed() {
        // A = [[1,2,3],[4,5,6]] (2x3), B = [[7,8],[9,10],[11,12]] (3x2)
        // A@B = [[58,64],[139,154]]
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut out = [0.0f32; 4];
        unsafe {
            gemm_f32(
                a.as_ptr(), 0, 3, 1, // A: row_stride=3, col_stride=1 (row-major 2x3)
                b.as_ptr(), 0, 2, 1, // B: row_stride=2, col_stride=1 (row-major 3x2)
                out.as_mut_ptr(), 0, 2, 1, // out: row-major 2x2
                2, 2, 3, 1.0, 0.0,
            )
        };
        assert_eq!(out, [58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn gemm_f32_transposed_a_via_strides_no_copy() {
        // A physically stored as its transpose (3x2, row-major): swap the
        // row/col strides to read it as if it were (2x3) -- proves the
        // kernel never needs a packed copy of a transposed operand.
        let a_t = [1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0]; // A^T, row-major (3x2)
        let b = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut out = [0.0f32; 4];
        unsafe {
            gemm_f32(
                a_t.as_ptr(), 0, 1, 2, // read a_t as (2x3): row_stride=1, col_stride=2
                b.as_ptr(), 0, 2, 1,
                out.as_mut_ptr(), 0, 2, 1,
                2, 2, 3, 1.0, 0.0,
            )
        };
        assert_eq!(out, [58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn solve_f32_matches_hand_computed() {
        // 2x + y = 3, x + 3y = 5 -> x=0.8, y=1.4
        let a = [2.0f32, 1.0, 1.0, 3.0];
        let b = [3.0f32, 5.0];
        let mut out = [0.0f32; 2];
        unsafe { solve_f32(a.as_ptr(), 0, 2, 1, b.as_ptr(), 0, 1, out.as_mut_ptr(), 0, 1, 2) };
        assert!((out[0] - 0.8).abs() < 1e-5, "x={}", out[0]);
        assert!((out[1] - 1.4).abs() < 1e-5, "y={}", out[1]);
    }

    #[test]
    fn solve_f32_requires_partial_pivoting() {
        // 0*x + 1*y = 2, 1*x + 1*y = 3 -- a[0][0]=0 forces a row swap, or a
        // naive no-pivoting elimination would divide by zero. x=1, y=2.
        let a = [0.0f32, 1.0, 1.0, 1.0];
        let b = [2.0f32, 3.0];
        let mut out = [0.0f32; 2];
        unsafe { solve_f32(a.as_ptr(), 0, 2, 1, b.as_ptr(), 0, 1, out.as_mut_ptr(), 0, 1, 2) };
        assert!((out[0] - 1.0).abs() < 1e-5, "x={}", out[0]);
        assert!((out[1] - 2.0).abs() < 1e-5, "y={}", out[1]);
    }

    #[test]
    fn solve_f32_transposed_a_via_strides_no_copy() {
        // Same system as solve_f32_matches_hand_computed, but A stored as its
        // transpose and read via swapped row/col strides -- proves the
        // kernel never needs a packed copy of a transposed operand (A is
        // symmetric here would trivially pass, so use an asymmetric system).
        // 2x + 4y = 10, x + 3y = 7 -> x=1, y=2
        let a = [2.0f32, 1.0, 4.0, 3.0]; // A^T, row-major (2x2): col-major A
        let b = [10.0f32, 7.0];
        let mut out = [0.0f32; 2];
        unsafe {
            // read a_t as A (2x2): row_stride=1, col_stride=2
            solve_f32(a.as_ptr(), 0, 1, 2, b.as_ptr(), 0, 1, out.as_mut_ptr(), 0, 1, 2)
        };
        assert!((out[0] - 1.0).abs() < 1e-5, "x={}", out[0]);
        assert!((out[1] - 2.0).abs() < 1e-5, "y={}", out[1]);
    }

    #[test]
    fn solve_f32_3x3() {
        // x + y + z = 6, 2y + 5z = -4, 2x + 5y - z = 27 -> x=5, y=3, z=-2
        // (classic textbook example)
        let a = [1.0f32, 1.0, 1.0, 0.0, 2.0, 5.0, 2.0, 5.0, -1.0];
        let b = [6.0f32, -4.0, 27.0];
        let mut out = [0.0f32; 3];
        unsafe { solve_f32(a.as_ptr(), 0, 3, 1, b.as_ptr(), 0, 1, out.as_mut_ptr(), 0, 1, 3) };
        assert!((out[0] - 5.0).abs() < 1e-3, "x={}", out[0]);
        assert!((out[1] - 3.0).abs() < 1e-3, "y={}", out[1]);
        assert!((out[2] - (-2.0)).abs() < 1e-3, "z={}", out[2]);
    }

    #[test]
    fn alloc_honors_requested_alignment() {
        for align in [1usize, 2, 4, 8, 16] {
            let ptr = alloc(37, align);
            assert_eq!(ptr as usize % align, 0, "align {align}");
            unsafe { dealloc(ptr, 37, align) };
        }
    }
}
