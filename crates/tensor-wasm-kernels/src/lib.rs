//! Flat-numeric kernel ABI for mallory-tensor-wasm.
//!
//! Exports use plain `extern "C"` with pointer/offset/stride params (no
//! wasm-bindgen object marshalling) — callers on the JS side own buffer
//! layout and pass offsets into linear memory. Offsets/strides are in
//! *elements*, not bytes (matches mallory-tensor-core's `Tensor.strides`
//! convention on the JS side).
//!
//! Alignment contract (issue #7): `alloc` takes an explicit `align` and
//! guarantees the returned pointer satisfies it, via `std::alloc` +
//! `Layout` rather than relying on the allocator's incidental behavior.

use std::alloc::{alloc as std_alloc, dealloc as std_dealloc, Layout};

/// Allocate `len` bytes aligned to `align` (must be a power of two; 4 or 8
/// for f32/f64 buffers). Ownership passes to the caller; free with
/// `dealloc(ptr, len, align)` using the SAME align used here.
#[no_mangle]
pub extern "C" fn alloc(len: usize, align: usize) -> *mut u8 {
    let layout = Layout::from_size_align(len.max(1), align).expect("invalid alloc layout");
    // SAFETY: layout has non-zero size (len.max(1)) and a validated alignment.
    unsafe { std_alloc(layout) }
}

/// Free a buffer previously returned by `alloc(len, align)`.
///
/// # Safety
/// `ptr` must come from `alloc(len, align)` with the SAME `len`/`align`, and
/// not have been freed already.
#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: *mut u8, len: usize, align: usize) {
    let layout = Layout::from_size_align(len.max(1), align).expect("invalid dealloc layout");
    std_dealloc(ptr, layout);
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn alloc_honors_requested_alignment() {
        for align in [1usize, 2, 4, 8, 16] {
            let ptr = alloc(37, align);
            assert_eq!(ptr as usize % align, 0, "align {align}");
            unsafe { dealloc(ptr, 37, align) };
        }
    }
}
