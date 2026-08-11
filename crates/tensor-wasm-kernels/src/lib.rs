//! Flat-numeric kernel ABI for mallory-tensor-wasm.
//!
//! Exports use plain `extern "C"` with pointer/length params (no wasm-bindgen
//! object marshalling) — callers on the JS side own buffer layout and pass
//! offsets into linear memory. Strides/offsets become explicit params as
//! kernels grow (BLAS-style ABI per docs/PLAN.md §6.1).

/// Allocate `len` bytes in linear memory, returning the offset.
/// Ownership passes to the caller; free with `dealloc(ptr, len)`.
#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    let mut buf: Vec<u8> = Vec::with_capacity(len);
    let ptr = buf.as_mut_ptr();
    core::mem::forget(buf);
    ptr
}

/// Free a buffer previously returned by `alloc`.
///
/// # Safety
/// `ptr` must come from `alloc(len)` and not have been freed already.
#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: *mut u8, len: usize) {
    drop(Vec::from_raw_parts(ptr, 0, len));
}

/// out[i] = a[i] + b[i] for i in 0..len (contiguous f32).
///
/// # Safety
/// All three pointers must reference `len` valid, non-overlapping-with-out
/// f32 slots in linear memory.
#[no_mangle]
pub unsafe extern "C" fn add_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    for i in 0..len {
        *out.add(i) = *a.add(i) + *b.add(i);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn add_f32_adds() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [10.0f32, 20.0, 30.0];
        let mut out = [0.0f32; 3];
        unsafe { super::add_f32(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 3) };
        assert_eq!(out, [11.0, 22.0, 33.0]);
    }
}
