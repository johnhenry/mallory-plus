import type { DType, Tensor } from "mallory-tensor-core";

/**
 * Flatten a Tensor's contiguous values into a plain Float64Array, row-major.
 * `.data` (after `.contiguous()`) is already the flat, row-major storage for
 * any ndim -- no boxed-nested-array walk needed. Only converts (copies) when
 * the source isn't already f64 (e.g. f32, per `assertFloatDtype`'s v1
 * scope); when it already is f64, returns the tensor's own backing store
 * directly (callers only read it, never mutate it).
 */
export function flattenToFloat64(t: Tensor): Float64Array {
  const data = t.contiguous().data as Float32Array | Float64Array;
  return data instanceof Float64Array ? data : Float64Array.from(data);
}

/** resize/normalize v1 scope: float dtypes only (f32/f64) -- not uint8 raw pixel data, which would need explicit rounding/clamping semantics this package doesn't design yet. Clear error rather than silently mishandling an integer/bool dtype. */
export function assertFloatDtype(dtype: DType, fnName: string): asserts dtype is "f32" | "f64" {
  if (dtype !== "f32" && dtype !== "f64") {
    throw new TypeError(
      `${fnName}: v1 supports f32/f64 tensors only, got dtype "${dtype}" -- cast to a float dtype first`,
    );
  }
}

/** Convert a Float64Array working buffer to the requested float dtype's TypedArray. */
export function toDtypeArray(data: Float64Array, dtype: "f32" | "f64"): Float32Array | Float64Array {
  return dtype === "f32" ? Float32Array.from(data) : data;
}
