import type { DType, Tensor } from "mallory-tensor-core";

/** Flatten a Tensor's contiguous values into a plain Float64Array, row-major. Reference-speed (walks the nested toArray() output), not a hot-path kernel. */
export function flattenToFloat64(t: Tensor): Float64Array {
  const nested = t.contiguous().toArray();
  const flat: number[] = [];
  const walk = (x: unknown): void => {
    if (Array.isArray(x)) {
      for (const v of x) walk(v);
    } else {
      flat.push(Number(x));
    }
  };
  walk(nested);
  return Float64Array.from(flat);
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
