/**
 * Fixed-width dtypes for @johnhenry/math-plus-tensor-core.
 *
 * Includes i64/u64 (BigInt64Array/BigUint64Array-backed) — resolving the
 * source design's own inconsistency where the DType union stopped at 32-bit
 * ints while its ONNX example required int64 input_ids (docs/PLAN.md §9 #4,
 * decided at kickoff). f16/bf16 are declared but stored in Uint16Array until
 * Float16Array is universally available; kernels handle conversion.
 */
export type DType =
  | "bool"
  | "u8"
  | "i8"
  | "u16"
  | "i16"
  | "u32"
  | "i32"
  | "u64"
  | "i64"
  | "f16"
  | "bf16"
  | "f32"
  | "f64";

export type TypedArrayFor<D extends DType> = D extends "f64"
  ? Float64Array
  : D extends "f32"
    ? Float32Array
    : D extends "i64"
      ? BigInt64Array
      : D extends "u64"
        ? BigUint64Array
        : D extends "i32"
          ? Int32Array
          : D extends "u32"
            ? Uint32Array
            : D extends "i16"
              ? Int16Array
              : D extends "u16" | "f16" | "bf16"
                ? Uint16Array
                : D extends "i8"
                  ? Int8Array
                  : Uint8Array; // u8, bool

export type AnyTypedArray =
  | Float64Array
  | Float32Array
  | BigInt64Array
  | BigUint64Array
  | Int32Array
  | Uint32Array
  | Int16Array
  | Uint16Array
  | Int8Array
  | Uint8Array;

const CONSTRUCTORS: Record<DType, new (length: number) => AnyTypedArray> = {
  bool: Uint8Array,
  u8: Uint8Array,
  i8: Int8Array,
  u16: Uint16Array,
  i16: Int16Array,
  u32: Uint32Array,
  i32: Int32Array,
  u64: BigUint64Array,
  i64: BigInt64Array,
  f16: Uint16Array,
  bf16: Uint16Array,
  f32: Float32Array,
  f64: Float64Array,
};

export const BYTES_PER_ELEMENT: Record<DType, number> = {
  bool: 1,
  u8: 1,
  i8: 1,
  u16: 2,
  i16: 2,
  u32: 4,
  i32: 4,
  u64: 8,
  i64: 8,
  f16: 2,
  bf16: 2,
  f32: 4,
  f64: 8,
};

/** True for i64/u64 — element access uses bigint, not number. */
export function isBigIntDType(dtype: DType): boolean {
  return dtype === "i64" || dtype === "u64";
}

export function allocate(dtype: DType, length: number): AnyTypedArray {
  return new CONSTRUCTORS[dtype](length);
}
