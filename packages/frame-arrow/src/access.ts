/**
 * Null-safe, precision-safe column access helpers.
 *
 * Every read in this module goes through `Vector.isValid()`/`Vector.get()`
 * (which consult the validity bitmap) — NEVER `Vector.toArray()`, whose null
 * slots hold garbage (docs/spikes/arrow-parity.md sharp edge #5).
 *
 * Timestamp columns get special handling (sharp edge #3): `Vector.get()`
 * returns a `number` of epoch-*milliseconds* for every unit, which silently
 * truncates microsecond precision. `timestampExactAt` instead reads the
 * underlying int64 buffer directly (chunk-transparent) to preserve exactness;
 * `timestampDateAt` is the explicit, documented-lossy `Date` convenience path.
 */
import type { Vector } from "apache-arrow";
import { type DType } from "./dtype.ts";

/**
 * Read the raw int64 buffer value backing a timestamp (or int64) Vector at
 * `index`, without going through `Vector.get()`'s float-ms coercion.
 * Chunk-transparent: walks `vector.data` (one entry per Arrow record batch)
 * and reads straight from each chunk's already offset-adjusted `.values`
 * BigInt64Array — verified empirically (chunk-probe in package history) that
 * `Data.values` is pre-sliced to `[0, chunk.length)`, so no additional
 * `chunk.offset` arithmetic is needed once you've found the right chunk.
 */
export function rawInt64At(vector: Vector, index: number): bigint | null {
  if (index < 0 || index >= vector.length) {
    throw new RangeError(`index ${index} out of bounds for vector of length ${vector.length}`);
  }
  if (!vector.isValid(index)) return null;
  let acc = 0;
  for (const chunk of vector.data) {
    if (index < acc + chunk.length) {
      const values = chunk.values as unknown as BigInt64Array;
      return values[index - acc] as bigint;
    }
    acc += chunk.length;
  }
  throw new RangeError(`index ${index} out of bounds`);
}

/** Exact epoch value (in the column's own unit — ms or us) as a bigint, or null. */
export function timestampExactAt(vector: Vector, index: number): bigint | null {
  return rawInt64At(vector, index);
}

/** Convenience, ms-truncated: reads via Vector.get(), which apache-arrow always
 * returns as epoch-milliseconds (a `number`) for every timestamp unit. */
export function timestampDateAt(vector: Vector, index: number): Date | null {
  const ms = vector.get(index) as number | null;
  return ms === null ? null : new Date(ms);
}

/** A single materialized cell value, dtype-aware. List/struct cells are
 * flattened to plain arrays/objects (never Arrow's Vector/StructRow proxies). */
export function cellAt(vector: Vector, index: number, dtype: DType): unknown {
  if (dtype === "timestamp_ms" || dtype === "timestamp_us") {
    return timestampExactAt(vector, index);
  }
  const raw = vector.get(index) as unknown;
  if (raw === null || raw === undefined) return null;
  if (dtype === "list" || dtype === "struct") {
    return (raw as { toJSON(): unknown }).toJSON();
  }
  return raw;
}

/** Materialize an entire column to a plain JS array (dtype-aware, null-safe). */
export function columnToArray(vector: Vector, dtype: DType): unknown[] {
  const out: unknown[] = new Array(vector.length);
  for (let i = 0; i < vector.length; i++) out[i] = cellAt(vector, i, dtype);
  return out;
}
