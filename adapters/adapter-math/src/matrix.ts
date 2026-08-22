/**
 * Matrix/Vector <-> Tensor conversion (issue #14) — the first and cheapest
 * @johnhenry/math bridge deliverable; validates the adapter-isolation pattern
 * (math-plus -> @johnhenry/math, never the reverse; core packages carry no
 * dependency on this adapter).
 *
 * Always copies at the edge: @johnhenry/math's `Vector<T> extends Array<T>`
 * nested representation (a `Matrix<T>` is a `Vector<Vector<T>>`) cannot
 * alias a Tensor's flat TypedArray storage, so there is no view path here —
 * unlike Tensor's own slice/permute/etc., which are views by design.
 */
import { isBigIntDType, Tensor, type DType } from "@johnhenry/math-plus-tensor-core";
import type { Matrix, Vector } from "@johnhenry/math";

export interface ConvertOptions {
  dtype?: DType;
}

function assertNotBigInt(dtype: DType, fnName: string): void {
  if (isBigIntDType(dtype)) {
    throw new TypeError(
      `${fnName}: bigint dtypes (${dtype}) can't be represented as plain JS numbers; cast() to a non-bigint dtype first`,
    );
  }
}

function toRows(m: Matrix<number> | number[][]): number[][] {
  return [...(m as Iterable<Iterable<number>>)].map((row) => [...row]);
}

/** `Matrix<number>` (or plain `number[][]`) -> a 2-D Tensor. Copies. Throws on a ragged (non-rectangular) input. */
export function fromMatrix(m: Matrix<number> | number[][], opts: ConvertOptions = {}): Tensor {
  const rows = toRows(m);
  const height = rows.length;
  const width = height > 0 ? (rows[0] as number[]).length : 0;
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i] as number[];
    if (row.length !== width) {
      throw new RangeError(
        `fromMatrix: ragged input — row 0 has ${width} columns but row ${i} has ${row.length}`,
      );
    }
  }
  return Tensor.from(rows.flat(), { dtype: opts.dtype }).reshape([height, width]);
}

/** `Vector<number>` (or plain `number[]`) -> a 1-D Tensor. Copies. */
export function fromVector(v: Vector<number> | readonly number[], opts: ConvertOptions = {}): Tensor {
  return Tensor.from([...v], { dtype: opts.dtype });
}

/** A 2-D Tensor -> plain `number[][]`. Copies. Throws on ndim != 2 (including ndim > 2 — this adapter's v1 scope is 2-D matrices only). */
export function toMatrix(t: Tensor): number[][] {
  if (t.ndim !== 2) {
    throw new RangeError(`toMatrix: expected a 2-D tensor, got ndim=${t.ndim} (shape [${t.shape}])`);
  }
  assertNotBigInt(t.dtype, "toMatrix");
  return t.toArray() as number[][];
}

/** A 1-D Tensor -> plain `number[]`. Copies. Throws on ndim != 1. */
export function toVector(t: Tensor): number[] {
  if (t.ndim !== 1) {
    throw new RangeError(`toVector: expected a 1-D tensor, got ndim=${t.ndim} (shape [${t.shape}])`);
  }
  assertNotBigInt(t.dtype, "toVector");
  return t.toArray() as number[];
}
