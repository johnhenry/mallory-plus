/**
 * mallory-tensor-core — typed n-dimensional arrays.
 *
 * Skeleton (kickoff Task 3). The M1 slice (strided views, broadcasting,
 * elementwise ops, reductions, .npy I/O) lands per docs/PLAN.md §6.1.
 * Architectural rules enforced from day one: no Proxy-based indexing,
 * views vs. contiguous are semantically distinct, no implicit copies.
 */
import {
  allocate,
  isBigIntDType,
  type AnyTypedArray,
  type DType,
} from "./dtype.ts";

export {
  allocate,
  isBigIntDType,
  BYTES_PER_ELEMENT,
  type AnyTypedArray,
  type DType,
  type TypedArrayFor,
} from "./dtype.ts";

export type Shape = readonly number[];

function shapeSize(shape: Shape): number {
  let size = 1;
  for (const dim of shape) {
    if (!Number.isInteger(dim) || dim < 0) {
      throw new RangeError(`invalid dimension ${dim} in shape [${shape}]`);
    }
    size *= dim;
  }
  return size;
}

/** Row-major (C-order) strides, in elements. */
function contiguousStrides(shape: Shape): number[] {
  const strides = new Array<number>(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = acc;
    acc *= shape[i] as number;
  }
  return strides;
}

export interface TensorOptions {
  dtype?: DType;
}

export class Tensor {
  readonly shape: Shape;
  readonly strides: readonly number[];
  readonly dtype: DType;
  readonly size: number;
  /** Backing storage; may be shared between views. */
  readonly data: AnyTypedArray;
  /** Element offset of this tensor's [0, 0, ...] within `data`. */
  readonly offset: number;

  private constructor(
    data: AnyTypedArray,
    shape: Shape,
    strides: readonly number[],
    dtype: DType,
    offset: number,
  ) {
    this.data = data;
    this.shape = Object.freeze([...shape]);
    this.strides = Object.freeze([...strides]);
    this.dtype = dtype;
    this.size = shapeSize(shape);
    this.offset = offset;
  }

  get ndim(): number {
    return this.shape.length;
  }

  static zeros(shape: Shape, options: TensorOptions = {}): Tensor {
    const dtype = options.dtype ?? "f32";
    const data = allocate(dtype, shapeSize(shape));
    return new Tensor(data, shape, contiguousStrides(shape), dtype, 0);
  }

  static from(values: readonly number[], options: TensorOptions = {}): Tensor {
    const dtype = options.dtype ?? "f32";
    const data = allocate(dtype, values.length);
    if (isBigIntDType(dtype)) {
      const big = data as BigInt64Array | BigUint64Array;
      for (let i = 0; i < values.length; i++) big[i] = BigInt(values[i] as number);
    } else {
      (data as Exclude<AnyTypedArray, BigInt64Array | BigUint64Array>).set(
        values,
      );
    }
    return new Tensor(data, [values.length], [1], dtype, 0);
  }

  /** Scalar element read by multi-index. Returns bigint for i64/u64. */
  at(...indices: number[]): number | bigint {
    if (indices.length !== this.ndim) {
      throw new RangeError(
        `expected ${this.ndim} indices, got ${indices.length}`,
      );
    }
    let elementOffset = this.offset;
    for (let axis = 0; axis < indices.length; axis++) {
      const dim = this.shape[axis] as number;
      let index = indices[axis] as number;
      if (index < 0) index += dim;
      if (index < 0 || index >= dim) {
        throw new RangeError(`index ${indices[axis]} out of bounds for axis ${axis} (size ${dim})`);
      }
      elementOffset += index * (this.strides[axis] as number);
    }
    return this.data[elementOffset] as number | bigint;
  }
}
