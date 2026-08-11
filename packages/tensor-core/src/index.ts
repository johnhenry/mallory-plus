/**
 * mallory-tensor-core — typed n-dimensional arrays (M1 slice).
 *
 * Pure-JS/TypedArray execution path; WASM kernels swap in underneath via
 * mallory-tensor-wasm without changing this API. Architectural rules
 * enforced from day one (docs/PLAN.md §2, §6.1): no Proxy-based indexing,
 * views vs. contiguous are semantically distinct (permute/transpose/reshape
 * never copy; contiguous() copies iff needed), no implicit copies, no
 * implicit dtype promotion (M1: elementwise ops require matching dtypes).
 */
import {
  allocate,
  isBigIntDType,
  type AnyTypedArray,
  type DType,
} from "./dtype.ts";
import { parseNpy, serializeNpy } from "./npy.ts";

export {
  allocate,
  isBigIntDType,
  BYTES_PER_ELEMENT,
  type AnyTypedArray,
  type DType,
  type TypedArrayFor,
} from "./dtype.ts";

export type Shape = readonly number[];
export type Axis = number;

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

/**
 * NumPy broadcasting: align from trailing axes; dims must match or be 1.
 */
export function broadcastShapes(a: Shape, b: Shape): number[] {
  const ndim = Math.max(a.length, b.length);
  const out = new Array<number>(ndim);
  for (let i = 0; i < ndim; i++) {
    const da = a[a.length - 1 - i] ?? 1;
    const db = b[b.length - 1 - i] ?? 1;
    if (da !== db && da !== 1 && db !== 1) {
      throw new RangeError(
        `shapes [${a}] and [${b}] are not broadcast-compatible at axis ${ndim - 1 - i}`,
      );
    }
    out[ndim - 1 - i] = Math.max(da, db);
  }
  return out;
}

/** Strides for reading `shape`-shaped data as if it were `target`-shaped (0-stride on broadcast axes). */
function broadcastStrides(
  shape: Shape,
  strides: readonly number[],
  target: Shape,
): number[] {
  const out = new Array<number>(target.length).fill(0);
  for (let i = 0; i < shape.length; i++) {
    const targetAxis = target.length - shape.length + i;
    if (shape[i] === target[targetAxis]) {
      out[targetAxis] = strides[i] as number;
    } else {
      // shape[i] === 1 (validated by broadcastShapes): stride 0 repeats it.
      out[targetAxis] = 0;
    }
  }
  return out;
}

type BinaryOp = "add" | "sub" | "mul" | "div";

const NUMBER_OPS: Record<BinaryOp, (a: number, b: number) => number> = {
  add: (a, b) => a + b,
  sub: (a, b) => a - b,
  mul: (a, b) => a * b,
  div: (a, b) => a / b,
};

const BIGINT_OPS: Partial<Record<BinaryOp, (a: bigint, b: bigint) => bigint>> =
  {
    add: (a, b) => a + b,
    sub: (a, b) => a - b,
    mul: (a, b) => a * b,
    // div omitted: integer division semantics differ from NumPy's true
    // division (i64/i64 -> f64); cast to a float dtype first.
  };

export interface TensorOptions {
  dtype?: DType;
}

export class Tensor {
  readonly shape: Shape;
  readonly strides: readonly number[];
  readonly dtype: DType;
  readonly size: number;
  /** Backing storage; shared between views (compare identity to detect views). */
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

  get isContiguous(): boolean {
    const expected = contiguousStrides(this.shape);
    for (let i = 0; i < expected.length; i++) {
      // A dim of size 1 makes its stride irrelevant.
      if (this.shape[i] !== 1 && this.strides[i] !== expected[i]) return false;
    }
    return true;
  }

  // ---- constructors -------------------------------------------------------

  static zeros(shape: Shape, options: TensorOptions = {}): Tensor {
    const dtype = options.dtype ?? "f32";
    return new Tensor(
      allocate(dtype, shapeSize(shape)),
      shape,
      contiguousStrides(shape),
      dtype,
      0,
    );
  }

  static ones(shape: Shape, options: TensorOptions = {}): Tensor {
    return Tensor.full(shape, 1, options);
  }

  static full(shape: Shape, value: number, options: TensorOptions = {}): Tensor {
    const dtype = options.dtype ?? "f32";
    const data = allocate(dtype, shapeSize(shape));
    if (isBigIntDType(dtype)) {
      (data as BigInt64Array | BigUint64Array).fill(BigInt(value));
    } else {
      (data as Float64Array).fill(value);
    }
    return new Tensor(data, shape, contiguousStrides(shape), dtype, 0);
  }

  static arange(
    start: number,
    stop?: number,
    step = 1,
    options: TensorOptions = {},
  ): Tensor {
    if (stop === undefined) {
      stop = start;
      start = 0;
    }
    if (step === 0) throw new RangeError("arange step must be non-zero");
    const dtype = options.dtype ?? "f32";
    const length = Math.max(0, Math.ceil((stop - start) / step));
    const data = allocate(dtype, length);
    if (isBigIntDType(dtype)) {
      const big = data as BigInt64Array | BigUint64Array;
      for (let i = 0; i < length; i++) big[i] = BigInt(start + i * step);
    } else {
      const num = data as Float64Array;
      for (let i = 0; i < length; i++) num[i] = start + i * step;
    }
    return new Tensor(data, [length], [1], dtype, 0);
  }

  static from(values: readonly number[], options: TensorOptions = {}): Tensor {
    const dtype = options.dtype ?? "f32";
    const data = allocate(dtype, values.length);
    if (isBigIntDType(dtype)) {
      const big = data as BigInt64Array | BigUint64Array;
      for (let i = 0; i < values.length; i++) {
        big[i] = BigInt(values[i] as number);
      }
    } else {
      (data as Float64Array).set(values);
    }
    return new Tensor(data, [values.length], [1], dtype, 0);
  }

  /** Wrap an existing TypedArray WITHOUT copying; caller owns aliasing. */
  static fromTypedArray(
    data: AnyTypedArray,
    shape: Shape,
    options: { dtype: DType },
  ): Tensor {
    if (shapeSize(shape) !== data.length) {
      throw new RangeError(
        `shape [${shape}] (${shapeSize(shape)} elements) does not match data length ${data.length}`,
      );
    }
    return new Tensor(data, shape, contiguousStrides(shape), options.dtype, 0);
  }

  // ---- views (never copy) -------------------------------------------------

  /** View with a new shape. Supports one -1 (inferred). Requires contiguity. */
  reshape(shape: Shape): Tensor {
    if (!this.isContiguous) {
      throw new TypeError(
        "reshape of a non-contiguous tensor is not a view; call .contiguous() first (no implicit copies)",
      );
    }
    const inferred = [...shape];
    const negatives = inferred.filter((d) => d === -1).length;
    if (negatives > 1) {
      throw new RangeError("reshape allows at most one -1 dimension");
    }
    if (negatives === 1) {
      const known = inferred.reduce((a, d) => (d === -1 ? a : a * d), 1);
      if (known === 0 || this.size % known !== 0) {
        throw new RangeError(
          `cannot infer -1: ${this.size} elements do not divide by ${known}`,
        );
      }
      inferred[inferred.indexOf(-1)] = this.size / known;
    }
    if (shapeSize(inferred) !== this.size) {
      throw new RangeError(
        `cannot reshape ${this.size} elements into [${shape}]`,
      );
    }
    return new Tensor(
      this.data,
      inferred,
      contiguousStrides(inferred),
      this.dtype,
      this.offset,
    );
  }

  /** Axis-permutation view (shares storage; never copies). */
  permute(axes: readonly number[]): Tensor {
    if (axes.length !== this.ndim) {
      throw new RangeError(`permute needs ${this.ndim} axes, got ${axes.length}`);
    }
    const seen = new Set(axes);
    if (seen.size !== this.ndim || axes.some((a) => a < 0 || a >= this.ndim)) {
      throw new RangeError(`invalid permutation [${axes}] for ndim ${this.ndim}`);
    }
    const shape = axes.map((a) => this.shape[a] as number);
    const strides = axes.map((a) => this.strides[a] as number);
    return new Tensor(this.data, shape, strides, this.dtype, this.offset);
  }

  /** Reverse (or permute) axes — a view. */
  transpose(axes?: readonly number[]): Tensor {
    if (axes) return this.permute(axes);
    const reversed = Array.from({ length: this.ndim }, (_, i) => this.ndim - 1 - i);
    return this.permute(reversed);
  }

  /** Packed C-order copy — or `this` (identical object) when already packed. */
  contiguous(): Tensor {
    if (this.isContiguous && this.offset === 0 && this.size === this.data.length) {
      return this;
    }
    const data = allocate(this.dtype, this.size);
    const source = this.data;
    let i = 0;
    for (const elementOffset of this.elementOffsets()) {
      data[i++] = source[elementOffset] as never;
    }
    return new Tensor(
      data,
      this.shape,
      contiguousStrides(this.shape),
      this.dtype,
      0,
    );
  }

  // ---- element access ------------------------------------------------------

  /** Scalar element read by multi-index. Returns bigint for i64/u64. */
  at(...indices: number[]): number | bigint {
    if (indices.length !== this.ndim) {
      throw new RangeError(`expected ${this.ndim} indices, got ${indices.length}`);
    }
    let elementOffset = this.offset;
    for (let axis = 0; axis < indices.length; axis++) {
      const dim = this.shape[axis] as number;
      let index = indices[axis] as number;
      if (index < 0) index += dim;
      if (index < 0 || index >= dim) {
        throw new RangeError(
          `index ${indices[axis]} out of bounds for axis ${axis} (size ${dim})`,
        );
      }
      elementOffset += index * (this.strides[axis] as number);
    }
    return this.data[elementOffset] as number | bigint;
  }

  /** The single element of a size-1 tensor. */
  item(): number | bigint {
    if (this.size !== 1) {
      throw new RangeError(`item() requires size 1, got ${this.size}`);
    }
    return this.data[this.offset] as number | bigint;
  }

  /** Nested plain-array copy (edge/API use only — never a kernel format). */
  toArray(): unknown[] {
    const build = (axis: number, offset: number): unknown[] => {
      const dim = this.shape[axis] as number;
      const stride = this.strides[axis] as number;
      const out = new Array(dim);
      for (let i = 0; i < dim; i++) {
        out[i] =
          axis === this.ndim - 1
            ? this.data[offset + i * stride]
            : build(axis + 1, offset + i * stride);
      }
      return out;
    };
    if (this.ndim === 0) return [];
    return build(0, this.offset);
  }

  /** Iterate storage offsets of every element in C-order of this view. */
  private *elementOffsets(): Generator<number> {
    if (this.size === 0) return;
    if (this.ndim === 0) {
      yield this.offset;
      return;
    }
    const index = new Array<number>(this.ndim).fill(0);
    let offset = this.offset;
    outer: for (;;) {
      yield offset;
      for (let axis = this.ndim - 1; axis >= 0; axis--) {
        index[axis] = (index[axis] as number) + 1;
        offset += this.strides[axis] as number;
        if ((index[axis] as number) < (this.shape[axis] as number)) {
          continue outer;
        }
        index[axis] = 0;
        offset -=
          (this.shape[axis] as number) * (this.strides[axis] as number);
      }
      return;
    }
  }

  // ---- elementwise (broadcasting) -----------------------------------------

  #binary(op: BinaryOp, other: Tensor | number): Tensor {
    const rhs =
      typeof other === "number"
        ? Tensor.full([], other, { dtype: this.dtype })
        : other;
    if (rhs.dtype !== this.dtype) {
      throw new TypeError(
        `dtype mismatch: ${this.dtype} vs ${rhs.dtype} (M1 has no implicit promotion; cast() first)`,
      );
    }
    const outShape = broadcastShapes(this.shape, rhs.shape);
    const out = Tensor.zeros(outShape, { dtype: this.dtype });
    const aStrides = broadcastStrides(this.shape, this.strides, outShape);
    const bStrides = broadcastStrides(rhs.shape, rhs.strides, outShape);

    const size = out.size;
    const ndim = outShape.length;
    const index = new Array<number>(ndim).fill(0);
    let aOff = this.offset;
    let bOff = rhs.offset;

    if (isBigIntDType(this.dtype)) {
      const fn = BIGINT_OPS[op];
      if (!fn) {
        throw new TypeError(
          `${op} is not defined for ${this.dtype} (NumPy true-division returns f64; cast() first)`,
        );
      }
      const a = this.data as BigInt64Array;
      const b = rhs.data as BigInt64Array;
      const o = out.data as BigInt64Array;
      for (let i = 0; i < size; i++) {
        o[i] = fn(a[aOff] as bigint, b[bOff] as bigint);
        for (let axis = ndim - 1; axis >= 0; axis--) {
          index[axis] = (index[axis] as number) + 1;
          aOff += aStrides[axis] as number;
          bOff += bStrides[axis] as number;
          if ((index[axis] as number) < (outShape[axis] as number)) break;
          index[axis] = 0;
          aOff -= (outShape[axis] as number) * (aStrides[axis] as number);
          bOff -= (outShape[axis] as number) * (bStrides[axis] as number);
        }
      }
    } else {
      const fn = NUMBER_OPS[op];
      const a = this.data as Float64Array;
      const b = rhs.data as Float64Array;
      const o = out.data as Float64Array;
      for (let i = 0; i < size; i++) {
        o[i] = fn(a[aOff] as number, b[bOff] as number);
        for (let axis = ndim - 1; axis >= 0; axis--) {
          index[axis] = (index[axis] as number) + 1;
          aOff += aStrides[axis] as number;
          bOff += bStrides[axis] as number;
          if ((index[axis] as number) < (outShape[axis] as number)) break;
          index[axis] = 0;
          aOff -= (outShape[axis] as number) * (aStrides[axis] as number);
          bOff -= (outShape[axis] as number) * (bStrides[axis] as number);
        }
      }
    }
    return out;
  }

  add(other: Tensor | number): Tensor {
    return this.#binary("add", other);
  }
  sub(other: Tensor | number): Tensor {
    return this.#binary("sub", other);
  }
  mul(other: Tensor | number): Tensor {
    return this.#binary("mul", other);
  }
  div(other: Tensor | number): Tensor {
    return this.#binary("div", other);
  }

  // ---- reductions -----------------------------------------------------------

  /** Sum over all elements (axis omitted) or along one axis. */
  sum(axis?: Axis): Tensor {
    return this.#reduce(axis, false);
  }

  /** Mean over all elements (axis omitted) or along one axis. Always float output. */
  mean(axis?: Axis): Tensor {
    return this.#reduce(axis, true);
  }

  #reduce(axis: Axis | undefined, mean: boolean): Tensor {
    const big = isBigIntDType(this.dtype);
    // NumPy: mean of any integer dtype yields f64; sum keeps the dtype.
    const isFloat = this.dtype === "f32" || this.dtype === "f64";
    const outDtype: DType = mean && !isFloat ? "f64" : this.dtype;

    if (axis === undefined) {
      const count = this.size;
      const out = Tensor.zeros([], { dtype: outDtype });
      if (big) {
        let acc = 0n;
        for (const off of this.elementOffsets()) {
          acc += this.data[off] as bigint;
        }
        if (mean) (out.data as Float64Array)[0] = Number(acc) / count;
        else (out.data as BigInt64Array)[0] = acc;
      } else {
        let acc = 0;
        for (const off of this.elementOffsets()) {
          acc += this.data[off] as number;
        }
        (out.data as Float64Array)[0] = mean ? acc / count : acc;
      }
      return out;
    }

    let ax = axis;
    if (ax < 0) ax += this.ndim;
    if (ax < 0 || ax >= this.ndim) {
      throw new RangeError(`axis ${axis} out of range for ndim ${this.ndim}`);
    }
    const outShape = this.shape.filter((_, i) => i !== ax);
    const reduceDim = this.shape[ax] as number;
    const reduceStride = this.strides[ax] as number;
    const out = Tensor.zeros(outShape, { dtype: outDtype });

    // Walk the non-reduced axes in C-order; for each position, accumulate
    // along the pinned axis via its stride.
    const leadShape = outShape;
    const leadStrides = this.strides.filter((_, i) => i !== ax);
    const leadSize = shapeSize(outShape);
    const index = new Array<number>(leadShape.length).fill(0);
    let base = this.offset;
    for (let i = 0; i < leadSize; i++) {
      if (big) {
        let acc = 0n;
        for (let j = 0; j < reduceDim; j++) {
          acc += this.data[base + j * reduceStride] as bigint;
        }
        if (mean) (out.data as Float64Array)[i] = Number(acc) / reduceDim;
        else (out.data as BigInt64Array)[i] = acc;
      } else {
        let acc = 0;
        for (let j = 0; j < reduceDim; j++) {
          acc += this.data[base + j * reduceStride] as number;
        }
        (out.data as Float64Array)[i] = mean ? acc / reduceDim : acc;
      }
      for (let a = leadShape.length - 1; a >= 0; a--) {
        index[a] = (index[a] as number) + 1;
        base += leadStrides[a] as number;
        if ((index[a] as number) < (leadShape[a] as number)) break;
        index[a] = 0;
        base -= (leadShape[a] as number) * (leadStrides[a] as number);
      }
    }
    return out;
  }

  // ---- .npy I/O -------------------------------------------------------------

  /** Serialize to NPY v1.0 bytes (packs a non-contiguous view first). */
  toNpy(): Uint8Array {
    const packed = this.contiguous();
    return serializeNpy({
      data:
        packed.data.length === packed.size
          ? packed.data
          : (packed.data.slice(
              packed.offset,
              packed.offset + packed.size,
            ) as AnyTypedArray),
      shape: [...packed.shape],
      dtype: packed.dtype,
    });
  }

  static fromNpy(bytes: Uint8Array): Tensor {
    const { data, shape, dtype } = parseNpy(bytes);
    return Tensor.fromTypedArray(data, shape, { dtype });
  }
}
