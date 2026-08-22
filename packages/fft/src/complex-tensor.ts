/**
 * ComplexTensor (issue #40) — the boundary contract decided in
 * docs/PLAN.md §6.3's "Complex-number dependency for fft" note: storage
 * stays split TypedArrays (a `real` Tensor and an `imag` Tensor, same
 * shape/dtype), `at()`/`item()` return a @johnhenry/math `ComplexNumber`, and
 * constructors accept `ComplexNumber[]` at the edges. Boxed at edges, flat
 * in kernels — consistent with non-goals 3 and 9.
 *
 * Deliberately NOT a full arithmetic/broadcasting tensor type in v1 — no
 * `.add()`/`.mul()`/etc. Just enough to be `fft`/`ifft`'s natural input and
 * output type and to round-trip against `ComplexNumber[]` at the edges.
 * Real arithmetic on complex tensors is separate, unscoped future work.
 */
import { ComplexNumber } from "@johnhenry/math-plus-scalar-types";
import { Tensor, type DType, type Shape } from "@johnhenry/math-plus-tensor-core";

export class ComplexTensor {
  readonly real: Tensor;
  readonly imag: Tensor;

  private constructor(real: Tensor, imag: Tensor) {
    if (real.shape.length !== imag.shape.length || real.shape.some((d, i) => d !== imag.shape[i])) {
      throw new RangeError(`ComplexTensor: real shape [${real.shape}] and imag shape [${imag.shape}] must match`);
    }
    if (real.dtype !== imag.dtype) {
      throw new TypeError(`ComplexTensor: real dtype "${real.dtype}" and imag dtype "${imag.dtype}" must match`);
    }
    this.real = real;
    this.imag = imag;
  }

  /** Wrap an existing `real`/`imag` Tensor pair (same shape/dtype) — no copy. */
  static fromParts(real: Tensor, imag: Tensor): ComplexTensor {
    return new ComplexTensor(real, imag);
  }

  /** All-zero-imaginary ComplexTensor wrapping a real Tensor — no copy of `real`. */
  static fromReal(real: Tensor): ComplexTensor {
    return new ComplexTensor(real, Tensor.zeros(real.shape, { dtype: real.dtype }));
  }

  /** A 1-D ComplexTensor from a flat `ComplexNumber[]` — the boxed-at-the-edge entry point. */
  static fromComplexArray(values: readonly ComplexNumber[], options: { dtype?: "f32" | "f64" } = {}): ComplexTensor {
    const dtype = options.dtype ?? "f64";
    const real = Tensor.from(
      values.map((z) => z.re),
      { dtype },
    );
    const imag = Tensor.from(
      values.map((z) => z.im),
      { dtype },
    );
    return new ComplexTensor(real, imag);
  }

  static zeros(shape: Shape, options: { dtype?: DType } = {}): ComplexTensor {
    const dtype = options.dtype ?? "f64";
    return new ComplexTensor(Tensor.zeros(shape, { dtype }), Tensor.zeros(shape, { dtype }));
  }

  get shape(): Shape {
    return this.real.shape;
  }

  get size(): number {
    return this.real.size;
  }

  get dtype(): DType {
    return this.real.dtype;
  }

  /** Boxed element read by multi-index — the `at()` half of the decided boundary contract. */
  at(...indices: number[]): ComplexNumber {
    return new ComplexNumber(this.real.at(...indices) as number, this.imag.at(...indices) as number);
  }

  /** The single element of a size-1 ComplexTensor. */
  item(): ComplexNumber {
    return new ComplexNumber(this.real.item() as number, this.imag.item() as number);
  }

  /** Flatten to a plain `ComplexNumber[]` — the boxed-at-the-edge exit point (row-major order for ndim > 1). */
  toComplexArray(): ComplexNumber[] {
    // `.data` (after `.contiguous()`) is already the flat, row-major storage
    // for any ndim -- no need to build a nested array via toArray() and
    // re-flatten it with `.flat(Infinity)`. Read-only here (never mutated),
    // so no defensive copy is needed even when `.data` aliases the source
    // tensor's own backing store.
    const reFlat = this.real.contiguous().data;
    const imFlat = this.imag.contiguous().data;
    const out = new Array<ComplexNumber>(reFlat.length);
    for (let i = 0; i < reFlat.length; i++) {
      out[i] = new ComplexNumber(reFlat[i] as number, imFlat[i] as number);
    }
    return out;
  }
}
