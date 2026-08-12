/**
 * mallory-scalar-types — the single import point for mallory-math scalar
 * types inside Mallory Plus (docs/PLAN.md §B.1 of the integration plan).
 *
 * Boxed scalars appear only at tensor API edges (at()/item()/constructors),
 * never in tensor storage or kernels (non-goals 3 and 9). If the scalar
 * layer ever changes (e.g. accelerated scalars), only this package moves.
 */
export { ComplexNumber, Rational, Decimal } from "mallory-math";

// "Fraction" is math.js vocabulary; the Mallory implementation is Rational.
// Alias provided for adapter-mathjs familiarity only.
export { Rational as Fraction } from "mallory-math";

/**
 * `Interval` (issue #36): rigorous interval arithmetic `[lo, hi]` -- no
 * Tensor-boundary conversion of its own (it has no natural tensor-of-values
 * shape the way ComplexNumber does), so it's re-exported verbatim like the
 * scalars above rather than getting a matrix.ts-style converter. Useful as
 * a rounding-error bound oracle wherever an f32 (e.g. tensor-webgpu) result
 * needs to be checked against an f64 reference's known error bound, rather
 * than just a bit-for-bit/tolerance comparison -- see
 * packages/tensor-webgpu/test/fusion.test.ts for that usage.
 */
export { Interval } from "mallory-math";

/**
 * `Quaternion` (issue #37): rotation/3D-transform value type -- ready-made
 * if a 3D/graphics-adjacent tensor-webgpu use case ever shows up; no
 * concrete consumer yet, so this is a plain re-export like the scalars
 * above, not a Tensor-boundary converter (nothing to convert to/from until
 * a batched-quaternion tensor use case actually exists).
 */
export { Quaternion } from "mallory-math";

import { ComplexNumber } from "mallory-math";

/** Split-storage parts of a complex vector — the ComplexTensor edge format. */
export interface ComplexParts {
  real: Float64Array;
  imag: Float64Array;
}

/** Boxed → flat: extract split typed storage from boxed complex scalars. */
export function complexToParts(values: readonly ComplexNumber[]): ComplexParts {
  const real = new Float64Array(values.length);
  const imag = new Float64Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const z = values[i] as ComplexNumber;
    real[i] = z.re;
    imag[i] = z.im;
  }
  return { real, imag };
}

/** Flat → boxed: box split typed storage into ComplexNumber instances. */
export function partsToComplex(parts: ComplexParts): ComplexNumber[] {
  const { real, imag } = parts;
  if (real.length !== imag.length) {
    throw new RangeError(
      `real (${real.length}) and imag (${imag.length}) lengths differ`,
    );
  }
  const out = new Array<ComplexNumber>(real.length);
  for (let i = 0; i < real.length; i++) {
    out[i] = new ComplexNumber(real[i] as number, imag[i] as number);
  }
  return out;
}
