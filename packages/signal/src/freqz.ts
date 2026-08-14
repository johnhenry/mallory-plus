/**
 * freqz (issue #70) — evaluate an SOS filter's frequency response `H(e^jw)`
 * at `worN` points, direct evaluation of the existing `SosSection`
 * coefficients (no new filter representation). `butter()` DESIGNS a
 * filter; this is the natural next step — inspecting what it actually
 * does in the frequency domain, which was previously impossible.
 *
 * Frequency grid matches `scipy.signal.sosfreqz`'s default (`whole=false`,
 * `fs=2*pi`) exactly: `w[i] = i * pi / worN` for `i` in `[0, worN)` —
 * verified numerically against scipy before writing this, not assumed.
 */
import { ComplexNumber } from "mallory-scalar-types";
import type { Sos } from "./sos-filter.ts";

export interface FreqzResult {
  /** Radians/sample, `[0, pi)`. */
  readonly frequencies: readonly number[];
  /** Complex frequency response `H(e^{jw})` at each frequency. */
  readonly response: readonly ComplexNumber[];
}

/** `c0 + c1*z + c2*z^2` at complex `z`. */
function complexPoly(c0: number, c1: number, c2: number, z: ComplexNumber): ComplexNumber {
  return new ComplexNumber(c0).add(z.multiply(c1)).add(z.multiply(z).multiply(c2));
}

export function freqz(sos: Sos, options: { worN?: number } = {}): FreqzResult {
  if (sos.length === 0) throw new RangeError("freqz: sos must have at least one section");
  const worN = options.worN ?? 512;
  const frequencies = Array.from({ length: worN }, (_, i) => (i * Math.PI) / worN);
  const response = frequencies.map((w) => {
    // z^-1 = e^{-jw}; the SOS numerator/denominator are polynomials in z^-1.
    const zInv = new ComplexNumber(Math.cos(-w), Math.sin(-w));
    let h = new ComplexNumber(1, 0);
    for (const [b0, b1, b2, a0, a1, a2] of sos) {
      if (a0 === 0) throw new RangeError("freqz: a section's a0 coefficient must be nonzero");
      const num = complexPoly(b0 / a0, b1 / a0, b2 / a0, zInv);
      const den = complexPoly(1, a1 / a0, a2 / a0, zInv);
      h = h.multiply(num).divide(den);
    }
    return h;
  });
  return { frequencies, response };
}
