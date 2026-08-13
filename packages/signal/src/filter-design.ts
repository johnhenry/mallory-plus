/**
 * butter (issue #44) — digital Butterworth filter design, `output: "sos"`
 * only (the numerically-stable form `sosFilter` consumes). Matches
 * `scipy.signal.butter(N, Wn, btype, output="sos", fs=None)`'s algorithm
 * exactly: analog Butterworth lowpass prototype (`buttap`) -> lowpass/
 * highpass frequency transform (`lp2lp_zpk`/`lp2hp_zpk`) with bilinear-
 * pre-warped cutoff -> bilinear transform (`bilinear_zpk`) -> pole/zero
 * pairing into second-order sections (`zpk2sos`).
 *
 * v1 scope: `btype: "lowpass" | "highpass"` only -- `"bandpass"`/
 * `"bandstop"` need a second frequency transform (`lp2bp_zpk`/`lp2bs_zpk`)
 * this module doesn't implement yet (real, documented gap, not silently
 * dropped). `Wn` is always the normalized convention (`fs=None`): a value
 * in `(0, 1)` where `1` is the Nyquist frequency, matching scipy's default
 * when `fs` isn't passed.
 *
 * The `zpk2sos` step here is a SPECIALIZED pairing, not scipy's fully
 * general one: Butterworth lowpass/highpass output has a very constrained
 * shape (all N digital zeros equal the same real value -- `-1` for
 * lowpass, `+1` for highpass -- and poles are complex-conjugate pairs plus,
 * for odd N, exactly one real pole), so pairing is unambiguous. This
 * produces a mathematically equivalent SOS cascade to scipy's, but not
 * necessarily identical section ORDERING/coefficients -- differential
 * tests below verify end-to-end filtering behavior (via `sosFilter`)
 * against scipy's own `sosfilt`, not byte-for-byte coefficient equality,
 * which is the property that actually matters.
 */
import type { Sos, SosSection } from "./sos-filter.ts";

export type FilterType = "lowpass" | "highpass";

export interface ButterOptions {
  readonly btype: FilterType;
}

interface Complex {
  readonly re: number;
  readonly im: number;
}

function c(re: number, im = 0): Complex {
  return { re, im };
}

/** Analog Butterworth lowpass prototype poles (scipy's `buttap`): unit-magnitude poles at angles `pi*(2k+N-1)/(2N) + pi/2`, all in the left half-plane. Gain 1, no zeros. */
function buttap(order: number): { poles: Complex[] } {
  const poles: Complex[] = [];
  for (let k = -order + 1; k <= order - 1; k += 2) {
    const angle = (Math.PI * k) / (2 * order);
    // p = -exp(1j*angle)
    poles.push(c(-Math.cos(angle), -Math.sin(angle)));
  }
  return { poles };
}

/** Digital Butterworth filter design, SOS output only. `order` >= 1, `wn` in (0, 1) (normalized to Nyquist=1). */
export function butter(order: number, wn: number, options: ButterOptions): Sos {
  if (!Number.isInteger(order) || order < 1) throw new RangeError(`butter: order must be a positive integer, got ${order}`);
  if (!(wn > 0 && wn < 1)) throw new RangeError(`butter: wn must be in (0, 1) (normalized to Nyquist=1), got ${wn}`);

  const fs = 2.0; // scipy's internal digital-filter reference (matches Wn's Nyquist=1 convention)
  const warped = 4 * Math.tan((Math.PI * wn) / 2); // 2*fs*tan(pi*Wn/fs) with fs=2

  const { poles: protoPoles } = buttap(order);

  // Lowpass/highpass frequency transform (analog domain).
  let zeros: Complex[];
  let poles: Complex[];
  let gain: number;
  if (options.btype === "lowpass") {
    zeros = [];
    poles = protoPoles.map((p) => c(warped * p.re, warped * p.im));
    gain = warped ** order; // k=1 * warped^degree, degree = order - 0
  } else {
    // highpass: z_hp = warped / p (poles), extra zeros at s=0 for the degree difference; k_hp = k * real(prod(-z)/prod(-p)) with z=[] (prod(-z) over empty = 1).
    poles = protoPoles.map((p) => cDiv(c(warped), p));
    zeros = protoPoles.map(() => c(0, 0));
    let prodNegP = c(1, 0);
    for (const p of protoPoles) prodNegP = cMul(prodNegP, c(-p.re, -p.im));
    gain = 1 / prodNegP.re; // prodNegP is real for a real-coefficient prototype (conjugate pairs cancel imaginary parts)
  }

  // Bilinear transform (analog -> digital), fs2 = 2*fs = 4.
  const fs2 = 2 * fs;
  const degree = poles.length - zeros.length; // extra digital zeros at z=-1
  const digitalPoles = poles.map((p) => cDiv(c(fs2 + p.re, p.im), c(fs2 - p.re, -p.im)));
  const digitalZeros: Complex[] = zeros.map((z) => cDiv(c(fs2 + z.re, z.im), c(fs2 - z.re, -z.im)));
  for (let i = 0; i < degree; i++) digitalZeros.push(c(-1, 0));

  let prodNumer = c(1, 0);
  for (const z of zeros) prodNumer = cMul(prodNumer, c(fs2 - z.re, -z.im));
  let prodDenom = c(1, 0);
  for (const p of poles) prodDenom = cMul(prodDenom, c(fs2 - p.re, -p.im));
  const digitalGain = gain * cDiv(prodNumer, prodDenom).re;

  return zpk2sos(digitalZeros, digitalPoles, digitalGain);
}

function cMul(a: Complex, b: Complex): Complex {
  return c(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

function cDiv(a: Complex, b: Complex): Complex {
  const denom = b.re * b.re + b.im * b.im;
  return c((a.re * b.re + a.im * b.im) / denom, (a.im * b.re - a.re * b.im) / denom);
}

const EPS = 1e-9;

/**
 * Group poles into second-order sections (complex-conjugate pairs, plus at
 * most one leftover real pole for odd order) and pair each with the right
 * COUNT of zeros -- NOT by matching real/complex-ness between zeros and
 * poles (a real zpk2sos would need to, but ours doesn't): every digital
 * zero `butter` produces is real AND identical (`-1` for lowpass, `+1` for
 * highpass -- confirmed by construction in `butter` above), so which
 * specific zero "index" feeds which section is irrelevant; only the COUNT
 * consumed per section (2 for a pole pair, 1 for a leftover real pole)
 * matters. See this module's own doc comment for why this specialized
 * shortcut is valid for Butterworth's constrained lowpass/highpass output.
 */
function zpk2sos(zeros: readonly Complex[], poles: readonly Complex[], gain: number): Sos {
  if (zeros.length !== poles.length) {
    throw new Error(`zpk2sos: expected equal zero/pole counts (Butterworth-shaped input), got ${zeros.length} zeros, ${poles.length} poles`);
  }
  const z0 = (zeros[0] ?? c(0)).re;
  if (zeros.some((z) => Math.abs(z.im) > EPS || Math.abs(z.re - z0) > EPS)) {
    throw new Error("zpk2sos: expected all zeros real and identical (Butterworth lowpass/highpass shape)");
  }

  const poleComplex = poles.filter((p) => Math.abs(p.im) > EPS);
  const poleReal = poles.filter((p) => Math.abs(p.im) <= EPS);
  const polePairs = poleComplex
    .filter((p) => p.im > 0)
    .sort((a, b) => a.im - b.im)
    .map((p): [Complex, Complex] => [p, c(p.re, -p.im)]);

  const sections: SosSection[] = [];
  for (const [p] of polePairs) {
    // Numerator (z-z0)^2, denominator (z-p)(z-conj(p)).
    sections.push([1, -2 * z0, z0 * z0, 1, -2 * p.re, p.re * p.re + p.im * p.im]);
  }
  for (const p of poleReal) {
    // Numerator (z-z0), denominator (z-p).
    sections.push([1, -z0, 0, 1, -p.re, 0]);
  }

  if (sections.length === 0) throw new Error("zpk2sos: no sections produced");
  const first = sections[0] as SosSection;
  sections[0] = [first[0] * gain, first[1] * gain, first[2] * gain, first[3], first[4], first[5]];
  return sections;
}
