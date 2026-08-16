/**
 * butter (issue #44, extended in #90) — digital Butterworth filter design,
 * `output: "sos"` only (the numerically-stable form `sosFilter` consumes).
 * Matches `scipy.signal.butter(N, Wn, btype, output="sos", fs=None)`'s
 * algorithm exactly: analog Butterworth lowpass prototype (`buttap`) ->
 * frequency transform (`lp2lp_zpk`/`lp2hp_zpk`/`lp2bp_zpk`/`lp2bs_zpk`,
 * with bilinear-pre-warped cutoff(s)) -> bilinear transform
 * (`bilinear_zpk`) -> pole/zero pairing into second-order sections
 * (`zpk2sos`).
 *
 * `Wn` is always the normalized convention (`fs=None`): a value in `(0, 1)`
 * where `1` is the Nyquist frequency, matching scipy's default when `fs`
 * isn't passed. For `"bandpass"`/`"bandstop"`, `Wn` is a `[low, high]` pair
 * (both normalized, `low < high`) instead of a single cutoff -- expressed
 * as a discriminated union via function overloads (not a always-a-tuple
 * API) so the existing `"lowpass"`/`"highpass"` call shape stays exactly
 * as it was pre-#90, and the compiler enforces the right `Wn` shape per
 * `btype` at the call site.
 *
 * The `zpk2sos` step here is a GENERAL real-coefficient pairing (issue
 * #90), not scipy's own byte-for-byte "nearest" algorithm -- see its own
 * doc comment below for why a simpler grouping is both sufficient and
 * provably always produces a matching section count. Differential tests
 * verify end-to-end filtering behavior (via `sosFilter`) against scipy's
 * own `sosfilt`, not section-by-section coefficient equality, which this
 * module's own tests already established is the property that actually
 * matters (scipy's own pairing isn't fixed across orders/versions either).
 */
import type { Sos, SosSection } from "./sos-filter.ts";

export type FilterType = "lowpass" | "highpass" | "bandpass" | "bandstop";

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

/** `2*fs*tan(pi*Wn/fs)` with `fs=2` (scipy's own internal digital-filter reference, matching `Wn`'s Nyquist=1 convention) -- the bilinear-transform pre-warp, applied to each cutoff independently. */
function prewarp(wn: number): number {
  return 4 * Math.tan((Math.PI * wn) / 2);
}

export function butter(order: number, wn: number, options: { btype: "lowpass" | "highpass" }): Sos;
export function butter(order: number, wn: readonly [number, number], options: { btype: "bandpass" | "bandstop" }): Sos;
export function butter(order: number, wn: number | readonly [number, number], options: { btype: FilterType }): Sos {
  if (!Number.isInteger(order) || order < 1) throw new RangeError(`butter: order must be a positive integer, got ${order}`);

  const { poles: protoPoles } = buttap(order);
  const isBand = options.btype === "bandpass" || options.btype === "bandstop";

  let zeros: Complex[];
  let poles: Complex[];
  let gain: number;

  if (!isBand) {
    if (typeof wn !== "number") throw new RangeError(`butter: wn must be a single number for btype ${options.btype}`);
    if (!(wn > 0 && wn < 1)) throw new RangeError(`butter: wn must be in (0, 1) (normalized to Nyquist=1), got ${wn}`);
    const warped = prewarp(wn);
    if (options.btype === "lowpass") {
      zeros = [];
      poles = protoPoles.map((p) => c(warped * p.re, warped * p.im));
      gain = warped ** order; // k=1 * warped^degree, degree = order - 0
    } else {
      // highpass: z_hp = warped / p (poles), extra zeros at s=0 for the degree difference; k_hp = k * real(prod(-z)/prod(-p)) with z=[] (prod(-z) over empty = 1).
      poles = protoPoles.map((p) => cDiv(c(warped), p));
      zeros = protoPoles.map(() => c(0, 0));
      gain = 1 / prodNeg(protoPoles).re;
    }
  } else {
    if (!Array.isArray(wn) || wn.length !== 2) throw new RangeError(`butter: wn must be a [low, high] pair for btype ${options.btype}`);
    const [wnLo, wnHi] = wn as readonly [number, number];
    if (!(wnLo > 0 && wnLo < wnHi && wnHi < 1)) {
      throw new RangeError(`butter: wn must satisfy 0 < wn[0] < wn[1] < 1 (normalized to Nyquist=1), got [${wnLo}, ${wnHi}]`);
    }
    const warpedLo = prewarp(wnLo);
    const warpedHi = prewarp(wnHi);
    const bw = warpedHi - warpedLo;
    const wo = Math.sqrt(warpedLo * warpedHi);
    const degree = protoPoles.length; // relative degree of the lowpass prototype (order poles, 0 zeros)

    if (options.btype === "bandpass") {
      // lp2bp_zpk with z=[] (buttap has no zeros): p_lp = p*bw/2, then
      // duplicate+shift each pole to +-wo: p_bp = p_lp +- sqrt(p_lp^2 - wo^2).
      // z_bp is `degree` zeros at the origin (z_lp is empty, so the
      // "duplicate+shift" step contributes nothing); k_bp = k * bw^degree.
      const pLp = protoPoles.map((p) => c(p.re * (bw / 2), p.im * (bw / 2)));
      poles = [...pLp.map((p) => bandShift(p, wo, 1)), ...pLp.map((p) => bandShift(p, wo, -1))];
      zeros = protoPoles.map(() => c(0, 0));
      gain = bw ** degree;
    } else {
      // lp2bs_zpk with z=[]: p_hp = (bw/2)/p (inversion, not the bandpass
      // case's scaling), same duplicate+shift by +-wo. The `degree` zeros
      // that would sit at the origin for a highpass instead sit at +-j*wo
      // (the stopband's own center) -- "any zeros that were at infinity
      // moved to the center of the stopband". k_bs = k*real(prod(-z)/
      // prod(-p)) with z=[], identical formula shape to the highpass gain.
      const pHp = protoPoles.map((p) => cDiv(c(bw / 2), p));
      poles = [...pHp.map((p) => bandShift(p, wo, 1)), ...pHp.map((p) => bandShift(p, wo, -1))];
      zeros = [...Array.from({ length: degree }, () => c(0, wo)), ...Array.from({ length: degree }, () => c(0, -wo))];
      gain = 1 / prodNeg(protoPoles).re;
    }
  }

  // Bilinear transform (analog -> digital), fs2 = 2*fs = 4 (fs=2, scipy's own internal digital-filter reference).
  const fs2 = 4;
  const degree = poles.length - zeros.length; // extra digital zeros at z=-1 (any analog zeros "at infinity")
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

/** One child of `lp2bp_zpk`/`lp2bs_zpk`'s "duplicate and shift by +-wo" step: `p +- sqrt(p^2 - wo^2)`. `sign` is `+1` for the first duplicate, `-1` for the second. */
function bandShift(p: Complex, wo: number, sign: 1 | -1): Complex {
  const discriminant = cSub(cMul(p, p), c(wo * wo));
  const root = cSqrt(discriminant);
  return sign === 1 ? cAdd(p, root) : cSub(p, root);
}

/** `prod(-p)` over a list of poles/zeros -- the common factor in both the highpass and bandstop gain formulas (both have an empty analog zero list `z=[]` at the point this is used, so `k = k_proto * real(prod(-z)/prod(-p)) = real(1/prod(-p))`). */
function prodNeg(values: readonly Complex[]): Complex {
  let prod = c(1, 0);
  for (const v of values) prod = cMul(prod, c(-v.re, -v.im));
  return prod;
}

function cMul(a: Complex, b: Complex): Complex {
  return c(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

function cDiv(a: Complex, b: Complex): Complex {
  const denom = b.re * b.re + b.im * b.im;
  return c((a.re * b.re + a.im * b.im) / denom, (a.im * b.re - a.re * b.im) / denom);
}

function cAdd(a: Complex, b: Complex): Complex {
  return c(a.re + b.re, a.im + b.im);
}

function cSub(a: Complex, b: Complex): Complex {
  return c(a.re - b.re, a.im - b.im);
}

/** Principal square root of a complex number (`numpy`/`cmath`'s own branch convention: a negative real number's root has a NON-NEGATIVE imaginary part). `im >= 0` (not `Math.sign`, which maps `+0` to `0`) is what keeps that convention correct for a real negative input (`im` exactly `0`). */
function cSqrt(z: Complex): Complex {
  const r = Math.hypot(z.re, z.im);
  const re = Math.sqrt((r + z.re) / 2);
  const imMag = Math.sqrt(Math.max(0, (r - z.re) / 2));
  return c(re, z.im >= 0 ? imMag : -imMag);
}

const EPS = 1e-9;

/**
 * General real-coefficient zpk-to-SOS pairing (issue #90) -- replaces the
 * old Butterworth-lowpass/highpass-only shortcut (which assumed every
 * zero was real AND identical, a shape bandpass/bandstop's zeros don't
 * have: e.g. bandpass digital zeros split between `z=+1` and `z=-1`,
 * bandstop's between two complex-conjugate values).
 *
 * The key fact that makes a SIMPLE (not scipy's own "nearest worst-pole-
 * first" algorithm) general grouping provably correct: partition zeros
 * into complex-conjugate-pair representatives plus leftover real values
 * (greedily paired two-at-a-time, at most one real value left over), and
 * do the same for poles. Since real-coefficient inputs have
 * `2*complexPairs + reals = totalCount` on EACH side, and `zeros.length
 * === poles.length` is enforced below, `reals` on the zero side and the
 * pole side always have the SAME PARITY -- so `ceil(reals/2)` (the real
 * group count) matches on both sides too, meaning the total GROUP COUNT
 * (complex pairs + real groups) is always IDENTICAL between zeros and
 * poles. Zipping the two group lists index-by-index therefore always
 * produces a valid cascade: every zero and every pole is consumed exactly
 * once, and each section's own numerator/denominator are independently
 * real-coefficient (a lone unpaired complex value never happens, since
 * conjugates are grouped together by construction) -- a section's
 * zero-group and pole-group don't need to match in ARITY (a 2-pole
 * section can pair with a single leftover real zero, expressed as a
 * degree-1 numerator in the fixed 3-coefficient SOS slot, same as this
 * module's pre-#90 real-pole/real-zero case already did) since `sosFilter`
 * only cares about the 6 raw coefficients, not that they came from
 * "matching" zero/pole counts.
 *
 * For lowpass/highpass's own shape (all zeros real and identical), this
 * produces byte-identical section coefficients to the pre-#90 specialized
 * version -- verified by inspection, not just behaviorally: pairing N
 * identical real zeros two-at-a-time and taking `(z-z0)(z-z0)` /
 * `(z-z0)` is exactly the old code's own `(z-z0)^2`/`(z-z0)` formulas.
 */
function zpk2sos(zeros: readonly Complex[], poles: readonly Complex[], gain: number): Sos {
  if (zeros.length !== poles.length) {
    throw new Error(`zpk2sos: expected equal zero/pole counts (post-bilinear-transform invariant), got ${zeros.length} zeros, ${poles.length} poles`);
  }
  if (zeros.length === 0) return [[gain, 0, 0, 1, 0, 0]];

  const zeroGroups = groupConjugatePairs(zeros, "zeros");
  const poleGroups = groupConjugatePairs(poles, "poles");
  if (zeroGroups.length !== poleGroups.length) {
    // Guaranteed not to happen for real-coefficient input (see doc comment) -- a defensive check, not a reachable branch for this module's own callers.
    throw new Error(`zpk2sos: internal error -- zero group count (${zeroGroups.length}) != pole group count (${poleGroups.length})`);
  }

  const sections: SosSection[] = zeroGroups.map((zg, i) => {
    const pg = poleGroups[i] as readonly Complex[];
    const [b0, b1, b2] = groupToQuadratic(zg);
    const [a0, a1, a2] = groupToQuadratic(pg);
    return [b0, b1, b2, a0, a1, a2];
  });

  const first = sections[0] as SosSection;
  sections[0] = [first[0] * gain, first[1] * gain, first[2] * gain, first[3], first[4], first[5]];
  return sections;
}

/**
 * Splits a real-coefficient value list into groups of at most 2: one
 * group per complex-conjugate pair (the positive-imaginary representative
 * only -- its conjugate is implied and never separately grouped), plus
 * leftover real values paired two-at-a-time (at most one real group of
 * size 1). `label` is only for the error message.
 */
function groupConjugatePairs(values: readonly Complex[], label: string): Complex[][] {
  const complexReps = values.filter((v) => v.im > EPS);
  const complexConjugates = values.filter((v) => v.im < -EPS).length;
  const reals = values.filter((v) => Math.abs(v.im) <= EPS).map((v) => v.re);
  if (complexReps.length !== complexConjugates) {
    throw new Error(`zpk2sos: ${label} aren't closed under conjugation (${complexReps.length} with positive imaginary part, ${complexConjugates} with negative) -- not a real-coefficient system`);
  }

  const groups: Complex[][] = complexReps.map((v) => [v]);
  for (let i = 0; i < reals.length; i += 2) {
    const a = reals[i] as number;
    const b = reals[i + 1];
    groups.push(b === undefined ? [c(a)] : [c(a), c(b)]);
  }
  return groups;
}

/** A group's real quadratic factor: `(z-a)(z-conj(a)) = z^2 - 2*re(a)*z + |a|^2` for a complex-conjugate-pair representative, `(z-a)(z-b) = z^2 - (a+b)*z + a*b` for two real values, or `(z-a) = z - a` (degree-1, in the fixed 3-slot form with the `z^-2` coefficient `0`) for a single leftover real value. */
function groupToQuadratic(group: readonly Complex[]): readonly [number, number, number] {
  if (group.length === 1) {
    const v = group[0] as Complex;
    if (Math.abs(v.im) > EPS) return [1, -2 * v.re, v.re * v.re + v.im * v.im];
    return [1, -v.re, 0];
  }
  const [a, b] = group as [Complex, Complex];
  return [1, -(a.re + b.re), a.re * b.re];
}
