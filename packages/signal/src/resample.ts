/**
 * resamplePoly (issue #44) — rational up/down resampling via an
 * anti-aliasing FIR lowpass filter, matching `scipy.signal.resample_poly`'s
 * overall structure: reduce `up`/`down` by their GCD, design a windowed-
 * sinc lowpass filter at `1/max(up,down)` (normalized to the up-sampled
 * rate's Nyquist), then upsample (zero-stuff) -> filter (convolve) ->
 * downsample, trimming the filter's own group delay so the output stays
 * time-aligned with the input.
 *
 * Two real, DISCLOSED simplifications relative to scipy's default:
 * 1. **Hamming window, not Kaiser(beta=5.0).** scipy's default anti-aliasing
 *    filter uses a Kaiser window (needs the modified Bessel function I0,
 *    not implemented here); Hamming gives a similar-shaped, still genuinely
 *    anti-aliasing lowpass, just a different transition-band/stopband-
 *    ripple tradeoff. Per-sample output will NOT bit-match
 *    `scipy.signal.resample_poly`'s default -- tests below verify
 *    reconstruction QUALITY (does a resampled sinusoid still look like the
 *    same sinusoid at the new rate, within a real tolerance), which is the
 *    property that actually matters for resampling correctness, not exact
 *    coefficient agreement.
 * 2. **Direct upsample-then-convolve, not a true polyphase decomposition.**
 *    Produces the identical numerical result (no zero multiplications are
 *    skipped, just wastefully performed) but is NOT the efficient
 *    "polyphase" algorithm real-time/large-ratio use cases would want --
 *    a real performance gap, not a correctness one, disclosed here rather
 *    than silently claimed as fast.
 */
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { hammingWindow } from "./window.ts";

function gcdInt(a: number, b: number): number {
  let x = a;
  let y = b;
  while (y !== 0) {
    const t = y;
    y = x % y;
    x = t;
  }
  return x;
}

/** Ideal (infinite-length) lowpass filter impulse response at sample offset `n`, cutoff angular frequency `wc` radians/sample (0, pi). */
function idealLowpassImpulse(n: number, wc: number): number {
  if (n === 0) return wc / Math.PI;
  return Math.sin(wc * n) / (Math.PI * n);
}

/** A Hamming-windowed-sinc lowpass FIR filter, `2*halfLen+1` taps, unity DC gain, `cutoff` normalized to Nyquist=1. */
function designLowpassFir(cutoff: number, halfLen: number): Float64Array {
  const length = 2 * halfLen + 1;
  const window = hammingWindow(length);
  const wc = cutoff * Math.PI;
  const h = new Float64Array(length);
  for (let i = 0; i < length; i++) {
    const nRel = i - halfLen;
    h[i] = idealLowpassImpulse(nRel, wc) * (window[i] as number);
  }
  let sum = 0;
  for (const v of h) sum += v;
  for (let i = 0; i < length; i++) h[i] = (h[i] as number) / sum;
  return h;
}

export function resamplePoly(signal: Tensor, up: number, down: number): Tensor {
  if (signal.shape.length !== 1) throw new RangeError("resamplePoly: v1 supports 1-D Tensor only");
  if (!Number.isInteger(up) || up < 1) throw new RangeError(`resamplePoly: up must be a positive integer, got ${up}`);
  if (!Number.isInteger(down) || down < 1) throw new RangeError(`resamplePoly: down must be a positive integer, got ${down}`);

  // Read-only below (only `.slice()`d into fresh arrays), so no defensive
  // copy is needed even when `.data` aliases `signal`'s own storage.
  const x = signal.contiguous().data as Float64Array;
  const g = gcdInt(up, down);
  const u = up / g;
  const d = down / g;
  const nOut = Math.ceil((x.length * up) / down);

  if (u === 1 && d === 1) {
    return Tensor.fromTypedArray(x.slice(0, nOut), [nOut], { dtype: "f64" });
  }

  const maxRate = Math.max(u, d);
  const halfLen = 10 * maxRate;
  const cutoff = 1 / maxRate;
  const h = designLowpassFir(cutoff, halfLen);
  const filterLen = h.length;
  // Interpolation gain: compensates for the average-power drop introduced by zero-stuffing (matches scipy's `h *= up`).
  for (let i = 0; i < filterLen; i++) h[i] = (h[i] as number) * u;

  const xUpLen = x.length * u;
  const xUp = new Float64Array(xUpLen);
  for (let i = 0; i < x.length; i++) xUp[i * u] = x[i] as number;

  const convLen = xUpLen + filterLen - 1;
  const conv = new Float64Array(convLen);
  for (let i = 0; i < xUpLen; i++) {
    const xv = xUp[i] as number;
    if (xv === 0) continue;
    for (let j = 0; j < filterLen; j++) conv[i + j] = (conv[i + j] as number) + xv * (h[j] as number);
  }

  const out = new Float64Array(nOut);
  for (let i = 0; i < nOut; i++) {
    const idx = i * d + halfLen; // trims the symmetric FIR's own group delay (halfLen samples)
    out[i] = idx < convLen ? (conv[idx] as number) : 0;
  }
  return Tensor.fromTypedArray(out, [nOut], { dtype: "f64" });
}
