/**
 * correlate (issue #70) — cross-correlation, `convolve`'s direct dual (no
 * kernel flip). SciPy's own convention: `correlate(a, v) == convolve(a,
 * reverse(v))` for real signals (no conjugate needed) — verified
 * numerically against `scipy.signal.correlate` across all three modes
 * before writing this, not assumed. Built entirely on the existing
 * `convolve1D`, not a second O(n*m) loop.
 */
import { Tensor } from "mallory-tensor-core";
import { applyTimeDomainOp, convolve1D, type ConvolveMode, type ConvolveOptions } from "./convolve.ts";

/** Cross-correlation of two plain `Float64Array`s: `correlate(a, b) === convolve(a, reverse(b))`. */
export function correlate1D(a: Float64Array, b: Float64Array, mode: ConvolveMode = "full"): Float64Array {
  const reversed = new Float64Array(b.length);
  for (let i = 0; i < b.length; i++) reversed[i] = b[b.length - 1 - i] as number;
  return convolve1D(a, reversed, mode);
}

/** Cross-correlation of a 1-D Tensor, or each row/column of a 2-D `[N, T]` Tensor, with a 1-D Tensor — same shape/axis contract as {@link convolve} (they share the batching logic via `applyTimeDomainOp`). */
export function correlate(input: Tensor, other: Tensor, options: ConvolveOptions = {}): Tensor {
  const mode = options.mode ?? "full";
  return applyTimeDomainOp("correlate", (a, b) => correlate1D(a, b, mode), input, other, options);
}
