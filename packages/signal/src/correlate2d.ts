/**
 * correlate2D (issue #84) — true 2-D cross-correlation via FFT, upstream
 * for the generalized Wang tile laboratory's autocorrelation-surface
 * analysis (johnhenry/mallory-graph#92): the standard repetitivity
 * picture, bright peaks at every translation vector under which a tiling
 * approximately recurs. `correlate` (correlate.ts) is 1-D with row/column
 * BATCHING of a 2-D input, per its own doc comment — there is no genuine
 * 2-D correlation there.
 *
 * Same convention as the existing `correlate1D`: `correlate2D(a, b) ===
 * convolve2D(a, flip(b))` (no conjugate needed for real inputs) — this is
 * exactly what makes an FFT-based implementation cheap: convolution is a
 * spectral MULTIPLY (`ifft2(fft2(a) * fft2(b))`), so correlation is that
 * same multiply with `b` pre-flipped on both axes. Verified against
 * `scipy.signal.correlate2d(..., mode="full")` before writing this (see
 * test/correlate2d.test.ts's differential test) — full-mode output shape
 * `[Ma+Mb-1, Na+Nb-1]`, zero-padded internally to a power-of-two size per
 * axis (both `fft2`'s own requirement and what avoids circular-wraparound
 * aliasing, the standard "pad before FFT-convolve" technique).
 */
import { fft2, ComplexTensor, ifft2 } from "mallory-fft";
import { Tensor } from "mallory-tensor-core";

function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p *= 2;
  return p;
}

function zeroPad2D(t: Tensor, rows: number, cols: number): Tensor {
  const [r, c] = t.shape as [number, number];
  return t.pad([
    [0, rows - r],
    [0, cols - c],
  ]);
}

/**
 * "Full" 2-D cross-correlation of `a` and `b` (both real, 2-D): output
 * shape `[Ma+Mb-1, Na+Nb-1]`. Real-valued (the imaginary part of the
 * inverse FFT is discarded — mathematically ~0 for real inputs, up to
 * floating-point noise, matching `irfft`'s own "real input in, real
 * output out" contract elsewhere in this monorepo).
 */
export function correlate2D(a: Tensor, b: Tensor): Tensor {
  if (a.shape.length !== 2) throw new RangeError(`correlate2D: a must be 2-D, got shape [${a.shape}]`);
  if (b.shape.length !== 2) throw new RangeError(`correlate2D: b must be 2-D, got shape [${b.shape}]`);
  const [ma, na] = a.shape as [number, number];
  const [mb, nb] = b.shape as [number, number];
  const outRows = ma + mb - 1;
  const outCols = na + nb - 1;
  const padRows = nextPow2(outRows);
  const padCols = nextPow2(outCols);

  const flippedB = b.flip();
  const aPadded = zeroPad2D(a, padRows, padCols);
  const bPadded = zeroPad2D(flippedB, padRows, padCols);

  const aSpectrum = fft2(ComplexTensor.fromParts(aPadded, Tensor.zeros(aPadded.shape, { dtype: "f64" })));
  const bSpectrum = fft2(ComplexTensor.fromParts(bPadded, Tensor.zeros(bPadded.shape, { dtype: "f64" })));

  // Complex multiply: (ar+i*ai)(br+i*bi) = (ar*br - ai*bi) + i(ar*bi + ai*br)
  const productReal = aSpectrum.real.mul(bSpectrum.real).sub(aSpectrum.imag.mul(bSpectrum.imag));
  const productImag = aSpectrum.real.mul(bSpectrum.imag).add(aSpectrum.imag.mul(bSpectrum.real));

  const full = ifft2(ComplexTensor.fromParts(productReal, productImag));
  return full.real.slice({ end: outRows }, { end: outCols }).contiguous();
}
