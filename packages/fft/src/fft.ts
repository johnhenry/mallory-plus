/**
 * fft/ifft/fftPadded/rfft/irfft (issue #40) — a fresh, tensor-shaped radix-2
 * Cooley-Tukey implementation over flat Float64Array real/imag pairs, NOT
 * routing through adapter-math's existing boxed-ComplexNumber-based FFT
 * bridge (#33) at this hot path -- matches tensor-compile's own "stays
 * dependency-free of @johnhenry/math" precedent for kernel-shaped code.
 * @johnhenry/math's FFT class serves as the differential-test oracle instead
 * (devDependency only, same pattern as the erf cross-check test, #34).
 *
 * Reference-speed JS, no WASM kernel in v1 -- same framing as
 * adapter-math's linalg.ts and this repo's other "reference now, native
 * later" precedents.
 */
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { ComplexTensor } from "./complex-tensor.ts";

function isPowerOfTwo(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0;
}

/** In-place iterative radix-2 Cooley-Tukey (bit-reversal permutation, then butterfly passes). `invert` runs the inverse transform (and normalizes by 1/n). */
function transform(re: Float64Array, im: Float64Array, invert: boolean): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i] as number;
      re[i] = re[j] as number;
      re[j] = tr;
      const ti = im[i] as number;
      im[i] = im[j] as number;
      im[j] = ti;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((2 * Math.PI) / len) * (invert ? 1 : -1);
    const wlenR = Math.cos(ang);
    const wlenI = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let wr = 1;
      let wi = 0;
      const half = len / 2;
      for (let j = 0; j < half; j++) {
        const uR = re[i + j] as number;
        const uI = im[i + j] as number;
        const vR = (re[i + j + half] as number) * wr - (im[i + j + half] as number) * wi;
        const vI = (re[i + j + half] as number) * wi + (im[i + j + half] as number) * wr;
        re[i + j] = uR + vR;
        im[i + j] = uI + vI;
        re[i + j + half] = uR - vR;
        im[i + j + half] = uI - vI;
        const nwr = wr * wlenR - wi * wlenI;
        wi = wr * wlenI + wi * wlenR;
        wr = nwr;
      }
    }
  }
  if (invert) {
    for (let i = 0; i < n; i++) {
      re[i] = (re[i] as number) / n;
      im[i] = (im[i] as number) / n;
    }
  }
}

// `transform` mutates `re`/`im` in place, and `contiguous()` returns the
// SAME Tensor (aliasing its live backing store) when the input is already
// packed -- so `.data` must still be defensively copied here (unlike the
// other toArray() sites in this repo, which are read-only). `Float64Array.
// from(...)` over a flat typed array is still the fix: it drops toArray()'s
// expensive boxed-nested-array walk while keeping the copy `transform`
// needs to avoid mutating the caller's tensor storage.
function toFlatParts(t: ComplexTensor): { re: Float64Array; im: Float64Array } {
  return {
    re: Float64Array.from(t.real.contiguous().data as Float64Array),
    im: Float64Array.from(t.imag.contiguous().data as Float64Array),
  };
}

function fromFlatParts(re: Float64Array, im: Float64Array): ComplexTensor {
  const n = re.length;
  return ComplexTensor.fromParts(
    Tensor.fromTypedArray(re, [n], { dtype: "f64" }),
    Tensor.fromTypedArray(im, [n], { dtype: "f64" }),
  );
}

/** Fast Fourier transform. `input` must be 1-D with a power-of-two length (see {@link fftPadded} otherwise). */
export function fft(input: ComplexTensor): ComplexTensor {
  if (input.shape.length !== 1) throw new RangeError("fft: v1 supports 1-D ComplexTensor only");
  if (!isPowerOfTwo(input.size)) {
    throw new RangeError(`fft: input length must be a power of two (got ${input.size}); see fftPadded.`);
  }
  const { re, im } = toFlatParts(input);
  transform(re, im, false);
  return fromFlatParts(re, im);
}

/** Inverse fast Fourier transform. `input` must be 1-D with a power-of-two length. */
export function ifft(input: ComplexTensor): ComplexTensor {
  if (input.shape.length !== 1) throw new RangeError("ifft: v1 supports 1-D ComplexTensor only");
  if (!isPowerOfTwo(input.size)) {
    throw new RangeError(`ifft: input length must be a power of two (got ${input.size}).`);
  }
  const { re, im } = toFlatParts(input);
  transform(re, im, true);
  return fromFlatParts(re, im);
}

/** FFT with the input zero-padded up to the next power of two — no length restriction. */
export function fftPadded(input: ComplexTensor): ComplexTensor {
  if (input.shape.length !== 1) throw new RangeError("fftPadded: v1 supports 1-D ComplexTensor only");
  const n0 = input.size;
  let n = 1;
  while (n < n0) n <<= 1;
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  re.set(input.real.contiguous().data as Float64Array);
  im.set(input.imag.contiguous().data as Float64Array);
  transform(re, im, false);
  return fromFlatParts(re, im);
}

// ---- fft2/ifft2 + fftshift/ifftshift (issue #69) ---------------------------
//
// The one bridge between @johnhenry/math-plus-fft and @johnhenry/math-plus-image that didn't exist:
// no 2-D FFT, and critically no fftshift -- the function everyone pairs
// with a 2-D FFT for spectral-image work (centering the zero-frequency
// component).

/**
 * FFT each 1-D line varying along `fftAxis`, holding the other axis fixed
 * -- iterate the OTHER axis, `select()` drops it (leaving a 1-D line along
 * `fftAxis`), transform, then `Tensor.stack` re-introduces the dropped
 * axis at the same position. No in-place tensor mutation needed (this
 * repo's Tensors have no public mutating API by design) -- pure
 * select+transform+stack.
 */
function fftAlongAxis(input: ComplexTensor, fftAxis: 0 | 1, inverse: boolean): ComplexTensor {
  const otherAxis = fftAxis === 0 ? 1 : 0;
  const count = input.shape[otherAxis] as number;
  const realLines: Tensor[] = [];
  const imagLines: Tensor[] = [];
  for (let i = 0; i < count; i++) {
    const line = ComplexTensor.fromParts(input.real.select(otherAxis, i), input.imag.select(otherAxis, i));
    const transformed = inverse ? ifft(line) : fft(line);
    realLines.push(transformed.real);
    imagLines.push(transformed.imag);
  }
  return ComplexTensor.fromParts(
    Tensor.stack(realLines, { axis: otherAxis }),
    Tensor.stack(imagLines, { axis: otherAxis }),
  );
}

/**
 * 2-D FFT: separable into two passes of 1-D `fft` (each axis independently
 * — order doesn't affect the result, the 2-D DFT is separable), reusing
 * `fft` rather than any new transform math. Both dimensions must be
 * powers of two (inherited from `fft`'s own requirement; see
 * {@link fftPadded} for the 1-D escape hatch — no 2-D padded variant yet).
 */
export function fft2(input: ComplexTensor): ComplexTensor {
  if (input.shape.length !== 2) throw new RangeError(`fft2: input must be 2-D, got ${input.shape.length}-D`);
  return fftAlongAxis(fftAlongAxis(input, 1, false), 0, false);
}

/** Inverse of {@link fft2}. */
export function ifft2(input: ComplexTensor): ComplexTensor {
  if (input.shape.length !== 2) throw new RangeError(`ifft2: input must be 2-D, got ${input.shape.length}-D`);
  return fftAlongAxis(fftAlongAxis(input, 1, true), 0, true);
}

function resolveAxes(shape: readonly number[], axes: number | readonly number[] | undefined): number[] {
  if (axes === undefined) return shape.map((_, i) => i);
  const list = Array.isArray(axes) ? axes : [axes];
  return list.map((a) => (a < 0 ? a + shape.length : a));
}

/**
 * FFT every 1-D line along `axis` of an arbitrary-rank tensor, holding
 * every other axis fixed — the n-D generalization of {@link fftAlongAxis}
 * (which is hardcoded to exactly 2 axes). Recursive descent: pick the
 * first axis that ISN'T `axis`, iterate its indices via `select` (which
 * drops that axis, shifting `axis`'s own index down by one if the dropped
 * axis was before it), recurse on the resulting rank-(n-1) tensor, then
 * `Tensor.stack` the results back at the dropped axis's position. Base
 * case: a 1-D tensor (axis must be 0 there) — apply {@link fft}/{@link ifft}
 * directly, same as {@link fftAlongAxis}'s own base transform.
 */
function fftAlongAxisN(input: ComplexTensor, axis: number, inverse: boolean): ComplexTensor {
  if (input.shape.length === 1) {
    return inverse ? ifft(input) : fft(input);
  }
  const pivotAxis = axis === 0 ? 1 : 0;
  const count = input.shape[pivotAxis] as number;
  const nextAxis = axis > pivotAxis ? axis - 1 : axis;
  const realSlices: Tensor[] = [];
  const imagSlices: Tensor[] = [];
  for (let i = 0; i < count; i++) {
    const slice = ComplexTensor.fromParts(input.real.select(pivotAxis, i), input.imag.select(pivotAxis, i));
    const transformed = fftAlongAxisN(slice, nextAxis, inverse);
    realSlices.push(transformed.real);
    imagSlices.push(transformed.imag);
  }
  return ComplexTensor.fromParts(
    Tensor.stack(realSlices, { axis: pivotAxis }),
    Tensor.stack(imagSlices, { axis: pivotAxis }),
  );
}

/**
 * n-D FFT (issue #84, upstream for the generalized Wang tile laboratory's
 * diffraction-spectrum machinery — 3-D for Wang cubes, `fft2` already
 * covers the 2-D case today). Separable, like `fft2`: one 1-D `fft` pass
 * per axis in `axes` (default: every axis), order doesn't affect the
 * result. Every transformed axis's length must be a power of two
 * (inherited from `fft`).
 */
export function fftn(input: ComplexTensor, axes?: number | readonly number[]): ComplexTensor {
  let out = input;
  for (const axis of resolveAxes(input.shape, axes)) out = fftAlongAxisN(out, axis, false);
  return out;
}

/** Inverse of {@link fftn}. */
export function ifftn(input: ComplexTensor, axes?: number | readonly number[]): ComplexTensor {
  let out = input;
  for (const axis of resolveAxes(input.shape, axes)) out = fftAlongAxisN(out, axis, true);
  return out;
}

/**
 * Circularly shift the zero-frequency component to the center of the
 * spectrum along `axes` (default: every axis) — `roll(floor(n/2))` per
 * axis, matching NumPy's `fftshift` exactly (including its even/odd-length
 * asymmetry with {@link ifftshift}). Works on any dimensionality, not just
 * 2-D.
 */
export function fftshift(input: ComplexTensor, axes?: number | readonly number[]): ComplexTensor {
  let real = input.real;
  let imag = input.imag;
  for (const axis of resolveAxes(input.shape, axes)) {
    const shift = Math.floor((input.shape[axis] as number) / 2);
    real = real.roll(shift, { axis });
    imag = imag.roll(shift, { axis });
  }
  return ComplexTensor.fromParts(real, imag);
}

/** Exact inverse of {@link fftshift} — `roll(-floor(n/2))` per axis. For EVEN-length axes this is identical to `fftshift`; for ODD-length axes it isn't (the center element differs), matching NumPy's own `fftshift`/`ifftshift` asymmetry. */
export function ifftshift(input: ComplexTensor, axes?: number | readonly number[]): ComplexTensor {
  let real = input.real;
  let imag = input.imag;
  for (const axis of resolveAxes(input.shape, axes)) {
    const shift = -Math.floor((input.shape[axis] as number) / 2);
    real = real.roll(shift, { axis });
    imag = imag.roll(shift, { axis });
  }
  return ComplexTensor.fromParts(real, imag);
}

/**
 * Real-valued convenience wrapper: `fft` of a real `Tensor` (zero-imaginary
 * input), returning the FULL N-point complex spectrum. This is NOT the
 * compute/memory-optimized N/2+1 half-spectrum a true `rfft` (e.g. NumPy's)
 * would produce — a documented v1 simplification, since a real input's
 * spectrum is Hermitian-symmetric and the upper half is fully redundant.
 * Revisit if that redundancy actually matters for a concrete workload.
 */
export function rfft(real: Tensor): ComplexTensor {
  if (real.shape.length !== 1) throw new RangeError("rfft: v1 supports 1-D Tensor only");
  return fft(ComplexTensor.fromReal(real));
}

/** Inverse of {@link rfft} — `ifft`, then discards the imaginary part (assumes the Hermitian symmetry a real-valued original signal's spectrum has, exactly what `rfft`'s output has). */
export function irfft(spectrum: ComplexTensor): Tensor {
  return ifft(spectrum).real;
}
