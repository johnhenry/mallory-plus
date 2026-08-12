/**
 * FFT bridge (issue #33). mallory-math's `FFT` class operates on plain
 * `(ComplexNumber | number)[]` arrays; this wraps it in the split-storage
 * `{real, imag}` Float64Array convention `mallory-scalar-types`'s
 * `complexToParts`/`partsToComplex` already established for tensor edges,
 * so the same shape works for a real-only signal (`imag` omitted) or a
 * full complex one.
 *
 * v1 is a thin, reference-speed wrapper -- no native WASM FFT kernel yet,
 * same "reference-speed now, native kernels later" framing as this
 * package's own `linalg.ts`. No batching/2-D FFT (mallory-math's own `FFT`
 * doesn't have it either).
 */
import { ComplexNumber, FFT } from "mallory-math";

export interface ComplexSignal {
  readonly real: Float64Array;
  readonly imag: Float64Array;
}

/** Wrap a real-only signal as a {@link ComplexSignal} (zero imaginary part) -- for {@link convolve}, which takes two full signals. */
export function realSignal(data: Float64Array): ComplexSignal {
  return { real: data, imag: new Float64Array(data.length) };
}

function toComplexArray(real: Float64Array, imag?: Float64Array): ComplexNumber[] {
  if (imag && imag.length !== real.length) {
    throw new RangeError(`real (${real.length}) and imag (${imag.length}) lengths differ`);
  }
  const out = new Array<ComplexNumber>(real.length);
  for (let i = 0; i < real.length; i++) {
    out[i] = new ComplexNumber(real[i] as number, imag ? (imag[i] as number) : 0);
  }
  return out;
}

function fromComplexArray(values: readonly ComplexNumber[]): ComplexSignal {
  const real = new Float64Array(values.length);
  const imag = new Float64Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const z = values[i] as ComplexNumber;
    real[i] = z.re;
    imag[i] = z.im;
  }
  return { real, imag };
}

/** Fast Fourier transform. Length must be a power of two -- use {@link fftPadded} otherwise. `imag` optional (defaults to an all-real signal). */
export function fft(real: Float64Array, imag?: Float64Array): ComplexSignal {
  return fromComplexArray(FFT.fft(toComplexArray(real, imag)));
}

/** Inverse fast Fourier transform. */
export function ifft(real: Float64Array, imag: Float64Array): ComplexSignal {
  return fromComplexArray(FFT.ifft(toComplexArray(real, imag)));
}

/** FFT with the input zero-padded up to the next power of two -- no length restriction. */
export function fftPadded(real: Float64Array, imag?: Float64Array): ComplexSignal {
  return fromComplexArray(FFT.fftPadded(toComplexArray(real, imag)));
}

/** Linear convolution of two signals via the FFT. Use {@link realSignal} to wrap a plain real array. */
export function convolve(a: ComplexSignal, b: ComplexSignal): ComplexSignal {
  return fromComplexArray(FFT.convolve(toComplexArray(a.real, a.imag), toComplexArray(b.real, b.imag)));
}
