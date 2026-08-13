/**
 * stft/istft (issue #44). Standard windowed short-time Fourier transform:
 * each frame is `window * signal[start:start+nperseg]`, transformed via
 * mallory-fft's `fft`; `istft` reconstructs via weighted overlap-add (WOLA)
 * with window-power normalization -- the standard COLA (constant
 * overlap-add) reconstruction, exact (to floating-point precision) for the
 * default Hann-window-at-50%-overlap configuration, which is COLA-compliant
 * by construction.
 *
 * v1 scope: `nperseg` must be a power of two (mallory-fft's `fft`/`ifft`
 * requirement -- no per-frame zero-padding to a larger FFT size, to avoid
 * reconciling that with windowing/overlap-add). Covers the overwhelming
 * majority of practical STFT use (256/512/1024-sample windows); a
 * non-power-of-two `nperseg` throws a clear error rather than silently
 * rounding or padding.
 */
import { ComplexTensor, fft, ifft } from "mallory-fft";
import { Tensor } from "mallory-tensor-core";
import { hannWindow } from "./window.ts";

export interface StftOptions {
  readonly window?: Float64Array;
  readonly nperseg?: number;
  readonly noverlap?: number;
}

const DEFAULT_NPERSEG = 256;

function isPowerOfTwo(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0;
}

function resolveParams(
  fallbackNperseg: number,
  options: StftOptions,
): { window: Float64Array; nperseg: number; noverlap: number; hop: number } {
  const nperseg = options.nperseg ?? fallbackNperseg;
  if (!isPowerOfTwo(nperseg)) {
    throw new RangeError(`stft/istft: nperseg must be a power of two in v1, got ${nperseg}`);
  }
  const noverlap = options.noverlap ?? Math.floor(nperseg / 2);
  if (noverlap < 0 || noverlap >= nperseg) {
    throw new RangeError(`stft/istft: noverlap (${noverlap}) must be in [0, nperseg) (nperseg=${nperseg})`);
  }
  const window = options.window ?? hannWindow(nperseg);
  if (window.length !== nperseg) {
    throw new RangeError(`stft/istft: window length (${window.length}) must equal nperseg (${nperseg})`);
  }
  return { window, nperseg, noverlap, hop: nperseg - noverlap };
}

/** `signal[T]` -> a `[numFrames, nperseg]` ComplexTensor spectrogram (one full-nperseg-point FFT per windowed frame). */
export function stft(signal: Tensor, options: StftOptions = {}): ComplexTensor {
  if (signal.shape.length !== 1) throw new RangeError("stft: v1 supports 1-D Tensor only");
  const data = Float64Array.from(signal.contiguous().toArray() as number[]);
  const { window, nperseg, hop } = resolveParams(Math.min(DEFAULT_NPERSEG, data.length), options);
  const numFrames = Math.floor((data.length - nperseg) / hop) + 1;
  if (numFrames < 1) {
    throw new RangeError(`stft: signal too short (${data.length} samples) for nperseg=${nperseg}`);
  }

  const realOut = new Float64Array(numFrames * nperseg);
  const imagOut = new Float64Array(numFrames * nperseg);
  for (let f = 0; f < numFrames; f++) {
    const start = f * hop;
    const frameReal = new Float64Array(nperseg);
    for (let i = 0; i < nperseg; i++) frameReal[i] = (data[start + i] as number) * (window[i] as number);
    const spectrum = fft(ComplexTensor.fromReal(Tensor.fromTypedArray(frameReal, [nperseg], { dtype: "f64" })));
    const specReal = spectrum.real.toArray() as number[];
    const specImag = spectrum.imag.toArray() as number[];
    for (let k = 0; k < nperseg; k++) {
      realOut[f * nperseg + k] = specReal[k] as number;
      imagOut[f * nperseg + k] = specImag[k] as number;
    }
  }
  return ComplexTensor.fromParts(
    Tensor.fromTypedArray(realOut, [numFrames, nperseg], { dtype: "f64" }),
    Tensor.fromTypedArray(imagOut, [numFrames, nperseg], { dtype: "f64" }),
  );
}

/** Inverse of `stft` -- weighted overlap-add reconstruction. `options` should match what `stft` was called with (nperseg is inferred from `spectrogram`'s own shape if omitted). */
export function istft(spectrogram: ComplexTensor, options: StftOptions = {}): Tensor {
  if (spectrogram.shape.length !== 2) throw new RangeError("istft: expected a [numFrames, nperseg] ComplexTensor");
  const [numFrames, spectrogramNperseg] = spectrogram.shape as [number, number];
  const { window, nperseg, hop } = resolveParams(spectrogramNperseg, options);
  if (nperseg !== spectrogramNperseg) {
    throw new RangeError(`istft: nperseg option (${nperseg}) doesn't match the spectrogram's own frame length (${spectrogramNperseg})`);
  }

  const outLen = (numFrames - 1) * hop + nperseg;
  const out = new Float64Array(outLen);
  const norm = new Float64Array(outLen);

  for (let f = 0; f < numFrames; f++) {
    const frameReal = new Float64Array(nperseg);
    const frameImag = new Float64Array(nperseg);
    for (let k = 0; k < nperseg; k++) {
      frameReal[k] = spectrogram.real.at(f, k) as number;
      frameImag[k] = spectrogram.imag.at(f, k) as number;
    }
    const reconstructed = ifft(
      ComplexTensor.fromParts(
        Tensor.fromTypedArray(frameReal, [nperseg], { dtype: "f64" }),
        Tensor.fromTypedArray(frameImag, [nperseg], { dtype: "f64" }),
      ),
    ).real.toArray() as number[];
    const start = f * hop;
    for (let i = 0; i < nperseg; i++) {
      out[start + i] = (out[start + i] as number) + (reconstructed[i] as number) * (window[i] as number);
      norm[start + i] = (norm[start + i] as number) + (window[i] as number) * (window[i] as number);
    }
  }

  for (let i = 0; i < outLen; i++) {
    out[i] = (norm[i] as number) > 1e-10 ? (out[i] as number) / (norm[i] as number) : 0;
  }
  return Tensor.fromTypedArray(out, [outLen], { dtype: "f64" });
}
