/**
 * welch (issue #70) — power spectral density via Welch's method: reuses
 * the existing windowed-segment machinery (mirroring `stft`'s framing, not
 * calling `stft` directly since a raw complex spectrum per frame is needed
 * before squaring — `stft` already returns exactly that shape, reused via
 * `stft` itself, see below), then averages `|X_i(f)|^2` across segments
 * with window-power normalization.
 *
 * v1 scope, disclosed: returns the TWO-SIDED PSD over all `nperseg`
 * frequency bins (matching `scipy.signal.welch(..., return_onesided=
 * False)` exactly, verified numerically before writing this) — NOT
 * SciPy's default one-sided-with-doubling convention for real inputs.
 * Simpler, and a real, correctly-normalized PSD either way; revisit if a
 * concrete workload wants the halved one-sided form. `fs` is fixed at 1.0
 * (SciPy's own default), not exposed as an option, matching this issue's
 * scoped `{window?, nperseg?, noverlap?}` signature.
 */
import { ComplexTensor, fft } from "mallory-fft";
import { Tensor } from "mallory-tensor-core";
import { hannWindow } from "./window.ts";

export interface WelchOptions {
  readonly window?: Float64Array;
  readonly nperseg?: number;
  readonly noverlap?: number;
}

const DEFAULT_NPERSEG = 256;

function isPowerOfTwo(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0;
}

export function welch(signal: Tensor, options: WelchOptions = {}): { frequencies: number[]; psd: Tensor } {
  if (signal.shape.length !== 1) throw new RangeError("welch: v1 supports 1-D Tensor only");
  // Read-only below, so no defensive copy is needed even when `.data`
  // aliases `signal`'s own storage.
  const data = signal.contiguous().data as Float64Array;

  const nperseg = options.nperseg ?? Math.min(DEFAULT_NPERSEG, data.length);
  if (!isPowerOfTwo(nperseg)) {
    throw new RangeError(`welch: nperseg must be a power of two in v1, got ${nperseg}`);
  }
  const noverlap = options.noverlap ?? Math.floor(nperseg / 2);
  if (noverlap < 0 || noverlap >= nperseg) {
    throw new RangeError(`welch: noverlap (${noverlap}) must be in [0, nperseg) (nperseg=${nperseg})`);
  }
  const window = options.window ?? hannWindow(nperseg);
  if (window.length !== nperseg) {
    throw new RangeError(`welch: window length (${window.length}) must equal nperseg (${nperseg})`);
  }
  const hop = nperseg - noverlap;
  const numFrames = Math.floor((data.length - nperseg) / hop) + 1;
  if (numFrames < 1) {
    throw new RangeError(`welch: signal too short (${data.length} samples) for nperseg=${nperseg}`);
  }

  let windowPower = 0;
  for (const w of window) windowPower += w * w;

  const accumulator = new Float64Array(nperseg);
  for (let f = 0; f < numFrames; f++) {
    const start = f * hop;
    const frame = new Float64Array(nperseg);
    for (let i = 0; i < nperseg; i++) frame[i] = (data[start + i] as number) * (window[i] as number);
    const spectrum = fft(ComplexTensor.fromReal(Tensor.fromTypedArray(frame, [nperseg], { dtype: "f64" })));
    const specReal = spectrum.real.contiguous().data as Float64Array;
    const specImag = spectrum.imag.contiguous().data as Float64Array;
    for (let k = 0; k < nperseg; k++) {
      const re = specReal[k] as number;
      const im = specImag[k] as number;
      accumulator[k] = (accumulator[k] as number) + (re * re + im * im);
    }
  }

  const psdData = new Float64Array(nperseg);
  for (let k = 0; k < nperseg; k++) {
    psdData[k] = (accumulator[k] as number) / numFrames / windowPower;
  }
  // Two-sided frequency labeling matches numpy.fft.fftfreq's wraparound
  // convention, NOT a naive k/nperseg: bins [0, nperseg/2) are positive,
  // bins [nperseg/2, nperseg) wrap to negative (k-nperseg)/nperseg -- the
  // Nyquist bin itself (k=nperseg/2, even nperseg only) is -0.5, not +0.5.
  const frequencies = Array.from({ length: nperseg }, (_, k) =>
    k < nperseg / 2 ? k / nperseg : (k - nperseg) / nperseg,
  );

  return { frequencies, psd: Tensor.fromTypedArray(psdData, [nperseg], { dtype: "f64" }) };
}
