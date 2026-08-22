/**
 * convolve (issue #44). Direct O(n*m) linear convolution over a 1-D or
 * batched-2-D `[N, T]` Tensor (the `axis` option selects which axis is the
 * time axis for a 2-D input — "explicit channel/batch-axis options" per
 * docs/PLAN.md §6.3, video-relevant, not 1-D-audio-only). `mode` matches
 * NumPy/SciPy's `convolve`/`np.convolve` convention exactly: `"full"`
 * (length `n+m-1`, default), `"same"` (length `max(n,m)`, centered on
 * `"full"`), `"valid"` (length `max(n,m)-min(n,m)+1`, only fully-overlapping
 * output).
 *
 * Direct (not FFT-based) -- `@johnhenry/math-plus-fft`'s own `convolve` note in
 * adapter-math's fft.ts already covers the FFT-based route for large
 * kernels via `@johnhenry/math`; this one is for typical signal-processing
 * kernel sizes where O(n*m) is fine and no complex-tensor round-trip is
 * needed for a purely real operation.
 */
import { Tensor } from "@johnhenry/math-plus-tensor-core";

export type ConvolveMode = "full" | "same" | "valid";

export interface ConvolveOptions {
  readonly mode?: ConvolveMode;
  /** Which axis is the time/signal axis, for a 2-D `[N, T]` batched input. Ignored for 1-D input. Default 1 (last axis). */
  readonly axis?: number;
}

function directConvolve1D(a: Float64Array, b: Float64Array): Float64Array {
  const n = a.length;
  const m = b.length;
  const out = new Float64Array(n + m - 1);
  for (let i = 0; i < n; i++) {
    const av = a[i] as number;
    if (av === 0) continue;
    for (let j = 0; j < m; j++) {
      out[i + j] = (out[i + j] as number) + av * (b[j] as number);
    }
  }
  return out;
}

function trimMode(full: Float64Array, n: number, m: number, mode: ConvolveMode): Float64Array {
  if (mode === "full") return full;
  if (mode === "valid") {
    const outLen = Math.max(n, m) - Math.min(n, m) + 1;
    const start = Math.min(n, m) - 1;
    return full.slice(start, start + outLen);
  }
  // "same": length max(n, m), centered within "full".
  const outLen = Math.max(n, m);
  const start = Math.floor((full.length - outLen) / 2);
  return full.slice(start, start + outLen);
}

// Read-only downstream (directConvolve1D/correlate1D never mutate `a`/`b`),
// so `.data` can be used directly without a defensive copy even when
// `contiguous()` aliases the source tensor's own backing store.
function toFlat1D(t: Tensor): Float64Array {
  return t.contiguous().data as Float64Array;
}

/** 1-D linear convolution of two plain `Float64Array`s. The core primitive `convolve` (Tensor-oriented) delegates to. */
export function convolve1D(a: Float64Array, b: Float64Array, mode: ConvolveMode = "full"): Float64Array {
  if (a.length === 0 || b.length === 0) throw new RangeError("convolve1D: inputs must be non-empty");
  return trimMode(directConvolve1D(a, b), a.length, b.length, mode);
}

/**
 * Apply a `Float64Array, Float64Array -> Float64Array` time-domain op
 * (`convolve1D` or, for issue #70, `correlate1D`) to a 1-D Tensor or each
 * row/column of a 2-D `[N, T]` Tensor — the batching/axis-handling shared
 * by {@link convolve} and `correlate` (see correlate.ts), extracted so
 * neither reimplements the other's tested batching loop.
 */
export function applyTimeDomainOp(
  opName: string,
  op: (a: Float64Array, b: Float64Array) => Float64Array,
  input: Tensor,
  kernel: Tensor,
  options: ConvolveOptions,
): Tensor {
  if (kernel.shape.length !== 1) throw new RangeError(`${opName}: kernel must be 1-D, got shape [${kernel.shape}]`);
  const kernelFlat = toFlat1D(kernel);

  if (input.shape.length === 1) {
    const out = op(toFlat1D(input), kernelFlat);
    return Tensor.fromTypedArray(out, [out.length], { dtype: "f64" });
  }

  if (input.shape.length === 2) {
    const axis = options.axis ?? 1;
    if (axis !== 0 && axis !== 1) throw new RangeError(`${opName}: axis must be 0 or 1 for a 2-D input, got ${axis}`);
    const [d0, d1] = input.shape as [number, number];
    const numRows = axis === 1 ? d0 : d1;
    const timeLen = axis === 1 ? d1 : d0;
    const rows: Float64Array[] = [];
    const full = input.contiguous();
    for (let r = 0; r < numRows; r++) {
      const row = new Float64Array(timeLen);
      for (let i = 0; i < timeLen; i++) {
        row[i] = (axis === 1 ? (full.at(r, i) as number) : (full.at(i, r) as number));
      }
      rows.push(op(row, kernelFlat));
    }
    const outTimeLen = rows[0]?.length ?? 0;
    const outShape: readonly number[] = axis === 1 ? [numRows, outTimeLen] : [outTimeLen, numRows];
    const out = new Float64Array(numRows * outTimeLen);
    for (let r = 0; r < numRows; r++) {
      for (let i = 0; i < outTimeLen; i++) {
        const idx = axis === 1 ? r * outTimeLen + i : i * numRows + r;
        out[idx] = (rows[r] as Float64Array)[i] as number;
      }
    }
    return Tensor.fromTypedArray(out, outShape, { dtype: "f64" });
  }

  throw new RangeError(`${opName}: input must be 1-D or 2-D, got rank ${input.shape.length}`);
}

/** Linear convolution of a 1-D Tensor, or each row/column of a 2-D `[N, T]` Tensor, with a 1-D kernel Tensor. */
export function convolve(input: Tensor, kernel: Tensor, options: ConvolveOptions = {}): Tensor {
  const mode = options.mode ?? "full";
  return applyTimeDomainOp("convolve", (a, b) => convolve1D(a, b, mode), input, kernel, options);
}
