/**
 * normalize (issue #41, v2 "practical ML/media compute" bundle). Per-channel
 * `(x - mean[c]) / std[c]` on a `[H, W, C]` or `[N, H, W, C]` tensor
 * (channel-last, `C` the fastest-varying/last axis).
 */
import { Tensor, type DType } from "@johnhenry/math-plus-tensor-core";
import { assertFloatDtype, flattenToFloat64, toDtypeArray } from "./util.ts";

export interface NormalizeOptions {
  readonly mean: readonly number[];
  readonly std: readonly number[];
}

export function normalize(input: Tensor, options: NormalizeOptions): Tensor {
  const { mean, std } = options;
  const rank = input.shape.length;
  if (rank !== 3 && rank !== 4) {
    throw new RangeError(
      `normalize: expected a [H,W,C] or [N,H,W,C] tensor, got rank ${rank} (shape [${input.shape}])`,
    );
  }
  assertFloatDtype(input.dtype, "normalize");
  const dtype: DType = input.dtype;

  const c = input.shape[rank - 1] as number;
  if (mean.length !== c || std.length !== c) {
    throw new RangeError(`normalize: mean (${mean.length}) and std (${std.length}) must each have length ${c} (the channel count)`);
  }
  if (std.some((s) => s === 0)) {
    throw new RangeError("normalize: std must be nonzero for every channel");
  }

  const data = flattenToFloat64(input);
  const out = new Float64Array(data.length);
  for (let i = 0; i < data.length; i++) {
    const ch = i % c;
    out[i] = ((data[i] as number) - (mean[ch] as number)) / (std[ch] as number);
  }

  return Tensor.fromTypedArray(toDtypeArray(out, dtype), input.shape, { dtype });
}
