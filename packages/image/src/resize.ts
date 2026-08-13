/**
 * resize (issue #41, v2 "practical ML/media compute" bundle). Operates on a
 * `[H, W, C]` or `[N, H, W, C]` tensor (channel-last, the common ML
 * preprocessing layout). `nearest` uses simple index-scaling; `bilinear`
 * (the default, matching common ML preprocessing conventions) uses
 * half-pixel-center coordinate mapping, the same convention TensorFlow/
 * PyTorch's `align_corners=false` resize uses.
 */
import { Tensor, type DType } from "mallory-tensor-core";
import { assertFloatDtype, flattenToFloat64, toDtypeArray } from "./util.ts";

export type ResizeMethod = "nearest" | "bilinear";

export interface ResizeSize {
  height: number;
  width: number;
}

export interface ResizeOptions {
  method?: ResizeMethod;
}

export function resize(input: Tensor, size: ResizeSize, options: ResizeOptions = {}): Tensor {
  const method = options.method ?? "bilinear";
  const rank = input.shape.length;
  if (rank !== 3 && rank !== 4) {
    throw new RangeError(`resize: expected a [H,W,C] or [N,H,W,C] tensor, got rank ${rank} (shape [${input.shape}])`);
  }
  assertFloatDtype(input.dtype, "resize");
  const dtype: DType = input.dtype;

  const batched = rank === 4;
  const shape = input.shape as readonly number[];
  const n = batched ? (shape[0] as number) : 1;
  const inH = (batched ? shape[1] : shape[0]) as number;
  const inW = (batched ? shape[2] : shape[1]) as number;
  const c = (batched ? shape[3] : shape[2]) as number;
  const { height: outH, width: outW } = size;
  if (outH <= 0 || outW <= 0) throw new RangeError(`resize: target size must be positive, got ${outH}x${outW}`);

  const data = flattenToFloat64(input);
  const out = new Float64Array(n * outH * outW * c);

  for (let img = 0; img < n; img++) {
    const inBase = img * inH * inW * c;
    const outBase = img * outH * outW * c;
    for (let oy = 0; oy < outH; oy++) {
      for (let ox = 0; ox < outW; ox++) {
        const outPixel = outBase + (oy * outW + ox) * c;
        if (method === "nearest") {
          const sy = Math.min(inH - 1, Math.floor((oy * inH) / outH));
          const sx = Math.min(inW - 1, Math.floor((ox * inW) / outW));
          const inPixel = inBase + (sy * inW + sx) * c;
          for (let ch = 0; ch < c; ch++) out[outPixel + ch] = data[inPixel + ch] as number;
        } else {
          const scaleY = inH / outH;
          const scaleX = inW / outW;
          const srcY = Math.min(Math.max((oy + 0.5) * scaleY - 0.5, 0), inH - 1);
          const srcX = Math.min(Math.max((ox + 0.5) * scaleX - 0.5, 0), inW - 1);
          const y0 = Math.floor(srcY);
          const y1 = Math.min(y0 + 1, inH - 1);
          const x0 = Math.floor(srcX);
          const x1 = Math.min(x0 + 1, inW - 1);
          const wy = srcY - y0;
          const wx = srcX - x0;
          for (let ch = 0; ch < c; ch++) {
            const v00 = data[inBase + (y0 * inW + x0) * c + ch] as number;
            const v01 = data[inBase + (y0 * inW + x1) * c + ch] as number;
            const v10 = data[inBase + (y1 * inW + x0) * c + ch] as number;
            const v11 = data[inBase + (y1 * inW + x1) * c + ch] as number;
            const top = v00 * (1 - wx) + v01 * wx;
            const bottom = v10 * (1 - wx) + v11 * wx;
            out[outPixel + ch] = top * (1 - wy) + bottom * wy;
          }
        }
      }
    }
  }

  const outShape = batched ? [n, outH, outW, c] : [outH, outW, c];
  return Tensor.fromTypedArray(toDtypeArray(out, dtype), outShape, { dtype });
}
