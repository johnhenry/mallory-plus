/**
 * mallory-tensor-wasm — persistent WASM-resident tensor storage + the
 * `...Into` performance interface (issue #3, the M2 gate).
 *
 * docs/spikes/wasm-baseline.md measured the actual problem this solves: a
 * WASM kernel over RESIDENT buffers is 1.78x faster than pure JS at N=1e6
 * (0.864ms vs 1.539ms, scalar, no SIMD) — but a wrapper that allocates and
 * copies in/out per call turns that into a 2.27x REGRESSION (4.046ms). The
 * fix is this file: `WasmTensor` storage lives in linear memory for its
 * whole lifetime, and `addInto`/`mulInto`/`matmulInto` write into an
 * already-allocated `out` with zero WASM allocation per call.
 *
 * v1 scope: f32 only, matching the current kernel set in
 * crates/tensor-wasm-kernels. `WasmTensor` is NOT yet wired into
 * mallory-tensor-core's `Tensor` (that storage-model merge is bigger scope,
 * tracked separately) — this package proves the seam and its performance
 * in isolation first.
 */
import { readFile } from "node:fs/promises";

interface KernelExports {
  memory: WebAssembly.Memory;
  alloc(len: number, align: number): number;
  dealloc(ptr: number, len: number, align: number): void;
  add_f32_strided(
    aPtr: number,
    aOffset: number,
    aStride: number,
    bPtr: number,
    bOffset: number,
    bStride: number,
    outPtr: number,
    outOffset: number,
    outStride: number,
    len: number,
  ): void;
  mul_f32_strided(
    aPtr: number,
    aOffset: number,
    aStride: number,
    bPtr: number,
    bOffset: number,
    bStride: number,
    outPtr: number,
    outOffset: number,
    outStride: number,
    len: number,
  ): void;
  gemm_f32(
    aPtr: number,
    aOffset: number,
    aRowStride: number,
    aColStride: number,
    bPtr: number,
    bOffset: number,
    bRowStride: number,
    bColStride: number,
    outPtr: number,
    outOffset: number,
    outRowStride: number,
    outColStride: number,
    m: number,
    n: number,
    k: number,
    alpha: number,
    beta: number,
  ): void;
}

const F32_ALIGN = 4;

/**
 * A tensor whose f32 storage lives in WASM linear memory for its whole
 * lifetime. Call `.free()` when done — this is manual memory management,
 * not GC'd JS storage.
 */
export class WasmTensor {
  readonly kernels: Kernels;
  /** Byte pointer returned by `alloc` — the buffer's true start, for `dealloc`. */
  readonly bufferPtr: number;
  readonly byteLength: number;
  readonly shape: readonly number[];
  /** Strides in ELEMENTS (not bytes), matching mallory-tensor-core's convention. */
  readonly strides: readonly number[];
  /** Element offset from `bufferPtr` — always 0 for tensors created here; view() can differ. */
  readonly elementOffset: number;
  #freed = false;

  private constructor(
    kernels: Kernels,
    bufferPtr: number,
    byteLength: number,
    shape: readonly number[],
    strides: readonly number[],
    elementOffset: number,
  ) {
    this.kernels = kernels;
    this.bufferPtr = bufferPtr;
    this.byteLength = byteLength;
    this.shape = shape;
    this.strides = strides;
    this.elementOffset = elementOffset;
  }

  static allocate(kernels: Kernels, shape: readonly number[]): WasmTensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const byteLength = size * 4;
    const ptr = kernels.exports.alloc(byteLength, F32_ALIGN);
    if (ptr === 0) throw new Error("WASM allocation failed");
    return new WasmTensor(kernels, ptr, byteLength, shape, contiguousStrides(shape), 0);
  }

  /** Allocate and copy `data` in once. */
  static fromArray(kernels: Kernels, data: Float32Array, shape: readonly number[]): WasmTensor {
    const t = WasmTensor.allocate(kernels, shape);
    new Float32Array(kernels.exports.memory.buffer, t.bufferPtr, data.length).set(data);
    return t;
  }

  /** A transposed 2-D VIEW — swaps strides, shares the same WASM buffer, zero copy. */
  transposed(): WasmTensor {
    if (this.shape.length !== 2) {
      throw new RangeError("transposed() only supports 2-D tensors in v1");
    }
    return new WasmTensor(
      this.kernels,
      this.bufferPtr,
      this.byteLength,
      [this.shape[1] as number, this.shape[0] as number],
      [this.strides[1] as number, this.strides[0] as number],
      this.elementOffset,
    );
  }

  /** Copy the buffer out as a plain Float32Array (for inspection/testing only). */
  toFloat32Array(): Float32Array {
    if (this.#freed) throw new Error("WasmTensor: use after free");
    const size = this.shape.reduce((a, b) => a * b, 1);
    if (this.strides.length === 1 && this.strides[0] === 1 && this.elementOffset === 0) {
      return new Float32Array(this.kernels.exports.memory.buffer, this.bufferPtr, size).slice();
    }
    // Non-contiguous or offset view: read element-by-element via strides.
    const out = new Float32Array(size);
    const view = new Float32Array(this.kernels.exports.memory.buffer);
    const base = this.bufferPtr / 4 + this.elementOffset;
    let flatIndex = 0;
    const walk = (axis: number, offset: number): void => {
      const dim = this.shape[axis] as number;
      const stride = this.strides[axis] as number;
      for (let i = 0; i < dim; i++) {
        if (axis === this.shape.length - 1) {
          out[flatIndex++] = view[offset + i * stride] as number;
        } else {
          walk(axis + 1, offset + i * stride);
        }
      }
    };
    walk(0, base);
    return out;
  }

  free(): void {
    if (this.#freed) return;
    this.kernels.exports.dealloc(this.bufferPtr, this.byteLength, F32_ALIGN);
    this.#freed = true;
  }
}

function contiguousStrides(shape: readonly number[]): number[] {
  const strides = new Array<number>(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = acc;
    acc *= shape[i] as number;
  }
  return strides;
}

function flatSpec(t: WasmTensor): { bufferPtr: number; offset: number; stride: number } {
  if (t.strides.length !== 1) {
    throw new RangeError(
      `addInto/mulInto require 1-D tensors in v1 (got ${t.strides.length}-D); flatten first`,
    );
  }
  // bufferPtr stays a raw BYTE address: the Rust `*const f32` parameter is a
  // byte pointer at the WASM ABI level, and its `.offset(n)` already scales
  // `n` by sizeof(f32) internally. Only `offset`/`stride` are element units.
  return {
    bufferPtr: t.bufferPtr,
    offset: t.elementOffset,
    stride: t.strides[0] as number,
  };
}

export class Kernels {
  readonly exports: KernelExports;
  readonly #allocCounter: { count: number };

  /** Calls to the WASM `alloc` export since load() — proves the `...Into` path allocates zero times. */
  get allocCallCount(): number {
    return this.#allocCounter.count;
  }

  private constructor(exports: KernelExports, allocCounter: { count: number }) {
    this.exports = exports;
    this.#allocCounter = allocCounter;
  }

  static async load(wasmBytes?: Uint8Array): Promise<Kernels> {
    const bytes =
      wasmBytes ??
      (await readFile(new URL("../wasm/tensor_wasm_kernels.wasm", import.meta.url)));
    const { instance } = await WebAssembly.instantiate(bytes as BufferSource, {});
    const rawExports = instance.exports as unknown as KernelExports;
    const rawAlloc = rawExports.alloc.bind(rawExports);
    const counter = { count: 0 };
    // Build a fresh exports object rather than mutating properties on the
    // WASM-provided one (whether those are writable is spec-uncertain and
    // varies by engine) -- every other export is just rebound, unchanged.
    const wrapped: KernelExports = {
      memory: rawExports.memory,
      alloc: (len: number, align: number) => {
        counter.count++;
        return rawAlloc(len, align);
      },
      dealloc: rawExports.dealloc.bind(rawExports),
      add_f32_strided: rawExports.add_f32_strided.bind(rawExports),
      mul_f32_strided: rawExports.mul_f32_strided.bind(rawExports),
      gemm_f32: rawExports.gemm_f32.bind(rawExports),
    };
    return new Kernels(wrapped, counter);
  }

  zeros(shape: readonly number[]): WasmTensor {
    return WasmTensor.allocate(this, shape);
  }

  fromArray(data: Float32Array, shape: readonly number[]): WasmTensor {
    return WasmTensor.fromArray(this, data, shape);
  }

  /** out[i] = a[i] + b[i], writing directly into `out`'s WASM buffer. Zero allocation. */
  addInto(out: WasmTensor, a: WasmTensor, b: WasmTensor): WasmTensor {
    const A = flatSpec(a);
    const B = flatSpec(b);
    const O = flatSpec(out);
    this.exports.add_f32_strided(
      A.bufferPtr,
      A.offset,
      A.stride,
      B.bufferPtr,
      B.offset,
      B.stride,
      O.bufferPtr,
      O.offset,
      O.stride,
      out.shape.reduce((x, y) => x * y, 1),
    );
    return out;
  }

  /** out[i] = a[i] * b[i], writing directly into `out`'s WASM buffer. Zero allocation. */
  mulInto(out: WasmTensor, a: WasmTensor, b: WasmTensor): WasmTensor {
    const A = flatSpec(a);
    const B = flatSpec(b);
    const O = flatSpec(out);
    this.exports.mul_f32_strided(
      A.bufferPtr,
      A.offset,
      A.stride,
      B.bufferPtr,
      B.offset,
      B.stride,
      O.bufferPtr,
      O.offset,
      O.stride,
      out.shape.reduce((x, y) => x * y, 1),
    );
    return out;
  }

  /**
   * out = a @ b (2-D only in v1), writing directly into `out`'s WASM buffer.
   * `a`/`b` may be `.transposed()` views — read via strides, never copied.
   */
  matmulInto(out: WasmTensor, a: WasmTensor, b: WasmTensor): WasmTensor {
    if (a.shape.length !== 2 || b.shape.length !== 2 || out.shape.length !== 2) {
      throw new RangeError("matmulInto supports 2-D tensors only in v1");
    }
    const [m, k] = a.shape as [number, number];
    const [k2, n] = b.shape as [number, number];
    if (k !== k2) {
      throw new RangeError(`matmulInto: inner dims ${k} and ${k2} don't match`);
    }
    this.exports.gemm_f32(
      a.bufferPtr,
      a.elementOffset,
      a.strides[0] as number,
      a.strides[1] as number,
      b.bufferPtr,
      b.elementOffset,
      b.strides[0] as number,
      b.strides[1] as number,
      out.bufferPtr,
      out.elementOffset,
      out.strides[0] as number,
      out.strides[1] as number,
      m,
      n,
      k,
      1.0,
      0.0,
    );
    return out;
  }
}
