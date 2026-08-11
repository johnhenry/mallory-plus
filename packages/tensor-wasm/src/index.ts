/**
 * mallory-tensor-wasm — loader + typed wrappers over the flat-numeric
 * extern-C kernel ABI in crates/tensor-wasm-kernels.
 *
 * The ABI takes byte offsets into linear memory (no wasm-bindgen object
 * marshalling); this wrapper owns allocation bookkeeping. Kernel signatures
 * grow strides/offsets as they generalize (docs/PLAN.md §6.1).
 */
import { readFile } from "node:fs/promises";

interface KernelExports {
  memory: WebAssembly.Memory;
  alloc(len: number): number;
  dealloc(ptr: number, len: number): void;
  add_f32(a: number, b: number, out: number, len: number): void;
}

export class Kernels {
  #exports: KernelExports;

  private constructor(exports: KernelExports) {
    this.#exports = exports;
  }

  static async load(wasmBytes?: Uint8Array): Promise<Kernels> {
    const bytes =
      wasmBytes ??
      (await readFile(
        new URL("../wasm/tensor_wasm_kernels.wasm", import.meta.url),
      ));
    const { instance } = await WebAssembly.instantiate(bytes as BufferSource, {});
    return new Kernels(instance.exports as unknown as KernelExports);
  }

  /** out[i] = a[i] + b[i], contiguous f32. Copies in/out of linear memory. */
  addF32(a: Float32Array, b: Float32Array): Float32Array {
    if (a.length !== b.length) {
      throw new RangeError(`length mismatch: ${a.length} vs ${b.length}`);
    }
    const { alloc, dealloc, add_f32, memory } = this.#exports;
    const byteLength = a.length * 4;
    const aPtr = alloc(byteLength);
    const bPtr = alloc(byteLength);
    const outPtr = alloc(byteLength);
    try {
      new Float32Array(memory.buffer, aPtr, a.length).set(a);
      new Float32Array(memory.buffer, bPtr, b.length).set(b);
      add_f32(aPtr, bPtr, outPtr, a.length);
      return new Float32Array(memory.buffer, outPtr, a.length).slice();
    } finally {
      dealloc(aPtr, byteLength);
      dealloc(bPtr, byteLength);
      dealloc(outPtr, byteLength);
    }
  }
}
