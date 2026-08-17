/**
 * WebGPU capability detection + the GPU-resident tensor type (issue #12, v1
 * scope items 4 & 5: "`await x.to('webgpu')` stays explicit and async").
 *
 * Design decision (documented per the issue's "you decide the exact shape"):
 * this package does NOT monkey-patch `mallory-tensor-core`'s `Tensor` class
 * with a `.to()` method. `Tensor` has no device-transfer hook today, and
 * adding one from a downstream package would mean either (a) mutating
 * `Tensor.prototype` from outside its own module — fragile, and invisible to
 * anyone reading tensor-core in isolation — or (b) tensor-core growing a
 * dependency on this package to define it as a real instance method, which
 * inverts the intended dependency direction (tensor-webgpu depends on
 * tensor-core, never the reverse, matching every other adapter/accelerator
 * package in this repo). Instead, the explicit-and-async transfer the issue
 * asks for is a free function: `toWebGPU(tensor)` returns a `Promise<GPUTensor>`,
 * mirroring `mallory-frame-arrow`'s `Series.toTensor()` / `Frame.toTensor()`
 * pattern of "device/format transfer is always an awaited call, never a
 * property access" — same spirit as `x.to("webgpu")`, different spelling.
 * `GPUTensor.toTensor()` is the inverse (GPU -> CPU), completing the pair.
 *
 * v1 dtype scope: f32 only, matching `mallory-tensor-wasm`'s `WasmTensor`
 * (WGSL's only convenient float type is `f32`; f64 has no WebGPU
 * representation, and integer dtypes aren't part of v1's GEMM/attention/
 * fusion surface).
 */
import { Tensor, type Shape } from "mallory-tensor-core";

export interface WebGPUCapability {
  available: boolean;
  /** Present only when `available` is true. */
  adapter?: GPUAdapter;
  /** Present only when `available` is true. */
  device?: GPUDevice;
  /** Present only when `available` is false — always a human-readable explanation, never silently empty. */
  reason?: string;
}

/**
 * Feature-detect WebGPU and, if present, actually request an adapter +
 * device (not just check `"gpu" in navigator`) — a browser can expose
 * `navigator.gpu` while `requestAdapter()` still resolves `null` (no
 * compatible adapter, software or hardware), which is exactly the failure
 * mode this function's `reason` string distinguishes from "API not present
 * at all" so callers/tests can tell the two apart.
 */
export async function detectWebGPU(): Promise<WebGPUCapability> {
  const nav = (globalThis as { navigator?: { gpu?: GPU } }).navigator;
  const gpu = nav?.gpu;
  if (!gpu) {
    return {
      available: false,
      reason:
        "navigator.gpu is not present — this requires a Chromium-family browser (v1 scope) " +
        "or a Node WebGPU polyfill (see README.md \"Node support\"); it is not available in plain Node",
    };
  }
  let adapter: GPUAdapter | null;
  try {
    adapter = await gpu.requestAdapter();
  } catch (err) {
    return { available: false, reason: `navigator.gpu.requestAdapter() threw: ${String(err)}` };
  }
  if (!adapter) {
    return { available: false, reason: "navigator.gpu.requestAdapter() resolved null (no compatible GPUAdapter)" };
  }
  const device = await adapter.requestDevice();
  return { available: true, adapter, device };
}

function shapeSize(shape: Shape): number {
  return shape.reduce((a, b) => a * b, 1);
}

/**
 * A tensor whose f32 data lives in a `GPUBuffer` (STORAGE | COPY_SRC |
 * COPY_DST usage) rather than a JS `TypedArray`. Created via {@link toWebGPU};
 * `.free()` releases the underlying `GPUBuffer` — WebGPU buffers are NOT
 * garbage collected on a predictable schedule, so (like `mallory-tensor-wasm`'s
 * `WasmTensor`) this is manual memory management, not GC'd JS storage.
 */
export class GPUTensor {
  readonly device: GPUDevice;
  readonly buffer: GPUBuffer;
  readonly shape: Shape;
  readonly dtype = "f32" as const;
  #freed = false;

  private constructor(device: GPUDevice, buffer: GPUBuffer, shape: Shape) {
    this.device = device;
    this.buffer = buffer;
    this.shape = Object.freeze([...shape]);
  }

  static fromFloat32Array(device: GPUDevice, data: Float32Array, shape: Shape): GPUTensor {
    if (data.length !== shapeSize(shape)) {
      throw new RangeError(
        `GPUTensor.fromFloat32Array: shape [${shape}] (${shapeSize(shape)} elements) does not match data length ${data.length}`,
      );
    }
    const buffer = device.createBuffer({
      size: Math.max(4, data.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    // See gpu-runtime.ts's `uploadStorageBuffer` for why `.buffer as ArrayBuffer` is needed here.
    device.queue.writeBuffer(buffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
    return new GPUTensor(device, buffer, shape);
  }

  /**
   * Wrap an ALREADY-POPULATED GPU buffer as a `GPUTensor` with no host
   * round-trip (issue #100) — for op implementations (attention.ts, gemm.ts,
   * elementwise.ts) that compute directly into a buffer they allocated (e.g.
   * a compute shader's output) and want the result to stay GPU-resident for
   * chaining into further dispatches, rather than reading it back to a
   * `Float32Array` just to re-upload it via {@link fromFloat32Array}. `buffer`
   * must already be sized for `shape` (`shapeSize(shape) * 4` bytes, f32) and
   * usable both as a dispatch output and, if the caller ever calls
   * {@link toTensor}/{@link toFloat32Array} on the result or reuses it as an
   * upload target, as a copy source/destination too — i.e. it should carry at
   * least `STORAGE`, and typically `COPY_SRC`/`COPY_DST` as well, matching
   * what {@link fromFloat32Array} itself allocates (`gpu-runtime.ts`'s
   * `allocateGPUResidentBuffer` returns exactly that combination).
   */
  static fromBuffer(device: GPUDevice, buffer: GPUBuffer, shape: Shape): GPUTensor {
    return new GPUTensor(device, buffer, shape);
  }

  /** Read the buffer back into a plain `Float32Array` (host copy — for `.toTensor()` or inspection/testing). */
  async toFloat32Array(): Promise<Float32Array> {
    if (this.#freed) throw new Error("GPUTensor: use after free()");
    const byteLength = this.buffer.size;
    const staging = this.device.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.buffer, 0, staging, 0, byteLength);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0), 0, shapeSize(this.shape));
    staging.unmap();
    staging.destroy();
    return out;
  }

  /**
   * GPU -> CPU: the inverse of {@link toWebGPU}. Always a copy (never aliases
   * the `GPUBuffer`), and always explicit/async — no implicit CPU<->GPU
   * copying (this repo's non-goal 5), same as `toWebGPU` itself.
   */
  async toTensor(): Promise<Tensor> {
    const data = await this.toFloat32Array();
    return Tensor.fromTypedArray(data, this.shape, { dtype: "f32" });
  }

  free(): void {
    if (this.#freed) return;
    this.buffer.destroy();
    this.#freed = true;
  }
}

/**
 * CPU -> GPU: the explicit, awaited device transfer the issue calls for
 * (v1 non-goal 5 — no implicit copying). Requires `tensor.dtype === "f32"`
 * and a contiguous tensor (call `.contiguous()` first on a view/transposed
 * tensor — matches `mallory-tensor-wasm`'s `WasmTensor.fromArray` contract).
 */
export async function toWebGPU(tensor: Tensor, device: GPUDevice): Promise<GPUTensor> {
  if (tensor.dtype !== "f32") {
    throw new TypeError(`toWebGPU: v1 supports f32 only, got ${tensor.dtype} (cast() first)`);
  }
  if (!tensor.isContiguous) {
    throw new TypeError("toWebGPU: tensor must be contiguous (call .contiguous() first)");
  }
  const data =
    tensor.offset === 0 && tensor.data.length === tensor.size
      ? (tensor.data as Float32Array)
      : (tensor.data as Float32Array).subarray(tensor.offset, tensor.offset + tensor.size);
  return GPUTensor.fromFloat32Array(device, data, tensor.shape);
}
