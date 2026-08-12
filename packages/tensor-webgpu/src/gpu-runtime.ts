/**
 * Small shared helpers for dispatching a WGSL compute shader over
 * `array<f32>` storage buffers and reading the result back. Used by
 * fusion-wgsl's elementwise kernels, gemm.ts, and attention.ts — factored out
 * once instead of duplicated per kernel (buffer creation, staging-buffer
 * readback, and the map/unmap dance are identical across all of them).
 *
 * These functions call real `GPUDevice` methods — they only *type*-check in
 * Node (via `@webgpu/types`, no runtime browser globals needed to import this
 * module), but actually RUNNING them requires a real `GPUDevice`, which only
 * exists behind `navigator.gpu` in a Chromium-family browser (or a Node WebGPU
 * polyfill — see README "Node support"). test/helpers.ts drives a real one
 * via headless Chrome + CDP.
 */

/** A `GPUBuffer` plus the byte length it was created with — every helper here needs both, and `GPUBuffer` alone doesn't expose its size. */
export interface SizedBuffer {
  buffer: GPUBuffer;
  byteLength: number;
}

/** Upload `data` into a new STORAGE buffer usable as a shader input. */
export function uploadStorageBuffer(device: GPUDevice, data: Float32Array): SizedBuffer {
  const byteLength = data.byteLength;
  const buffer = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  // `data.buffer as ArrayBuffer`: @webgpu/types' `writeBuffer` wants a
  // `BufferSource` typed over a plain `ArrayBuffer`, but TS 5.7's typed-array
  // generics widen `Float32Array#buffer` to `ArrayBufferLike` (which includes
  // `SharedArrayBuffer`) — a real cast, not a bug workaround, since every
  // `Float32Array` this package constructs is backed by a plain `ArrayBuffer`.
  device.queue.writeBuffer(buffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
  return { buffer, byteLength };
}

/** Allocate a zeroed STORAGE buffer for a shader to write into (also COPY_SRC so it can be staged out afterward). */
export function allocateOutputBuffer(device: GPUDevice, elementCount: number): SizedBuffer {
  const byteLength = elementCount * 4;
  const buffer = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  return { buffer, byteLength };
}

/**
 * Copy a GPU buffer to a MAP_READ staging buffer, submit, await the map, and
 * return a plain (detached-from-GPU-memory) `Float32Array` copy. The
 * `.slice(0)` on the mapped range matters: `getMappedRange()` returns a view
 * backed by the mapping, which becomes invalid the instant `unmap()` runs.
 */
export async function readBackFloat32(
  device: GPUDevice,
  source: SizedBuffer,
): Promise<Float32Array> {
  const staging = device.createBuffer({
    size: source.byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source.buffer, 0, staging, 0, source.byteLength);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}

/** Dispatch a compute shader with `bindings` bound at group 0 in binding-index order, `workgroupCount` groups along X only (every kernel in this package is a flat 1-D or 2-D-encoded-as-1-D dispatch). */
export function dispatchCompute(
  device: GPUDevice,
  code: string,
  bindings: readonly SizedBuffer[],
  workgroupCountX: number,
): void {
  const module = device.createShaderModule({ code });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: bindings.map((b, i) => ({ binding: i, resource: { buffer: b.buffer } })),
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.max(1, workgroupCountX));
  pass.end();
  device.queue.submit([encoder.finish()]);
}

/** `Math.ceil(total / groupSize)`, minimum 1 (WebGPU rejects a 0-workgroup dispatch on some backends). */
export function workgroupsFor(total: number, groupSize: number): number {
  return Math.max(1, Math.ceil(total / groupSize));
}
