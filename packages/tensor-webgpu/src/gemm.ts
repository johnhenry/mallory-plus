/**
 * WebGPU GEMM (issue #12, v1 scope item 1): `out[m,n] = sum_k a[m,k] * b[k,n]`,
 * row-major f32, one thread per output element. Naive (no shared-memory
 * tiling) — v1's job was proving the seam and measuring the crossover
 * against `mallory-tensor-wasm`'s `matmulInto`, not squeezing out the last
 * GFLOP. The measurement (docs/spikes/webgpu-baseline.md) came back honest
 * rather than flattering: on this machine (integrated GPU via ANGLE's GL
 * backend, no discrete GPU) this naive kernel never beats WASM, at any size
 * tested up to 768x768 — `chooseGemmBackend` (threshold.ts) reflects that
 * directly (`GEMM_ELEMENT_THRESHOLD = Infinity`). Shared-memory tiling — the
 * standard next optimization for a naive per-element GEMM kernel like this
 * one — is documented future work, not attempted here; re-measure after it
 * lands (or on real discrete-GPU hardware) before changing that threshold.
 *
 * Shapes/strides are NOT WGSL-generic here (unlike fusion-wgsl.ts, which
 * compiles a fresh shader per traced IR): GEMM's shader source is fixed, and
 * `m`/`n`/`k` are passed as uniform buffer data, matching how a real kernel
 * library (cuBLAS, etc.) treats GEMM as one shader with runtime dimensions
 * rather than one shader per shape.
 */
import {
  acquireBuffer,
  allocateOutputBuffer,
  getOrCreateComputePipeline,
  readBackFloat32,
  releaseBuffer,
  uploadStorageBuffer,
  type SizedBuffer,
} from "./gpu-runtime.ts";

const TILE = 8;

const GEMM_WGSL = `
struct Dims {
  m: u32,
  n: u32,
  k: u32,
  _pad: u32,
};
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.y;
  let col = gid.x;
  if (row >= dims.m || col >= dims.n) {
    return;
  }
  var acc: f32 = 0.0;
  for (var p: u32 = 0u; p < dims.k; p = p + 1u) {
    acc = acc + a[row * dims.k + p] * b[p * dims.n + col];
  }
  out[row * dims.n + col] = acc;
}
`;

/** `GEMM_WGSL`'s Dims struct, `std140`-ish layout: 4 x u32 = 16 bytes. */
function dimsUniform(device: GPUDevice, m: number, n: number, k: number): SizedBuffer {
  const sized = acquireBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(sized.buffer, 0, new Uint32Array([m, n, k, 0]));
  return sized;
}

/**
 * `a` is `m x k`, `b` is `k x n`, both row-major contiguous `Float32Array`s;
 * returns `m x n` row-major. Matches `mallory-tensor-wasm`'s `matmulInto`
 * shape contract (2-D only in v1) so the two backends are drop-in comparable
 * for the threshold measurement.
 */
export async function runGemmWGSL(
  device: GPUDevice,
  a: Float32Array,
  b: Float32Array,
  m: number,
  k: number,
  n: number,
): Promise<Float32Array> {
  if (a.length !== m * k) throw new RangeError(`runGemmWGSL: a.length ${a.length} !== m*k ${m * k}`);
  if (b.length !== k * n) throw new RangeError(`runGemmWGSL: b.length ${b.length} !== k*n ${k * n}`);
  const bufA = uploadStorageBuffer(device, a);
  const bufB = uploadStorageBuffer(device, b);
  const bufOut = allocateOutputBuffer(device, m * n);
  const dims = dimsUniform(device, m, n, k);
  try {
    // gpu-runtime's shared `dispatchCompute` only covers a 1-D workgroup
    // count; GEMM's natural dispatch shape is 2-D (one thread per (row,
    // col)), so it issues its own `dispatchWorkgroups(x, y)` call rather than
    // stretching the shared helper to support a shape only this one kernel
    // needs.
    return await runGemm2D(device, GEMM_WGSL, [bufA, bufB, bufOut, dims], m, n);
  } finally {
    releaseBuffer(device, bufA);
    releaseBuffer(device, bufB);
    releaseBuffer(device, bufOut);
    releaseBuffer(device, dims);
  }
}

async function runGemm2D(
  device: GPUDevice,
  code: string,
  bindings: readonly SizedBuffer[],
  m: number,
  n: number,
): Promise<Float32Array> {
  const pipeline = getOrCreateComputePipeline(device, code);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: bindings.map((b, i) => ({ binding: i, resource: { buffer: b.buffer } })),
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.max(1, Math.ceil(n / TILE)), Math.max(1, Math.ceil(m / TILE)));
  pass.end();
  const outBuffer = bindings[2] as SizedBuffer;
  device.queue.submit([encoder.finish()]);
  return readBackFloat32(device, outBuffer);
}
