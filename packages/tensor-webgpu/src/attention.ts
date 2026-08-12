/**
 * WebGPU attention-adjacent primitives (issue #12, v1 scope item 2): the
 * three ops that make up scaled-dot-product-attention's core —
 * `runQKT` (batched Q @ K^T), `runSoftmax` (numerically stable, last axis),
 * `runWeightedSum` (batched weights @ V) — kept as separate dispatches
 * rather than one fused flash-attention-style kernel, matching the issue's
 * explicit v1 scope ("these three primitives, not a fused kernel").
 *
 * Shape convention throughout: `Q`/`K`/`V` are `(batch, seq, dim)` row-major
 * contiguous `Float32Array`s (batch folds leading axes the same way
 * `Tensor.matmul`'s batch broadcasting does, but v1 requires Q/K/V to already
 * share one batch size — no broadcasting inside the kernel).
 */
import {
  allocateOutputBuffer,
  readBackFloat32,
  uploadStorageBuffer,
  type SizedBuffer,
} from "./gpu-runtime.ts";

const TILE = 8;

function dims4Uniform(device: GPUDevice, values: readonly [number, number, number, number]): SizedBuffer {
  const buffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, new Uint32Array(values));
  return { buffer, byteLength: 16 };
}

async function dispatch3D(
  device: GPUDevice,
  code: string,
  bindings: readonly SizedBuffer[],
  outputIndex: number,
  x: number,
  y: number,
  z: number,
): Promise<Float32Array> {
  const module = device.createShaderModule({ code });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: bindings.map((b, i) => ({ binding: i, resource: { buffer: b.buffer } })),
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.max(1, x), Math.max(1, y), Math.max(1, z));
  pass.end();
  device.queue.submit([encoder.finish()]);
  return readBackFloat32(device, bindings[outputIndex] as SizedBuffer);
}

const QKT_WGSL = `
struct Dims { seqQ: u32, seqK: u32, dim: u32, batch: u32 };
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(${TILE}, ${TILE}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x; // index into seqK
  let row = gid.y; // index into seqQ
  let b = gid.z;   // batch
  if (col >= dims.seqK || row >= dims.seqQ || b >= dims.batch) {
    return;
  }
  let qBase = (b * dims.seqQ + row) * dims.dim;
  let kBase = (b * dims.seqK + col) * dims.dim;
  var acc: f32 = 0.0;
  for (var d: u32 = 0u; d < dims.dim; d = d + 1u) {
    acc = acc + q[qBase + d] * k[kBase + d];
  }
  out[(b * dims.seqQ + row) * dims.seqK + col] = acc;
}
`;

/**
 * `scores[b, i, j] = sum_d Q[b, i, d] * K[b, j, d]` — `Q @ K^T` per batch,
 * unscaled (callers apply `1/sqrt(dim)` themselves, e.g. by pre-scaling `Q`,
 * matching how most reference attention implementations separate the scale
 * from the matmul rather than baking it into the kernel).
 */
export async function runQKT(
  device: GPUDevice,
  q: Float32Array,
  k: Float32Array,
  batch: number,
  seqQ: number,
  seqK: number,
  dim: number,
): Promise<Float32Array> {
  if (q.length !== batch * seqQ * dim) throw new RangeError("runQKT: Q shape mismatch");
  if (k.length !== batch * seqK * dim) throw new RangeError("runQKT: K shape mismatch");
  const bufQ = uploadStorageBuffer(device, q);
  const bufK = uploadStorageBuffer(device, k);
  const bufOut = allocateOutputBuffer(device, batch * seqQ * seqK);
  const dims = dims4Uniform(device, [seqQ, seqK, dim, batch]);
  try {
    return await dispatch3D(
      device,
      QKT_WGSL,
      [bufQ, bufK, bufOut, dims],
      2,
      Math.ceil(seqK / TILE),
      Math.ceil(seqQ / TILE),
      batch,
    );
  } finally {
    bufQ.buffer.destroy();
    bufK.buffer.destroy();
    bufOut.buffer.destroy();
    dims.buffer.destroy();
  }
}

const SOFTMAX_WGSL = `
struct Dims { rows: u32, cols: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= dims.rows) {
    return;
  }
  let base = row * dims.cols;
  var m: f32 = x[base];
  for (var j: u32 = 1u; j < dims.cols; j = j + 1u) {
    m = max(m, x[base + j]);
  }
  var sum: f32 = 0.0;
  for (var j: u32 = 0u; j < dims.cols; j = j + 1u) {
    sum = sum + exp(x[base + j] - m);
  }
  for (var j: u32 = 0u; j < dims.cols; j = j + 1u) {
    out[base + j] = exp(x[base + j] - m) / sum;
  }
}
`;

/**
 * Numerically stable softmax along the LAST axis of a `(rows, cols)`
 * row-major buffer (one GPU invocation per row; `cols` is walked serially
 * within the invocation, matching `Tensor.softmax`'s per-row reduction
 * shape — a parallel-reduction version is future work once profiling shows
 * this naive one is the bottleneck, not before).
 */
export async function runSoftmax(
  device: GPUDevice,
  x: Float32Array,
  rows: number,
  cols: number,
): Promise<Float32Array> {
  if (x.length !== rows * cols) throw new RangeError("runSoftmax: shape mismatch");
  const bufX = uploadStorageBuffer(device, x);
  const bufOut = allocateOutputBuffer(device, rows * cols);
  const dims = dims4Uniform(device, [rows, cols, 0, 0]);
  try {
    return await dispatch3D(device, SOFTMAX_WGSL, [bufX, bufOut, dims], 1, Math.ceil(rows / 64), 1, 1);
  } finally {
    bufX.buffer.destroy();
    bufOut.buffer.destroy();
    dims.buffer.destroy();
  }
}

const WEIGHTED_SUM_WGSL = `
struct Dims { seqQ: u32, seqK: u32, dim: u32, batch: u32 };
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(${TILE}, ${TILE}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let d = gid.x;   // index into dim
  let row = gid.y; // index into seqQ
  let b = gid.z;   // batch
  if (d >= dims.dim || row >= dims.seqQ || b >= dims.batch) {
    return;
  }
  let wBase = (b * dims.seqQ + row) * dims.seqK;
  var acc: f32 = 0.0;
  for (var j: u32 = 0u; j < dims.seqK; j = j + 1u) {
    acc = acc + weights[wBase + j] * v[(b * dims.seqK + j) * dims.dim + d];
  }
  out[(b * dims.seqQ + row) * dims.dim + d] = acc;
}
`;

/**
 * `out[b, i, d] = sum_j weights[b, i, j] * V[b, j, d]` — `weights @ V` per
 * batch (the softmax'd attention weights times the value tensor; `weights`
 * is typically `runSoftmax`'s output, but this function takes plain data so
 * it composes with any `(batch, seqQ, seqK)` weight tensor).
 */
export async function runWeightedSum(
  device: GPUDevice,
  weights: Float32Array,
  v: Float32Array,
  batch: number,
  seqQ: number,
  seqK: number,
  dim: number,
): Promise<Float32Array> {
  if (weights.length !== batch * seqQ * seqK) throw new RangeError("runWeightedSum: weights shape mismatch");
  if (v.length !== batch * seqK * dim) throw new RangeError("runWeightedSum: V shape mismatch");
  const bufW = uploadStorageBuffer(device, weights);
  const bufV = uploadStorageBuffer(device, v);
  const bufOut = allocateOutputBuffer(device, batch * seqQ * dim);
  const dims = dims4Uniform(device, [seqQ, seqK, dim, batch]);
  try {
    return await dispatch3D(
      device,
      WEIGHTED_SUM_WGSL,
      [bufW, bufV, bufOut, dims],
      2,
      Math.ceil(dim / TILE),
      Math.ceil(seqQ / TILE),
      batch,
    );
  } finally {
    bufW.buffer.destroy();
    bufV.buffer.destroy();
    bufOut.buffer.destroy();
    dims.buffer.destroy();
  }
}
