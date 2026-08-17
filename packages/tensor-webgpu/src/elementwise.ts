/**
 * Dispatch side of the IR -> WGSL fusion (issue #12 / #11's IR). Pairs with
 * `fusion-wgsl.ts`'s `compileIRToWGSL`, which only produces shader SOURCE —
 * this module does the actual GPU work: upload inputs, dispatch, read back.
 *
 * `runElementwiseWGSL`'s signature deliberately mirrors `CompiledFn.forward`'s
 * *shape*: `numInputs` flat `Float32Array`s in, one flat `Float32Array` out,
 * all the SAME length (`elementCount`) — broadcasting to a common output
 * shape is the caller's job (see tensor.ts's `GPUTensor`-facing wrapper),
 * exactly as noted in fusion-wgsl.ts's module doc.
 */
import type { IRNode } from "mallory-tensor-compile";
import { compileIRToWGSL } from "./fusion-wgsl.ts";
import {
  allocateOutputBuffer,
  dispatchCompute,
  readBackFloat32,
  releaseBuffer,
  uploadStorageBuffer,
  workgroupsFor,
  type SizedBuffer,
} from "./gpu-runtime.ts";

const WORKGROUP_SIZE = 64;

/**
 * Run a compiled elementwise expression (a `CompiledFn`'s traced `IRNode`,
 * reused verbatim from `mallory-tensor-compile`) on the GPU: one shader
 * dispatch touches every output element once, fusing however many ops the
 * expression chained together — no intermediate GPU buffer per op.
 */
export async function runElementwiseWGSL(
  device: GPUDevice,
  node: IRNode,
  inputs: readonly Float32Array[],
  elementCount: number,
): Promise<Float32Array> {
  if (inputs.some((a) => a.length !== elementCount)) {
    throw new RangeError(
      `runElementwiseWGSL: all inputs and the output must share elementCount ${elementCount} (broadcast first)`,
    );
  }
  const { code } = compileIRToWGSL(node, inputs.length);
  const inputBuffers: SizedBuffer[] = inputs.map((data) => uploadStorageBuffer(device, data));
  const outputBuffer = allocateOutputBuffer(device, elementCount);
  try {
    dispatchCompute(device, code, [...inputBuffers, outputBuffer], workgroupsFor(elementCount, WORKGROUP_SIZE));
    return await readBackFloat32(device, outputBuffer);
  } finally {
    for (const b of inputBuffers) releaseBuffer(device, b);
    releaseBuffer(device, outputBuffer);
  }
}
