/**
 * @johnhenry/math-plus-tensor-webgpu (issue #12) — WebGPU-accelerated GEMM and
 * attention-adjacent primitives for Math Plus tensors. Chromium-family
 * browsers only in v1 (see README.md); Node needs a documented WebGPU
 * polyfill (a `webgpu` npm package wrapping Dawn) that this package does not
 * bundle or require.
 *
 * See docs/spikes/webgpu-baseline.md for the measured GEMM
 * WASM-vs-WebGPU crossover this package's `chooseGemmBackend` is built on.
 */
export { detectWebGPU, toWebGPU, GPUTensor, type WebGPUCapability } from "./device.ts";
export { GEMM_ELEMENT_THRESHOLD, chooseGemmBackend } from "./threshold.ts";
export { runGemmWGSL } from "./gemm.ts";
export { runQKT, runSoftmax, runWeightedSum } from "./attention.ts";
export { compileIRToWGSL, type ElementwiseWGSL } from "./fusion-wgsl.ts";
export { runElementwiseWGSL } from "./elementwise.ts";
export {
  uploadStorageBuffer,
  allocateOutputBuffer,
  allocateGPUResidentBuffer,
  acquireBuffer,
  releaseBuffer,
  destroyBufferPool,
  readBackFloat32,
  dispatchCompute,
  getOrCreateComputePipeline,
  pipelineCacheSize,
  workgroupsFor,
  type SizedBuffer,
} from "./gpu-runtime.ts";
