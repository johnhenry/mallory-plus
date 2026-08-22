/**
 * Deno conditional-export entry (issue #55 Phase 2): everything the default
 * entry exports, PLUS the native FFI surface. Importing this on Deno does
 * NOT auto-switch anything -- the WASM `Kernels` path works under Deno too
 * (via node: compat) and stays the zero-install default; native is opt-in:
 *
 *   import { Kernels, NativeKernels } from "@johnhenry/math-plus-tensor-wasm";
 *   const native = NativeKernels.load();          // undefined -> use wasm
 *   const kernels = native ?? await Kernels.load();
 */
export * from "./index.ts";
export * from "./native.ts";
