/**
 * Deno-side verification of the native FFI path (issue #55 Phase 2):
 * NativeKernels results must agree with the WASM Kernels on the same
 * inputs, and the load/fallback contract must hold. Not part of `npm test`
 * (Deno isn't a repo dependency); run manually or in CI as:
 *
 *   cargo build --release -p tensor-wasm-kernels
 *   deno run --allow-ffi --allow-read --allow-env packages/tensor-wasm/scripts/deno-native-test.ts
 */
import { Kernels } from "../src/index.ts";
import { matrix, NativeKernels } from "../src/native.ts";

function assert(cond: boolean, msg: string): void {
  if (!cond) throw new Error(`FAIL: ${msg}`);
}
function close(a: number, b: number): boolean {
  return Math.abs(a - b) <= 1e-5 + 1e-4 * Math.max(Math.abs(a), Math.abs(b));
}

const native = NativeKernels.load();
assert(native !== undefined, "NativeKernels.load() found no binary — build the cdylib first");
const nk = native!;
console.log(`native kernels loaded from ${nk.libraryPath}`);
const wasm = await Kernels.load();

// addInto / mulInto agree with wasm
{
  const N = 1000;
  const a = Float32Array.from({ length: N }, (_, i) => Math.sin(i));
  const b = Float32Array.from({ length: N }, (_, i) => Math.cos(i) * 2);
  const nativeOut = nk.addInto(new Float32Array(N), a, b);
  const wa = wasm.fromArray(a, [N]);
  const wb = wasm.fromArray(b, [N]);
  const wout = wasm.zeros([N]);
  wasm.addInto(wout, wa, wb);
  const wasmOut = wout.toFloat32Array();
  for (let i = 0; i < N; i++) assert(close(nativeOut[i]!, wasmOut[i]!), `addInto[${i}]: ${nativeOut[i]} vs ${wasmOut[i]}`);
  const nativeMul = nk.mulInto(new Float32Array(N), a, b);
  for (let i = 0; i < N; i++) assert(close(nativeMul[i]!, a[i]! * b[i]!), `mulInto[${i}]`);
}

// matmulInto: hand-checked product + transposed-view (swapped strides, no copy)
{
  const a = matrix(new Float32Array([1, 2, 3, 4, 5, 6]), 2, 3);
  const b = matrix(new Float32Array([7, 8, 9, 10, 11, 12]), 3, 2);
  const out = matrix(new Float32Array(4), 2, 2);
  nk.matmulInto(out, a, b);
  const expected = [58, 64, 139, 154];
  for (let i = 0; i < 4; i++) assert(close(out.data[i]!, expected[i]!), `matmul[${i}]: ${out.data[i]}`);

  const aT = matrix(new Float32Array([1, 4, 2, 5, 3, 6]), 3, 2); // A^T stored
  const aViaStrides = { data: aT.data, offset: 0, rowStride: 1, colStride: 2, rows: 2, cols: 3 };
  const out2 = matrix(new Float32Array(4), 2, 2);
  nk.matmulInto(out2, aViaStrides, b);
  for (let i = 0; i < 4; i++) assert(close(out2.data[i]!, expected[i]!), `matmul-transposed[${i}]: ${out2.data[i]}`);
}

// solveInto agrees with wasm's solve on a well-conditioned system
{
  const n = 32;
  const aData = Float32Array.from({ length: n * n }, (_, i) => Math.sin(i) * 0.5);
  for (let i = 0; i < n; i++) aData[i * n + i] = 8 + Math.abs(aData[i * n + i]!);
  const bData = Float32Array.from({ length: n }, (_, i) => Math.cos(i));
  const nativeX = nk.solveInto(new Float32Array(n), matrix(aData, n, n), bData);
  const wa = wasm.fromArray(aData, [n, n]);
  const wb = wasm.fromArray(bData, [n]);
  const wx = wasm.zeros([n]);
  wasm.solveInto(wx, wa, wb);
  const wasmX = wx.toFloat32Array();
  for (let i = 0; i < n; i++) assert(close(nativeX[i]!, wasmX[i]!), `solve[${i}]: ${nativeX[i]} vs ${wasmX[i]}`);
}

// validation errors are plain RangeErrors, thrown before any FFI call
{
  let threw = false;
  try {
    nk.addInto(new Float32Array(3), new Float32Array(4), new Float32Array(3));
  } catch (e) {
    threw = e instanceof RangeError;
  }
  assert(threw, "length mismatch must throw RangeError");
}

nk.close();
console.log("deno-native-test: all checks passed");
