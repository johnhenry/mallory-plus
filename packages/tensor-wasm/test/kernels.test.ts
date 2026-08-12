import assert from "node:assert/strict";
import { test } from "node:test";
import { Kernels } from "../src/index.ts";

test("addInto writes the sum directly into a pre-allocated resident buffer", async () => {
  const kernels = await Kernels.load();
  const a = kernels.fromArray(new Float32Array([1, 2, 3, 4]), [4]);
  const b = kernels.fromArray(new Float32Array([10, 20, 30, 40]), [4]);
  const out = kernels.zeros([4]);
  kernels.addInto(out, a, b);
  assert.deepEqual([...out.toFloat32Array()], [11, 22, 33, 44]);
  a.free();
  b.free();
  out.free();
});

test("mulInto writes the product directly into a pre-allocated resident buffer", async () => {
  const kernels = await Kernels.load();
  const a = kernels.fromArray(new Float32Array([2, 3, 4]), [3]);
  const b = kernels.fromArray(new Float32Array([5, 6, 7]), [3]);
  const out = kernels.zeros([3]);
  kernels.mulInto(out, a, b);
  assert.deepEqual([...out.toFloat32Array()], [10, 18, 28]);
});

test("matmulInto matches a hand-computed 2x3 @ 3x2 product", async () => {
  const kernels = await Kernels.load();
  const a = kernels.fromArray(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
  const b = kernels.fromArray(new Float32Array([7, 8, 9, 10, 11, 12]), [3, 2]);
  const out = kernels.zeros([2, 2]);
  kernels.matmulInto(out, a, b);
  assert.deepEqual([...out.toFloat32Array()], [58, 64, 139, 154]);
});

test("matmulInto on a transposed view reads via strides — no copy needed", async () => {
  const kernels = await Kernels.load();
  // Store A^T directly (3x2); .transposed() reads it as (2x3) via strides.
  const aT = kernels.fromArray(new Float32Array([1, 4, 2, 5, 3, 6]), [3, 2]);
  const b = kernels.fromArray(new Float32Array([7, 8, 9, 10, 11, 12]), [3, 2]);
  const out = kernels.zeros([2, 2]);
  kernels.matmulInto(out, aT.transposed(), b);
  assert.deepEqual([...out.toFloat32Array()], [58, 64, 139, 154]);
});

test("matmulInto rejects inner-dimension mismatch and non-2-D operands", async () => {
  const kernels = await Kernels.load();
  const a = kernels.zeros([2, 3]);
  const bad = kernels.zeros([4, 2]);
  const out = kernels.zeros([2, 2]);
  assert.throws(() => kernels.matmulInto(out, a, bad), RangeError);
  const oneD = kernels.zeros([3]);
  assert.throws(() => kernels.matmulInto(out, oneD, bad), RangeError);
});

// ---- the M2 acceptance criteria (issue #3) ---------------------------------

test("addInto allocates ZERO times across repeated calls on resident buffers", async () => {
  const kernels = await Kernels.load();
  const a = kernels.fromArray(new Float32Array(1000).fill(1.5), [1000]);
  const b = kernels.fromArray(new Float32Array(1000).fill(2.5), [1000]);
  const out = kernels.zeros([1000]);
  const before = kernels.allocCallCount;
  for (let i = 0; i < 500; i++) {
    kernels.addInto(out, a, b);
  }
  assert.equal(
    kernels.allocCallCount,
    before,
    "addInto must not call alloc — the whole point of the ...Into interface",
  );
  assert.equal(out.toFloat32Array()[0], 4);
});

test("addInto over resident buffers beats a pure-JS loop at N=1e6 (reproduces the measured 1.78x)", async () => {
  const kernels = await Kernels.load();
  const N = 1_000_000;
  const aData = new Float32Array(N).fill(1.5);
  const bData = new Float32Array(N).fill(2.5);
  const a = kernels.fromArray(aData, [N]);
  const b = kernels.fromArray(bData, [N]);
  const out = kernels.zeros([N]);

  const bench = (fn: () => void, iters: number): number => {
    for (let i = 0; i < 3; i++) fn(); // warm up
    const start = process.hrtime.bigint();
    for (let i = 0; i < iters; i++) fn();
    return Number(process.hrtime.bigint() - start) / iters / 1e6; // ms/call
  };

  const jsOut = new Float32Array(N);
  const jsTime = bench(() => {
    for (let i = 0; i < N; i++) jsOut[i] = aData[i]! + bData[i]!;
  }, 20);

  const wasmTime = bench(() => kernels.addInto(out, a, b), 20);

  // docs/spikes/wasm-baseline.md measured 1.78x on this machine; require a
  // conservative >1.15x here so the test isn't flaky across CI hardware
  // while still failing if the ...Into path regresses back toward parity
  // (or worse, back toward the 2.27x-slower copying wrapper it replaced).
  const speedup = jsTime / wasmTime;
  assert.ok(
    speedup > 1.15,
    `expected addInto to beat pure JS meaningfully at N=1e6, got ${speedup.toFixed(2)}x (js=${jsTime.toFixed(3)}ms, wasm=${wasmTime.toFixed(3)}ms)`,
  );
});
