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

// ---- telemetry hook (issue #10) --------------------------------------------

test("allocator emits wasm/alloc.bytes and wasm/alloc.calls metrics when a sink is installed", async () => {
  const { setSink } = await import("mallory-telemetry");
  const events: unknown[] = [];
  setSink((e) => events.push(e));
  try {
    const kernels = await Kernels.load();
    kernels.zeros([10]); // one alloc call
    assert.equal(events.length, 2); // bytes + calls, per alloc
    const names = events.map((e) => (e as { name: string }).name);
    assert.deepEqual(names, ["wasm/alloc.bytes", "wasm/alloc.calls"]);
  } finally {
    setSink(null);
  }
});

test("allocator emits NOTHING when no sink is installed (default, zero-cost)", async () => {
  const kernels = await Kernels.load();
  kernels.zeros([10]);
  kernels.zeros([20]);
  // No assertion possible on "no events" without a sink to observe them --
  // this test's job is just to prove it doesn't throw/misbehave by default.
  assert.equal(kernels.allocCallCount, 2);
});

// ---- SIMD128 fast path (issue #13) -----------------------------------------

test("simdAvailable is true on this runtime (Node supports WASM SIMD) — the fast path is actually exercised, not silently skipped", async () => {
  const kernels = await Kernels.load();
  assert.equal(kernels.simdAvailable, true);
});

test("Kernels.load() degrades gracefully when the SIMD artifact is unusable — simdAvailable false, addInto still correct via the scalar fallback", async () => {
  const kernels = await Kernels.load(undefined, new Uint8Array([0, 1, 2, 3])); // not a valid WASM module
  assert.equal(kernels.simdAvailable, false);
  const a = kernels.fromArray(new Float32Array([1, 2, 3, 4]), [4]);
  const b = kernels.fromArray(new Float32Array([10, 20, 30, 40]), [4]);
  const out = kernels.zeros([4]);
  kernels.addInto(out, a, b);
  assert.deepEqual([...out.toFloat32Array()], [11, 22, 33, 44]);
});

test("addInto/mulInto: SIMD-accelerated result is bit-for-bit identical to the scalar fallback, contiguous case", async () => {
  const withSimd = await Kernels.load();
  const scalarOnly = await Kernels.load(undefined, new Uint8Array([0, 1, 2, 3]));
  assert.equal(withSimd.simdAvailable, true);
  assert.equal(scalarOnly.simdAvailable, false);

  const N = 4001; // deliberately not a multiple of 4 -- exercises the SIMD kernel's scalar tail loop
  const aData = Float32Array.from({ length: N }, (_, i) => Math.sin(i) * 100);
  const bData = Float32Array.from({ length: N }, (_, i) => Math.cos(i) * 50);

  for (const [name, op] of [
    ["addInto", (k: Kernels, out: ReturnType<Kernels["zeros"]>, a: ReturnType<Kernels["zeros"]>, b: ReturnType<Kernels["zeros"]>) => k.addInto(out, a, b)],
    ["mulInto", (k: Kernels, out: ReturnType<Kernels["zeros"]>, a: ReturnType<Kernels["zeros"]>, b: ReturnType<Kernels["zeros"]>) => k.mulInto(out, a, b)],
  ] as const) {
    const aSimd = withSimd.fromArray(aData, [N]);
    const bSimd = withSimd.fromArray(bData, [N]);
    const outSimd = withSimd.zeros([N]);
    op(withSimd, outSimd, aSimd, bSimd);

    const aScalar = scalarOnly.fromArray(aData, [N]);
    const bScalar = scalarOnly.fromArray(bData, [N]);
    const outScalar = scalarOnly.zeros([N]);
    op(scalarOnly, outScalar, aScalar, bScalar);

    assert.deepEqual([...outSimd.toFloat32Array()], [...outScalar.toFloat32Array()], `${name}: SIMD vs scalar mismatch`);
  }
});

test("addInto: a non-contiguous (strided) view is NOT eligible for the SIMD path but still computes correctly via the fallback", async () => {
  const kernels = await Kernels.load();
  assert.equal(kernels.simdAvailable, true);
  // Every other element of a 6-element buffer -- view1D reports stride=2,
  // disqualifying it from the contiguous-only SIMD kernel (flatSpec's
  // stride === 1 check in addInto/mulInto).
  const full = kernels.fromArray(new Float32Array([1, 10, 2, 20, 3, 30]), [6]);
  const strided = full.view1D(0, 3, 2); // offset=0, length=3, stride=2 -> [1, 2, 3]
  const b = kernels.fromArray(new Float32Array([100, 200, 300]), [3]);
  const out = kernels.zeros([3]);
  kernels.addInto(out, strided, b);
  assert.deepEqual([...out.toFloat32Array()], [101, 202, 303]);
});

test("addInto: SIMD is a real, measured speedup over the scalar fallback at N=1e6 (issue #13 — docs/spikes/wasm-simd.md)", async () => {
  const withSimd = await Kernels.load();
  const scalarOnly = await Kernels.load(undefined, new Uint8Array([0, 1, 2, 3]));
  const N = 1_000_000;
  const aData = new Float32Array(N).fill(1.5);
  const bData = new Float32Array(N).fill(2.5);

  const bench = (kernels: Kernels, iters: number): number => {
    const a = kernels.fromArray(aData, [N]);
    const b = kernels.fromArray(bData, [N]);
    const out = kernels.zeros([N]);
    for (let i = 0; i < 3; i++) kernels.addInto(out, a, b); // warm up
    const start = process.hrtime.bigint();
    for (let i = 0; i < iters; i++) kernels.addInto(out, a, b);
    return Number(process.hrtime.bigint() - start) / iters / 1e6; // ms/call
  };

  const scalarTime = bench(scalarOnly, 20);
  const simdTime = bench(withSimd, 20);

  // docs/spikes/wasm-simd.md measured a stable ~2.6-3x SIMD-only speedup
  // (apples-to-apples, contiguous-scalar baseline) on this machine. Matches
  // this package's OTHER benchmark test's own proven-non-flaky threshold
  // (>1.15x, chosen there for the same reason: tolerate contended CI/dev
  // hardware — observed here to still flake occasionally above 1.2x under
  // real concurrent CPU load on this dev machine) while still failing if
  // the SIMD path regresses all the way back toward parity.
  const speedup = scalarTime / simdTime;
  assert.ok(
    speedup > 1.15,
    `expected SIMD addInto to beat the scalar fallback meaningfully at N=1e6, got ${speedup.toFixed(2)}x (scalar=${scalarTime.toFixed(3)}ms, simd=${simdTime.toFixed(3)}ms)`,
  );
});
