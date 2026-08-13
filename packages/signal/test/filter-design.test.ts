/**
 * butter (issue #44). Section-by-section SOS coefficients are NOT expected
 * to match scipy.signal.butter's exactly (scipy's own zero/pole-to-section
 * grouping isn't fixed either -- verified empirically: for some orders it
 * pairs 2 zeros with the real pole and 1 with each complex pair, for
 * others the opposite) -- so these tests verify END-TO-END FILTERING
 * BEHAVIOR (apply our own sosFilter to our own butter() output, compare
 * against scipy's sosfilt applied to scipy's own butter() output, on the
 * SAME input signal), which is invariant to section grouping and is the
 * property that actually matters.
 */
import assert from "node:assert/strict";
import { test } from "node:test";
import { Tensor } from "mallory-tensor-core";
import { butter, sosFilter } from "../src/index.ts";
import { runScipyOracle, SCIPY_SKIP_REASON } from "./helpers.ts";

function close(a: ArrayLike<number>, b: ArrayLike<number>, tol: number): void {
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++) assert.ok(Math.abs((a[i] as number) - (b[i] as number)) < tol, `index ${i}: ${a[i]} vs ${b[i]}`);
}

function testSignal(n: number): number[] {
  return Array.from({ length: n }, (_, i) => Math.sin(i * 0.3) + 0.3 * Math.sin(i * 1.7) + 0.1 * Math.sin(i * 2.9));
}

const CASES: Array<{ order: number; wn: number; btype: "lowpass" | "highpass" }> = [
  { order: 1, wn: 0.1, btype: "lowpass" },
  { order: 2, wn: 0.3, btype: "lowpass" },
  { order: 3, wn: 0.25, btype: "lowpass" },
  { order: 4, wn: 0.6, btype: "highpass" },
  { order: 5, wn: 0.4, btype: "highpass" },
  { order: 6, wn: 0.5, btype: "lowpass" },
];

for (const { order, wn, btype } of CASES) {
  test(`butter(${order}, ${wn}, ${btype}): filtered output matches scipy end-to-end`, { skip: SCIPY_SKIP_REASON }, () => {
    const sos = butter(order, wn, { btype });
    const x = testSignal(30);
    const mine = sosFilter(sos, Tensor.from(x, { dtype: "f64" })).toArray() as number[];

    const oracleSos = runScipyOracle<{ sos: number[][] }>({ op: "butter_sos", order, wn, btype }).sos;
    const oracleFiltered = runScipyOracle<{ y: number[] }>({ op: "sosfilt", sos: oracleSos, x }).y;

    close(mine, oracleFiltered, 1e-6);
  });
}

test("butter: an even order produces order/2 sections, an odd order produces (order+1)/2 sections", () => {
  assert.equal(butter(2, 0.3, { btype: "lowpass" }).length, 1);
  assert.equal(butter(4, 0.3, { btype: "lowpass" }).length, 2);
  assert.equal(butter(3, 0.3, { btype: "lowpass" }).length, 2);
  assert.equal(butter(5, 0.3, { btype: "lowpass" }).length, 3);
});

test("butter: lowpass attenuates a high-frequency sinusoid far more than a low-frequency one", () => {
  const sos = butter(4, 0.2, { btype: "lowpass" });
  const n = 200;
  const lowFreq = Array.from({ length: n }, (_, i) => Math.sin(i * 0.05)); // well within passband
  const highFreq = Array.from({ length: n }, (_, i) => Math.sin(i * 2.5)); // well into stopband

  const lowOut = sosFilter(sos, Tensor.from(lowFreq, { dtype: "f64" })).toArray() as number[];
  const highOut = sosFilter(sos, Tensor.from(highFreq, { dtype: "f64" })).toArray() as number[];

  // Compare steady-state (post-transient) RMS amplitude vs the input's own.
  const rms = (arr: number[]) => Math.sqrt(arr.reduce((s, v) => s + v * v, 0) / arr.length);
  const lowRatio = rms(lowOut.slice(100)) / rms(lowFreq.slice(100));
  const highRatio = rms(highOut.slice(100)) / rms(highFreq.slice(100));

  assert.ok(lowRatio > 0.9, `passband signal should pass through near-unchanged, ratio=${lowRatio}`);
  assert.ok(highRatio < 0.1, `stopband signal should be strongly attenuated, ratio=${highRatio}`);
});

test("butter: highpass attenuates a low-frequency sinusoid far more than a high-frequency one", () => {
  const sos = butter(4, 0.3, { btype: "highpass" });
  const n = 200;
  const lowFreq = Array.from({ length: n }, (_, i) => Math.sin(i * 0.02));
  const highFreq = Array.from({ length: n }, (_, i) => Math.sin(i * 2.8));

  const lowOut = sosFilter(sos, Tensor.from(lowFreq, { dtype: "f64" })).toArray() as number[];
  const highOut = sosFilter(sos, Tensor.from(highFreq, { dtype: "f64" })).toArray() as number[];

  const rms = (arr: number[]) => Math.sqrt(arr.reduce((s, v) => s + v * v, 0) / arr.length);
  const lowRatio = rms(lowOut.slice(100)) / rms(lowFreq.slice(100));
  const highRatio = rms(highOut.slice(100)) / rms(highFreq.slice(100));

  assert.ok(lowRatio < 0.1, `stopband signal should be strongly attenuated, ratio=${lowRatio}`);
  assert.ok(highRatio > 0.9, `passband signal should pass through near-unchanged, ratio=${highRatio}`);
});

test("butter: rejects a non-positive-integer order and a wn outside (0, 1)", () => {
  assert.throws(() => butter(0, 0.5, { btype: "lowpass" }), RangeError);
  assert.throws(() => butter(1.5, 0.5, { btype: "lowpass" }), RangeError);
  assert.throws(() => butter(2, 0, { btype: "lowpass" }), RangeError);
  assert.throws(() => butter(2, 1, { btype: "lowpass" }), RangeError);
  assert.throws(() => butter(2, 1.5, { btype: "lowpass" }), RangeError);
});
