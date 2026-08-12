import assert from "node:assert/strict";
import { test } from "node:test";
import {
  ComplexNumber,
  Decimal,
  Fraction,
  Interval,
  Rational,
  complexToParts,
  partsToComplex,
} from "../src/index.ts";

test("re-exports mallory-math scalar types", () => {
  const z = new ComplexNumber(3, 4);
  assert.equal(z.magnitude(), 5);
  const r = new Rational(2n, 4n);
  assert.equal(r.toString(), "1/2");
  const d = Decimal.fromString("1.5");
  assert.equal(d.add(Decimal.fromString("2.5")).toNumber(), 4);
});

test("Fraction is an alias of Rational", () => {
  assert.equal(Fraction, Rational);
});

test("Interval is re-exported and does real rigorous interval arithmetic", () => {
  const a = new Interval(1, 2);
  const b = new Interval(3, 5);
  const sum = a.add(b);
  assert.equal(sum.lo, 4);
  assert.equal(sum.hi, 7);
  const product = a.multiply(b);
  assert.equal(product.lo, 3); // 1*3
  assert.equal(product.hi, 10); // 2*5
  assert.ok(a.contains(1.5));
  assert.ok(!a.contains(2.5));
});

test("Interval: a point interval bounds a known f64 computation, useful as a rounding-error oracle", () => {
  // sin(pi/2) computed at f64 precision, wrapped as a degenerate interval,
  // still contains the exact value -- the basic property any f32-vs-f64
  // precision-bound check builds on (see tensor-webgpu's fusion tests for
  // the real usage against an actual GPU f32 result).
  const exact = Math.sin(Math.PI / 2);
  const bounded = Interval.point(exact);
  assert.ok(bounded.contains(exact));
  assert.equal(bounded.width, 0);
});

test("complex boxed<->flat round-trip (ComplexTensor edge format)", () => {
  const boxed = [new ComplexNumber(1, 2), new ComplexNumber(3, -4)];
  const parts = complexToParts(boxed);
  assert.deepEqual([...parts.real], [1, 3]);
  assert.deepEqual([...parts.imag], [2, -4]);
  const round = partsToComplex(parts);
  assert.ok(boxed[0]!.equals(round[0]!));
  assert.ok(boxed[1]!.equals(round[1]!));
});
