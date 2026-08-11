import assert from "node:assert/strict";
import { test } from "node:test";
import {
  ComplexNumber,
  Decimal,
  Fraction,
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

test("complex boxed<->flat round-trip (ComplexTensor edge format)", () => {
  const boxed = [new ComplexNumber(1, 2), new ComplexNumber(3, -4)];
  const parts = complexToParts(boxed);
  assert.deepEqual([...parts.real], [1, 3]);
  assert.deepEqual([...parts.imag], [2, -4]);
  const round = partsToComplex(parts);
  assert.ok(boxed[0]!.equals(round[0]!));
  assert.ok(boxed[1]!.equals(round[1]!));
});
