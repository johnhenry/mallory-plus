/**
 * DualNumber forward-mode oracle sanity suite (docs/PLAN.md §6.1: third
 * gradient oracle). Validates the oracle-vs-finite-difference plumbing
 * before tensor-autograd exists; the same helpers move to adapter-math
 * when that package lands. Pure JS — runs where Python isn't available.
 */
import assert from "node:assert/strict";
import { test } from "node:test";
import { DualNumber } from "mallory-math";

type ScalarFn = (x: DualNumber) => DualNumber;

function dualGrad(fn: ScalarFn, x: number): number {
  return DualNumber.derivative((d: DualNumber) => fn(d), x);
}

function finiteDifference(fn: (x: number) => number, x: number, h = 1e-6): number {
  return (fn(x + h) - fn(x - h)) / (2 * h);
}

const CASES: Array<{
  name: string;
  dual: ScalarFn;
  plain: (x: number) => number;
  points: number[];
}> = [
  {
    name: "x^2 + 3x",
    dual: (x) => x.multiply(x).add(x.multiply(DualNumber.constant(3))),
    plain: (x) => x * x + 3 * x,
    points: [-2, 0, 1.5, 10],
  },
  {
    name: "sin(x)",
    dual: (x) => DualNumber.sin(x),
    plain: Math.sin,
    points: [0, 0.5, Math.PI / 2, 3],
  },
  {
    name: "exp(x) * cos(x)",
    dual: (x) => DualNumber.exp(x).multiply(DualNumber.cos(x)),
    plain: (x) => Math.exp(x) * Math.cos(x),
    points: [-1, 0, 1, 2],
  },
  {
    name: "sqrt(x^2 + 1)",
    dual: (x) =>
      DualNumber.sqrt(x.multiply(x).add(DualNumber.constant(1))),
    plain: (x) => Math.sqrt(x * x + 1),
    points: [-3, 0, 0.25, 5],
  },
];

for (const { name, dual, plain, points } of CASES) {
  test(`DualNumber derivative matches finite differences: ${name}`, () => {
    for (const x of points) {
      const forward = dualGrad(dual, x);
      const numeric = finiteDifference(plain, x);
      const bound = 1e-5 + 1e-5 * Math.abs(numeric);
      assert.ok(
        Math.abs(forward - numeric) <= bound,
        `${name} at x=${x}: dual ${forward} vs finite-diff ${numeric}`,
      );
    }
  });
}

test("dualGradN: multivariate gradient via mallory's gradient driver", () => {
  // f(x, y) = x^2 * y + y^3 ; df/dx = 2xy ; df/dy = x^2 + 3y^2
  const grad = DualNumber.gradient(
    (xs: DualNumber[]) => {
      const [x, y] = xs as [DualNumber, DualNumber];
      return x.multiply(x).multiply(y).add(y.multiply(y).multiply(y));
    },
    [2, 3],
  );
  assert.ok(Math.abs((grad[0] as number) - 12) < 1e-9, `df/dx: ${grad[0]}`);
  assert.ok(Math.abs((grad[1] as number) - 31) < 1e-9, `df/dy: ${grad[1]}`);
});
