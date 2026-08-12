/**
 * Central finite-difference gradient checker (issue #8's acceptance
 * criterion) — perturbs each element of `x` by `eps` and compares against
 * the analytic gradient from `fn`'s backward pass. `fn` must return a
 * scalar (size-1) Variable; ops with non-scalar natural output should be
 * wrapped in `.sum()` by the caller, same as PyTorch's `gradcheck` convention.
 */
import assert from "node:assert/strict";
import { Tensor, type DType } from "mallory-tensor-core";
import { Variable } from "../src/index.ts";

function flattenNested(value: unknown): number[] {
  if (Array.isArray(value)) return value.flatMap(flattenNested);
  return [value as number];
}

function scalarValue(y: Variable): number {
  if (y.value.size !== 1) {
    throw new Error("checkGradient: fn must return a scalar Variable (wrap with .sum())");
  }
  return y.value.item() as number;
}

export function maxGradError(
  fn: (x: Variable) => Variable,
  x: Tensor,
  eps = 1e-4,
): number {
  const v = Variable.variable(x);
  const y = fn(v);
  y.backward();
  const analytic = v.grad;
  if (!analytic) throw new Error("checkGradient: no gradient was produced");

  const shape = [...x.shape];
  const dtype = x.dtype;
  const flatX = flattenNested(x.toArray());
  const flatGrad = flattenNested(analytic.contiguous().toArray());

  let maxAbsError = 0;
  for (let i = 0; i < flatX.length; i++) {
    const plus = [...flatX];
    plus[i] = (plus[i] as number) + eps;
    const minus = [...flatX];
    minus[i] = (minus[i] as number) - eps;
    const fPlus = scalarValue(fn(Variable.constant(Tensor.from(plus, { dtype }).reshape(shape))));
    const fMinus = scalarValue(
      fn(Variable.constant(Tensor.from(minus, { dtype }).reshape(shape))),
    );
    const numeric = (fPlus - fMinus) / (2 * eps);
    maxAbsError = Math.max(maxAbsError, Math.abs(numeric - (flatGrad[i] as number)));
  }
  return maxAbsError;
}

export function assertGradientMatches(
  fn: (x: Variable) => Variable,
  x: Tensor,
  tolerance = 1e-3,
): void {
  const err = maxGradError(fn, x);
  assert.ok(err < tolerance, `max gradient error ${err} exceeds tolerance ${tolerance}`);
}

export function randomTensor(shape: number[], dtype: DType = "f64", scale = 2): Tensor {
  const size = shape.reduce((a, b) => a * b, 1);
  const values = Array.from({ length: size }, () => (Math.random() * 2 - 1) * scale);
  return Tensor.from(values, { dtype }).reshape(shape);
}
