/**
 * Functional-style gradient API (issue #8's "support both styles"), built on
 * the same tape as `Variable.backward()`. `fn` must return a scalar
 * (size-1) Variable.
 */
import type { Tensor } from "mallory-tensor-core";
import { Variable, enableGrad, noGrad } from "./variable.ts";

function runAndDiff(fn: (x: Variable) => Variable, x: Tensor): { value: Tensor; grad: Tensor } {
  const v = Variable.variable(x);
  const y = fn(v);
  y.backward();
  if (!v.grad) {
    throw new Error(
      "grad: the function did not use its input in a differentiable way (no path back to it on the tape)",
    );
  }
  return { value: y.value, grad: v.grad };
}

export const grad = {
  noGrad,
  enable: enableGrad,

  /** Returns a function computing d(fn(x))/dx. */
  of(fn: (x: Variable) => Variable): (x: Tensor) => Tensor {
    return (x: Tensor) => runAndDiff(fn, x).grad;
  },

  /** Like `of`, but also returns fn's value — one forward pass instead of two. */
  valueAndGrad(fn: (x: Variable) => Variable): (x: Tensor) => { value: Tensor; grad: Tensor } {
    return (x: Tensor) => runAndDiff(fn, x);
  },
};
