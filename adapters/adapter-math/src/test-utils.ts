/**
 * DualNumber forward-mode gradient oracle (issue #17) — a third,
 * algorithmically independent way to check a gradient, alongside the
 * reverse-mode tape (`@johnhenry/math-plus-tensor-autograd`) and finite differences.
 * Forward-mode dual numbers propagate derivatives through the SAME
 * evaluation as the value (no separate backward pass, no tape), so a bug
 * shared between "the analytic formula" and "the tape's backward rule"
 * for some op won't also be present here — it's a genuinely different
 * algorithm, not just a second run of the same one.
 *
 * Also pure JS (no Python subprocess), so it runs anywhere the NumPy/torch
 * oracle (docs/TESTING.md) can't: browser CI, local watch mode, etc.
 *
 * Exported as a separate `@johnhenry/math-plus-adapter-math/test-utils` subpath — this
 * is test/oracle plumbing, not part of the adapter's main runtime surface
 * (`fromMatrix`/`compileExpr`/etc.), so consumers who don't need it don't
 * pull DualNumber-wrapping code into their main bundle.
 */
import { DualNumber } from "@johnhenry/math";

export type ScalarDualFn = (x: DualNumber) => DualNumber;
export type MultivariateDualFn = (xs: DualNumber[]) => DualNumber;

/** d(fn)/dx at `x`, via forward-mode automatic differentiation. `fn` must be built from `DualNumber`'s own primitives (add/multiply/divide/negate/exp/sin/cos/... — see `DualNumber`'s static helpers), not plain `Math.*` on `x` directly. */
export function dualGrad(fn: ScalarDualFn, x: number): number {
  return DualNumber.derivative(fn, x);
}

/** The full gradient of a scalar-valued multivariate `fn` at `xs`, via @johnhenry/math's forward-mode gradient driver (one dual-number sweep per input, matching how `DualNumber.gradient` itself is implemented). */
export function dualGradN(fn: MultivariateDualFn, xs: readonly number[]): number[] {
  return DualNumber.gradient(fn, [...xs]);
}
