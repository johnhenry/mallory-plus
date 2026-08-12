/**
 * Special functions / distributions / descriptive statistics bridge (issue
 * #35) for the future SciPy-equivalent tier (docs/PLAN.md §6.3). Same
 * "reference-speed now, native kernels later" framing as this package's own
 * `linalg.ts` -- these are direct delegations to mallory-math, not
 * reimplementations, and there is no native WASM kernel in v1.
 *
 * `SpecialFunctions` (gamma/lnGamma/beta/erf/erfc/regularizedGammaP,Q/
 * regularizedIncompleteBeta) and `Distributions`/`HypothesisTests` are pure
 * scalar functions with no tensor-shape aspect at all -- they're re-exported
 * verbatim below, the same "no wrapping needed" pattern
 * `mallory-scalar-types` already uses for `ComplexNumber`/`Rational`/
 * `Decimal`.
 *
 * `Statistics.ts`'s array-in descriptive-stats functions DO need a shape
 * adapter: they require mallory-math's own `Vector<number>` (an `Array`
 * subclass with `x`/`y`/`z`/`t` getters a plain array literal doesn't
 * structurally satisfy -- `Vector.fromArray` is used below rather than
 * spreading into `new Vector(...)`, which (per `Vector`'s own doc comment)
 * mis-handles a single-element input as a *length*, or the inherited
 * `Vector.from` (typed as returning a plain array, not narrowed to
 * `Vector<T>`, since it's `Array.from`'s static signature, not overridden)),
 * not a plain
 * `Float64Array`/`number[]` -- so the most commonly useful subset gets thin
 * typed-array-accepting wrappers below. Not every one of Statistics.ts's ~25
 * functions is wrapped (this is a v1 subset, not an exhaustive port) -- add
 * more the same way if a concrete need shows up.
 */
export { Distributions, HypothesisTests, SpecialFunctions } from "mallory-math";
export type { ContinuousDistribution, DiscreteDistribution } from "mallory-math";

import { Statistics, Vector } from "mallory-math";

export interface LinearFit {
  readonly slope: number;
  readonly intercept: number;
}

function toVec(data: Float64Array | readonly number[]): Vector<number> {
  return Vector.fromArray(data);
}

/** Arithmetic mean. */
export function mean(data: Float64Array | readonly number[]): number {
  return Statistics.mean(toVec(data));
}

/** Sample variance (N-1 denominator). */
export function variance(data: Float64Array | readonly number[]): number {
  return Statistics.variance(toVec(data));
}

/** Sample standard deviation (N-1 denominator). */
export function standardDeviation(data: Float64Array | readonly number[]): number {
  return Statistics.standardDeviation(toVec(data));
}

/** Population variance (N denominator). */
export function populationVariance(data: Float64Array | readonly number[]): number {
  return Statistics.populationVariance(toVec(data));
}

/** Population standard deviation (N denominator). */
export function populationStandardDeviation(data: Float64Array | readonly number[]): number {
  return Statistics.populationStandardDeviation(toVec(data));
}

export function median(data: Float64Array | readonly number[]): number {
  return Statistics.median(toVec(data));
}

/** The n-th percentile, as a FRACTION in [0, 1] (matching mallory-math's own `Statistics.percentile` convention exactly -- e.g. `0.5` for the median, not `50`). */
export function percentile(data: Float64Array | readonly number[], n: number): number {
  return Statistics.percentile(toVec(data), n);
}

/** Pearson correlation coefficient between two equal-length samples. */
export function correlation(x: Float64Array | readonly number[], y: Float64Array | readonly number[]): number {
  return Statistics.correlation(toVec(x), toVec(y));
}

/** Ordinary least-squares fit `y = slope*x + intercept`. */
export function linearRegression(x: Float64Array | readonly number[], y: Float64Array | readonly number[]): LinearFit {
  const [slope, intercept] = Statistics.linearRegression(toVec(x), toVec(y));
  return { slope: slope as number, intercept: intercept as number };
}
