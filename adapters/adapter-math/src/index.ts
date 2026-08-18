/**
 * mallory-adapter-math — the mallory-math bridge (docs/PLAN.md's adapter
 * cluster). v1: Matrix/Vector <-> Tensor conversion (issue #14). Later
 * issues land here too: Symbolic -> tensor-compile IR (#15), the DualNumber
 * forward-mode gradient oracle (#17).
 */
export { fromMatrix, fromVector, toMatrix, toVector, type ConvertOptions } from "./matrix.ts";
export { compileExpr, UnsupportedExprError, type CompileExprOptions } from "./expr.ts";
export { compileFrameExpr, UnsupportedFrameExprError } from "./frame-expr.ts";
export * as linalg from "./linalg.ts";
export { toCSR, toDense, type CSRGraph, type ToDenseOptions } from "./csr.ts";
export { convolve, fft, fftPadded, ifft, realSignal, type ComplexSignal } from "./fft.ts";
export {
  correlation,
  Distributions,
  HypothesisTests,
  linearRegression,
  mean,
  median,
  percentile,
  populationStandardDeviation,
  populationVariance,
  SpecialFunctions,
  standardDeviation,
  variance,
  type ContinuousDistribution,
  type DiscreteDistribution,
  type LinearFit,
} from "./stats.ts";
