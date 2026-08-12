/**
 * mallory-adapter-math — the mallory-math bridge (docs/PLAN.md's adapter
 * cluster). v1: Matrix/Vector <-> Tensor conversion (issue #14). Later
 * issues land here too: Symbolic -> tensor-compile IR (#15), the DualNumber
 * forward-mode gradient oracle (#17).
 */
export { fromMatrix, fromVector, toMatrix, toVector, type ConvertOptions } from "./matrix.ts";
export { compileExpr, UnsupportedExprError, type CompileExprOptions } from "./expr.ts";
export * as linalg from "./linalg.ts";
export { toCSR, toDense, type CSRGraph, type ToDenseOptions } from "./csr.ts";
