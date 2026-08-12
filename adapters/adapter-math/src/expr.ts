/**
 * Symbolic Expr -> tensor-compile IR bridge (issue #15) — the high-leverage
 * bridge. mallory-math's tagged-union `Expr` AST maps almost 1:1 onto
 * tensor-compile's IR, which makes vectorized symbolic expressions nearly
 * free: differentiate symbolically in mallory-math (`Symbolic.differentiate`),
 * then evaluate the result over million-element tensors here.
 *
 * This replaces the originally-planned bespoke string-expression parser:
 * reuse `Symbolic.parse`, so a string becomes an AST becomes validated IR —
 * never JS. Fully compatible with non-goal 2 (no `eval`): the IR and its
 * emitter (this file, `evalWithGrad` in tensor-compile) stay under our
 * control end to end.
 */
import { Symbolic, type Expr } from "mallory-math";
import { CompiledFn, type BinaryOp, type CmpOp, type IRNode, type UnaryOp } from "mallory-tensor-compile";

/**
 * Thrown for `Expr` node shapes tensor-compile's elementwise IR structurally
 * cannot represent — `sum`/`product` (reductions over a bound range) and the
 * integer-domain `gcd`/`lcm` call2 ops. No silent fallback: these fail loudly
 * rather than silently producing a wrong or approximate tensor program.
 */
export class UnsupportedExprError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UnsupportedExprError";
  }
}

export interface CompileExprOptions {
  /**
   * Declares the input-tensor order: `variables[i]` becomes `CompiledFn`
   * input index `i`. Defaults to `Symbolic.freeVariables(expr)` (every
   * variable referenced anywhere in `expr`, alphabetical) when omitted.
   * Always validated against `expr` via `Symbolic.assertVariables` — an
   * undeclared variable throws `UndeclaredVariableError` (from mallory-math)
   * rather than silently evaluating to `NaN`.
   */
  variables?: readonly string[];
}

// FuncName -> UnaryOp is the identity map except "ln" -> "log" (tensor-compile's
// existing name, predating this bridge). Every other mallory-math FuncName
// spells identically to a tensor-compile UnaryOp by construction.
const UNARY_FUNC_MAP: Record<string, UnaryOp> = {
  sin: "sin",
  cos: "cos",
  tan: "tan",
  exp: "exp",
  ln: "log",
  sqrt: "sqrt",
  asin: "asin",
  acos: "acos",
  atan: "atan",
  sinh: "sinh",
  cosh: "cosh",
  tanh: "tanh",
  cot: "cot",
  sec: "sec",
  csc: "csc",
  asinh: "asinh",
  acosh: "acosh",
  atanh: "atanh",
  coth: "coth",
  sech: "sech",
  csch: "csch",
  acot: "acot",
  asec: "asec",
  acsc: "acsc",
  acoth: "acoth",
  asech: "asech",
  acsch: "acsch",
  abs: "abs",
  log10: "log10",
  log2: "log2",
  cbrt: "cbrt",
  floor: "floor",
  ceil: "ceil",
  round: "round",
  sign: "sign",
  trunc: "trunc",
  expm1: "expm1",
  log1p: "log1p",
  sigmoid: "sigmoid",
  erf: "erf",
  relu: "relu",
};

// atan2/hypot/min/max carry over directly; gcd/lcm are integer-domain (no
// elementwise-tensor meaning) and are deliberately left out of this map so
// they fall into the UnsupportedExprError path below.
const BINARY_CALL_MAP: Record<string, BinaryOp> = {
  atan2: "atan2",
  hypot: "hypot",
  min: "min",
  max: "max",
};

const BINARY_OP_MAP: Record<"add" | "sub" | "mul" | "div", BinaryOp> = {
  add: "add",
  sub: "sub",
  mul: "mul",
  div: "div",
};

function translate(e: Expr, index: ReadonlyMap<string, number>): IRNode {
  switch (e.type) {
    case "const":
      return { kind: "const", value: e.value };
    case "var": {
      const i = index.get(e.name);
      if (i === undefined) {
        // Shouldn't happen once Symbolic.assertVariables has run in compileExpr,
        // but stays a defensive check for direct translate() misuse.
        throw new RangeError(`compileExpr: variable "${e.name}" is not in the declared variable list`);
      }
      return { kind: "input", index: i };
    }
    case "add":
    case "sub":
    case "mul":
    case "div":
      return {
        kind: "binary",
        op: BINARY_OP_MAP[e.type],
        left: translate(e.left, index),
        right: translate(e.right, index),
      };
    case "pow":
      return { kind: "binary", op: "pow", left: translate(e.base, index), right: translate(e.exp, index) };
    case "neg":
      return { kind: "unary", op: "neg", arg: translate(e.arg, index) };
    case "func": {
      const op = UNARY_FUNC_MAP[e.name];
      if (!op) {
        throw new UnsupportedExprError(`compileExpr: unsupported function "${e.name}"`);
      }
      return { kind: "unary", op, arg: translate(e.arg, index) };
    }
    case "call2": {
      const op = BINARY_CALL_MAP[e.name];
      if (!op) {
        throw new UnsupportedExprError(
          `compileExpr: unsupported two-argument function "${e.name}" (gcd/lcm are integer-domain operations with no elementwise-tensor meaning)`,
        );
      }
      return { kind: "binary", op, left: translate(e.left, index), right: translate(e.right, index) };
    }
    case "cmp":
      return { kind: "cmp", op: e.op as CmpOp, left: translate(e.left, index), right: translate(e.right, index) };
    case "piecewise": {
      // branches[] is checked in order, first true wins; otherwise is the
      // final fallback — right-fold into nested (cond ? then : else) select
      // nodes so the LAST branch is the innermost "else".
      let node = translate(e.otherwise, index);
      for (let i = e.branches.length - 1; i >= 0; i--) {
        const branch = e.branches[i] as { cond: Expr; expr: Expr };
        node = { kind: "select", cond: translate(branch.cond, index), then: translate(branch.expr, index), else: node };
      }
      return node;
    }
    case "sum":
    case "product":
      throw new UnsupportedExprError(
        `compileExpr: "${e.type}" is a reduction over a bound range — outside tensor-compile's elementwise-only IR (issue #11's v1 scope). No silent fallback; evaluate it in mallory-math first if it can be resolved to a closed form.`,
      );
  }
}

/**
 * Compile a mallory-math `Expr` (or an already-parsed one — pass a string
 * through `Symbolic.parse` yourself, or let `Symbolic` do it for you
 * elsewhere) into a `CompiledFn` that evaluates over tensors.
 */
export function compileExpr(expr: Expr, options: CompileExprOptions = {}): CompiledFn {
  const variables = options.variables ?? Symbolic.freeVariables(expr);
  Symbolic.assertVariables(expr, [...variables]);
  const index = new Map(variables.map((name, i) => [name, i]));
  const node = translate(expr, index);
  return new CompiledFn(variables.length, node);
}
