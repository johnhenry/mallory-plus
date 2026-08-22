/**
 * Symbolic Expr -> frame-arrow Expr bridge (issue #38) — a second
 * `Symbolic`-`Expr` compile target alongside `compileExpr`'s `tensor-compile`
 * IR bridge (expr.ts, issue #15).
 *
 * Once `frame-arrow`'s `fn.*` namespace grew real elementary math functions
 * (`packages/frame-arrow/src/expr.ts`'s `SCALAR_MATH_FUNCS`, on its own
 * merits — computed columns like `withColumns({ y: fn.sin(col('x')) })`),
 * this bridge fell out largely for free, exactly as issue #38 predicted:
 * `frame-arrow`'s `fn.*` names were deliberately spelled to match
 * `@johnhenry/math`'s `Symbolic` `FuncName` 1:1, so `compileFrameExpr`'s
 * function-node translation is a plain identity lookup — no
 * `UNARY_FUNC_MAP`-style rename table needed here (unlike `expr.ts`'s
 * `ln` -> `log` rename for `tensor-compile`).
 *
 * The payoff described in the issue: differentiate a computed column's
 * formula symbolically in `@johnhenry/math` (`Symbolic.differentiate`), then
 * compile the derivative into a second computed column with
 * `compileFrameExpr` —
 * ```ts
 * const formula = Symbolic.parse("sin(x) * x");
 * const derivative = Symbolic.differentiate(formula, "x");
 * frame.withColumns({
 *   y: compileFrameExpr(formula),
 *   dy_dx: compileFrameExpr(derivative),
 * });
 * ```
 * — each free variable in the `Symbolic` `Expr` becomes a `frame-arrow`
 * `col()` reference of the same name; frame-arrow's own "no such column"
 * error covers the undeclared-variable case (no separate validation pass
 * needed here, unlike `compileExpr`'s `Symbolic.assertVariables` — there is
 * no fixed positional input order to validate against).
 */
import { type Expr as SymbolicExpr } from "@johnhenry/math";
import {
  col,
  fn,
  lit,
  type Expr as FrameExpr,
  type ScalarMathFuncName,
  SCALAR_MATH_FUNCS,
} from "@johnhenry/math-plus-frame-arrow";

/**
 * Thrown for `Expr` node shapes `frame-arrow`'s `Expr` algebra structurally
 * cannot represent: `pow` and the two-argument functions (`atan2`/`hypot`/
 * `min`/`max`/`gcd`/`lcm` — `frame-arrow` has no binary-function or
 * exponentiation combinator, only `add`/`sub`/`mul`/`div`), `piecewise` (no
 * conditional-select combinator), and `sum`/`product` (a reduction over a
 * bound range — same structural gap `compileExpr` hits, see `UnsupportedExprError`
 * in expr.ts). No silent fallback: these fail loudly rather than silently
 * dropping part of the formula.
 */
export class UnsupportedFrameExprError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UnsupportedFrameExprError";
  }
}

const SCALAR_MATH_FUNC_SET: ReadonlySet<string> = new Set(SCALAR_MATH_FUNCS);

function isScalarMathFuncName(name: string): name is ScalarMathFuncName {
  return SCALAR_MATH_FUNC_SET.has(name);
}

function translate(e: SymbolicExpr): FrameExpr {
  switch (e.type) {
    case "const":
      return lit(e.value);
    case "var":
      return col(e.name);
    case "add":
      return translate(e.left).add(translate(e.right));
    case "sub":
      return translate(e.left).sub(translate(e.right));
    case "mul":
      return translate(e.left).mul(translate(e.right));
    case "div":
      return translate(e.left).div(translate(e.right));
    case "neg":
      return translate(e.arg).mul(-1);
    case "pow":
      throw new UnsupportedFrameExprError(
        `compileFrameExpr: "pow" has no frame-arrow equivalent (ArithOp is only add/sub/mul/div)`,
      );
    case "func": {
      if (!isScalarMathFuncName(e.name)) {
        // Structurally unreachable today (frame-arrow's SCALAR_MATH_FUNCS covers every
        // @johnhenry/math FuncName 1:1), but guarded rather than assumed in case the two
        // unions ever drift apart.
        throw new UnsupportedFrameExprError(`compileFrameExpr: unsupported function "${e.name}"`);
      }
      return fn[e.name](translate(e.arg));
    }
    case "call2":
      throw new UnsupportedFrameExprError(
        `compileFrameExpr: two-argument function "${e.name}" has no frame-arrow equivalent ` +
          `(no binary-function combinator; gcd/lcm are additionally integer-domain, with no elementwise-column meaning)`,
      );
    case "cmp": {
      const left = translate(e.left);
      const right = translate(e.right);
      switch (e.op) {
        case "lt":
          return left.lt(right);
        case "le":
          return left.lte(right);
        case "gt":
          return left.gt(right);
        case "ge":
          return left.gte(right);
        case "eq":
          return left.eq(right);
        case "ne":
          return left.ne(right);
      }
      break;
    }
    case "piecewise":
      throw new UnsupportedFrameExprError(
        `compileFrameExpr: "piecewise" has no frame-arrow equivalent (no conditional-select combinator)`,
      );
    case "sum":
    case "product":
      throw new UnsupportedFrameExprError(
        `compileFrameExpr: "${e.type}" is a reduction over a bound range — outside frame-arrow's per-row ` +
          `Expr algebra. No silent fallback; evaluate it in @johnhenry/math first if it can be resolved to a closed form.`,
      );
  }
  throw new UnsupportedFrameExprError(`compileFrameExpr: unhandled Symbolic Expr node`);
}

/**
 * Compile a `@johnhenry/math` `Symbolic` `Expr` into a `frame-arrow` `Expr`
 * usable inside `withColumns`/`filter` — each free variable becomes a
 * `col()` reference of the same name. See this module's doc comment for the
 * symbolic-differentiate-then-compile-a-second-column workflow issue #38
 * describes.
 */
export function compileFrameExpr(expr: SymbolicExpr): FrameExpr {
  return translate(expr);
}
