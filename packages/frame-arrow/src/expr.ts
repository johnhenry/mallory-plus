/**
 * Expression algebra for Frame's plan-building API: `col('age').gte(18).and(col('active').eq(true))`.
 *
 * Expr nodes are plain immutable data (never executed at construction time —
 * see plan.ts's module doc comment for the `.collect()` materialization
 * boundary this feeds into). Every Expr knows its own `requiredColumns()`,
 * which the planner uses for column pruning / predicate pushdown (execute.ts).
 *
 * Comparison/logical combinators (`eq/ne/lt/lte/gt/gte`, `and/or/not`) are
 * exactly what issue #19 asks for. Arithmetic combinators (`add/sub/mul/div`)
 * are a documented ADDITION beyond the issue's literal text: `withColumns`
 * is close to useless without them (`withColumns({ deviation: col('x').sub(fn.mean(col('x')).overAll()) })`
 * is the issue's own example and requires `.sub()` to exist), so this is a
 * judgment call made explicit rather than left implicit.
 */

export type Scalar = number | bigint | string | boolean | null;

function isExpr(x: unknown): x is Expr {
  return x instanceof Expr;
}

function toExpr(x: Expr | Scalar): Expr {
  return isExpr(x) ? x : new LiteralExpr(x);
}

export type CompareOp = "eq" | "ne" | "lt" | "lte" | "gt" | "gte";
export type LogicalOp = "and" | "or";
export type ArithOp = "add" | "sub" | "mul" | "div";
export type AggOp = "count" | "sum" | "mean" | "stddev";
export type ScalarFnOp = "month";

export abstract class Expr {
  abstract readonly kind: string;
  abstract requiredColumns(): ReadonlySet<string>;

  eq(other: Expr | Scalar): Expr {
    return new CompareExpr("eq", this, toExpr(other));
  }
  ne(other: Expr | Scalar): Expr {
    return new CompareExpr("ne", this, toExpr(other));
  }
  lt(other: Expr | Scalar): Expr {
    return new CompareExpr("lt", this, toExpr(other));
  }
  lte(other: Expr | Scalar): Expr {
    return new CompareExpr("lte", this, toExpr(other));
  }
  gt(other: Expr | Scalar): Expr {
    return new CompareExpr("gt", this, toExpr(other));
  }
  gte(other: Expr | Scalar): Expr {
    return new CompareExpr("gte", this, toExpr(other));
  }
  and(other: Expr): Expr {
    return new LogicalExpr("and", this, other);
  }
  or(other: Expr): Expr {
    return new LogicalExpr("or", this, other);
  }
  not(): Expr {
    return new NotExpr(this);
  }
  add(other: Expr | Scalar): Expr {
    return new ArithExpr("add", this, toExpr(other));
  }
  sub(other: Expr | Scalar): Expr {
    return new ArithExpr("sub", this, toExpr(other));
  }
  mul(other: Expr | Scalar): Expr {
    return new ArithExpr("mul", this, toExpr(other));
  }
  div(other: Expr | Scalar): Expr {
    return new ArithExpr("div", this, toExpr(other));
  }
}

export class ColumnExpr extends Expr {
  readonly kind = "column" as const;
  readonly name: string;
  constructor(name: string) {
    super();
    this.name = name;
  }
  requiredColumns(): ReadonlySet<string> {
    return new Set([this.name]);
  }
}

export class LiteralExpr extends Expr {
  readonly kind = "literal" as const;
  readonly value: Scalar;
  constructor(value: Scalar) {
    super();
    this.value = value;
  }
  requiredColumns(): ReadonlySet<string> {
    return new Set();
  }
}

export class CompareExpr extends Expr {
  readonly kind = "compare" as const;
  readonly op: CompareOp;
  readonly left: Expr;
  readonly right: Expr;
  constructor(op: CompareOp, left: Expr, right: Expr) {
    super();
    this.op = op;
    this.left = left;
    this.right = right;
  }
  requiredColumns(): ReadonlySet<string> {
    return union(this.left.requiredColumns(), this.right.requiredColumns());
  }
}

export class LogicalExpr extends Expr {
  readonly kind = "logical" as const;
  readonly op: LogicalOp;
  readonly left: Expr;
  readonly right: Expr;
  constructor(op: LogicalOp, left: Expr, right: Expr) {
    super();
    this.op = op;
    this.left = left;
    this.right = right;
  }
  requiredColumns(): ReadonlySet<string> {
    return union(this.left.requiredColumns(), this.right.requiredColumns());
  }
}

export class NotExpr extends Expr {
  readonly kind = "not" as const;
  readonly operand: Expr;
  constructor(operand: Expr) {
    super();
    this.operand = operand;
  }
  requiredColumns(): ReadonlySet<string> {
    return this.operand.requiredColumns();
  }
}

export class ArithExpr extends Expr {
  readonly kind = "arith" as const;
  readonly op: ArithOp;
  readonly left: Expr;
  readonly right: Expr;
  constructor(op: ArithOp, left: Expr, right: Expr) {
    super();
    this.op = op;
    this.left = left;
    this.right = right;
  }
  requiredColumns(): ReadonlySet<string> {
    return union(this.left.requiredColumns(), this.right.requiredColumns());
  }
}

/** `fn.count()/sum()/mean()/stddev()` — an aggregate reduction. Only valid inside
 * `groupBy().aggregate({...})` or wrapped in `.overAll()` inside `withColumns`. */
export class AggExpr extends Expr {
  readonly kind = "agg" as const;
  readonly op: AggOp;
  readonly inner: Expr | null;
  constructor(op: AggOp, inner: Expr | null) {
    super();
    this.op = op;
    this.inner = inner;
  }
  requiredColumns(): ReadonlySet<string> {
    return this.inner ? this.inner.requiredColumns() : new Set();
  }
  /** Broadcast this whole-column aggregate back as a same-length column inside `withColumns`. */
  overAll(): Expr {
    return new OverAllExpr(this);
  }
}

export class OverAllExpr extends Expr {
  readonly kind = "overAll" as const;
  readonly agg: AggExpr;
  constructor(agg: AggExpr) {
    super();
    this.agg = agg;
  }
  requiredColumns(): ReadonlySet<string> {
    return this.agg.requiredColumns();
  }
}

/** `fn.month(expr)` — a per-row date-part extraction, NOT a reduction (used inside withColumns). */
export class ScalarFnExpr extends Expr {
  readonly kind = "scalarFn" as const;
  readonly op: ScalarFnOp;
  readonly inner: Expr;
  constructor(op: ScalarFnOp, inner: Expr) {
    super();
    this.op = op;
    this.inner = inner;
  }
  requiredColumns(): ReadonlySet<string> {
    return this.inner.requiredColumns();
  }
}

function union(a: ReadonlySet<string>, b: ReadonlySet<string>): Set<string> {
  return new Set([...a, ...b]);
}

export function col(name: string): ColumnExpr {
  return new ColumnExpr(name);
}

export function lit(value: Scalar): LiteralExpr {
  return new LiteralExpr(value);
}

export const fn = {
  count: (expr?: Expr): AggExpr => new AggExpr("count", expr ?? null),
  sum: (expr: Expr): AggExpr => new AggExpr("sum", expr),
  mean: (expr: Expr): AggExpr => new AggExpr("mean", expr),
  stddev: (expr: Expr): AggExpr => new AggExpr("stddev", expr),
  /**
   * Calendar month (1-12) of a timestamp column. Documented v1 simplification:
   * extraction uses the UTC calendar day of the epoch value, NOT localized to
   * the column's timezone metadata — correct localization needs an IANA tz
   * database resolver, which is out of scope for this issue. Naive (no-tz)
   * timestamps are unaffected by this caveat (there's no tz to localize to).
   */
  month: (expr: Expr): ScalarFnExpr => new ScalarFnExpr("month", expr),
};
