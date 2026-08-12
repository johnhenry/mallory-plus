/**
 * Metadata-only Arrow-type inference for Expr trees, used both to compute
 * plan schemas without executing (plan.ts's `planArrowSchema`) and to know
 * what Arrow type to build result vectors as during execution (execute.ts) —
 * one source of truth, so the two never disagree. Never touches column
 * values, only Field types from a Schema.
 *
 * v1 simplifications (documented, not spec'd by issue #19):
 * - Arithmetic promotes to float64 unless BOTH operands are int64, in which
 *   case the result stays int64. No wider numeric-type lattice (e.g.
 *   int32 + int32 -> float64, not int32) — keeps the type-inference surface
 *   small; users who need int32 arithmetic can `.cast()` back down.
 * - `fn.sum()` mirrors that rule (int64 in -> int64 out, everything else ->
 *   float64). `fn.count()` is always int64. `fn.mean()`/`fn.stddev()` are
 *   always float64.
 * - `fn.month()` produces int32.
 */
import { Bool, DataType, Field, Int32, Int64, Schema, Utf8, Float64 } from "apache-arrow";
import {
  AggExpr,
  ArithExpr,
  ColumnExpr,
  CompareExpr,
  Expr,
  LiteralExpr,
  LogicalExpr,
  NotExpr,
  OverAllExpr,
  ScalarFnExpr,
} from "./expr.ts";

export function fieldFor(schema: Schema, name: string): Field {
  const f = (schema.fields as Field[]).find((field) => field.name === name);
  if (!f) {
    throw new Error(
      `no such column "${name}" (available: ${(schema.names as string[]).join(", ")})`,
    );
  }
  return f;
}

function isInt64(t: DataType): boolean {
  return DataType.isInt(t) && t.bitWidth === 64 && t.isSigned;
}

export function inferExprType(expr: Expr, schema: Schema): DataType {
  if (expr instanceof ColumnExpr) return fieldFor(schema, expr.name).type;
  if (expr instanceof LiteralExpr) {
    const v = expr.value;
    if (typeof v === "bigint") return new Int64();
    if (typeof v === "string") return new Utf8();
    if (typeof v === "boolean") return new Bool();
    return new Float64();
  }
  if (expr instanceof CompareExpr || expr instanceof LogicalExpr || expr instanceof NotExpr) {
    return new Bool();
  }
  if (expr instanceof ArithExpr) {
    const l = inferExprType(expr.left, schema);
    const r = inferExprType(expr.right, schema);
    return isInt64(l) && isInt64(r) ? new Int64() : new Float64();
  }
  if (expr instanceof AggExpr) {
    if (expr.op === "count") return new Int64();
    if (expr.op === "sum") {
      const inner = expr.inner ? inferExprType(expr.inner, schema) : new Float64();
      return isInt64(inner) ? new Int64() : new Float64();
    }
    return new Float64(); // mean, stddev
  }
  if (expr instanceof OverAllExpr) return inferExprType(expr.agg, schema);
  if (expr instanceof ScalarFnExpr) return new Int32(); // month()
  throw new Error(`inferExprType: unhandled expr (kind=${(expr as Expr).kind})`);
}
