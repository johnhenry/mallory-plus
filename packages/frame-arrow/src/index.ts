/**
 * mallory-frame-arrow — an immutable, expression-oriented Frame/Series
 * dataframe library on top of apache-arrow (pinned exactly 21.2.0).
 *
 * See docs/spikes/arrow-parity.md for the v1 type claims and the JS-API
 * sharp edges this package's internals are built to route around, and
 * src/plan.ts's module doc comment for the `.collect()` materialization
 * boundary / column-pruning design.
 */
export { Frame, GroupBy, desc } from "./frame.ts";
export type { FieldDescriptor, FillNullValue, JoinHow, JoinOptions, SortKey } from "./frame.ts";
export { Series } from "./series.ts";
export {
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
  col,
  fn,
  lit,
  type AggOp,
  type ArithOp,
  type CompareOp,
  type LogicalOp,
  type Scalar,
  type ScalarFnOp,
} from "./expr.ts";
export { UnsupportedTypeError, describeField, describeSchema, type DType } from "./dtype.ts";
export { bigintSafeReplacer, stringifyRows } from "./safe-json.ts";
