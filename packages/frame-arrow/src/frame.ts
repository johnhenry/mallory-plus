/**
 * Frame — immutable, expression-oriented dataframe. See plan.ts's module doc
 * comment for the full explanation of the `.collect()` materialization
 * boundary and which accessors force it.
 */
import { type Field, Schema, Table, tableFromIPC, tableToIPC, type Vector } from "apache-arrow";
import { columnToArray } from "./access.ts";
import { tableToCSV } from "./csv.ts";
import { describeField, describeSchema, type DType, type FieldDescriptor } from "./dtype.ts";
import { collectPlan, collectPlanAsync } from "./execute.ts";
import type { Expr } from "./expr.ts";
import {
  type FillNullValue,
  type JoinOptions,
  planArrowSchema,
  planColumnNames,
  type PlanNode,
  type SortKey,
  type Wanted,
} from "./plan.ts";
import { Series } from "./series.ts";
import { frameToTensor } from "./tensor.ts";

export type { FillNullValue, JoinOptions, JoinHow, SortKey, Wanted } from "./plan.ts";
export type { FieldDescriptor } from "./dtype.ts";

/** `sortBy(desc('age'))` — descending sort key convenience. */
export function desc(column: string): SortKey {
  return { column, descending: true };
}

export class Frame {
  private readonly plan: PlanNode;
  private cachedTable: Table | undefined;

  private constructor(plan: PlanNode, cachedTable?: Table) {
    this.plan = plan;
    this.cachedTable = cachedTable;
  }

  /**
   * @internal Used by Frame's own transform methods and GroupBy.aggregate()
   * to wrap a new plan node. Not private-to-the-class because GroupBy (a
   * separate class) needs it too; prefer `Frame.fromArrow`/`Frame.fromIPC`
   * as actual entry points — this does no validation of the node itself.
   */
  static internalFromPlan(plan: PlanNode): Frame {
    return new Frame(plan);
  }

  static fromArrow(table: Table): Frame {
    return new Frame({ kind: "source", table }, table);
  }

  static fromIPC(bytes: Uint8Array): Frame {
    const table = tableFromIPC(bytes) as Table;
    return Frame.fromArrow(table);
  }

  /**
   * Build a Frame backed by a LAZY source (issue #32) — data isn't read
   * until something downstream actually needs it (`collectAsync()`, not
   * the synchronous `collect()`/terminal accessors; see plan.ts's module
   * doc for why the two are separate). `schema` must be resolvable
   * up-front and cheaply (e.g. from a Parquet file's footer metadata,
   * never its row data) — `read(wanted)` is what actually materializes
   * rows, honoring the SAME column-pruning `wanted` conveys to every other
   * plan node kind.
   *
   * Meant for adapter/format packages (e.g. `mallory-frame-parquet`'s
   * `scanParquetLazy`) to build on, not typical application code.
   */
  static fromLazySource(schema: Schema, read: (wanted: Wanted) => Promise<Table>): Frame {
    return Frame.internalFromPlan({ kind: "lazySource", schema, read });
  }

  /** Static form. See also the `.concat(...others)` instance method (a thin wrapper around this). */
  static concat(frames: readonly Frame[]): Frame {
    if (frames.length === 0) throw new Error("Frame.concat: no frames given");
    if (frames.length === 1) return frames[0] as Frame;
    return Frame.internalFromPlan({ kind: "concat", inputs: frames.map((f) => f.plan) });
  }

  /** Instance form of `Frame.concat([this, ...others])`. Both exist (issue #19 left the choice open); use whichever reads better at the call site. */
  concat(...others: readonly Frame[]): Frame {
    return Frame.concat([this, ...others]);
  }

  // ---- metadata: never collects (see plan.ts doc comment) ----

  get schema(): FieldDescriptor[] {
    return describeSchema(planArrowSchema(this.plan));
  }

  get columns(): string[] {
    return planColumnNames(this.plan);
  }

  // ---- plan-building: lazy, never touches the underlying Table ----

  select(...names: string[]): Frame {
    return Frame.internalFromPlan({ kind: "select", input: this.plan, columns: names });
  }

  drop(...names: string[]): Frame {
    return Frame.internalFromPlan({ kind: "drop", input: this.plan, columns: names });
  }

  rename(mapping: Readonly<Record<string, string>>): Frame {
    return Frame.internalFromPlan({ kind: "rename", input: this.plan, mapping });
  }

  withColumns(exprs: Readonly<Record<string, Expr>>): Frame {
    return Frame.internalFromPlan({
      kind: "withColumns",
      input: this.plan,
      exprs: new Map(Object.entries(exprs)),
    });
  }

  filter(predicate: Expr): Frame {
    return Frame.internalFromPlan({ kind: "filter", input: this.plan, predicate });
  }

  sortBy(...keys: readonly (string | SortKey)[]): Frame {
    const normalized = keys.map((k) => (typeof k === "string" ? { column: k } : k));
    return Frame.internalFromPlan({ kind: "sortBy", input: this.plan, keys: normalized });
  }

  limit(n: number): Frame {
    return Frame.internalFromPlan({ kind: "limit", input: this.plan, n });
  }

  slice(start: number, end?: number): Frame {
    return Frame.internalFromPlan({ kind: "slice", input: this.plan, start, end });
  }

  groupBy(...keys: readonly string[]): GroupBy {
    return new GroupBy(this.plan, keys);
  }

  join(other: Frame, options: JoinOptions): Frame {
    return Frame.internalFromPlan({ kind: "join", left: this.plan, right: other.plan, options });
  }

  /** Drop rows with a null in any of `columns` (default: any column). */
  dropNull(...columns: readonly string[]): Frame {
    return Frame.internalFromPlan({
      kind: "dropNull",
      input: this.plan,
      columns: columns.length > 0 ? columns : undefined,
    });
  }

  fillNull(value: FillNullValue): Frame {
    return Frame.internalFromPlan({ kind: "fillNull", input: this.plan, value });
  }

  // ---- the collect() boundary ----

  /**
   * Explicit materialization. Memoizes its result on this Frame instance —
   * calling it again (including implicitly, from a terminal accessor below)
   * is a no-op. This is an internal cache, not a mutation of the Frame's
   * logical content (plan/schema are unaffected and this Frame's identity
   * is unchanged — `collect()` returns `this`).
   */
  collect(): Frame {
    if (!this.cachedTable) this.cachedTable = collectPlan(this.plan);
    return this;
  }

  /**
   * Async twin of `collect()` (issue #32) — required (instead of `collect()`)
   * for any Frame whose plan contains a lazy source (e.g. from
   * `scanParquetLazy`); safe to call on any OTHER Frame too, since a plan
   * with no lazy source resolves through unchanged. Populates the SAME
   * cache `collect()` does, so once this resolves, every synchronous
   * terminal accessor below (`toRows()`, `length`, `toArrow()`, ...) works
   * normally on the returned Frame — there's no need for `toRowsAsync()`/
   * `lengthAsync()`/etc. twins of each one.
   */
  async collectAsync(): Promise<Frame> {
    if (!this.cachedTable) this.cachedTable = await collectPlanAsync(this.plan);
    return this;
  }

  private materialize(): Table {
    this.collect();
    return this.cachedTable as Table;
  }

  // ---- terminal accessors: implicitly collect (see plan.ts doc comment) ----

  get length(): number {
    return this.materialize().numRows;
  }

  toArrow(): Table {
    return this.materialize();
  }

  toRows(): Record<string, unknown>[] {
    const table = this.materialize();
    const fields = table.schema.fields as Field[];
    const columns = fields.map((f) => {
      const desc = describeField(f);
      return { name: f.name, values: columnToArray(table.getChild(f.name) as Vector, desc.dtype) };
    });
    const rows: Record<string, unknown>[] = new Array(table.numRows);
    for (let i = 0; i < table.numRows; i++) {
      const row: Record<string, unknown> = {};
      for (const c of columns) row[c.name] = c.values[i];
      rows[i] = row;
    }
    return rows;
  }

  toCSV(): string {
    return tableToCSV(this.materialize());
  }

  toIPC(): Uint8Array {
    return tableToIPC(this.materialize(), "file");
  }

  nullCount(): Record<string, number> {
    const table = this.materialize();
    const out: Record<string, number> = {};
    for (const name of table.schema.names as string[]) {
      out[name] = (table.getChild(name) as Vector).nullCount;
    }
    return out;
  }

  /** Not in issue #19's explicit list, but needed for anything to get a Series out of a Frame — a documented addition. */
  getSeries(name: string): Series {
    const table = this.materialize();
    const vector = table.getChild(name);
    const field = (table.schema.fields as Field[]).find((f) => f.name === name);
    if (!vector || !field) throw new Error(`no such column "${name}"`);
    return new Series(name, vector, field.nullable);
  }

  async toTensor(options?: { dtype?: DType }): Promise<unknown> {
    return frameToTensor(this, options);
  }
}

export class GroupBy {
  private readonly input: PlanNode;
  private readonly keys: readonly string[];

  constructor(input: PlanNode, keys: readonly string[]) {
    this.input = input;
    this.keys = keys;
  }

  aggregate(spec: Readonly<Record<string, Expr>>): Frame {
    return Frame.internalFromPlan({
      kind: "aggregate",
      input: this.input,
      keys: this.keys,
      aggs: new Map(Object.entries(spec)),
    });
  }
}
