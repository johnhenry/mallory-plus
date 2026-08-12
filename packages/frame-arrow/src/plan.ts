/**
 * Frame's logical plan: an immutable expression DAG.
 *
 * ## The `.collect()` boundary
 *
 * Every Frame wraps a `PlanNode`. Plan-BUILDING methods (`select`, `drop`,
 * `rename`, `withColumns`, `filter`, `sortBy`, `limit`, `slice`, `groupBy`,
 * `join`, `concat`, `fillNull`, `dropNull`) never touch the underlying Arrow
 * `Table` — they just wrap the current plan in a new node and return a new
 * Frame. Nothing executes.
 *
 * `.collect()` is the explicit materialization boundary: it walks the plan,
 * computes required-column sets top-down (column pruning + predicate
 * pushdown, see execute.ts), and executes bottom-up into a single Arrow
 * `Table`. The result is memoized on the Frame instance (a private cache,
 * not a change to the Frame's logical identity/plan/schema) so repeated
 * `.collect()` calls — including implicit ones, see below — are cheap.
 *
 * DESIGN DECISION (not fully specified by issue #19): rather than force
 * every caller to remember `.collect()`, TERMINAL/materializing accessors —
 * `length`, `toRows()`, `toArrow()`, `toCSV()`, `toIPC()`, `nullCount()`,
 * `toTensor()`, and `Series` value accessors reached through them — call
 * `.collect()` implicitly on first use. `schema`/`columns` do NOT collect:
 * they're answered by walking the plan's *metadata* only (`planArrowSchema`
 * below), which never reads a column's values and therefore never throws on
 * an unsupported dtype in a column nobody asked for (this is what makes the
 * column-pruning proof in test/pruning.test.ts observable). Plan-building
 * methods remain lazy either way — only terminal reads ever execute.
 *
 * ## Column pruning & predicate pushdown
 *
 * Pruning is real, not cosmetic: `execute.ts`'s `requiredInputColumns()`
 * walks the plan top-down from whatever columns a terminal accessor actually
 * asked for, intersecting through `select`/`drop`/`rename`/`withColumns`/
 * `filter`/`sortBy` (whose predicates/expressions contribute their own
 * `Expr.requiredColumns()`), down to the `source` leaf, where only the
 * surviving column names are ever pulled out of the underlying Arrow Table
 * (`Table.select()`, which is zero-copy — it slices the Table's field/child
 * list, never decodes column values). A column that's pruned away is never
 * read, never type-checked, and — if it holds a dtype this package doesn't
 * support — never throws. `join` does not push pruning through its two
 * input plans in v1 (see execute.ts for why); everything else does.
 *
 * ## Lazy sources (issue #32) and why `collect()` stays synchronous
 *
 * `"lazySource"` is a second kind of leaf, alongside `"source"`: instead of
 * wrapping an already-materialized `Table`, it wraps a `schema` (resolved
 * EAGERLY, at node-construction time — see `Frame.fromLazySource`) plus a
 * `read(wanted)` callback that materializes the actual data lazily, on
 * demand, honoring the same column pruning every other node kind gets.
 * This exists for `mallory-frame-parquet`'s `scanParquetLazy` (a Parquet
 * file's rows shouldn't be read from disk until something downstream
 * actually needs them — the whole point of #32).
 *
 * `read()` returns a `Promise<Table>` — genuinely async, since real I/O is
 * involved. But `Frame`'s entire execution model (`execute()`/`collect()`,
 * and every terminal accessor built on them: `length`, `toRows()`,
 * `toArrow()`, ...) is synchronous, and changing that would be a breaking
 * change to the ALREADY-SHIPPED contract every existing Frame relies on
 * (issue #19). Rather than force that migration, lazy sources are handled
 * with a small ADDITIVE async layer instead: `Frame.collectAsync()`
 * pre-resolves every `"lazySource"` leaf in the plan into a real
 * `"source"` node (honoring the exact same `requiredInputColumns()`-driven
 * pruning `execute()` itself uses — see `resolveLazySources` in
 * execute.ts), THEN hands the now-fully-synchronous, lazy-source-free tree
 * to the existing, unmodified `execute()`/`collectPlan()`. Every case
 * body's actual Arrow manipulation logic is reused completely unchanged;
 * only a small tree-rewriting pre-pass is new. `schema`/`columns` need no
 * such pre-pass at all — a `"lazySource"` node's `schema` is already a
 * plain value by the time it exists, so `planArrowSchema()` (fully
 * synchronous, below) handles it exactly like a `"source"` node's own
 * `table.schema`, meaning `schema`/`columns` still never collect, even on
 * a Frame built from `scanParquetLazy` — the same guarantee eager Frames
 * already have.
 *
 * Calling the SYNCHRONOUS `collect()` (or any terminal accessor) directly
 * on a Frame whose plan still contains an unresolved `"lazySource"` throws
 * a clear error pointing at `collectAsync()`, rather than silently hanging
 * or producing wrong results.
 */
import { Field, Schema, type Table } from "apache-arrow";
import type { Expr } from "./expr.ts";
import { inferExprType } from "./schema-infer.ts";

/** Which columns a plan node's OUTPUT must supply — `"all"`, or an explicit set (used to prune everything upstream that isn't in it). */
export type Wanted = "all" | ReadonlySet<string>;

export interface SortKey {
  readonly column: string;
  readonly descending?: boolean;
}

export type JoinHow = "inner" | "left" | "right" | "outer";

export interface JoinOptions {
  /** Column name(s) present in both frames to equi-join on. */
  readonly on: string | readonly string[];
  readonly how?: JoinHow;
  /** Suffix appended to non-key right-side columns that collide with left-side names. Default "_right". */
  readonly suffix?: string;
}

export type FillNullValue = unknown | Readonly<Record<string, unknown>>;

export type PlanNode =
  | { readonly kind: "source"; readonly table: Table }
  | {
      readonly kind: "lazySource";
      /** Resolved eagerly at construction time (see this file's module doc) — never a function/Promise, so `planArrowSchema` stays fully synchronous. */
      readonly schema: Schema;
      readonly read: (wanted: Wanted) => Promise<Table>;
    }
  | { readonly kind: "select"; readonly input: PlanNode; readonly columns: readonly string[] }
  | { readonly kind: "drop"; readonly input: PlanNode; readonly columns: readonly string[] }
  | { readonly kind: "rename"; readonly input: PlanNode; readonly mapping: Readonly<Record<string, string>> }
  | { readonly kind: "withColumns"; readonly input: PlanNode; readonly exprs: ReadonlyMap<string, Expr> }
  | { readonly kind: "filter"; readonly input: PlanNode; readonly predicate: Expr }
  | { readonly kind: "sortBy"; readonly input: PlanNode; readonly keys: readonly SortKey[] }
  | { readonly kind: "limit"; readonly input: PlanNode; readonly n: number }
  | { readonly kind: "slice"; readonly input: PlanNode; readonly start: number; readonly end: number | undefined }
  | {
      readonly kind: "aggregate";
      readonly input: PlanNode;
      readonly keys: readonly string[];
      readonly aggs: ReadonlyMap<string, Expr>;
    }
  | { readonly kind: "join"; readonly left: PlanNode; readonly right: PlanNode; readonly options: JoinOptions }
  | { readonly kind: "concat"; readonly inputs: readonly PlanNode[] }
  | { readonly kind: "dropNull"; readonly input: PlanNode; readonly columns: readonly string[] | undefined }
  | { readonly kind: "fillNull"; readonly input: PlanNode; readonly value: FillNullValue };

/**
 * Output column names of a plan node, computed from metadata only (never
 * touches column values — reading a Field's `.name` never throws, unlike
 * reading its `.type` through describeField for an unsupported dtype).
 */
export function planColumnNames(node: PlanNode): string[] {
  return planArrowSchema(node).names as string[];
}

/** Output Arrow Schema of a plan node — metadata-only, never executes. */
export function planArrowSchema(node: PlanNode): Schema {
  switch (node.kind) {
    case "source":
      return node.table.schema;

    case "lazySource":
      return node.schema;

    case "select":
      return planArrowSchema(node.input).select(node.columns as string[]);

    case "drop": {
      const input = planArrowSchema(node.input);
      const drop = new Set(node.columns);
      const keep = (input.names as string[]).filter((n) => !drop.has(n));
      return input.select(keep);
    }

    case "rename": {
      const input = planArrowSchema(node.input);
      const fields = (input.fields as Field[]).map((f) =>
        node.mapping[f.name] ? f.clone({ name: node.mapping[f.name] }) : f,
      );
      return new Schema(fields, input.metadata);
    }

    case "withColumns": {
      const input = planArrowSchema(node.input);
      const overwritten = (input.fields as Field[]).map((f) => {
        const expr = node.exprs.get(f.name);
        return expr ? new Field(f.name, inferExprType(expr, input), true) : f;
      });
      const newNames = new Set(input.names as string[]);
      const appended: Field[] = [];
      for (const [name, expr] of node.exprs) {
        if (!newNames.has(name)) appended.push(new Field(name, inferExprType(expr, input), true));
      }
      return new Schema([...overwritten, ...appended], input.metadata);
    }

    case "filter":
    case "sortBy":
    case "limit":
    case "slice":
    case "dropNull":
    case "fillNull":
      return planArrowSchema(node.input);

    case "aggregate": {
      const input = planArrowSchema(node.input);
      const keyFields = node.keys.map((k) => {
        const f = (input.fields as Field[]).find((field) => field.name === k);
        if (!f) throw new Error(`groupBy: no such column "${k}"`);
        return f;
      });
      const aggFields = [...node.aggs].map(
        ([name, expr]) => new Field(name, inferExprType(expr, input), true),
      );
      return new Schema([...keyFields, ...aggFields]);
    }

    case "join": {
      const leftSchema = planArrowSchema(node.left);
      const rightSchema = planArrowSchema(node.right);
      const keys = typeof node.options.on === "string" ? [node.options.on] : [...node.options.on];
      const suffix = node.options.suffix ?? "_right";
      const how = node.options.how ?? "inner";
      const leftNames = new Set(leftSchema.names as string[]);
      const nullifyLeft = how === "right" || how === "outer";
      const nullifyRight = how === "left" || how === "outer";
      const leftFields = (leftSchema.fields as Field[]).map((f) =>
        nullifyLeft ? f.clone({ nullable: true }) : f,
      );
      const rightFields: Field[] = [];
      for (const f of rightSchema.fields as Field[]) {
        if (keys.includes(f.name)) continue; // join key carried once, from the left side
        const renamed = leftNames.has(f.name) ? f.clone({ name: f.name + suffix }) : f;
        rightFields.push(nullifyRight ? renamed.clone({ nullable: true }) : renamed);
      }
      return new Schema([...leftFields, ...rightFields]);
    }

    case "concat": {
      if (node.inputs.length === 0) throw new Error("Frame.concat: no frames given");
      const schemas = node.inputs.map((n) => planArrowSchema(n));
      const first = schemas[0] as Schema;
      for (let i = 1; i < schemas.length; i++) {
        const s = schemas[i] as Schema;
        const a = first.names as string[];
        const b = s.names as string[];
        if (a.length !== b.length || a.some((n, idx) => n !== b[idx])) {
          throw new Error(
            `Frame.concat: schema mismatch — frame 0 has columns [${a.join(", ")}], frame ${i} has [${b.join(", ")}]`,
          );
        }
      }
      return first;
    }
  }
}
