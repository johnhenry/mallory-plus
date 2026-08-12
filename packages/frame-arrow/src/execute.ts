/**
 * Plan execution: column pruning + predicate pushdown, then materialization.
 *
 * `requiredInputColumns(node, wanted)` answers "given that a consumer wants
 * these OUTPUT columns from `node`, which columns does `node.input` need to
 * produce them?" — recursing this top-down from whatever a terminal accessor
 * asked for (see plan.ts's module doc) is what makes pruning real: by the
 * time execution reaches the `source` leaf, `wanted` has been narrowed to
 * exactly the columns some downstream consumer can actually observe, and
 * `Table.select()` (zero-copy — it slices the field list, never decodes
 * column values) is used to drop the rest before anything touches them.
 *
 * `execute(node, wanted)` does the actual bottom-up materialization.
 *
 * `join` is the one exception: both input plans are executed with
 * `wanted: "all"` regardless of what the join's own consumer asked for.
 * Pushing pruning through a join needs to track which output columns came
 * from which side across renames/suffixing and is a well-defined but
 * separable piece of work; v1 scopes it out (see plan.ts's doc comment) —
 * everything else (select/drop/rename/withColumns/filter/sortBy/limit/
 * slice/groupBy/concat/fillNull/dropNull) prunes fully.
 */
import { type Field, Table, type Vector } from "apache-arrow";
import { columnToArray } from "./access.ts";
import { describeField } from "./dtype.ts";
import { AggExpr } from "./expr.ts";
import { evalRowwise, reduceGroup } from "./eval-expr.ts";
import { planArrowSchema, planColumnNames, type PlanNode, type Wanted } from "./plan.ts";
import { fieldFor, inferExprType } from "./schema-infer.ts";
import { buildVector, gatherRows } from "./vector-build.ts";

export type { Wanted } from "./plan.ts";

/** What must `node`'s single input supply, given that `wanted` is requested from `node` itself. */
function requiredInputColumns(
  node: Extract<PlanNode, { input: PlanNode }>,
  wanted: Wanted,
): Wanted {
  switch (node.kind) {
    case "select": {
      return wanted === "all"
        ? new Set(node.columns)
        : new Set([...wanted].filter((c) => node.columns.includes(c)));
    }
    case "drop":
    case "limit":
    case "slice":
      return wanted; // these never add/remove column *requirements* beyond what's asked

    case "rename": {
      if (wanted === "all") return "all";
      const inverse = new Map<string, string>();
      for (const [oldName, newName] of Object.entries(node.mapping)) inverse.set(newName, oldName);
      return new Set([...wanted].map((name) => inverse.get(name) ?? name));
    }

    case "withColumns": {
      if (wanted === "all") return "all"; // every passthrough input column is needed
      const req = new Set<string>();
      for (const name of wanted) {
        const expr = node.exprs.get(name);
        if (expr) {
          for (const c of expr.requiredColumns()) req.add(c);
        } else {
          req.add(name); // passthrough
        }
      }
      return req;
    }

    case "filter": {
      const predCols = node.predicate.requiredColumns();
      if (wanted === "all") return "all";
      return new Set([...wanted, ...predCols]);
    }

    case "sortBy": {
      const keyCols = node.keys.map((k) => k.column);
      if (wanted === "all") return "all";
      return new Set([...wanted, ...keyCols]);
    }

    case "dropNull": {
      if (!node.columns) return "all"; // must inspect every column's nullity
      if (wanted === "all") return "all";
      return new Set([...wanted, ...node.columns]);
    }

    case "fillNull":
      return wanted; // only touches columns already in `wanted`

    case "aggregate": {
      const req = new Set<string>(node.keys);
      const names = wanted === "all" ? [...node.keys, ...node.aggs.keys()] : [...wanted];
      for (const name of names) {
        if (node.keys.includes(name)) continue;
        const expr = node.aggs.get(name);
        if (expr) for (const c of expr.requiredColumns()) req.add(c);
      }
      return req;
    }
  }
}

function isFillMap(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v) && !(v instanceof Date);
}

export function execute(node: PlanNode, wanted: Wanted): Table {
  switch (node.kind) {
    case "source": {
      if (wanted === "all") return node.table;
      const names = (node.table.schema.names as string[]).filter((n) => (wanted as ReadonlySet<string>).has(n));
      return node.table.select(names);
    }

    case "lazySource":
      // resolveLazySources() (below) always rewrites every "lazySource" leaf
      // into a "source" node before collectPlanAsync() calls execute() on
      // the result -- reaching this case means the SYNCHRONOUS collect()/a
      // terminal accessor was called directly on a Frame whose plan still
      // has an unresolved lazy source (e.g. built via scanParquetLazy).
      throw new Error(
        "this Frame contains an unresolved lazy source (e.g. from scanParquetLazy) -- " +
          "use collectAsync() instead of collect(), or await a terminal accessor's async equivalent",
      );

    case "select": {
      const inputWanted = requiredInputColumns(node, wanted);
      const input = execute(node.input, inputWanted);
      const outNames = wanted === "all" ? node.columns : node.columns.filter((c) => wanted.has(c));
      return input.select(outNames as string[]);
    }

    case "drop": {
      const inputWanted = requiredInputColumns(node, wanted);
      const input = execute(node.input, inputWanted);
      const drop = new Set(node.columns);
      const keep = (input.schema.names as string[]).filter((n) => !drop.has(n));
      return input.select(keep);
    }

    case "rename": {
      const inputWanted = requiredInputColumns(node, wanted);
      const input = execute(node.input, inputWanted);
      const cols: Record<string, Vector> = {};
      for (const name of input.schema.names as string[]) {
        const outName = node.mapping[name] ?? name;
        cols[outName] = input.getChild(name) as Vector;
      }
      return new Table(cols);
    }

    case "withColumns": {
      const inputWanted = requiredInputColumns(node, wanted);
      const input = execute(node.input, inputWanted);
      const outputNames = wanted === "all" ? planColumnNames(node) : [...wanted];
      const cols: Record<string, Vector> = {};
      for (const name of outputNames) {
        const expr = node.exprs.get(name);
        if (expr) {
          const values = evalRowwise(expr, input);
          const type = inferExprType(expr, input.schema);
          cols[name] = buildVector(values, type);
        } else {
          const v = input.getChild(name);
          if (!v) throw new Error(`withColumns: no such passthrough column "${name}"`);
          cols[name] = v;
        }
      }
      return new Table(cols);
    }

    case "filter": {
      const inputWanted = requiredInputColumns(node, wanted);
      const input = execute(node.input, inputWanted);
      const mask = evalRowwise(node.predicate, input) as (boolean | null)[];
      const indices: number[] = [];
      for (let i = 0; i < mask.length; i++) if (mask[i] === true) indices.push(i);
      const gathered = gatherRows(input, indices);
      return wanted === "all" ? gathered : gathered.select([...wanted]);
    }

    case "sortBy": {
      const inputWanted = requiredInputColumns(node, wanted);
      const input = execute(node.input, inputWanted);
      const keyArrays = node.keys.map((k) => {
        const v = input.getChild(k.column);
        if (!v) throw new Error(`sortBy: no such column "${k.column}"`);
        const arr: unknown[] = new Array(v.length);
        for (let i = 0; i < v.length; i++) arr[i] = v.get(i);
        return { values: arr, descending: !!k.descending };
      });
      const indices = Array.from({ length: input.numRows }, (_, i) => i);
      indices.sort((ia, ib) => {
        for (const { values, descending } of keyArrays) {
          const a = values[ia];
          const b = values[ib];
          const an = a === null || a === undefined;
          const bn = b === null || b === undefined;
          if (an && bn) continue;
          if (an) return 1; // nulls sort last regardless of direction (documented v1 convention)
          if (bn) return -1;
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          let cmp = (a as any) < (b as any) ? -1 : (a as any) > (b as any) ? 1 : 0;
          if (descending) cmp = -cmp;
          if (cmp !== 0) return cmp;
        }
        return 0;
      });
      const gathered = gatherRows(input, indices);
      return wanted === "all" ? gathered : gathered.select([...wanted]);
    }

    case "limit": {
      const input = execute(node.input, wanted);
      return input.slice(0, Math.min(node.n, input.numRows));
    }

    case "slice": {
      const input = execute(node.input, wanted);
      return input.slice(node.start, node.end);
    }

    case "dropNull": {
      const inputWanted = requiredInputColumns(node, wanted);
      const input = execute(node.input, inputWanted);
      const checkCols = node.columns ?? (input.schema.names as string[]);
      const vectors = checkCols.map((c) => {
        const v = input.getChild(c);
        if (!v) throw new Error(`dropNull: no such column "${c}"`);
        return v;
      });
      const indices: number[] = [];
      for (let i = 0; i < input.numRows; i++) {
        if (vectors.every((v) => v.isValid(i))) indices.push(i);
      }
      const gathered = gatherRows(input, indices);
      return wanted === "all" ? gathered : gathered.select([...wanted]);
    }

    case "fillNull": {
      const inputWanted = requiredInputColumns(node, wanted);
      const input = execute(node.input, inputWanted);
      const isMap = isFillMap(node.value);
      const cols: Record<string, Vector> = {};
      for (const field of input.schema.fields as Field[]) {
        const name = field.name;
        const vector = input.getChild(name) as Vector;
        const fillValue = isMap ? (node.value as Record<string, unknown>)[name] : node.value;
        if (fillValue === undefined) {
          cols[name] = vector;
          continue;
        }
        const desc = describeField(field);
        const values = columnToArray(vector, desc.dtype).map((v) => (v === null ? fillValue : v));
        cols[name] = buildVector(values, field.type);
      }
      return new Table(cols);
    }

    case "aggregate": {
      const inputWanted = requiredInputColumns(node, wanted);
      const input = execute(node.input, inputWanted);
      const outputNames = wanted === "all" ? [...node.keys, ...node.aggs.keys()] : [...wanted];
      const wantedAggNames = outputNames.filter((n) => !node.keys.includes(n));

      const keyVectors = node.keys.map((k) => {
        const v = input.getChild(k);
        if (!v) throw new Error(`groupBy: no such column "${k}"`);
        return v;
      });

      const groupOrder: string[] = [];
      const groupIndices = new Map<string, number[]>();
      const groupKeyValues = new Map<string, unknown[]>();
      for (let i = 0; i < input.numRows; i++) {
        const keyVals = keyVectors.map((v) => v.get(i));
        const keyStr = keyVals
          .map((v) => (typeof v === "bigint" ? `b:${v.toString()}` : JSON.stringify(v)))
          .join(" ");
        let idxs = groupIndices.get(keyStr);
        if (!idxs) {
          idxs = [];
          groupIndices.set(keyStr, idxs);
          groupKeyValues.set(keyStr, keyVals);
          groupOrder.push(keyStr);
        }
        idxs.push(i);
      }

      const aggValueArrays = new Map<string, unknown[] | null>();
      for (const name of wantedAggNames) {
        const expr = node.aggs.get(name);
        if (!expr) throw new Error(`groupBy: no such aggregate "${name}"`);
        if (!(expr instanceof AggExpr)) {
          throw new Error(
            `groupBy().aggregate(): "${name}" must be built from fn.count/sum/mean/stddev, not a plain Expr`,
          );
        }
        aggValueArrays.set(name, expr.inner ? evalRowwise(expr.inner, input) : null);
      }

      const keyCols: Record<string, unknown[]> = {};
      for (const k of node.keys) keyCols[k] = [];
      const aggCols: Record<string, unknown[]> = {};
      for (const name of wantedAggNames) aggCols[name] = [];

      for (const keyStr of groupOrder) {
        const idxs = groupIndices.get(keyStr) as number[];
        const keyVals = groupKeyValues.get(keyStr) as unknown[];
        node.keys.forEach((k, ki) => (keyCols[k] as unknown[]).push(keyVals[ki]));
        for (const name of wantedAggNames) {
          const expr = node.aggs.get(name) as AggExpr;
          const values = aggValueArrays.get(name) ?? null;
          (aggCols[name] as unknown[]).push(reduceGroup(expr.op, values, idxs));
        }
      }

      const vectors: Record<string, Vector> = {};
      for (const k of node.keys) {
        const type = fieldFor(input.schema, k).type;
        vectors[k] = buildVector(keyCols[k] as unknown[], type);
      }
      for (const name of wantedAggNames) {
        const expr = node.aggs.get(name) as AggExpr;
        const type = inferExprType(expr, input.schema);
        vectors[name] = buildVector(aggCols[name] as unknown[], type);
      }
      return new Table(vectors);
    }

    case "join": {
      const leftTable = execute(node.left, "all");
      const rightTable = execute(node.right, "all");
      const keys = typeof node.options.on === "string" ? [node.options.on] : [...node.options.on];
      const how = node.options.how ?? "inner";
      const suffix = node.options.suffix ?? "_right";

      const leftKeyVectors = keys.map((k) => {
        const v = leftTable.getChild(k);
        if (!v) throw new Error(`join: no such column "${k}" on the left frame`);
        return v;
      });
      const rightKeyVectors = keys.map((k) => {
        const v = rightTable.getChild(k);
        if (!v) throw new Error(`join: no such column "${k}" on the right frame`);
        return v;
      });

      const keyStrAt = (vectors: readonly Vector[], i: number): string =>
        vectors
          .map((v) => {
            const val = v.get(i);
            if (val === null || val === undefined) return " null";
            return typeof val === "bigint" ? `b:${val}` : JSON.stringify(val);
          })
          .join("");

      const rightByKey = new Map<string, number[]>();
      for (let i = 0; i < rightTable.numRows; i++) {
        const ks = keyStrAt(rightKeyVectors, i);
        let arr = rightByKey.get(ks);
        if (!arr) {
          arr = [];
          rightByKey.set(ks, arr);
        }
        arr.push(i);
      }

      const leftIdx: number[] = []; // -1 sentinel: no matching left row (right/outer-only rows)
      const rightIdx: (number | null)[] = [];
      const matchedRight = new Set<number>();
      for (let i = 0; i < leftTable.numRows; i++) {
        const ks = keyStrAt(leftKeyVectors, i);
        const matches = rightByKey.get(ks);
        if (matches && matches.length > 0) {
          for (const rj of matches) {
            leftIdx.push(i);
            rightIdx.push(rj);
            matchedRight.add(rj);
          }
        } else if (how === "left" || how === "outer") {
          leftIdx.push(i);
          rightIdx.push(null);
        }
      }
      if (how === "right" || how === "outer") {
        for (let rj = 0; rj < rightTable.numRows; rj++) {
          if (!matchedRight.has(rj)) {
            leftIdx.push(-1);
            rightIdx.push(rj);
          }
        }
      }

      const rowCount = leftIdx.length;
      const leftNames = new Set(leftTable.schema.names as string[]);
      const cols: Record<string, Vector> = {};

      for (const field of leftTable.schema.fields as Field[]) {
        const desc = describeField(field);
        const values = columnToArray(leftTable.getChild(field.name) as Vector, desc.dtype);
        let rightKeyValues: unknown[] | null = null;
        if (keys.includes(field.name)) {
          const rField = (rightTable.schema.fields as Field[]).find((f) => f.name === field.name);
          const rVec = rField ? (rightTable.getChild(field.name) as Vector) : null;
          rightKeyValues = rVec && rField ? columnToArray(rVec, describeField(rField).dtype) : null;
        }
        const gathered: unknown[] = new Array(rowCount);
        for (let r = 0; r < rowCount; r++) {
          const li = leftIdx[r] as number;
          if (li >= 0) {
            gathered[r] = values[li];
          } else if (rightKeyValues) {
            const rj = rightIdx[r];
            gathered[r] = rj === null || rj === undefined ? null : rightKeyValues[rj];
          } else {
            gathered[r] = null;
          }
        }
        cols[field.name] = buildVector(gathered, field.type);
      }

      for (const field of rightTable.schema.fields as Field[]) {
        if (keys.includes(field.name)) continue;
        const outName = leftNames.has(field.name) ? field.name + suffix : field.name;
        const desc = describeField(field);
        const values = columnToArray(rightTable.getChild(field.name) as Vector, desc.dtype);
        const gathered: unknown[] = new Array(rowCount);
        for (let r = 0; r < rowCount; r++) {
          const rj = rightIdx[r];
          gathered[r] = rj === null || rj === undefined ? null : values[rj];
        }
        cols[outName] = buildVector(gathered, field.type);
      }

      const joined = new Table(cols);
      return wanted === "all" ? joined : joined.select([...wanted]);
    }

    case "concat": {
      if (node.inputs.length === 0) throw new Error("Frame.concat: no frames given");
      const tables = node.inputs.map((n) => execute(n, wanted));
      // apache-arrow 21.2.0's Table.concat() breaks getChild() ("Vector
      // constructor expects an Array of Data instances") once ANY concatenated
      // input has zero rows — reproducible directly against apache-arrow, no
      // frame-arrow logic involved (issue #31). numRows is unaffected, so this
      // surfaces later, at whatever terminal accessor first calls getChild().
      // Filter zero-row tables out before concatenating; if every input is
      // empty, return the first one as-is (still the right schema, nothing to
      // concat).
      const nonEmpty = tables.filter((t) => t.numRows > 0);
      if (nonEmpty.length === 0) return tables[0] as Table;
      const [first, ...rest] = nonEmpty as [Table, ...Table[]];
      return rest.length > 0 ? first.concat(...rest) : first;
    }
  }
}

/** Public entry point: materialize a plan node's full declared output. */
export function collectPlan(node: PlanNode): Table {
  return execute(node, "all");
}

/**
 * Async pre-pass for `Frame.collectAsync()` (issue #32) — rewrites every
 * `"lazySource"` leaf in `node`'s tree into a real `"source"` node (its
 * `read(wanted)` awaited and the result wrapped), honoring the EXACT same
 * per-node-kind pruning `requiredInputColumns()`/`execute()` already
 * implement, by literally reusing `requiredInputColumns()` here instead of
 * re-deriving the rules. The rest of the tree is copied structurally
 * unchanged (a node with no lazy source anywhere below it comes back
 * identical, just a new object).
 *
 * Deliberately does NOT reimplement any of `execute()`'s actual Arrow
 * manipulation (select/filter/join/aggregate/...) — this only resolves
 * WHICH DATA gets fetched and with WHICH COLUMN SET, exactly mirroring
 * execute()'s own top-down wanted-propagation; the real materialization
 * still happens via the single, already-tested, unmodified `execute()`
 * once every lazy leaf has been replaced.
 */
async function resolveLazySources(node: PlanNode, wanted: Wanted): Promise<PlanNode> {
  switch (node.kind) {
    case "source":
      return node; // nothing lazy here, structurally unchanged

    case "lazySource": {
      const table = await node.read(wanted);
      return { kind: "source", table };
    }

    case "join": {
      // Matches execute()'s own "join" case: both sides are always resolved
      // with wanted="all", regardless of what the join's own consumer
      // asked for (v1 doesn't push pruning through a join — see plan.ts).
      const [left, right] = await Promise.all([
        resolveLazySources(node.left, "all"),
        resolveLazySources(node.right, "all"),
      ]);
      return { ...node, left, right };
    }

    case "concat": {
      // Matches execute()'s own "concat" case: every input is resolved with
      // the SAME wanted concat itself was asked for (no narrowing).
      const inputs = await Promise.all(node.inputs.map((n) => resolveLazySources(n, wanted)));
      return { ...node, inputs };
    }

    default: {
      // Every remaining PlanNode kind has exactly one `.input` — the same
      // `Extract<PlanNode, {input: PlanNode}>` shape requiredInputColumns()
      // itself is typed against.
      const inputWanted = requiredInputColumns(node, wanted);
      const input = await resolveLazySources(node.input, inputWanted);
      return { ...node, input };
    }
  }
}

/** Async public entry point (issue #32): like `collectPlan`, but resolves any lazy sources in the tree first. Safe to call on a plan with NO lazy sources too — `resolveLazySources` just returns it structurally unchanged in that case, so this is a strict superset of `collectPlan`, never slower in spirit, only in the (currently unavoidable) cost of one extra tree walk. */
export async function collectPlanAsync(node: PlanNode): Promise<Table> {
  const resolved = await resolveLazySources(node, "all");
  return execute(resolved, "all");
}
