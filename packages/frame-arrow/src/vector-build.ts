/**
 * Plain-JS-array -> Arrow Vector construction, used whenever execute.ts needs
 * to materialize a NEW or row-reordered column (filter/sortBy/join/groupBy/
 * withColumns/fillNull all funnel through here).
 *
 * Two rules enforced everywhere in this module, both straight from
 * docs/spikes/arrow-parity.md:
 * - Every call passes an EXPLICIT Arrow type (sharp edge #4) — never
 *   `vectorFromArray(values)` type inference, which would silently
 *   dictionary-encode plain string arrays.
 * - Timestamp columns NEVER go through `vectorFromArray`/a Builder (sharp
 *   edge #2 — the timestamp[us]/[ns] builder does `BigInt(value * 1000)`,
 *   which throws on realistic inputs). They're built via `makeData` +
 *   `BigInt64Array` directly instead.
 */
import { DataType, type Field, makeData, Table, type Timestamp, Vector, vectorFromArray } from "apache-arrow";
import { columnToArray } from "./access.ts";
import { describeField } from "./dtype.ts";

/** Build a Vector of the given Arrow type from plain JS values (bigint for timestamps/int64, null for nulls). */
export function buildVector(values: readonly unknown[], type: DataType): Vector {
  if (DataType.isTimestamp(type)) {
    return buildTimestampVector(values as readonly (bigint | null | undefined)[], type);
  }
  // Explicit type always passed — never inference. Safe for utf8 (stays Utf8,
  // not silently dictionary-encoded), int64 (plain Int64 builder handles
  // bigint arrays fine; only the timestamp builder is broken), dictionary,
  // list, struct, and all fixed-width numeric/bool types.
  return vectorFromArray(values as never[], type as never);
}

function buildTimestampVector(values: readonly (bigint | null | undefined)[], type: Timestamp): Vector {
  const n = values.length;
  const data = new BigInt64Array(n);
  const nullBitmap = new Uint8Array(Math.max(1, Math.ceil(n / 8)));
  let nullCount = 0;
  for (let i = 0; i < n; i++) {
    const v = values[i];
    if (v === null || v === undefined) {
      nullCount++;
    } else {
      data[i] = v;
      nullBitmap[i >> 3] |= 1 << (i & 7); // bit=1 means valid, per Arrow validity-bitmap convention
    }
  }
  return new Vector([makeData({ type, length: n, nullCount, nullBitmap, data })]);
}

/** Build a new Table containing only the given row indices of `table`, in that order.
 * Used for filter (indices = rows where the predicate is true) and sortBy (indices =
 * the sorted permutation). Rebuilds every surviving column via `columnToArray` +
 * `buildVector`, which is why a column must be dtype-supported to survive this path —
 * exactly why pruning it away earlier (never routing it through gatherRows) matters. */
export function gatherRows(table: Table, indices: readonly number[]): Table {
  const cols: Record<string, Vector> = {};
  for (const field of table.schema.fields as Field[]) {
    const vector = table.getChild(field.name) as Vector;
    const desc = describeField(field);
    const values = columnToArray(vector, desc.dtype);
    const gathered = indices.map((i) => values[i]);
    cols[field.name] = buildVector(gathered, field.type);
  }
  return new Table(cols);
}
