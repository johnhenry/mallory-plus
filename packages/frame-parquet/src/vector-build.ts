/**
 * Plain-JS-array -> Arrow Vector construction for the read path.
 *
 * Same two rules as frame-arrow's own src/vector-build.ts (not reusable
 * directly — it isn't exported from @johnhenry/math-plus-frame-arrow's public surface,
 * see schema.ts's doc comment), ported here because they apply equally to
 * data hyparquet hands back:
 * - Always pass an EXPLICIT Arrow type — never `vectorFromArray(values)`
 *   type inference, which would silently dictionary-encode plain string
 *   arrays (docs/spikes/arrow-parity.md sharp edge #4).
 * - Timestamp columns never go through `vectorFromArray`/a Builder (sharp
 *   edge #2); built via `makeData` + `BigInt64Array` directly. hyparquet
 *   already hands back exact BigInt epoch values for timestamp columns (see
 *   read.ts), so no precision is lost feeding them straight into this path.
 */
import { DataType, makeData, type Timestamp, Vector, vectorFromArray } from "apache-arrow";

export function buildVector(values: readonly unknown[], type: DataType): Vector {
  if (DataType.isTimestamp(type)) {
    return buildTimestampVector(values as readonly (bigint | null | undefined)[], type);
  }
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
