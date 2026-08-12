/**
 * Test-only Arrow construction helpers. Deliberately independent of
 * src/vector-build.ts (which is what's under test) — these mirror what a
 * real apache-arrow consumer would write, following docs/spikes/arrow-parity.md's
 * recipes (explicit types always; timestamps via makeData, never a Builder).
 */
import {
  type DataType,
  Field,
  makeData,
  Timestamp,
  Vector,
} from "apache-arrow";

/** Build a timestamp Vector from exact epoch values in the column's own unit (ms or us). */
export function timestampVector(values: readonly (bigint | null)[], type: Timestamp): Vector {
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
      nullBitmap[i >> 3] |= 1 << (i & 7);
    }
  }
  return new Vector([makeData({ type, length: n, nullCount, nullBitmap, data })]);
}

/** Build a Data chunk with an explicit type, for constructing intentionally
 * "poisoned" columns (e.g. decimal128) used in pruning tests. */
export function field(name: string, type: DataType, nullable = true): Field {
  return new Field(name, type, nullable);
}
