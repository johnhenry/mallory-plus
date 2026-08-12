/**
 * bigint-safe JSON helpers.
 *
 * apache-arrow surfaces int64 columns (and, internally, raw timestamp[us]
 * buffers) as JS `bigint`. `JSON.stringify` throws on bigint
 * (`Do not know how to serialize a BigInt`) — see docs/spikes/arrow-parity.md
 * sharp edge #1. Any Frame path that might reach JSON.stringify with a row
 * object must go through here instead of calling JSON.stringify directly.
 */

/** JSON.stringify replacer that renders bigint as a decimal string, e.g. `"123"`. */
export function bigintSafeReplacer(_key: string, value: unknown): unknown {
  return typeof value === "bigint" ? value.toString() : value;
}

/** Bigint-safe JSON.stringify — bigints are rendered as decimal strings. */
export function stringifyRows(rows: readonly Record<string, unknown>[]): string {
  return JSON.stringify(rows, bigintSafeReplacer);
}
