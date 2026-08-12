/**
 * Frame.toCSV(). Not part of Arrow's own IPC machinery — a small
 * hand-rolled writer, since apache-arrow doesn't ship one.
 *
 * bigint-safe (int64 renders as a decimal string, never reaches
 * JSON.stringify raw — see safe-json.ts) and renders timestamp cells as
 * exact ISO-8601 strings (reading the raw int64 buffer via access.ts, not
 * apache-arrow's lossy-for-microseconds Vector.get()).
 */
import type { Field, Table, Vector } from "apache-arrow";
import { columnToArray } from "./access.ts";
import { describeField, type DType } from "./dtype.ts";
import { bigintSafeReplacer } from "./safe-json.ts";

function escapeCsvField(raw: string): string {
  if (/[",\n\r]/.test(raw)) return `"${raw.replace(/"/g, '""')}"`;
  return raw;
}

function timestampToISO(epoch: bigint, dtype: "timestamp_ms" | "timestamp_us"): string {
  if (dtype === "timestamp_ms") return new Date(Number(epoch)).toISOString();
  let msPart = epoch / 1000n;
  let rem = epoch % 1000n;
  if (rem < 0n) {
    rem += 1000n;
    msPart -= 1n;
  }
  const base = new Date(Number(msPart)).toISOString(); // "....SSSZ"
  const micros = rem.toString().padStart(3, "0");
  return base.slice(0, -1) + micros + "Z";
}

function formatCell(value: unknown, dtype: DType): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "bigint") {
    if (dtype === "timestamp_ms" || dtype === "timestamp_us") return timestampToISO(value, dtype);
    return value.toString();
  }
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "object") return JSON.stringify(value, bigintSafeReplacer);
  return String(value);
}

export function tableToCSV(table: Table): string {
  const fields = table.schema.fields as Field[];
  const columns = fields.map((f) => {
    const desc = describeField(f);
    return { dtype: desc.dtype, values: columnToArray(table.getChild(f.name) as Vector, desc.dtype) };
  });
  const lines: string[] = [fields.map((f) => escapeCsvField(f.name)).join(",")];
  for (let i = 0; i < table.numRows; i++) {
    lines.push(columns.map((c) => escapeCsvField(formatCell(c.values[i], c.dtype))).join(","));
  }
  return lines.join("\n") + "\n";
}
