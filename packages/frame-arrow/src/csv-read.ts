/**
 * Frame.fromCSV() — the reader counterpart to csv.ts's tableToCSV() writer.
 * mallory-frame-arrow had a CSV writer since v1 but no reader at all (an
 * asymmetry consumers hit directly -- see johnhenry/mallory-plus#86, filed
 * by mallory-graph after having to hand-roll its own app-side CSV parser
 * for exactly this gap).
 *
 * Two stages: `tokenizeCsv` (RFC-4180 tokenizing -- quoted fields, doubled-
 * quote escapes, commas/newlines inside quotes, CRLF or LF line endings;
 * ragged rows rejected with a clear error rather than silently padded,
 * since for a data-import flow a column-count mismatch almost always means
 * a quoting bug in the source) produces a plain header + string-rows shape,
 * then `parseCsvToTable` infers a per-column dtype and builds the Arrow
 * Table via vector-build.ts's `buildVector` (never `vectorFromArray`'s bare
 * type inference -- same sharp-edge-#4 rule every other column-construction
 * path in this package follows).
 */
import { Bool, Float64, Int64, Utf8 } from "apache-arrow";
import { buildVector } from "./vector-build.ts";

interface TokenizedCsv {
  header: string[];
  rows: string[][];
}

function tokenizeCsv(text: string): TokenizedCsv {
  const records: string[][] = [];
  let field = "";
  let record: string[] = [];
  let inQuotes = false;
  let i = 0;

  const pushField = () => {
    record.push(field);
    field = "";
  };
  const pushRecord = () => {
    pushField();
    records.push(record);
    record = [];
  };

  while (i < text.length) {
    const ch = text[i]!;
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i += 2;
          continue;
        }
        inQuotes = false;
        i++;
        continue;
      }
      field += ch;
      i++;
      continue;
    }
    if (ch === '"' && field.length === 0) {
      // A quote only OPENS a quoted section at field start (RFC 4180 forbids
      // quotes in unquoted fields entirely; the lenient convention -- what
      // Python's csv module does -- is to treat a mid-field quote as a
      // literal character, which the `else` fallthrough below handles).
      inQuotes = true;
      i++;
      continue;
    }
    if (ch === ",") {
      pushField();
      i++;
      continue;
    }
    if (ch === "\r" && text[i + 1] === "\n") {
      pushRecord();
      i += 2;
      continue;
    }
    if (ch === "\n" || ch === "\r") {
      pushRecord();
      i++;
      continue;
    }
    field += ch;
    i++;
  }
  if (inQuotes) throw new Error("Unterminated quoted field.");
  // Final record, unless the text ended exactly on a record boundary.
  if (field.length > 0 || record.length > 0) pushRecord();

  // A trailing newline leaves no phantom empty record (handled above), but a
  // genuinely empty line mid-file parses as a single empty field -- drop those.
  const nonEmpty = records.filter((r) => !(r.length === 1 && r[0] === ""));
  if (nonEmpty.length < 1) throw new Error("Need at least a header row.");
  const header = nonEmpty[0]!;
  const rows = nonEmpty.slice(1);
  for (const [rowIndex, row] of rows.entries()) {
    if (row.length !== header.length) {
      throw new Error(`Row ${rowIndex + 1} has ${row.length} fields, but the header has ${header.length}.`);
    }
  }
  return { header, rows };
}

type InferredDType = "bool" | "int64" | "float64" | "utf8";

const INT_PATTERN = /^-?\d+$/;

/**
 * Classifies one column's non-empty cell strings into the narrowest dtype
 * they all agree on -- bool only if EVERY non-empty cell is exactly "true"/
 * "false" (case-insensitive), int64 only if every one matches a strict
 * integer pattern (parsed via `BigInt`, not `Number`, so large integers
 * outside `Number.MAX_SAFE_INTEGER` stay exact -- a real precision
 * advantage over the `Number()`-based inference mallory-graph's own
 * app-side parser used), float64 if every one parses as a finite JS
 * number (a mix of integer- and decimal-looking cells widens to float64,
 * the same int64-only-if-BOTH-operands-int64 promotion rule schema-infer.ts
 * uses for arithmetic), utf8 otherwise (including an all-empty column,
 * which carries no evidence either way).
 */
function inferColumnDType(cells: readonly string[]): InferredDType {
  let sawAny = false;
  let allBool = true;
  let allInt = true;
  let allFloat = true;
  for (const raw of cells) {
    const cell = raw.trim();
    if (cell === "") continue;
    sawAny = true;
    const lower = cell.toLowerCase();
    if (lower !== "true" && lower !== "false") allBool = false;
    if (!INT_PATTERN.test(cell)) allInt = false;
    if (!Number.isFinite(Number(cell))) allFloat = false;
  }
  if (!sawAny) return "utf8";
  if (allBool) return "bool";
  if (allInt) return "int64";
  if (allFloat) return "float64";
  return "utf8";
}

function cellToTypedValue(raw: string, dtype: InferredDType): bigint | number | boolean | string | null {
  const cell = raw.trim();
  if (cell === "") return null;
  switch (dtype) {
    case "bool":
      return cell.toLowerCase() === "true";
    case "int64":
      return BigInt(cell);
    case "float64":
      return Number(cell);
    case "utf8":
      return raw;
  }
}

function arrowTypeForInferred(dtype: InferredDType) {
  switch (dtype) {
    case "bool":
      return new Bool();
    case "int64":
      return new Int64();
    case "float64":
      return new Float64();
    case "utf8":
      return new Utf8();
  }
}

/** `{name -> Vector}` columns for a `new Table(cols)` call, with per-column dtype inferred from the CSV text itself. Used by `Frame.fromCSV`. */
export function parseCsvToColumns(text: string): Record<string, ReturnType<typeof buildVector>> {
  const { header, rows } = tokenizeCsv(text);
  const cols: Record<string, ReturnType<typeof buildVector>> = {};
  header.forEach((name, columnIndex) => {
    const cells = rows.map((row) => row[columnIndex] ?? "");
    const dtype = inferColumnDType(cells);
    const values = cells.map((cell) => cellToTypedValue(cell, dtype));
    cols[name] = buildVector(values, arrowTypeForInferred(dtype));
  });
  return cols;
}
