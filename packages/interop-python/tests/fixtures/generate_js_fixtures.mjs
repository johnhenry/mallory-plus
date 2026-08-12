#!/usr/bin/env node
/**
 * Generates the "JS writes, Python reads" half of interop-python's
 * bidirectional conformance suite (issue #21). Run from the repo root
 * (`node packages/interop-python/tests/fixtures/generate_js_fixtures.mjs`)
 * so Node's module resolution finds the workspace-hoisted `apache-arrow`/
 * `mallory-frame-arrow`/`mallory-frame-parquet` in the root `node_modules` —
 * this script isn't part of the npm workspace itself (interop-python has no
 * package.json, deliberately: see docs/RELEASING.md, it's a PyPI package
 * outside the npm/Cargo workspaces), it just borrows from it to generate
 * fixtures.
 *
 * Output is committed (small binary fixtures, matching this repo's existing
 * convention for frame-parquet's own fixtures) rather than regenerated on
 * every Python test run, so the Python suite has no Node/npm dependency at
 * test time.
 */
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import {
  Bool,
  Float64,
  Int32,
  Table,
  Utf8,
  vectorFromArray,
} from "apache-arrow";
import { Frame } from "mallory-frame-arrow";
import { writeParquet } from "mallory-frame-parquet";

const DIR = fileURLToPath(new URL(".", import.meta.url));

const table = new Table({
  id: vectorFromArray([1, 2, 3, 4, 5], new Int32()),
  value: vectorFromArray([1.5, null, 3.25, -0.5, 100], new Float64()),
  label: vectorFromArray(["alpha", "beta", null, "delta", ""], new Utf8()),
  active: vectorFromArray([true, false, null, true, false], new Bool()),
});
const frame = Frame.fromArrow(table);

// 1. Arrow IPC file
writeFileSync(`${DIR}js_written.arrow`, frame.toIPC());

// 2. Parquet (snappy — universally readable by pyarrow with zero setup)
await writeParquet(frame, `${DIR}js_written.parquet`, { compression: "snappy" });

// Expected values, compared by the Python suite (schema + values, not a
// byte-diff — matches this repo's established fixture-verification style).
const expected = {
  columns: ["id", "value", "label", "active"],
  rows: frame.toRows().map((row) => ({
    id: row.id,
    value: row.value,
    label: row.label,
    active: row.active,
  })),
};
writeFileSync(`${DIR}js_written_expected.json`, JSON.stringify(expected, null, 2));

console.log("wrote js_written.arrow, js_written.parquet, js_written_expected.json");
