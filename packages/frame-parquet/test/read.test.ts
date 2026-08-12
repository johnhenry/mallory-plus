/**
 * Read-path correctness against real pyarrow-written fixtures (not
 * hand-rolled buffers — test/fixtures/generate.py, run via nix-shell
 * pyarrow, see its module doc). Covers: value/null fidelity across every
 * v1-supported dtype, zstd-compressed input, column projection, filter
 * correctness, dictionary-column-decodes-as-utf8 (documented v1
 * simplification), and the clear-throw on nested (struct) columns.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { readParquet, UnsupportedParquetTypeError } from "../src/index.ts";
import { FIXTURES_DIR } from "./helpers.ts";

interface BasicExpected {
  f64: (number | null)[];
  i64: (number | null)[];
  i32: (number | null)[];
  u32: (number | null)[];
  f32: (number | null)[];
  utf8: (string | null)[];
  bool: (boolean | null)[];
  ts_ms: (number | null)[];
  ts_us: (number | null)[];
}

function loadExpected(name: string): BasicExpected {
  return JSON.parse(readFileSync(new URL(name, FIXTURES_DIR), "utf8")) as BasicExpected;
}

async function assertBasicFidelity(path: string) {
  const expected = loadExpected("basic_expected.json");
  const frame = await readParquet(path);
  assert.deepEqual(frame.columns, ["f64", "i64", "i32", "u32", "f32", "utf8", "bool", "ts_ms", "ts_us"]);
  assert.equal(frame.length, 100);
  const rows = frame.toRows();

  for (let i = 0; i < 100; i++) {
    const row = rows[i] as Record<string, unknown>;
    assert.equal(row.f64, expected.f64[i], `f64[${i}]`);
    assert.equal(row.i32, expected.i32[i], `i32[${i}]`);
    assert.equal(row.u32, expected.u32[i], `u32[${i}]`);
    assert.equal(row.utf8, expected.utf8[i], `utf8[${i}]`);
    assert.equal(row.bool, expected.bool[i], `bool[${i}]`);

    const wantI64 = expected.i64[i];
    assert.equal(row.i64, wantI64 === null ? null : BigInt(wantI64), `i64[${i}]`);

    const wantF32 = expected.f32[i];
    if (wantF32 === null) assert.equal(row.f32, null, `f32[${i}]`);
    else assert.ok(Math.abs((row.f32 as number) - wantF32) < 1e-4, `f32[${i}] ${row.f32} vs ${wantF32}`);

    const wantMs = expected.ts_ms[i];
    assert.equal(row.ts_ms, wantMs === null ? null : BigInt(wantMs), `ts_ms[${i}]`);
    const wantUs = expected.ts_us[i];
    assert.equal(row.ts_us, wantUs === null ? null : BigInt(wantUs), `ts_us[${i}]`);
  }
}

test("readParquet: value+null fidelity across every v1 dtype (snappy fixture)", async () => {
  await assertBasicFidelity(new URL("fixtures/basic.parquet", import.meta.url).pathname);
});

test("readParquet: zstd-compressed input reads transparently (compressors always passed)", async () => {
  await assertBasicFidelity(new URL("fixtures/basic_zstd.parquet", import.meta.url).pathname);
});

test("readParquet: schema dtypes match the v1 mapping", async () => {
  const frame = await readParquet(new URL("fixtures/basic.parquet", import.meta.url).pathname);
  const byName = Object.fromEntries(frame.schema.map((f) => [f.name, f.dtype]));
  assert.deepEqual(byName, {
    f64: "float64",
    i64: "int64",
    i32: "int32",
    u32: "uint32",
    f32: "float32",
    utf8: "utf8",
    bool: "bool",
    ts_ms: "timestamp_ms",
    ts_us: "timestamp_us",
  });
});

test("readParquet: columns option projects — only requested columns present", async () => {
  const frame = await readParquet(new URL("fixtures/basic.parquet", import.meta.url).pathname, {
    columns: ["utf8", "i64"],
  });
  assert.deepEqual(frame.columns, ["utf8", "i64"]);
  assert.equal(frame.length, 100);
});

test("readParquet: unknown column in `columns` throws a clear error", async () => {
  await assert.rejects(
    () => readParquet(new URL("fixtures/basic.parquet", import.meta.url).pathname, { columns: ["nope"] }),
    /no such column "nope"/,
  );
});

test("readParquet: filter option returns exactly the matching rows", async () => {
  const frame = await readParquet(new URL("fixtures/pushdown.parquet", import.meta.url).pathname, {
    columns: ["id", "value"],
    filter: { value: { $gt: 5990.5 } },
  });
  const rows = frame.toRows();
  assert.equal(rows.length, 9); // ids 5991..5999
  assert.deepEqual(
    rows.map((r) => Number(r.id)),
    [5991, 5992, 5993, 5994, 5995, 5996, 5997, 5998, 5999],
  );
});

test("readParquet: dictionary-encoded column decodes as plain utf8 (documented v1 simplification)", async () => {
  const expected = JSON.parse(readFileSync(new URL("fixtures/quirks_expected.json", import.meta.url), "utf8")) as {
    cat: string[];
  };
  const frame = await readParquet(new URL("fixtures/quirks.parquet", import.meta.url).pathname);
  const catField = frame.schema.find((f) => f.name === "cat");
  assert.equal(catField?.dtype, "utf8"); // NOT "dictionary" — see schema.ts's module doc
  const rows = frame.toRows();
  assert.deepEqual(
    rows.map((r) => r.cat),
    expected.cat,
  );
});

test("readParquet: a nested (struct) column throws UnsupportedParquetTypeError naming the column", async () => {
  await assert.rejects(
    () => readParquet(new URL("fixtures/nested.parquet", import.meta.url).pathname),
    (err: unknown) => {
      assert.ok(err instanceof UnsupportedParquetTypeError);
      assert.match((err as Error).message, /"point"/);
      return true;
    },
  );
});

test("readParquet: selecting around a nested column avoids the throw (projection happens before the schema check fails on it)", async () => {
  const frame = await readParquet(new URL("fixtures/nested.parquet", import.meta.url).pathname, {
    columns: ["id"],
  });
  assert.deepEqual(frame.columns, ["id"]);
  assert.equal(frame.length, 10);
});
