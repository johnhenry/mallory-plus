/**
 * Write-path correctness — critically, verified against REAL pyarrow
 * (`pyarrow.parquet.read_table`), not just round-tripped through
 * frame-parquet's own reader (self-consistency isn't proof of a valid
 * Parquet file — see write.ts's module doc and docs/spikes/parquet-bakeoff.md
 * §4's verified zstd footgun). pyarrow-dependent tests SKIP (don't fail)
 * when no interpreter with pyarrow is available (see test/helpers.ts).
 */
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
  Bool,
  Dictionary,
  Field,
  Float32,
  Float64,
  Int16,
  Int32,
  Int64,
  Int8,
  List,
  makeData,
  Table,
  type Timestamp,
  TimestampMicrosecond,
  TimestampMillisecond,
  Uint16,
  Uint32,
  Uint8,
  Utf8,
  Vector,
  vectorFromArray,
} from "apache-arrow";
import { Frame } from "mallory-frame-arrow";
import { readParquet, writeParquet, type WriteCodec } from "../src/index.ts";
import { PYARROW_SKIP_REASON, PYTHON, runPyarrowJson } from "./helpers.ts";

const N = 40;

function tsVector(values: readonly (bigint | null)[], type: Timestamp): Vector {
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

function sampleFrame(): Frame {
  const bool = Array.from({ length: N }, (_, i) => (i % 6 === 0 ? null : i % 2 === 0));
  const int8 = Array.from({ length: N }, (_, i) => (i % 7 === 0 ? null : (i % 200) - 100));
  const int16 = Array.from({ length: N }, (_, i) => (i % 8 === 0 ? null : i * 100 - 500));
  const int32 = Array.from({ length: N }, (_, i) => (i % 9 === 0 ? null : i * 100_000));
  const uint8 = Array.from({ length: N }, (_, i) => (i % 5 === 0 ? null : i % 250));
  const uint16 = Array.from({ length: N }, (_, i) => (i % 11 === 0 ? null : i * 1000));
  const uint32 = Array.from({ length: N }, (_, i) => (i % 13 === 0 ? null : i * 100_000));
  const int64 = Array.from({ length: N }, (_, i) => (i % 4 === 0 ? null : BigInt(i) * 10_000_000_000n));
  const float32 = Array.from({ length: N }, (_, i) => (i % 6 === 0 ? null : i + 0.25));
  const float64 = Array.from({ length: N }, (_, i) => (i % 10 === 0 ? null : i + 0.125));
  const utf8 = Array.from({ length: N }, (_, i) => (i % 8 === 0 ? null : `w-${i}`));
  const tsMs = Array.from({ length: N }, (_, i) => (i % 7 === 0 ? null : 1_700_000_000_000n + BigInt(i) * 1000n));
  const tsUs = Array.from({ length: N }, (_, i) => (i % 9 === 0 ? null : 1_700_000_000_000_000n + BigInt(i) * 137n));

  const table = new Table({
    bool: vectorFromArray(bool, new Bool()),
    int8: vectorFromArray(int8, new Int8()),
    int16: vectorFromArray(int16, new Int16()),
    int32: vectorFromArray(int32, new Int32()),
    uint8: vectorFromArray(uint8, new Uint8()),
    uint16: vectorFromArray(uint16, new Uint16()),
    uint32: vectorFromArray(uint32, new Uint32()),
    int64: vectorFromArray(int64, new Int64()),
    float32: vectorFromArray(float32, new Float32()),
    float64: vectorFromArray(float64, new Float64()),
    utf8: vectorFromArray(utf8, new Utf8()),
    ts_ms: tsVector(tsMs, new TimestampMillisecond(null)),
    ts_us: tsVector(tsUs, new TimestampMicrosecond(null)),
  });
  return Frame.fromArrow(table);
}

let tmpDir: string;
test.before(() => {
  tmpDir = mkdtempSync(join(tmpdir(), "frame-parquet-write-"));
});
test.after(() => {
  rmSync(tmpDir, { recursive: true, force: true });
});

for (const codec of ["snappy", "zstd", "uncompressed"] as WriteCodec[]) {
  test(`writeParquet(compression: "${codec}") round-trips through frame-parquet's own reader`, async () => {
    const frame = sampleFrame();
    const path = join(tmpDir, `roundtrip-${codec}.parquet`);
    await writeParquet(frame, path, { compression: codec });
    const readBack = await readParquet(path);
    assert.equal(readBack.length, N);
    assert.deepEqual(readBack.toRows(), frame.toRows());
  });

  test(`writeParquet(compression: "${codec}") produces a file pyarrow.parquet.read_table can read, with correct schema/values/nulls`, async (t) => {
    if (!PYTHON) {
      t.skip(PYARROW_SKIP_REASON);
      return;
    }
    const frame = sampleFrame();
    const path = join(tmpDir, `pyarrow-verify-${codec}.parquet`);
    await writeParquet(frame, path, { compression: codec });

    const result = runPyarrowJson(`
import json
import pyarrow.parquet as pq
pf = pq.ParquetFile(${JSON.stringify(path)})
md = pf.metadata
codecs = set()
for rg in range(md.num_row_groups):
    for c in range(md.num_columns):
        codecs.add(md.row_group(rg).column(c).compression)
t = pf.read()
out = {
    "schema": {f.name: str(f.type) for f in t.schema},
    "codecs": sorted(codecs),
    "columns": {name: t.column(name).to_pylist() for name in t.schema.names},
}
print(json.dumps(out, default=str))
`) as {
      schema: Record<string, string>;
      codecs: string[];
      columns: Record<string, unknown[]>;
    };

    // The codec pyarrow reports really was applied (this is the crux of the
    // footgun proof: a corrupt "ZSTD-labeled-but-actually-uncompressed" file
    // would still report codec ZSTD in metadata, but pyarrow.read() above
    // would already have thrown "Unknown frame descriptor" decoding it —
    // reaching this line at all is part of the proof, not just this assertion).
    const expectedCodec = codec === "snappy" ? "SNAPPY" : codec === "zstd" ? "ZSTD" : "UNCOMPRESSED";
    assert.deepEqual(result.codecs, [expectedCodec]);

    const rows = frame.toRows();
    for (const name of ["bool", "int32", "uint32", "float64", "utf8"]) {
      const want = rows.map((r) => r[name]);
      const got = result.columns[name];
      assert.deepEqual(got, want, `column "${name}" mismatch for codec ${codec}`);
    }
    // int64 / int8 / int16 / uint8 / uint16: pyarrow returns plain Python ints
    // (JSON numbers); compare against the bigint/number originals by value.
    for (const name of ["int64", "int8", "int16", "uint8", "uint16"]) {
      const want = rows.map((r) => {
        const v = r[name];
        return v === null ? null : typeof v === "bigint" ? Number(v) : v;
      });
      assert.deepEqual(result.columns[name], want, `column "${name}" mismatch for codec ${codec}`);
    }
    // timestamps: pyarrow reads ts_ms back as a UTC-tz-aware datetime — this
    // package writes TIMESTAMP_MILLIS via the plain converted_type (no
    // explicit logical_type isAdjustedToUTC flag), and pyarrow defaults such
    // legacy-annotated columns to UTC on read. `default=str` on that
    // datetime gives Python's `str(datetime)` format, e.g.
    // "2023-11-14 22:13:21+00:00" (space-separated, explicit +00:00 offset,
    // no trailing "Z") — swap the space for "T" and parse as-is.
    const tsMsWant = rows.map((r) => (r.ts_ms === null ? null : Number(r.ts_ms as bigint)));
    const tsMsGot = (result.columns.ts_ms as (string | null)[]).map((s) => (s === null ? null : Date.parse(s.replace(" ", "T"))));
    assert.deepEqual(tsMsGot, tsMsWant, `ts_ms mismatch for codec ${codec}`);
  });
}

test("writeParquet: unsupported compression string throws a clear error naming the supported codecs", async () => {
  const frame = sampleFrame();
  const path = join(tmpDir, "bad-codec.parquet");
  await assert.rejects(
    () => writeParquet(frame, path, { compression: "gzip" as WriteCodec }),
    /unsupported compression "gzip".*snappy, zstd, uncompressed/s,
  );
});

test("writeParquet: a list column throws a clear error naming the column, rather than silently dropping it", async () => {
  const listType = new List(new Field("item", new Float64(), true));
  const table = new Table({ values: vectorFromArray([[1.5, 2.5], null], listType) });
  const frame = Frame.fromArrow(table);
  const path = join(tmpDir, "list-column.parquet");
  await assert.rejects(() => writeParquet(frame, path), /column "values" has dtype "list"/);
});

test("writeParquet: a dictionary column writes as plain STRING data and reads back with the right values", async () => {
  const cats = ["alpha", "beta", "gamma"];
  const values = Array.from({ length: 12 }, (_, i) => cats[i % 3] as string);
  const dictType = new Dictionary(new Utf8(), new Int32());
  const table = new Table({ cat: vectorFromArray(values, dictType) });
  const frame = Frame.fromArrow(table);
  assert.equal(frame.schema[0]?.dtype, "dictionary");

  const path = join(tmpDir, "dictionary-column.parquet");
  await writeParquet(frame, path);
  const readBack = await readParquet(path);
  assert.equal(readBack.schema[0]?.dtype, "utf8"); // documented v1 simplification
  assert.deepEqual(
    readBack.toRows().map((r) => r.cat),
    values,
  );
});
