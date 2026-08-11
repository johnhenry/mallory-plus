# Parquet library bakeoff: parquet-wasm vs hyparquet (+hyparquet-writer)

**Spike date:** 2026-08-10 · **Scratch dir:** `/tmp/parquet-bakeoff-spike` · **Node v26.5.0, pyarrow 20.0.0**
**Versions tested:** parquet-wasm **0.7.2**, hyparquet **1.28.1**, hyparquet-writer **0.16.5**, hyparquet-compressors **1.1.1**, apache-arrow **21.2.0**

## Verdict summary

| Criterion | parquet-wasm 0.7.2 | hyparquet 1.28.1 (+writer 0.16.5) |
|---|---|---|
| 1. Read pyarrow file (zstd, 4 row groups, dict/list/struct/nulls) | ✅ All 7 columns verified value-identical | ✅ All 7 columns verified value-identical (**requires `hyparquet-compressors` for zstd** — core throws) |
| 2. Column projection | ⚠️ **Broken on the easy paths**: sync `readParquet` silently ignores `columns`; async `ParquetFile.read({columns})` returns a Table whose IPC is unparseable (GH #810). Works correctly only via `ParquetFile.stream({columns})` — verified 2.98 MB fetched vs 8.4 MB | ✅ First-class `columns` option; verified 2.98 MB fetched vs 8.4 MB, plan-level (only projected column chunks fetched) |
| 3. Row-group / predicate pushdown | ⚠️ Manual only: `rowGroups: [i]` + `limit`/`offset`. **No statistics exposed** on `ColumnChunkMetaData` (open GH #863/#864), so you cannot even implement stats-based pruning yourself from its metadata API. `stream()` applies `limit` per row group (quirk) | ✅ Real pushdown: `filter` option does statistics-based row-group skipping (`canSkipRowGroup` on min/max), bloom-filter checks, optional page-level pruning via offset/column indexes. Verified: filter touching 1/5 groups fetched 599 KB vs 8.4 MB |
| 4. Write + pyarrow round-trip | ✅ ZSTD write, pyarrow verified values+schema+nulls identical | ✅ SNAPPY write, pyarrow verified values+schema+nulls identical. ❌ No zstd compressor in the family (fzstd is decompress-only); **footgun: `codec:'ZSTD'` without a compressor silently writes a corrupt file** |
| 5. Arrow integration | ✅ Native: Arrow IPC out (`intoIPCStream` → `tableFromIPC`), or true zero-copy via FFI + arrow-js-ffi. Dictionary encoding preserved (`Dictionary<Int32,Utf8>`) | ❌ Plain JS objects/arrays (even `onChunk` yields plain `Array`, BigInt for i64, dict columns materialized to strings). Arrow conversion is an extra full pass: +108 ms for 2 cols × 500k rows |
| 6. Payload | ❌ WASM binary **6.49 MB raw / 1.82 MB gzip**; 20 MB installed (3 wasm builds) | ✅ hyparquet core 54 KB min / **16.7 KB gz**; +compressors 75 KB gz; +writer 16.5 KB gz. Installed: 488 K + 212 K + 912 K |
| 7. Maintenance | Last publish 2026-06-29 (0.7.2), prior gap Oct 2025→Jun 2026; 129k dl/wk; 658★, 42 open issues; solo author (kylebarron) | Very active: hyparquet 1.28.1 published 2026-08-07, ~monthly-or-faster cadence; 858k dl/wk (writer 339k); 847★, 31 open issues (mostly dependabot + features) |
| Speed (8.4 MB / 500k rows, local file) | Full read → Arrow Table: **195 ms** (sync buffer) / 400 ms (stream). Projected 2 cols: **143 ms** | Full read → row objects: 1046 ms. Projected 2 cols: 336 ms (+108 ms if you also need Arrow vectors ⇒ ~433 ms) |

## RECOMMENDATION for frame-parquet v1

**Overturn the prior lean. Use hyparquet + hyparquet-writer as the primary engine for frame-parquet v1.**

The prior plan leaned parquet-wasm for "Arrow-native zero-copy," and that story is real — but this spike found the surrounding reality worse than the marketing:

1. **parquet-wasm's projection is broken on 2 of its 3 read paths in the current release (0.7.2).** `readParquet(buf, {columns})` silently returns all columns; `ParquetFile.read({columns})` returns a Table whose `intoIPCStream()` crashes apache-arrow (`Cannot destructure property 'length'…`, upstream issue #810: batches are projected but the schema isn't). Only `ParquetFile.stream({columns})` works. Shipping v1 on a library where the obvious API calls are silently wrong is a support-burden magnet.
2. **parquet-wasm cannot do predicate pushdown, even manually-with-effort**: row-group `min/max` statistics are not exposed to JS (open issues #863/#864 since 2026-03), so stats-based group pruning would require parsing the footer with… hyparquet. hyparquet does statistics + bloom-filter + page-index pruning automatically from a mongo-style `filter` option, and we verified the I/O skipping is real (599 KB vs 8.4 MB).
3. **The 1.82 MB gzipped WASM payload** (vs ~91 KB gz for hyparquet+compressors read path) matters if frame-parquet ever runs in a browser/edge context, and 20 MB of node_modules vs 1.6 MB matters for install footprint.
4. **The Arrow gap is bridgeable; the pushdown/payload gaps are not.** hyparquet's cost is one conversion pass (`vectorFromArray`): measured +108 ms on 1M values. Total projected-read-to-Arrow is ~433 ms vs 143 ms for parquet-wasm — ~3× slower, but hundreds of ms at 500k rows, not seconds, and irrelevant for frame-sized data.

**Caveats / when to revisit:** if frame-parquet's dominant workload becomes "bulk-decode multi-hundred-MB files already in memory into Arrow," parquet-wasm's Rust decoder is genuinely ~3–5× faster and Arrow-native — consider it then as an optional accelerator behind the same interface (its `stream({columns, rowGroups})` path is solid). Also adopt parquet-wasm if zstd-*write* becomes a hard requirement (hyparquet-writer is snappy/uncompressed out of the box; zstd write needs a BYO compressor, at which point you're bundling WASM anyway).

**v1 implementation notes if we adopt hyparquet:**
- Always pass `compressors` from `hyparquet-compressors` (core hyparquet throws `parquet unsupported compression codec: ZSTD` on pyarrow's default-adjacent codec).
- Guard the writer: never pass `codec:'ZSTD'` without a compressor entry — it silently emits a corrupt file (verified: pyarrow fails with `ZSTD decompression failed: Unknown frame descriptor`).
- i64 comes back as `BigInt`; struct fields keep BigInt too. Optional struct/list fields may resolve `undefined` instead of `null` (upstream #168) — normalize at the frame-arrow boundary.
- Dictionary encoding is decoded to plain values (no `Dictionary<...>` Arrow type preservation). If frame-arrow wants dict-encoded vectors, re-encode with `vectorFromArray` on a dictionary type.

---

## 1. Reading a pyarrow-written file

Fixture: 100 rows, `row_group_size=30` → **4 row groups**, zstd, columns `f64/i64/utf8/bool` (each with nulls), dictionary-encoded `dict`, `list_f64` (`list<double>`, with null lists), `struct{x:int64,y:utf8}` (with null structs). Written by pyarrow 20.0.0; expected values exported to JSON and compared element-wise in JS.

### hyparquet

```js
import { parquetReadObjects, asyncBufferFromFile } from 'hyparquet'
import { compressors } from 'hyparquet-compressors'   // REQUIRED for zstd
const file = await asyncBufferFromFile('fixture_zstd.parquet')
const rows = await parquetReadObjects({ file, compressors })
// row[1]: {"f64":1.5,"i64":1000n,"utf8":"row-1","bool":false,"dict":"beta","list_f64":[1,2],"struct":{"x":1n,"y":"s1"}}
```

- **Result: 100 rows, all 7 columns × 100 rows value-identical** (including null positions, list contents, struct fields).
- Without `hyparquet-compressors`: `parquet unsupported compression codec: ZSTD`. (Snappy and uncompressed work in core.)
- Types: `f64→Number, i64→BigInt, utf8→String, bool→Boolean, dict→String (materialized), list→plain Array, struct→plain Object`.

### parquet-wasm

```js
import { readParquet } from 'parquet-wasm'     // node build self-initializes
import { tableFromIPC } from 'apache-arrow'
const table = tableFromIPC(readParquet(buf).intoIPCStream())
// schema: f64:Float64, i64:Int64, utf8:Utf8, bool:Bool,
//         dict:Dictionary<Int32, Utf8>, list_f64:List<Float64>, struct:Struct<{x:Int64, y:Utf8}>
```

- **Result: 100 rows, all 7 columns value-identical.** zstd handled natively (Rust arrow/parquet crates).
- Arrow types faithfully preserved, including the dictionary encoding (`Dictionary<Int32, Utf8>`) that hyparquet flattens.
- Types surfaced through Arrow JS: `list→arrow Vector`, `struct→StructRow` (BigInt inside), i.e. Arrow semantics, not POJO.

**Both pass criterion 1.**

## 2. Column projection

Measured on an 8.4 MB / 500k-row / 5-row-group zstd file, instrumenting the I/O layer (hyparquet `AsyncBuffer.slice`; a logging `Blob` subclass for parquet-wasm's `ParquetFile`). Metadata parsing excluded from byte counts.

| Read | fetches | bytes fetched | time |
|---|---|---|---|
| hyparquet full read | 5 | 8,394,247 | 1046 ms |
| hyparquet `columns:['id','value']` | 10 | **2,981,135** | 336 ms |
| parquet-wasm `stream()` full | 5 | 8,394,247 | 400 ms |
| parquet-wasm `stream({columns:['id','value']})` | 5 | **2,981,135** | 143 ms |

Both skip un-projected column chunks at the I/O level — byte-for-byte identical fetch totals (2,981,135). But parquet-wasm's other two read paths are broken in 0.7.2:

```js
// BUG 1: silently ignored (returns all 7 columns) — GH #758/#81
readParquet(buf, { columns: ['f64', 'utf8'] })

// BUG 2: batches ARE projected (batch.numColumns === 2) but the Table's schema
// isn't, so intoIPCStream() output crashes apache-arrow's tableFromIPC — GH #810
const wt = await parquetFile.read({ columns: ['f64', 'utf8'] })
tableFromIPC(wt.intoIPCStream())  // TypeError: Cannot destructure property 'length'…

// WORKS: per-batch IPC via stream()
for await (const rb of await parquetFile.stream({ columns: ['f64', 'utf8'] })) { … }
```

hyparquet's `columns` option works on every entry point (`parquetRead`, `parquetReadObjects`, `parquetQuery`, `onChunk`).

## 3. Row-group / predicate pushdown — what's real

### hyparquet: real, automatic, statistics-based

`parquetRead`/`parquetQuery` accept a mongo-style `filter` (`$gt/$gte/$lt/$lte/$ne/$in/$nin/$and/$or/$not`). The internal planner (`src/plan.js`) calls `canSkipRowGroup()` (`src/filter.js`), which compares the predicate against each row group's `min_value/max_value/null_count` statistics, optionally consults **bloom filters** for equality predicates, and can do **page-level pruning** via column/offset indexes (`useOffsetIndex`). Remaining rows are then exactly filtered (read-then-filter *within* surviving groups).

Verified on the 500k-row file (`value` is monotonic, so stats are decisive):

```js
await parquetReadObjects({ file, metadata, compressors,
  columns: ['id','value'], filter: { value: { $gt: 449999.5 } } })
// → 2 fetches, 598,709 bytes (vs 8.4 MB full) — 4 of 5 row groups skipped via stats
// → 50,000 rows, all correct
```

Row-range pushdown also works: `rowStart/rowEnd: 0–100` fetched only group 0's `id` chunk (309 KB — chunk-granular; page-granular needs `useOffsetIndex`). Full raw metadata (statistics, bloom offsets, key/value metadata) is exposed via `parquetMetadata()`, e.g. `row_groups[0].columns[0].meta_data.statistics = {min:1.5, max:29.5, null_count:5n, …}`.

### parquet-wasm: manual selection only, and stats are inaccessible

- `ReaderOptions` offers `rowGroups: number[]`, `limit`, `offset` — *you* decide which groups to read. No `filter` anywhere.
- `ParquetMetaData → RowGroupMetaData → ColumnChunkMetaData` exposes `numValues/columnPath/compression/compressedSize/uncompressedSize/encodings` — **no min/max statistics** (confirmed against the full .d.ts; open upstream issues #863 "Exposing ColumnChunkMetadata.statistics…" and #864). So stats-based pruning cannot be built on parquet-wasm's own metadata API today.
- Manual pruning does skip I/O when you know the group: `stream({columns:['id','value'], rowGroups:[4]})` fetched 1 range / 598,709 bytes — identical bytes to hyparquet's automatic filter.
- Quirk: `stream({limit: 100})` applies the limit **per row group** (returned 500 rows across 5 groups, fetched all groups); `read({limit: 100})` applies it globally (100 rows).

**Net:** "predicate pushdown" is real in hyparquet and absent in parquet-wasm; parquet-wasm's row-group machinery is a usable but manual building block missing the statistics needed to drive it.

## 4. Write + pyarrow round-trip

Fixture: 50 rows × 4 columns (`f64/i64/utf8/bool`), each with nulls at different strides. pyarrow verified values, arrow schema, codec, and null positions.

### parquet-wasm — zstd, verified

```js
const arrowTable = new ArrowTable({ f64: vectorFromArray(f64, new Float64()), /* … */ })
const wasmTable = WasmTable.fromIPCStream(tableToIPC(arrowTable, 'stream'))
const props = new WriterPropertiesBuilder().setCompression(Compression.ZSTD).build()
fs.writeFileSync('out.parquet', writeParquet(wasmTable, props))
```
pyarrow: schema `{f64:double, i64:int64, utf8:string, bool:bool}`, codec ZSTD on every column, **VALUES+NULLS OK**. Writer also supports per-column codecs, statistics levels (`EnabledStatistics`), writer version, encodings.

### hyparquet-writer — snappy, verified; zstd is a trap

```js
const bytes = parquetWriteBuffer({ columnData: [
  { name: 'f64', data: f64, type: 'DOUBLE', nullable: true }, /* … */
], codec: 'SNAPPY' })
```
pyarrow: same schema, codec SNAPPY, **VALUES+NULLS OK**. Statistics written by default (`statistics: true`), configurable `rowGroupSize`.

- **No zstd write out of the box**: `hyparquet-compressors` is decompression-only (fzstd has no compressor), and the writer's `compressors` option expects codec→compress functions you must supply yourself.
- **Verified footgun:** `codec:'ZSTD'` with no compressor does **not** throw — `compressors[codec]?.(bytes) ?? bytes` silently stores uncompressed bytes labeled ZSTD. pyarrow: `ZSTD decompression failed: Unknown frame descriptor`. frame-parquet must whitelist codecs at its API boundary.

## 5. Arrow integration

**parquet-wasm** is Arrow-native end to end. Decoded data lives in WASM memory as Rust Arrow arrays; two exits:
- `intoIPCStream()` → `tableFromIPC()`: one memcpy WASM→JS + IPC parse; the resulting Arrow JS vectors are views over the IPC buffer (near-zero-copy).
- `toFFI()/intoFFI()` + [arrow-js-ffi](https://github.com/kylebarron/arrow-js-ffi): true zero-copy views directly into `wasmMemory().buffer` (extra dep, manual `free()` lifetime management, "bleeding edge" per its own docs).
For frame-arrow this means Parquet→Arrow is essentially free, and Arrow types round-trip faithfully (dictionary, list, struct).

**hyparquet** decodes to plain JS: row objects from `parquetReadObjects`, or per-column chunks via `onChunk` — which are **plain `Array`s even for non-null f64 columns** (measured; typed-array output for nullable columns is an open upstream request, #157). i64→BigInt, dictionary columns pre-materialized to values. frame-arrow integration therefore needs an explicit conversion pass:

```js
vectorFromArray(idArr, new Int64())  // etc.
// measured: hyparquet columnar read of 2×500k cols = 325 ms; Arrow conversion = +108 ms
```

~33% overhead on read time for numeric columns; string-heavy data will be worse (arrow re-encodes to UTF-8 buffers). Fine for frame-sized tables; a real cost for bulk pipelines.

## 6. Payload cost (measured)

| Artifact | raw | gzip |
|---|---|---|
| parquet-wasm WASM binary (`esm/parquet_wasm_bg.wasm`) | **6,494,208 B (6.5 MB)** | **1,818,794 B (1.82 MB)** |
| parquet-wasm JS glue (esbuild min, wasm external) | 34 KB | 8.3 KB |
| hyparquet core (esbuild min) | 54 KB | 16.7 KB |
| hyparquet-compressors (min; embeds hysnappy wasm + fzstd) | 115 KB | 75 KB |
| hyparquet-writer (min) | 56 KB | 16.5 KB |

Installed (`du -sh node_modules/<pkg>`): parquet-wasm **20 MB** (ships node/esm/bundler wasm triplicate); hyparquet **488 K**; hyparquet-writer **912 K**; hyparquet-compressors **212 K** (+fzstd 96 K, hysnappy 28 K). apache-arrow itself: 8.4 MB installed, and is a shared dependency either way.

Read-path browser payload: **~91 KB gz (hyparquet+compressors) vs ~1.83 MB gz (parquet-wasm)** — 20×.

## 7. Maintenance signals (checked 2026-08-10)

| | parquet-wasm | hyparquet / hyparquet-writer |
|---|---|---|
| Version / last publish | 0.7.2 / 2026-06-29 | 1.28.1 / 2026-08-07 · writer 0.16.5 / 2026-08-02 |
| Cadence | Bursty: 0.6.1 May 2024 → 0.7.0 Sep 2025 → 0.7.2 Jun 2026 | ~19 releases in the last 8 months |
| Weekly downloads | 129,254 | 857,571 (writer 338,810; compressors 149,139) |
| GitHub | 658★, 42 open issues, pushed 2026-08-10 | 847★, 31 open issues (writer: 60★, 6 open), pushed 2026-08-07 |
| Issue vibe | Long-lived functional gaps: #81 (2022, "improve reader options"), #758/#810 (projection broken), #863/#864 (no statistics) — the exact problems this spike hit. Solo-maintainer (kylebarron), high-quality but attention-divided (geoarrow ecosystem) | Mostly dependabot noise + feature discussions (#143 "Hyparquet V2", #157 typed arrays, #167 native gzip). Known correctness wart: #168 optional struct/list fields yield `undefined` not `null`. Backed by hyparam (company), tests aim at 100% coverage |
| Compressors staleness | n/a | hyparquet-compressors last published 2025-03-20 — stale-ish but tiny/stable surface |

## Open risks

1. **hyparquet is hand-rolled JS decoding** (thrift, RLE/bit-pack, delta, dremel assembly). Broad-but-not-exhaustive format coverage; exotic pyarrow outputs (BYTE_STREAM_SPLIT, shredded VARIANT edge cases, unusual converted types) should get fixture tests in frame-parquet CI. This spike's coverage (plain/dict/RLE, zstd/snappy, list/struct, multi-group) all passed.
2. **Null-vs-undefined wart (#168)**: nested optional fields can come back `undefined`; normalize before frame equality checks.
3. **Writer silent-corruption on unknown codec** (measured, §4): frame-parquet must validate codec support at its boundary rather than pass through.
4. **No zstd write** without BYO compressor; if wire-compat with pyarrow-default zstd files is required *on write*, that's a follow-up decision (snappy is universally readable by pyarrow, so round-trip compat is not blocked).
5. **BigInt i64s** leak into results; frame-arrow needs a deliberate policy (BigInt64Array vs number-with-precision-loss).
6. **Arrow conversion pass** (~+33% read time on numerics, worse on strings) is a permanent tax vs parquet-wasm; if profiling later shows it dominating a real workload, parquet-wasm's `stream({columns, rowGroups})` path (the one that works) can be slotted in as an optional fast bulk decoder — keep frame-parquet's interface engine-agnostic to preserve that exit.
7. **parquet-wasm bugs may get fixed** (projection #810, stats #863): if 0.8.x lands statistics exposure + fixed projection, the calculus tightens — payload (1.8 MB gz) and no-predicate-pushdown would remain the deciding factors.

## Reproduction

All scripts live in `/tmp/parquet-bakeoff-spike/`: `make_fixtures.py` (pyarrow fixtures + expected JSON), `test_hyparquet_read.mjs`, `test_parquetwasm_read.mjs` (criterion 1), `test_projection.mjs`, `test_io_skipping.mjs` (criteria 2–3, instrumented I/O), `test_write.mjs` + `verify_writes.py` (criterion 4), bundle outputs `out_*.js` (criterion 6).
