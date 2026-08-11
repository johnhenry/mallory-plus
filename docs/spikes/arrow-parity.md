# Spike: apache-arrow JS ↔ PyArrow IPC round-trip parity

**Date:** 2026-08-10
**Purpose:** Validate whether `frame-arrow` can rely on Arrow IPC as the interchange format between mallory-plus Frames (JS) and pandas (Python), and which types v1 can safely claim.

**Versions tested (pinned, exact):**

| Component | Version |
|---|---|
| apache-arrow (npm) | **21.2.0** |
| pyarrow | **20.0.0** |
| pandas | **2.3.1** |
| Node.js | v26.5.0 |
| Python | 3.13.12 |

Scratch code lives in `/tmp/arrow-parity-spike/` (`js_write.mjs`, `py_side.py`, `js_read.mjs`). All tests used Arrow IPC **file** format (`tableToIPC(t, 'file')` / `pa.ipc.new_file`) unless noted; stream format was additionally verified for chunked data in both directions.

## Verdict summary

| Feature | JS → Py (pyarrow + `.to_pandas()`) | Py → JS | Notes |
|---|---|---|---|
| 1. f64 / i32 / utf8 / bool with nulls | ✅ works | ✅ works | Null bitmaps exact both ways. pandas coerces dtypes on `.to_pandas()` (see below) — an Arrow↔pandas issue, not a JS↔Py one |
| 2. int64 (BigInt) | ✅ works, bit-exact | ✅ works, bit-exact | JS `get()` returns `bigint`; `JSON.stringify` on rows throws; pandas `.to_pandas()` of nullable int64 coerces to float64 (lossy > 2^53) unless extension dtypes used |
| 3. Dictionary-encoded strings | ✅ works | ✅ works | pandas categorical ↔ Arrow dictionary clean both ways; index width differs (JS default int32, pandas emits int8) — both sides handle it; JS **dictionary-encodes plain string arrays by default** in `tableFromArrays` |
| 4. Chunked (multi-record-batch) tables | ✅ chunking preserved | ✅ chunking preserved | 3 batches in → 3 batches out, both file & stream format, both directions; JS exposes `table.batches` and per-column `vector.data[]`, random access is chunk-transparent |
| 5. list\<f64\> and struct\<a:f64, b:utf8\> | ✅ works incl. nulls at every level | ✅ works | JS returns `Vector` for list cells and `StructRow` proxy for struct cells (need `.toJSON()`/`toArray()` to plain-JS) |
| 6. timestamp[ms] / [us] (+tz) | ✅ works | ⚠️ works with caveats | Bytes + tz metadata exact both ways. JS **builder** for timestamp[us] is effectively broken for arbitrary µs values; JS `get()` returns float epoch-ms for all units (precision loss for ns display, buffer stays exact) |
| Schema metadata (`pandas` key) | ✅ survives JS read→rewrite | ✅ delivered to JS | Full fidelity: pandas restored nullable `Int32`/`boolean` extension dtypes from metadata after a trip through JS |

**Overall: full binary parity.** Every feature round-trips at the Arrow-buffer level in both directions with zero data loss. All caveats are *API-surface* issues (JS builder ergonomics, JS `get()` number coercion, pandas dtype coercion) — not interchange failures.

## Package size (apache-arrow 21.2.0)

- `node_modules/apache-arrow`: **8.4 MB** on disk (5.6 MB apparent) — includes CJS + ESM + UMD builds, `.d.ts`, and source maps.
- Full install incl. deps (`flatbuffers` 592 KB, `tslib` 132 KB, `json-with-bigint` 140 KB, `@types/node` 2.6 MB): **~13 MB** total `node_modules`.
- Shipped minified UMD bundles: `Arrow.es2015.min.js` **188 KB** (51.5 KB gzip), `Arrow.esnext.min.js` **178 KB** (48.5 KB gzip).
- The package is ESM tree-shakeable (`Arrow.mjs` + per-module files), so a bundler pulls substantially less than 188 KB if only IPC read/write + a subset of types is used. For a Node-side library the size is a non-issue; for browser use budget ~50 KB gzip worst case.

---

## 1. Basic columns (f64, i32, utf8, bool) with nulls

**Status: works both ways, null bitmaps exact.**

JS write:

```js
import { Table, vectorFromArray, tableToIPC, Float64, Int32, Utf8, Bool } from 'apache-arrow';

const basic = new Table({
  f64:  vectorFromArray([1.5, null, 3.25, -0.5], new Float64()),
  i32:  vectorFromArray([1, null, -3, 2147483647], new Int32()),
  str:  vectorFromArray(['alpha', null, 'gamma', ''], new Utf8()),
  flag: vectorFromArray([true, null, false, true], new Bool()),
});
fs.writeFileSync('basic.arrow', tableToIPC(basic, 'file'));
```

Python read — schema arrives as `double / int32 / string / bool`, all null counts = 1, empty string `''` distinct from null:

```python
t = pyarrow.ipc.open_file('basic.arrow').read_all()
df = t.to_pandas()
# nulls preserved: {'f64': 1, 'i32': 1, 'str': 1, 'flag': 1}
```

**Surprises / coercions (all on the pandas side, not the wire):**

- `.to_pandas()` on a nullable `int32` column yields **float64** (classic NumPy-pandas null coercion). Nullable `bool` yields **object** dtype. The Arrow table itself (`t.column(...).to_pylist()`) is exact.
- Reverse direction: writing from pandas with extension dtypes (`pd.array([...], dtype="Int32")`, `dtype="boolean"`) produces proper `int32`/`bool` Arrow columns with real null bitmaps — JS reads them perfectly (`nullCount: 1`, `get(1) === null` per column).
- **Explicit types matter on the JS side.** `tableFromArrays({...})` type inference: numbers → `Float64` (good), booleans → `Bool` (good), but **strings → `Dictionary<Int32, Utf8>`**, not `Utf8`. See §3.
- **`toArray()` null quirk (JS):** `vector.toArray()` returns the raw typed array; null slots contain whatever is in the buffer (pyarrow-written f64 nulls showed `NaN`; JS-written int64 nulls showed `0n`). Never use `toArray()` for null-aware access — use `get(i)`/iteration, which consult the validity bitmap.

## 2. int64 (BigInt64Array-backed)

**Status: bit-exact both ways, including values outside Number.MAX_SAFE_INTEGER.**

JS write:

```js
import { vectorFromArray, makeVector, Int64 } from 'apache-arrow';

// nullable, explicit type — values are JS BigInts
const i64 = vectorFromArray([1n, null, -9007199254740993n, 9223372036854775807n], new Int64());

// non-null fast path: makeVector infers Int64 directly from a BigInt64Array
const v = makeVector(new BigInt64Array([1n, 2n, 9223372036854775807n])); // type: Int64
```

Python read — `int64` schema, `to_pylist()` gives exact ints:

```python
t.column('i64').to_pylist()
# [1, None, -9007199254740993, 9223372036854775807]  — both sentinels EXACT
```

Py → JS: `pa.array([...], type=pa.int64())` read back in JS gives `get(i)` → **`bigint`** (`9223372036854775807n === get(3)` is `true`), null → `null`.

**JS number-vs-bigint behavior:**

- `vector.get(i)` on Int64 returns **`bigint`** in apache-arrow 21.x (older 9.x-era versions returned numbers/Int32Array pairs; that era is gone). No silent number coercion, no precision loss.
- `vector.toArray()` returns a `BigInt64Array` — **null slots read as `0n`** (validity bitmap not consulted).
- **`JSON.stringify(table.get(i))` and `row.toJSON()` both THROW** (`Do not know how to serialize a BigInt`) when the row contains an int64. Any Frame `.toJSON()`-style path must special-case bigint.
- pandas coercion: `.to_pandas()` on nullable int64 → **float64**, which silently mangles values > 2^53 (`-9007199254740993` became `-9007199254740992.0` in the DataFrame while the Arrow table held it exactly). Python users wanting exactness need `t.to_pandas(types_mapper=pd.ArrowDtype)` or non-null columns.

## 3. Dictionary-encoded strings ↔ pandas categorical

**Status: works cleanly both ways.**

JS write:

```js
import { Dictionary, Utf8, Int32, vectorFromArray } from 'apache-arrow';

const dictType = new Dictionary(new Utf8(), new Int32());
const color = vectorFromArray(['red', 'green', null, 'red', 'blue', 'green'], dictType);
```

Python read:

```python
t.schema  # color: dictionary<values=string, indices=int32, ordered=0>
df = t.to_pandas()
df['color'].dtype        # category
df['color'].dtype.categories  # Index(['red', 'green', 'blue'])  — first-appearance order kept
df['color'].tolist()     # ['red', 'green', nan, 'red', 'blue', 'green']
```

No re-encoding on read; dictionary order (first-appearance) survives, becomes the categorical's category order.

Py → JS (pandas categorical):

```python
df = pd.DataFrame({'color': pd.Categorical(['red', 'green', None, 'red', 'blue', 'green'])})
pa.Table.from_pandas(df)  # dictionary<values=string, indices=int8, ordered=0>
```

JS reads it as `Dictionary<Int8, Utf8>` — note pandas emits the **narrowest index type (int8** for few categories), not int32; JS handles any index width transparently. `get(i)` returns the decoded string, dictionary values accessible via `vector.data[0].dictionary`. pandas sorts categories lexically (`['blue','green','red']` here) and re-maps indices — values are identical, but **category/dictionary order is not a stable contract across the boundary**.

**Big JS-side surprise:** `vectorFromArray(['x','y','x'])` and `tableFromArrays` with a string column **dictionary-encode by default** → `Dictionary<Int32, Utf8>`, not `Utf8`. Usually harmless (pandas just sees a categorical), but it means a naive Frame→Arrow export changes the logical type users see in Python (`category` instead of `object`/string). Pass `new Utf8()` explicitly to get a plain string column.

## 4. Chunked data (multiple record batches)

**Status: chunk structure fully preserved, both directions, both IPC formats.**

JS write — batches are created per-`Table` and preserved by `concat`:

```js
const chunked = t1.concat(t2, t3);   // 3+2+4 rows
chunked.batches.length               // 3
chunked.batches.map(b => b.numRows)  // [3, 2, 4]
tableToIPC(chunked, 'file');    // and 'stream'
```

Python sees exactly the same structure:

```python
r = pa.ipc.open_file('chunked.arrow'); r.num_record_batches  # 3
t = r.read_all(); t.column('v').num_chunks                   # 3, sizes [3, 2, 4]
# stream format: list(pa.ipc.open_stream(...)) -> 3 batches [3, 2, 4]
```

Py → JS (`pa.Table.from_batches([...])`): JS reads **3 batches, sizes [3, 2, 4]** — chunking is not flattened on read.

**How apache-arrow JS exposes chunks:**

- `table.batches` → `RecordBatch[]` (one per IPC record batch).
- `table.getChild('v').data` → `Data[]`, one `Data` per chunk (`v.data.map(d => d.length)` → `[3, 2, 4]`).
- **Random access is chunk-transparent**: `v.get(6)` correctly resolves into the third chunk (binary search over chunk offsets). Iteration and `get()` need no chunk awareness.
- Incremental / streaming read: `RecordBatchReader.from(bytes)` is iterable batch-by-batch (verified against a pyarrow-written stream file).

## 5. Nested types: list\<f64\> and struct\<{a: f64, b: utf8}\>

**Status: supported both ways, including nulls at every level (null list, empty list, null element, null struct, null struct field).**

JS write:

```js
import { List, Struct, Field, Float64, Utf8, vectorFromArray } from 'apache-arrow';

const listVec = vectorFromArray([[1.5, 2.5], null, [], [3.5, null, 5.5]],
  new List(new Field('item', new Float64(), true)));
const structVec = vectorFromArray(
  [{ a: 1.5, b: 'one' }, null, { a: null, b: 'three' }, { a: 4.5, b: null }],
  new Struct([new Field('a', new Float64(), true), new Field('b', new Utf8(), true)]));
```

Python read: schema `list<item: double>` / `struct<a: double, b: string>`; `to_pandas()` gives object columns where list cells are **`numpy.ndarray`** (inner null → `nan`, so a null float inside a list is indistinguishable from NaN after `to_pandas` — `to_pylist()` keeps `None`) and struct cells are **`dict`** (`{'a': None, 'b': 'three'}` — inner nulls kept as `None`). Null rows → `None`, empty list ≠ null list.

Py → JS: identical schema arrives; **cell access is proxy-based**:

- `listVec.get(0)` → a `Vector` (call `.toJSON()`/`.toArray()` for plain JS); `get(3).toJSON()` → `[3.5, null, 5.5]` — inner null preserved.
- `structVec.get(0)` → a `StructRow` proxy with property access (`row.a === 1.5`) and `.toJSON()`.

No metadata or shape loss either way.

## 6. Timestamps (timestamp[ms], timestamp[us], timezones)

**Status: wire format + timezone metadata perfect both ways; the JS *builder and accessor* APIs have real sharp edges.**

JS write (ms via Dates works fine; **µs requires a workaround**):

```js
import { TimestampMillisecond, TimestampMicrosecond, vectorFromArray, makeData, Vector } from 'apache-arrow';

// ms: builder accepts Date objects (or epoch-ms numbers)
const tsMs = vectorFromArray([new Date('2026-01-15T12:34:56.789Z'), null, new Date(-1)],
  new TimestampMillisecond('UTC'));   // tz optional; omit for naive

// us: the builder is a trap — see below. Exact-µs workaround via makeData:
const tsUs = new Vector([makeData({
  type: new TimestampMicrosecond('America/Los_Angeles'),
  length: 3, nullCount: 1,
  nullBitmap: new Uint8Array([0b00000101]),           // row 1 null
  data: new BigInt64Array([1737000000123456n, 0n, -1n]),
})]);
```

Python read — exact, with correct dtype and tz:

```python
t.schema
# ts_ms_naive: timestamp[ms]            -> datetime64[ms]
# ts_ms_utc:   timestamp[ms, tz=UTC]    -> datetime64[ms, UTC]
# ts_us_la:    timestamp[us, tz=America/Los_Angeles] -> datetime64[us, America/Los_Angeles]
t.column('ts_us_la').to_pylist()[0]  # 2025-01-15 20:00:00.123456-08:00  — µs exact
```

Py → JS (pandas `datetime64[ns]`, tz-aware ns, explicit ms/us columns): all four schemas arrive intact (`Timestamp<NANOSECOND>`, `Timestamp<NANOSECOND, UTC>`, `Timestamp<MILLISECOND>`, `Timestamp<MICROSECOND, America/Los_Angeles>`); `type.timezone` exposes the tz string.

**Sharp edges found:**

1. **JS timestamp[us]/[ns] builder is effectively broken for exact sub-ms input.** `visitor/set.mjs` does `values[index] = BigInt(value * 1000)` for µs — input is assumed to be *epoch milliseconds as a JS number*:
   - Passing bigint µs values **throws** `Cannot mix BigInt and other types`.
   - Passing float ms works only when `ms * 1000` is float-exact; otherwise **throws** `The number 123456.7 cannot be converted to a BigInt because it is not an integer` (e.g. `123.4567` ms, and realistically most modern-epoch values with fractional ms: `1768480496789.1235 * 1000` → non-integer → throw).
   - Conclusion: for timestamp[us]/[ns] writes, construct the `BigInt64Array` buffer yourself via `makeData` (as above). Do not route through `vectorFromArray`.
2. **JS `get()` returns a `number` of epoch-*milliseconds* for every timestamp unit** (never a `Date`, never a bigint): `Timestamp<MICROSECOND>.get(0)` → `1737000000123.456` (float ms with fractional µs), `Timestamp<NANOSECOND>.get(0)` → `1768480496789.1235` — **ns precision is lost in the accessor** (float64 can't hold ns-since-epoch). The underlying buffer is still exact (`vector.data[0].values[0]` → `1768480496789123456n`), so a Frame accessor wanting exactness must read the int64 buffer directly.
3. **Timezone semantics match Arrow spec on both sides:** naive (no tz) stays wall-clock-naive in pandas (`datetime64[ms]`); tz-aware is stored UTC + tz metadata and pandas localizes for display. No shifting or tz loss in either direction. JS does no tz conversion at all — you always get the raw UTC-epoch value plus `type.timezone` metadata to interpret yourself.
4. pandas `datetime64[ns]` arrives in JS as `Timestamp<NANOSECOND>` — a unit JS mostly handles as "buffer you can read"; if Frame v1 only claims ms/us, consider converting ns → us at ingest (pyarrow users can also `.cast()` before writing).

## Schema metadata fidelity

The `pandas` schema-metadata blob (written by `pa.Table.from_pandas`) **survives a JS read → `tableToIPC` rewrite untouched**. After the trip through JS, `to_pandas()` still restored extension dtypes from it (`i32` → nullable `Int32`, `flag` → `boolean` instead of float64/object). JS exposes it as `table.schema.metadata` (a `Map`). So Frame can round-trip pandas metadata for free as long as it copies schema metadata when reconstructing tables.

---

## Implications for the frame-arrow v1 method list

**Safe to claim in v1 (verified bit-exact both directions):**

- `float64`, `float32` (by extension), `int32` (and other ≤32-bit ints), `utf8`, `bool` — all with nulls.
- `int64` — as **bigint** on the JS surface. Frame should standardize: `get()` returns `bigint`, provide an opt-in lossy `Number` accessor that throws or warns outside ±2^53, and make Frame's JSON serialization bigint-safe (apache-arrow's own `row.toJSON()` throws).
- `dictionary<utf8>` ↔ pandas categorical. Accept any index width on read (pandas emits int8/int16 depending on cardinality); emit int32 on write. Document that dictionary/category *order* is not a stable contract.
- Chunked tables: expose chunk-transparent row access (apache-arrow already provides it) plus an explicit `batches`/`chunks` accessor; preserve batching on write (write one record batch per chunk). Support both IPC file and stream formats — both verified.
- `timestamp[ms]` and `timestamp[us]`, naive and tz-aware. Internally store/emit int64 buffers directly (`makeData`), **never** apache-arrow's timestamp builders for µs. Frame accessors should return either bigint epoch-µs or `Date` (ms-truncated) explicitly, not apache-arrow's float-ms `get()`.
- `list<primitive>` and flat `struct` — with a materialization step (`Vector`/`StructRow` proxies → plain arrays/objects) at the Frame API boundary.

**Defer past v1:**

- `timestamp[ns]` as a first-class Frame type (JS accessor precision loss; fine to *pass through* or auto-cast to µs at ingest — pick one and document it).
- `date32/64`, `time32/64`, `duration`, `decimal128` — untested in this spike; decimal in particular has known JS ergonomic gaps.
- Deeply nested types (list<struct>, struct-of-list, map) — plain list/struct work, but combinatorial null handling wasn't validated here.
- Delta dictionaries / dictionary replacement across batches in streams — untested.
- `large_utf8`/`large_list` (64-bit offsets) — pandas doesn't emit them by default; untested.
- pandas extension-dtype fidelity guarantees (we verified metadata passthrough works; *claiming* it means owning the metadata contract).

## Open risks

1. **JS timestamp builder bug surface (apache-arrow 21.2.0).** The `BigInt(value * 1000)` pattern in `setTimestampMicrosecond` means any code path that lets user values reach `vectorFromArray`/Builder for µs/ns timestamps will throw on ~all realistic inputs. Frame must own its timestamp buffer construction; watch upstream for fixes that could change accepted input types under us.
2. **Bigint leakage.** int64 and timestamp buffers surface bigints; any Frame path that touches `JSON.stringify`, structured-clone-to-worker consumers expecting numbers, or arithmetic with numbers will throw or need conversion policy. This is the largest API-design risk, not an interop risk.
3. **Silent dictionary-encoding of strings** by apache-arrow's inference (`tableFromArrays`, bare `vectorFromArray`). If frame-arrow ever uses inference helpers, Python users will see `category` dtype where they expected strings. Always pass explicit types.
4. **pandas `.to_pandas()` coercions** (nullable int → float64 with silent >2^53 corruption, nullable bool → object) will be perceived as Frame bugs. Docs should prescribe `to_pandas(types_mapper=pd.ArrowDtype)` or ship pandas metadata (verified to survive) so extension dtypes are restored.
5. **`toArray()` vs validity bitmap**: raw typed arrays contain garbage (0, NaN, stale bytes) in null slots on both write and read sides. Any Frame fast path over typed arrays must mask with the null bitmap.
6. **Version drift.** All findings pinned to apache-arrow 21.2.0 / pyarrow 20.0.0 / pandas 2.3.1. apache-arrow JS has historically changed accessor return types across majors (e.g. int64 number→bigint); re-run this spike's scripts on upgrades (`/tmp/arrow-parity-spike/` has the full harness: `js_write.mjs` → `py_side.py` → `js_read.mjs`).
7. **Bundle weight for browser targets**: ~50 KB gzip if the whole UMD bundle lands; acceptable, but tree-shaking discipline (import from `apache-arrow`'s ESM entry, avoid the UMD build) should be a stated build requirement if frame-arrow targets browsers.
