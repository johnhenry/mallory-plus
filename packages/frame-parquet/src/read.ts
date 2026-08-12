/**
 * Parquet -> Frame read path (issue #20).
 *
 * Genuine pushdown: `columns`/`filter` are passed straight through to
 * hyparquet's own `parquetReadObjects`, which does statistics-based
 * row-group skipping for `filter` and never fetches column chunks outside
 * `columns` — see docs/spikes/parquet-bakeoff.md §3 for how this was
 * measured (599 KB vs 8.4 MB), and test/pushdown.test.ts for the analogous
 * proof against this package's own `readParquetFile`, not just hyparquet in
 * isolation.
 *
 * `compressors` (from hyparquet-compressors) is passed UNCONDITIONALLY —
 * core hyparquet throws `parquet unsupported compression codec: ZSTD`
 * without it, and every file this package reads should transparently handle
 * zstd input regardless of what codec produced it (this is a read-side
 * concern, distinct from write.ts's zstd-compressor footgun).
 *
 * A quirk NOT called out in docs/spikes/parquet-bakeoff.md (found while
 * building this package, not just in the spike): hyparquet's DEFAULT
 * `timestampFromMilliseconds`/`timestampFromMicroseconds` parsers convert to
 * a JS `Date` — silently truncating microsecond precision, exactly the
 * lossy-`Date` sharp edge frame-arrow's own access.ts (`timestampExactAt`)
 * was built to route around. `READ_PARSERS` below overrides those two
 * parsers to identity functions so timestamp columns come back as the raw
 * exact BigInt epoch value (in the column's own unit), matching int64's own
 * BigInt convention and frame-arrow's own precision-exactness policy — never
 * handed a lossy `Date` in the first place.
 *
 * `READ_PARSERS` is a FULLY populated `ParquetParsers`, not a partial
 * override — a second quirk, also found while building this package: while
 * `parquetMetadataAsync` cleanly merges a partial `parsers` option with
 * hyparquet's own defaults, `parquetReadObjects`'s underlying row-group
 * reader (hyparquet's `rowgroup.js`, `readRowGroup`) computes that same
 * merge and then immediately clobbers it with `...options` (which still
 * carries the ORIGINAL, unmerged `parsers` object) — so a partial override
 * reaches column decoding unmerged, and `parsers.stringFromBytes` (needed
 * for every utf8 column) ends up `undefined`. Supplying a complete object
 * sidesteps that clobber entirely. The four parsers below that this
 * package's v1 dtype set never actually reaches (JSON/GEOMETRY/GEOGRAPHY/
 * UUID/DATE all throw at the schema.ts mapping step before a read is even
 * attempted) throw defensively rather than being silently wrong if that
 * invariant is ever violated by a future change.
 *
 * hyparquet hands back plain JS objects (row-oriented via
 * `parquetReadObjects`), not Arrow — this function does the one conversion
 * pass into Arrow vectors that costs the ~33% read-time premium documented
 * in the spike (§5), accepted at frame scale. A columnar `onChunk`-based
 * path would avoid the row<->column transpose, but `filter` requires
 * `rowFormat: 'object'` in hyparquet and `onChunk` is explicitly NOT
 * filtered (per hyparquet's own docs) — using row objects throughout keeps
 * filter+projection+conversion in one straightforward, correct pass rather
 * than reimplementing hyparquet's own row assembly for the onChunk path.
 */
import { Table } from "apache-arrow";
import {
  asyncBufferFromFile,
  type AsyncBuffer,
  parquetMetadataAsync,
  parquetReadObjects,
  parquetSchema,
  type ParquetParsers,
  type ParquetQueryFilter,
} from "hyparquet";
import { compressors } from "hyparquet-compressors";
import { Frame } from "mallory-frame-arrow";
import { arrowTypeFor, mapNamedColumns, topLevelColumnNames } from "./schema.ts";
import { buildVector } from "./vector-build.ts";

/** See this module's doc comment for why this is a complete `ParquetParsers`. */
const utf8Decoder = new TextDecoder();

function unreachableParser(kind: string): () => never {
  return () => {
    throw new Error(`mallory-frame-parquet: ${kind} columns are not supported in v1 (should have thrown earlier, at schema mapping)`);
  };
}

const READ_PARSERS: ParquetParsers = {
  timestampFromMilliseconds: (millis: bigint) => millis,
  timestampFromMicroseconds: (micros: bigint) => micros,
  timestampFromNanoseconds: unreachableParser("TIMESTAMP[NANOS]"),
  dateFromDays: unreachableParser("DATE"),
  stringFromBytes: (bytes: Uint8Array) => bytes && utf8Decoder.decode(bytes),
  jsonFromBytes: unreachableParser("JSON"),
  geometryFromBytes: unreachableParser("GEOMETRY"),
  geographyFromBytes: unreachableParser("GEOGRAPHY"),
  uuidFromBytes: unreachableParser("UUID"),
};

export interface ReadParquetOptions {
  /** Column projection — only these columns are fetched from disk. Default: all columns. */
  readonly columns?: readonly string[];
  /** Mongo-style filter ($gt/$gte/$lt/$lte/$eq/$ne/$in/$nin/$and/$or/$not); drives hyparquet's
   * statistics-based row-group skipping. See hyparquet's own docs for the full operator set. */
  readonly filter?: ParquetQueryFilter;
  /** Passthrough to hyparquet's `parquetMetadataAsync` — how many bytes to
   * speculatively fetch for the footer in one round trip (default 512 KiB,
   * hyparquet's own default, tuned for remote/HTTP sources). Files smaller
   * than this fetch their ENTIRE contents as part of "parsing metadata" —
   * harmless for small local files, but worth lowering for small remote
   * files or when precisely measuring I/O (see test/pushdown.test.ts). */
  readonly initialFetchSize?: number;
}

/**
 * Core read path over hyparquet's `AsyncBuffer` abstraction rather than a
 * file path — works with any hyparquet-compatible source (local file via
 * `asyncBufferFromFile`, a URL via `asyncBufferFromUrl`, or an in-memory
 * buffer via a small manual wrapper), which keeps frame-parquet
 * engine-agnostic per the spike's own recommendation. It's also what makes
 * the pushdown-is-real proof possible: test/pushdown.test.ts wraps a real
 * file's AsyncBuffer to count bytes fetched and calls this exact function,
 * not a reimplementation of it.
 */
export async function readParquetFile(file: AsyncBuffer, options: ReadParquetOptions = {}): Promise<Frame> {
  const metadata = await parquetMetadataAsync(file, {
    parsers: READ_PARSERS,
    initialFetchSize: options.initialFetchSize,
  });
  const tree = parquetSchema(metadata);
  const allNames = topLevelColumnNames(tree);

  let wantedNames = allNames;
  if (options.columns) {
    const allNameSet = new Set(allNames);
    for (const name of options.columns) {
      if (!allNameSet.has(name)) {
        throw new Error(`readParquet: no such column "${name}" (available: ${allNames.join(", ")})`);
      }
    }
    wantedNames = [...options.columns];
  }
  // Only the actually-requested columns are dtype-mapped (and can throw
  // UnsupportedParquetTypeError) — a column that's pruned away by `columns`
  // is never even inspected, same pruning philosophy as frame-arrow itself.
  const columnSchemas = mapNamedColumns(tree, wantedNames);

  const rows = (await parquetReadObjects({
    file,
    metadata,
    compressors, // unconditional — see module doc
    parsers: READ_PARSERS,
    columns: options.columns ? [...options.columns] : undefined,
    filter: options.filter,
  })) as Record<string, unknown>[];

  const vectors: Record<string, ReturnType<typeof buildVector>> = {};
  for (const col of columnSchemas) {
    const type = arrowTypeFor(col);
    const n = rows.length;
    const values: unknown[] = new Array(n);
    for (let i = 0; i < n; i++) {
      const v = rows[i]?.[col.name];
      // Normalize hyparquet's undefined-for-null quirk (upstream #168, seen on
      // optional nested fields; guarded here unconditionally, cheaply, for any
      // column) before it ever reaches Arrow vector construction.
      values[i] = v === undefined ? null : v;
    }
    vectors[col.name] = buildVector(values, type);
  }
  return Frame.fromArrow(new Table(vectors));
}

/** Read a local Parquet file into a Frame, with genuine columns/filter pushdown. */
export async function readParquet(path: string, options: ReadParquetOptions = {}): Promise<Frame> {
  const file = await asyncBufferFromFile(path);
  return readParquetFile(file, options);
}
