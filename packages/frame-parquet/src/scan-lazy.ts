/**
 * `scanParquetLazy("/events/*.parquet")` — genuinely lazy glob/partitioned
 * Parquet scans (issue #32, part 2/2 — the frame-parquet half; see
 * mallory-frame-arrow's plan.ts module doc, "Lazy sources (issue #32)", for
 * the `"lazySource"` PlanNode/`Frame.fromLazySource()`/`collectAsync()`
 * machinery this builds on).
 *
 * `scan.ts`'s `scanParquet` is EAGER: it reads and concatenates every
 * matched file's data immediately, before `.collect()` is ever called. This
 * module is the true-laziness follow-up scan.ts's own doc comment pointed
 * at: row DATA for every matched file is deferred until something actually
 * calls `.collectAsync()` (or a sync terminal accessor, which throws — see
 * frame-arrow's plan.ts), and honors the same automatic column-pruning
 * `wanted` set every other lazy/eager plan node gets — so
 * `scanParquetLazy(pattern).select('a', 'b').collectAsync()` fetches only
 * columns `a`/`b` from every matched file, with no need to separately pass
 * `{columns: ['a', 'b']}` the way `scanParquet` requires.
 *
 * ## What's still NOT lazy here, and why
 *
 * `Frame.fromLazySource(schema, read)` requires its `schema` argument
 * resolved EAGERLY, synchronously-available, at construction time (frame-
 * arrow's plan.ts: "so `planArrowSchema` stays fully synchronous"). A
 * Parquet file's schema can only be known by actually reading its footer
 * metadata — there's no way to get it without I/O. So `scanParquetLazy`
 * itself is async and DOES fetch every matched file's metadata (footer)
 * up front, before returning — same as `scanParquet`. What's deferred is
 * strictly the ROW-GROUP DATA read (the expensive part), not the metadata
 * fetch (typically a few KB to ~512 KiB, see read.ts's `initialFetchSize`
 * doc). This is not a partial/broken laziness — it matches how every other
 * lazy Parquet scanner (e.g. Polars' own `scan_parquet`) works: schema
 * resolution is cheap and eager, data reads are the thing worth deferring.
 *
 * A second, real consequence of needing the full schema up front: unlike
 * `readParquet`/`scanParquet` (which only type-map the requested `columns`,
 * so an unsupported column nobody asked for is never even inspected — see
 * schema.ts's own pruning-philosophy doc), `scanParquetLazy` must type-map
 * EVERY top-level column of every matched file to build that file's
 * `Schema` — so a file containing an unsupported column type (e.g. INT96)
 * makes `scanParquetLazy` throw `UnsupportedParquetTypeError` immediately,
 * even if that column is never in any later `.select()`. This is an
 * unavoidable consequence of `.schema`/`.columns` needing to describe every
 * field synchronously (the same guarantee eager Frames give), not a bug.
 *
 * There is also no whole-file skipping (e.g. Hive-style partition-column
 * pruning that skips an entire file based on its path/partition value) —
 * every matched file's schema is always resolved, and every matched file's
 * row data is always read once `.collectAsync()` runs (mirroring
 * `scanParquet`'s own all-files-always-read model). Only within-file
 * pruning (columns/filter) is deferred+automatic here.
 */
import { Field, Schema } from "apache-arrow";
import { asyncBufferFromFile, parquetMetadataAsync, parquetSchema, type AsyncBuffer } from "hyparquet";
import { Frame, type Wanted } from "mallory-frame-arrow";
import { READ_PARSERS, readParquetFile, type ReadParquetOptions } from "./read.ts";
import { globSortedParquetPaths } from "./scan.ts";
import { arrowTypeFor, mapNamedColumns, topLevelColumnNames } from "./schema.ts";

export type ScanParquetLazyOptions = ReadParquetOptions;

/**
 * Build a lazy `Frame` for ONE already-opened Parquet `AsyncBuffer` — the
 * `AsyncBuffer`-oriented core `scanParquetLazy` delegates to per matched
 * file, split out the same way `readParquetFile`/`readParquet` are (see
 * read.ts) so I/O-instrumented tests can call this directly with a
 * byte-counting wrapper, without going through the filesystem glob.
 */
export async function lazyParquetFrame(file: AsyncBuffer, options: ReadParquetOptions = {}): Promise<Frame> {
  const metadata = await parquetMetadataAsync(file, { parsers: READ_PARSERS, initialFetchSize: options.initialFetchSize });
  const tree = parquetSchema(metadata);
  const allNames = topLevelColumnNames(tree);
  // Every top-level column is type-mapped here (not just `options.columns`)
  // -- this module's own doc comment explains why that's unavoidable given
  // Frame.fromLazySource's eager-schema requirement.
  const columnSchemas = mapNamedColumns(tree, allNames);
  const fields = columnSchemas.map((c) => new Field(c.name, arrowTypeFor(c), c.nullable));
  const schema = new Schema(fields);

  return Frame.fromLazySource(schema, async (wanted: Wanted) => {
    const columns = wanted === "all" ? options.columns : intersectColumns(wanted, options.columns, allNames);
    const frame = await readParquetFile(file, { ...options, columns });
    return frame.toArrow();
  });
}

/**
 * Combine the plan-level pruning request (`wanted`, computed from
 * `.select()`/`.filter()` above this lazy source) with any `columns` the
 * caller explicitly passed to `scanParquetLazy` itself — both are real
 * projections and should intersect, not override each other.
 */
function intersectColumns(
  wanted: ReadonlySet<string>,
  explicitColumns: readonly string[] | undefined,
  allNames: readonly string[],
): string[] {
  const base = explicitColumns ?? allNames;
  return base.filter((name) => wanted.has(name));
}

/**
 * Lazy counterpart to `scanParquet` (scan.ts) — see this module's doc
 * comment for exactly what's deferred (row data) and what isn't (schema
 * resolution). Row data across every matched file is read only once
 * `.collectAsync()` (or an equivalent terminal await) is called, honoring
 * whatever columns/filter pruning the resulting `Frame`'s plan ends up
 * needing.
 */
export async function scanParquetLazy(pattern: string, options: ScanParquetLazyOptions = {}): Promise<Frame> {
  const paths = await globSortedParquetPaths(pattern);
  if (paths.length === 0) {
    throw new Error(`scanParquetLazy: no files matched pattern "${pattern}"`);
  }
  const frames = await Promise.all(
    paths.map(async (p) => lazyParquetFrame(await asyncBufferFromFile(p), options)),
  );
  return frames.length > 1 ? Frame.concat(frames) : (frames[0] as Frame);
}
