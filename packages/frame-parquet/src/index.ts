/**
 * mallory-frame-parquet — a Parquet read/write bridge into
 * mallory-frame-arrow's Frame, built on hyparquet (issue #20; see
 * docs/spikes/parquet-bakeoff.md for why hyparquet over parquet-wasm).
 *
 * ## API shape — a deliberate deviation from the issue's literal phrasing
 *
 * Issue #20 sketches `Frame.readParquet(path, {columns, filter})` and
 * `frame.writeParquet(path, options)` — static/instance methods on Frame
 * itself. frame-arrow's `Frame` (packages/frame-arrow/src/frame.ts) has a
 * PRIVATE constructor and a closed set of public entry points
 * (`Frame.fromArrow`, `Frame.fromIPC`, `Frame.concat`); it is not designed
 * to be extended with new static/instance members from an external package,
 * and monkeypatching `Frame.readParquet = ...` from here would be exactly
 * that kind of extension — fragile, untyped without hand-written module
 * augmentation, and a global mutation on import. So this package instead
 * exports top-level functions, matching the free-function convention the
 * issue itself already uses for `scanParquet`:
 *
 * - `readParquet(path, options)` -> `Promise<Frame>`, built via the public
 *   `Frame.fromArrow(table)` entry point.
 * - `scanParquet(pattern, options)` -> `Promise<Frame>` (glob scans; see
 *   scan.ts's module doc for the eager-vs-lazy decision).
 * - `writeParquet(frame, path, options)` -> `Promise<void>`.
 *
 * `readParquetFile`/`writeParquetBuffer` are also exported: lower-level,
 * `AsyncBuffer`/`Uint8Array`-oriented variants useful for browser/edge
 * contexts (no filesystem) and for I/O-instrumented testing (see
 * test/pushdown.test.ts).
 */
export { readParquet, readParquetFile, type ReadParquetOptions } from "./read.ts";
export { scanParquet, type ScanParquetOptions } from "./scan.ts";
export { writeParquet, writeParquetBuffer, type WriteCodec, type WriteParquetOptions } from "./write.ts";
export { mapLeafElement, mapTopLevelColumns, UnsupportedParquetTypeError } from "./schema.ts";
export type { ParquetColumnSchema, ParquetColumnType } from "./schema.ts";
