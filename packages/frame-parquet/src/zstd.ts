/**
 * Real zstd COMPRESSION for the write path.
 *
 * The footgun (docs/spikes/parquet-bakeoff.md §4, verified): hyparquet-writer
 * accepts `codec: 'ZSTD'` and silently writes a CORRUPT file if no compressor
 * is registered for that codec — `compressors[codec]?.(bytes) ?? bytes`
 * (src/datapage.js, src/dictionary.js in hyparquet-writer) falls through to
 * storing the UNCOMPRESSED bytes mislabeled as ZSTD. pyarrow then fails with
 * "Unknown frame descriptor" trying to read it. `hyparquet-compressors`
 * bundles `fzstd`, which is DECOMPRESS-ONLY (no encoder) — so there is no
 * free lunch from the packages already in play; frame-parquet must supply a
 * real compressor itself.
 *
 * `@hpcc-js/wasm-zstd` was chosen (see write.ts's module doc for the
 * comparison against `@bokuweb/zstd-wasm`/`@foxglove/wasm-zstd`): actively
 * maintained (part of the hpcc-systems/hpcc-js-wasm monorepo, published
 * within days of this package being written), Apache-2.0, zero
 * dependencies, and a clean `Zstd.load()` (async, once) -> `.compress(data)`
 * (sync) API that drops straight into hyparquet-writer's
 * `Compressor = (input: Uint8Array) => Uint8Array` shape once the WASM
 * module has loaded. It compiles the real libzstd C library to WASM, so its
 * output is standard zstd — verified readable by `pyarrow.parquet.read_table`
 * in test/write.test.ts, not just self-consistent with frame-parquet's own
 * reader.
 */
import { Zstd } from "@hpcc-js/wasm-zstd";

/**
 * hyparquet-writer's `Compressor` type (`(input: Uint8Array) => Uint8Array`)
 * isn't part of its public `.d.ts` export surface (only `SchemaElement`,
 * `ColumnSource`, `ParquetWriteOptions`, etc. are — checked against
 * hyparquet-writer 0.16.6's `types/index.d.ts`), so it's redeclared locally
 * here to match the documented runtime shape (README's "Advanced Usage" /
 * `compressors: { SNAPPY: snappyCompresss }` example) rather than reaching
 * into an unexported subpath type.
 */
export type Compressor = (input: Uint8Array) => Uint8Array;

let loaded: Promise<Zstd> | undefined;

function loadZstd(): Promise<Zstd> {
  loaded ??= Zstd.load();
  return loaded;
}

/**
 * Resolve a synchronous zstd {@link Compressor} for hyparquet-writer's
 * `compressors: { ZSTD: ... }` option. The WASM module load is async and
 * happens once (memoized); the returned function itself is synchronous,
 * matching what hyparquet-writer requires.
 */
export async function zstdCompressor(): Promise<Compressor> {
  const zstd = await loadZstd();
  return (input: Uint8Array): Uint8Array => zstd.compress(input);
}
