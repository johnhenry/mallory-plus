/**
 * Frame -> Parquet write path (issue #20).
 *
 * ## The zstd footgun — how this package resolves it
 *
 * docs/spikes/parquet-bakeoff.md §4 verified that hyparquet-writer's
 * `codec: 'ZSTD'` WITHOUT a registered compressor does not throw — it
 * silently writes bytes labeled ZSTD that are actually uncompressed
 * (`compressors[codec]?.(pageBytes) ?? pageBytes`, hyparquet-writer's
 * src/datapage.js and src/dictionary.js), and pyarrow then fails to read it
 * ("Unknown frame descriptor"). `hyparquet-compressors` cannot help on
 * write — it bundles `fzstd`, decompression-only.
 *
 * This package resolves it with option (a) from the task: a REAL zstd
 * compressor, wired in via `@hpcc-js/wasm-zstd` (see zstd.ts's module doc
 * for why that package specifically, over `@bokuweb/zstd-wasm` /
 * `@foxglove/wasm-zstd`). `compression: "zstd"` therefore actually works —
 * verified against `pyarrow.parquet.read_table` in test/write.test.ts, not
 * just round-tripped through frame-parquet's own reader (self-consistency
 * isn't proof of a valid Parquet file).
 *
 * As defense in depth (in case a future refactor ever removes the zstd
 * wiring, or a caller somehow reaches this code with an unrecognized codec
 * string despite the TS type), `writeCodec` is validated at this module's
 * boundary against an explicit allow-list — "snappy" | "zstd" |
 * "uncompressed" — and anything else throws a clear error rather than being
 * passed through to hyparquet-writer's `codec` option unchecked.
 *
 * ## Dtype coverage
 *
 * Same v1 scope as the read path (see schema.ts's module doc): bool, all
 * int/uint widths, int64, float32/64, utf8, timestamp_ms/us. "dictionary"
 * columns are written as plain STRING data (Parquet's own dictionary
 * encoding is an internal, automatic optimization hyparquet-writer applies
 * to STRING columns on its own when it shrinks the encoded size — there's
 * no separate "write this as a logical dictionary type" concept in
 * Parquet, so no special-casing is needed beyond emitting plain strings).
 * "list"/"struct" columns throw a clear, named error — nested-type Parquet
 * write is a real gap, tracked as a follow-up issue, not silently dropped.
 */
import { writeFile } from "node:fs/promises";
import { parquetWriteBuffer, schemaFromColumnData } from "hyparquet-writer";
import type { ColumnSource, SchemaElement } from "hyparquet-writer";
import type { DType, FieldDescriptor, Frame } from "mallory-frame-arrow";
import { type Compressor, zstdCompressor } from "./zstd.ts";

/** Write-side compressor map — see zstd.ts's doc comment for why this is
 * redeclared locally rather than imported from hyparquet-writer. */
type Compressors = Partial<Record<"UNCOMPRESSED" | "SNAPPY" | "GZIP" | "LZO" | "BROTLI" | "LZ4" | "ZSTD" | "LZ4_RAW", Compressor>>;

export type WriteCodec = "snappy" | "zstd" | "uncompressed";
const SUPPORTED_CODECS: readonly WriteCodec[] = ["snappy", "zstd", "uncompressed"];

export interface WriteParquetOptions {
  /** Compression codec. Default "snappy" (universally readable by pyarrow with zero setup).
   * "zstd" is real (see module doc); "uncompressed" writes raw pages. Any other value throws. */
  readonly compression?: WriteCodec;
  /** Target rows per row group. Passed straight through to hyparquet-writer's `rowGroupSize`. */
  readonly rowGroupSize?: number;
}

const HYPARQUET_CODEC: Record<WriteCodec, "SNAPPY" | "ZSTD" | "UNCOMPRESSED"> = {
  snappy: "SNAPPY",
  zstd: "ZSTD",
  uncompressed: "UNCOMPRESSED",
};

interface FieldPlan {
  readonly column: ColumnSource;
  readonly override?: SchemaElement;
}

function planField(field: FieldDescriptor, values: unknown[]): FieldPlan {
  const dtype: DType = field.dtype;
  const nullable = field.nullable;
  const name = field.name;

  switch (dtype) {
    case "bool":
      return { column: { name, data: values, type: "BOOLEAN", nullable } };
    case "int32":
      return { column: { name, data: values, type: "INT32", nullable } };
    case "int64":
      return { column: { name, data: values, type: "INT64", nullable } };
    case "float32":
      return { column: { name, data: values, type: "FLOAT", nullable } };
    case "float64":
      return { column: { name, data: values, type: "DOUBLE", nullable } };
    case "utf8":
    case "dictionary":
      return { column: { name, data: values, type: "STRING", nullable } };
    case "timestamp_ms":
      return { column: { name, data: values, type: "TIMESTAMP", nullable } };
    case "int8":
    case "int16":
    case "uint8":
    case "uint16":
    case "uint32": {
      const convertedType = {
        int8: "INT_8",
        int16: "INT_16",
        uint8: "UINT_8",
        uint16: "UINT_16",
        uint32: "UINT_32",
      }[dtype] as "INT_8" | "INT_16" | "UINT_8" | "UINT_16" | "UINT_32";
      return {
        column: { name, data: values },
        override: {
          name,
          type: "INT32",
          converted_type: convertedType,
          repetition_type: nullable ? "OPTIONAL" : "REQUIRED",
        },
      };
    }
    case "timestamp_us":
      return {
        column: { name, data: values },
        override: {
          name,
          type: "INT64",
          converted_type: "TIMESTAMP_MICROS",
          repetition_type: nullable ? "OPTIONAL" : "REQUIRED",
        },
      };
    case "list":
    case "struct":
      throw new Error(
        `writeParquet: column "${name}" has dtype "${dtype}" — nested Parquet types (list/struct) are not ` +
          `supported by mallory-frame-parquet v1's write path. Drop or flatten this column before writing.`,
      );
    default: {
      const exhaustive: never = dtype;
      throw new Error(`writeParquet: unhandled dtype "${exhaustive as string}"`);
    }
  }
}

function validateCodec(compression: WriteCodec | undefined): WriteCodec {
  const codec = compression ?? "snappy";
  if (!SUPPORTED_CODECS.includes(codec)) {
    throw new Error(
      `writeParquet: unsupported compression "${codec as string}" — only ${SUPPORTED_CODECS.join(", ")} are ` +
        `supported in v1. (zstd write requires a real WASM zstd compressor, which this package wires in; there is ` +
        `no free-lunch zstd encoder from hyparquet-writer/hyparquet-compressors alone — see write.ts's module doc.)`,
    );
  }
  return codec;
}

/** Serialize a Frame to Parquet bytes. */
export async function writeParquetBuffer(frame: Frame, options: WriteParquetOptions = {}): Promise<Uint8Array> {
  const codec = validateCodec(options.compression);

  const columnData: ColumnSource[] = [];
  const schemaOverrides: Record<string, SchemaElement> = {};
  for (const field of frame.schema) {
    const series = frame.getSeries(field.name);
    const values = series.toArray();
    const { column, override } = planField(field, values);
    columnData.push(column);
    if (override) schemaOverrides[field.name] = override;
  }

  let compressors: Compressors | undefined;
  if (codec === "zstd") {
    compressors = { ZSTD: await zstdCompressor() };
  }

  // hyparquet-writer's parquetWrite() throws "cannot provide both schema and
  // columnData type" if any columnData entry still carries a `type`/`nullable`
  // once an explicit `schema` is passed (src/write.js) — which happens for
  // every write that has at least one int8/int16/uint8/uint16/uint32/
  // timestamp_us column (the only dtypes that need a schemaOverrides entry).
  // schemaFromColumnData({ columnData, schemaOverrides }) already computed a
  // full SchemaElement for every column (override or auto-detected-from-type),
  // so once we go explicit we strip type/nullable from ALL columns and rely
  // entirely on the computed `schema`.
  let schema: SchemaElement[] | undefined;
  let writeColumnData = columnData;
  if (Object.keys(schemaOverrides).length > 0) {
    schema = schemaFromColumnData({ columnData, schemaOverrides });
    writeColumnData = columnData.map((c) => ({ name: c.name, data: c.data }));
  }

  const buffer = parquetWriteBuffer({
    columnData: writeColumnData,
    schema,
    codec: HYPARQUET_CODEC[codec],
    compressors,
    rowGroupSize: options.rowGroupSize,
  });
  return new Uint8Array(buffer);
}

/** Write a Frame to a local Parquet file. */
export async function writeParquet(frame: Frame, path: string, options: WriteParquetOptions = {}): Promise<void> {
  const bytes = await writeParquetBuffer(frame, options);
  await writeFile(path, bytes);
}
