/**
 * Frame -> Parquet write path (issue #20; list/struct support added in #30).
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
 *
 * "list"/"struct" columns (issue #30) map to a standard Parquet LIST
 * (3-level list/element convention) or a flat STRUCT group. Nulls at every
 * level (null list vs. empty list vs. a null element; null struct vs. a
 * struct with a null field) fall straight out of hyparquet-writer's own
 * dremel encoder (`encodeNestedValues`) given a correctly-shaped explicit
 * schema and `Series.toArray()`'s already-null-preserving plain nested JS
 * values — this module doesn't reassemble or reinterpret nesting itself,
 * only builds the schema. Deeply nested types (list<struct>, struct<list>)
 * are NOT supported — matching frame-arrow's own limit (dtype.ts), a Frame
 * can't even produce such a column's `.schema` in the first place, so the
 * guards below are defense-in-depth rather than a reachable path through the
 * public `Frame` API.
 *
 * hyparquet-writer's `schemaOverrides` mechanism (used below for
 * int8/16/uint8/16/32/timestamp_us) explicitly rejects nested
 * (`num_children`-bearing) overrides ("schema override does not support
 * nested types", hyparquet-writer's src/schema.js) — so once any column
 * needs a LIST/STRUCT subtree, the whole explicit `schema` array is built by
 * hand in {@link writeParquetBuffer} instead of going through
 * `schemaFromColumnData`'s override slot for those columns (plain
 * auto-typed columns still reuse `schemaFromColumnData` per-column, so their
 * type inference isn't reimplemented here).
 */
import { writeFile } from "node:fs/promises";
import type { Field, List } from "apache-arrow";
import { parquetWriteBuffer, schemaFromColumnData } from "hyparquet-writer";
import type { ColumnSource, SchemaElement } from "hyparquet-writer";
import type { DType, FieldDescriptor, Frame, Series } from "@johnhenry/math-plus-frame-arrow";
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
  /** For "list"/"struct" columns: the whole flattened subtree (column node
   * itself plus its descendants, preorder) — hyparquet-writer's `override`
   * slot (above) can't carry `num_children`, so these are spliced into the
   * final explicit `schema` array by hand in {@link writeParquetBuffer}. */
  readonly overrideSubtree?: readonly SchemaElement[];
}

/** Build the flat (non-nested) SchemaElement for one scalar DType — shared by
 * the top-level scalar-override cases in {@link planField} and by the
 * list-item/struct-field leaves in {@link planListSchema}/{@link planStructSchema},
 * which need the exact same per-leaf type mapping one level down. */
function scalarSchemaElement(name: string, dtype: DType, nullable: boolean): SchemaElement {
  const repetition_type = nullable ? "OPTIONAL" : "REQUIRED";
  switch (dtype) {
    case "bool":
      return { name, type: "BOOLEAN", repetition_type };
    case "int32":
      return { name, type: "INT32", repetition_type };
    case "int64":
      return { name, type: "INT64", repetition_type };
    case "float32":
      return { name, type: "FLOAT", repetition_type };
    case "float64":
      return { name, type: "DOUBLE", repetition_type };
    case "utf8":
    case "dictionary":
      return { name, type: "BYTE_ARRAY", converted_type: "UTF8", repetition_type };
    case "timestamp_ms":
      return { name, type: "INT64", converted_type: "TIMESTAMP_MILLIS", repetition_type };
    case "timestamp_us":
      return { name, type: "INT64", converted_type: "TIMESTAMP_MICROS", repetition_type };
    case "int8":
      return { name, type: "INT32", converted_type: "INT_8", repetition_type };
    case "int16":
      return { name, type: "INT32", converted_type: "INT_16", repetition_type };
    case "uint8":
      return { name, type: "INT32", converted_type: "UINT_8", repetition_type };
    case "uint16":
      return { name, type: "INT32", converted_type: "UINT_16", repetition_type };
    case "uint32":
      return { name, type: "INT32", converted_type: "UINT_32", repetition_type };
    case "list":
    case "struct":
      // Unreachable through the public Frame API: frame-arrow's own
      // describeField() (dtype.ts) already throws UnsupportedTypeError for
      // list<struct>/list<list>/struct-with-a-nested-field before `frame.schema`
      // even returns such a field — this is defense-in-depth, not a real path.
      throw new Error(
        `writeParquet: a list item or struct field cannot itself have dtype "${dtype}" — deeply nested Parquet ` +
          `types (list<struct>, list<list>, struct<...list/struct...>) are not supported by @johnhenry/math-plus-frame-parquet ` +
          `v1's write path (frame-arrow itself only supports single-level list/struct, see its dtype.ts).`,
      );
    default: {
      const exhaustive: never = dtype;
      throw new Error(`writeParquet: unhandled scalar dtype "${exhaustive as string}"`);
    }
  }
}

/** Standard Parquet 3-level LIST convention: `optional|required group <name>
 * (LIST) { repeated group list { optional|required <item> element; } }` —
 * same shape hyparquet-writer's own VARIANT array-shredding builds
 * (src/schema.js's `buildVariantTypedValue`) and what pyarrow/hyparquet
 * recognize as list-like on read. Item nullability isn't carried on
 * frame-arrow's `FieldDescriptor` (only `itemDType` is — see dtype.ts), so
 * it's read off the Series' own Arrow `List` type instead. */
function planListSchema(field: FieldDescriptor, series: Series): readonly SchemaElement[] {
  const name = field.name;
  const itemDType = field.itemDType;
  if (!itemDType) throw new Error(`writeParquet: list column "${name}" is missing itemDType`);
  const listType = series.toVector().type as List;
  const itemField = listType.children[0] as Field;
  const elementSchema = scalarSchemaElement("element", itemDType, itemField.nullable);
  return [
    { name, converted_type: "LIST", repetition_type: field.nullable ? "OPTIONAL" : "REQUIRED", num_children: 1 },
    { name: "list", repetition_type: "REPEATED", num_children: 1 },
    elementSchema,
  ];
}

/** A flat STRUCT group: `optional|required group <name> { <fields...> }` —
 * frame-arrow's `FieldDescriptor.structFields` already carries each field's
 * own name/dtype/nullable (dtype.ts's `describeField` recursion), so no
 * extra Arrow-type inspection is needed here (unlike the list-item case). */
function planStructSchema(field: FieldDescriptor): readonly SchemaElement[] {
  const name = field.name;
  const structFields = field.structFields;
  if (!structFields) throw new Error(`writeParquet: struct column "${name}" is missing structFields`);
  const children = structFields.map((sf) => scalarSchemaElement(sf.name, sf.dtype, sf.nullable));
  return [{ name, repetition_type: field.nullable ? "OPTIONAL" : "REQUIRED", num_children: children.length }, ...children];
}

function planField(field: FieldDescriptor, series: Series): FieldPlan {
  const dtype: DType = field.dtype;
  const nullable = field.nullable;
  const name = field.name;
  const values = series.toArray();

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
    case "uint32":
    case "timestamp_us":
      return { column: { name, data: values }, override: scalarSchemaElement(name, dtype, nullable) };
    case "list":
      return { column: { name, data: values }, overrideSubtree: planListSchema(field, series) };
    case "struct":
      return { column: { name, data: values }, overrideSubtree: planStructSchema(field) };
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
  const nestedSubtrees: Record<string, readonly SchemaElement[]> = {};
  for (const field of frame.schema) {
    const series = frame.getSeries(field.name);
    const { column, override, overrideSubtree } = planField(field, series);
    columnData.push(column);
    if (override) schemaOverrides[field.name] = override;
    if (overrideSubtree) nestedSubtrees[field.name] = overrideSubtree;
  }

  let compressors: Compressors | undefined;
  if (codec === "zstd") {
    compressors = { ZSTD: await zstdCompressor() };
  }

  // hyparquet-writer's parquetWrite() throws "cannot provide both schema and
  // columnData type" if any columnData entry still carries a `type`/`nullable`
  // once an explicit `schema` is passed (src/write.js) — which happens for
  // every write that has at least one int8/int16/uint8/uint16/uint32/
  // timestamp_us/list/struct column (the dtypes that need an explicit schema
  // entry). schemaFromColumnData({ columnData, schemaOverrides }) already
  // computes a full SchemaElement for every column (override or
  // auto-detected-from-type), so once we go explicit we strip type/nullable
  // from ALL columns and rely entirely on the computed `schema`.
  let schema: SchemaElement[] | undefined;
  let writeColumnData = columnData;
  const hasNested = Object.keys(nestedSubtrees).length > 0;
  if (hasNested) {
    // schemaFromColumnData's schemaOverrides slot rejects nested overrides
    // outright (see this module's doc comment) — build the whole explicit
    // schema by hand: one flat element (scalar override) or subtree
    // (list/struct) per column, in original column order, or — for plain
    // columns that need neither — reuse schemaFromColumnData on that single
    // column alone so its existing type-inference logic (from the column's
    // own already-set `type`/`nullable`, per planField's scalar cases) isn't
    // duplicated here.
    const parts: SchemaElement[] = [];
    for (const col of columnData) {
      const subtree = nestedSubtrees[col.name];
      const override = schemaOverrides[col.name];
      if (subtree) {
        parts.push(...subtree);
      } else if (override) {
        parts.push(override);
      } else {
        const [, autoElement] = schemaFromColumnData({ columnData: [col] });
        parts.push(autoElement as SchemaElement);
      }
    }
    schema = [{ name: "root", num_children: columnData.length }, ...parts];
    writeColumnData = columnData.map((c) => ({ name: c.name, data: c.data }));
  } else if (Object.keys(schemaOverrides).length > 0) {
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
