/**
 * Parquet SchemaElement -> mallory-frame-arrow DType mapping.
 *
 * hyparquet decodes column *values* to plain JS (BigInt for i64, plain
 * strings for BYTE_ARRAY/UTF8 and — see the module doc in read.ts —
 * dictionary-encoded columns too), but it never hands back a "this is an
 * int8 vs int32" width tag: a decoded INT32 value is just a JS `number`.
 * To build a correctly-typed Arrow Table (and therefore a Frame whose
 * `.schema` matches what a human would expect from the source file) this
 * module inspects the *metadata* (`ParquetType` + `converted_type`/
 * `logical_type`) directly, once per column, via `parquetSchema()`'s
 * SchemaTree.
 *
 * v1 scope, deliberately narrow (mirrors frame-arrow's own dtype.ts, which
 * throws loudly on anything past its declared v1 set rather than silently
 * mishandling it — docs/spikes/parquet-bakeoff.md's "Open risks" #1 flags
 * hyparquet's format coverage as broad-but-not-exhaustive, so this boundary
 * matters doubly here):
 *
 * - Supported leaf types: BOOLEAN, INT32 (+INT_8/INT_16/UINT_8/UINT_16/
 *   UINT_32 converted/logical annotations), INT64 (+TIMESTAMP_MILLIS/
 *   TIMESTAMP_MICROS), FLOAT, DOUBLE, BYTE_ARRAY decoded as UTF8 (hyparquet's
 *   default `utf8: true` read option, which this package always sets).
 * - Supported group types (issue #30, follow-up to #20): a Parquet LIST
 *   column following the standard 3-level list/element repetition
 *   convention (`optional group name (LIST) { repeated group list {
 *   optional|required <leaf> element; } }` — legacy `converted_type: 'LIST'`
 *   and the newer `logical_type: { type: 'LIST' }` annotation are both
 *   recognized, matching what real writers emit) maps to frame-arrow's
 *   `list<T>` DType, and a flat STRUCT group (any group that isn't LIST/MAP,
 *   whose fields are themselves all leaves) maps to `struct<...>`. Nulls at
 *   every level — null list vs. empty list vs. a null element inside a
 *   non-null list; null struct vs. a struct with a null field — round-trip
 *   exactly, since hyparquet has already done the dremel repetition/
 *   definition-level reconstruction into plain nested JS arrays/objects by
 *   the time this package sees a row (see read.ts's module doc); this module
 *   only has to get the *type* right, not reassemble the nesting itself.
 *   Deeply nested types (list<struct>, struct<list>, list<list>) are NOT
 *   supported — frame-arrow's own Frame doesn't support them either (see
 *   its dtype.ts), so there's nothing to gain by accepting them here only to
 *   have Frame.fromArrow() reject them later; they throw
 *   {@link UnsupportedParquetTypeError} at this schema-mapping step instead,
 *   naming the column, same as every other unsupported type in this file.
 * - Deferred, throws {@link UnsupportedParquetTypeError}: INT96, INT64 with
 *   NANOS unit or unsigned width, DATE/TIME/DECIMAL/JSON/BSON/ENUM/UUID/
 *   FLOAT16/GEOMETRY/GEOGRAPHY/VARIANT converted or logical types, MAP
 *   columns (legacy `converted_type: 'MAP'`/`'MAP_KEY_VALUE'` or
 *   `logical_type: { type: 'MAP' }`) — a real gap, not attempted here since
 *   frame-arrow has no map DType to map it to — and the deeply-nested
 *   LIST/STRUCT combinations described above.
 *
 * Dictionary-encoded parquet columns are NOT surfaced as frame-arrow's
 * `"dictionary"` DType — hyparquet materializes them to plain strings with
 * no signal in the schema that they were dictionary-encoded on disk (it's a
 * page-level *encoding*, not a schema-level property), so they come back as
 * plain `"utf8"` Frame columns. Documented v1 simplification, matches the
 * spike's own recommendation.
 */
import {
  Bool,
  type DataType,
  Field,
  Float32,
  Float64,
  Int16,
  Int32,
  Int64,
  Int8,
  List,
  Struct,
  TimestampMicrosecond,
  TimestampMillisecond,
  Uint16,
  Uint32,
  Uint8,
  Utf8,
} from "apache-arrow";
import type { SchemaElement, SchemaTree } from "hyparquet";
import type { DType } from "mallory-frame-arrow";

export class UnsupportedParquetTypeError extends TypeError {
  constructor(columnName: string, detail: string) {
    super(
      `column "${columnName}" has a Parquet type mallory-frame-parquet v1 does not support: ${detail}. ` +
        `Cast or drop this column (e.g. in pyarrow) before reading it, or select around it.`,
    );
    this.name = "UnsupportedParquetTypeError";
  }
}

export interface ParquetColumnType {
  readonly dtype: DType;
  readonly timezone?: string | null;
  /** For dtype "list"/"struct" only: the exact Arrow `List`/`Struct` DataType
   * computed during schema mapping. Unlike the flat scalar dtypes, list/struct
   * can't be rebuilt generically from a dtype tag alone (the item type/struct
   * fields matter) — {@link arrowTypeFor} just returns this precomputed value
   * rather than trying to reconstruct it. */
  readonly nestedArrowType?: DataType;
}

function mapInt32(element: SchemaElement): ParquetColumnType {
  const { converted_type, logical_type, name } = element;
  if (converted_type === "INT_8") return { dtype: "int8" };
  if (converted_type === "INT_16") return { dtype: "int16" };
  if (converted_type === "UINT_8") return { dtype: "uint8" };
  if (converted_type === "UINT_16") return { dtype: "uint16" };
  if (converted_type === "UINT_32") return { dtype: "uint32" };
  if (logical_type?.type === "INTEGER") {
    const { bitWidth, isSigned } = logical_type;
    if (bitWidth === 8) return { dtype: isSigned ? "int8" : "uint8" };
    if (bitWidth === 16) return { dtype: isSigned ? "int16" : "uint16" };
    if (bitWidth === 32) return { dtype: isSigned ? "int32" : "uint32" };
    throw new UnsupportedParquetTypeError(name, `INT32 with logical INTEGER(bitWidth=${bitWidth})`);
  }
  if (converted_type !== undefined && converted_type !== "INT_32") {
    throw new UnsupportedParquetTypeError(name, `INT32 with converted_type ${converted_type} (DATE/DECIMAL deferred)`);
  }
  return { dtype: "int32" };
}

function mapInt64(element: SchemaElement): ParquetColumnType {
  const { converted_type, logical_type, name } = element;
  if (converted_type === "TIMESTAMP_MILLIS") return { dtype: "timestamp_ms", timezone: null };
  if (converted_type === "TIMESTAMP_MICROS") return { dtype: "timestamp_us", timezone: null };
  if (logical_type?.type === "TIMESTAMP") {
    const timezone = logical_type.isAdjustedToUTC ? "UTC" : null;
    if (logical_type.unit === "MILLIS") return { dtype: "timestamp_ms", timezone };
    if (logical_type.unit === "MICROS") return { dtype: "timestamp_us", timezone };
    throw new UnsupportedParquetTypeError(name, `INT64 TIMESTAMP with unit ${logical_type.unit} (NANOS deferred)`);
  }
  if (logical_type?.type === "INTEGER") {
    if (logical_type.bitWidth === 64 && !logical_type.isSigned) {
      throw new UnsupportedParquetTypeError(name, "INT64 UINT_64 (unsigned int64 has no frame-arrow dtype)");
    }
  }
  if (converted_type !== undefined && converted_type !== "INT_64") {
    throw new UnsupportedParquetTypeError(name, `INT64 with converted_type ${converted_type}`);
  }
  return { dtype: "int64" };
}

function mapByteArray(element: SchemaElement): ParquetColumnType {
  const { converted_type, logical_type, name } = element;
  if (converted_type === undefined && logical_type === undefined) return { dtype: "utf8" };
  if (converted_type === "UTF8" || logical_type?.type === "STRING") return { dtype: "utf8" };
  throw new UnsupportedParquetTypeError(
    name,
    `BYTE_ARRAY with converted_type=${converted_type ?? "none"}/logical_type=${logical_type?.type ?? "none"} (JSON/BSON/ENUM/DECIMAL/GEOMETRY/GEOGRAPHY/UUID/VARIANT deferred)`,
  );
}

/** Map one leaf (non-group) SchemaElement to a frame-arrow DType, or throw. */
export function mapLeafElement(element: SchemaElement): ParquetColumnType {
  switch (element.type) {
    case "BOOLEAN":
      return { dtype: "bool" };
    case "INT32":
      return mapInt32(element);
    case "INT64":
      return mapInt64(element);
    case "FLOAT":
      return { dtype: "float32" };
    case "DOUBLE":
      return { dtype: "float64" };
    case "BYTE_ARRAY":
      return mapByteArray(element);
    default:
      throw new UnsupportedParquetTypeError(
        element.name,
        `physical type ${element.type ?? "(group)"} (INT96/FIXED_LEN_BYTE_ARRAY deferred)`,
      );
  }
}

/**
 * A Parquet group node is LIST-like when it follows the standard 3-level
 * convention: exactly one REPEATED child ("list"), which itself has exactly
 * one child ("element", the actual item). Only the SHAPE is checked, not the
 * middle/leaf names — the convention doesn't mandate "list"/"element"
 * specifically, and real writers vary (mirrors hyparquet's own internal
 * `isListLike`, which this package deliberately does not import since it
 * lives under hyparquet's `src/` rather than its public `index.js` surface).
 */
function isListGroup(child: SchemaTree): boolean {
  return child.element.converted_type === "LIST" || child.element.logical_type?.type === "LIST";
}

/** Legacy `MAP_KEY_VALUE` shows up on the *middle* group in some old files,
 * but the outer group (what this function is called with) always carries
 * `converted_type: 'MAP'` and/or `logical_type: { type: 'MAP' }` — checking
 * both covers writers that only emit one. */
function isMapGroup(child: SchemaTree): boolean {
  return child.element.converted_type === "MAP" || child.element.logical_type?.type === "MAP";
}

/** Map a Parquet LIST group (3-level convention) to frame-arrow's `list<T>`. */
function mapListGroup(child: SchemaTree): ParquetColumnType {
  const name = child.element.name;
  const middle = child.children[0];
  if (child.children.length !== 1 || !middle || middle.element.repetition_type !== "REPEATED" || middle.children.length !== 1) {
    throw new UnsupportedParquetTypeError(name, "a LIST group not in the standard 3-level list/element repetition convention");
  }
  const elementNode = middle.children[0] as SchemaTree;
  if (elementNode.children.length > 0) {
    throw new UnsupportedParquetTypeError(
      name,
      "a list of a nested type (list<struct>/list<list>) — deeply nested Parquet types are not supported " +
        "(frame-arrow itself only supports single-level list/struct, see its dtype.ts)",
    );
  }
  const itemNullable = elementNode.element.repetition_type !== "REQUIRED";
  const item = mapLeafElement(elementNode.element);
  const nestedArrowType = new List(new Field("item", arrowTypeFor(item), itemNullable));
  return { dtype: "list", nestedArrowType };
}

/** Map a flat Parquet STRUCT group (every field a leaf) to frame-arrow's `struct<...>`. */
function mapStructGroup(child: SchemaTree): ParquetColumnType {
  const name = child.element.name;
  const fields = child.children.map((field) => {
    if (field.children.length > 0) {
      throw new UnsupportedParquetTypeError(
        name,
        `struct field "${field.element.name}" is itself a nested group — deeply nested Parquet types are not ` +
          "supported (frame-arrow itself only supports single-level list/struct, see its dtype.ts)",
      );
    }
    const leaf = mapLeafElement(field.element);
    const nullable = field.element.repetition_type !== "REQUIRED";
    return new Field(field.element.name, arrowTypeFor(leaf), nullable);
  });
  return { dtype: "struct", nestedArrowType: new Struct(fields) };
}

/** Map a group (non-leaf) SchemaElement to frame-arrow's `list`/`struct` DType, or throw. */
function mapGroupElement(child: SchemaTree): ParquetColumnType {
  if (isMapGroup(child)) {
    throw new UnsupportedParquetTypeError(
      child.element.name,
      "a MAP group — Parquet MAP columns are not supported in mallory-frame-parquet v1 (LIST/STRUCT are; " +
        "frame-arrow has no map DType to map a MAP column to)",
    );
  }
  if (isListGroup(child)) return mapListGroup(child);
  return mapStructGroup(child);
}

export interface ParquetColumnSchema extends ParquetColumnType {
  readonly name: string;
  readonly nullable: boolean;
}

/** Top-level column names only — never inspects a column's type, so (unlike
 * {@link mapNamedColumns}) it never throws on an unsupported/nested column
 * nobody asked for. Used by read.ts to validate a `columns` option and to
 * default to "all columns" without forcing dtype-mapping (and therefore
 * possibly throwing) on columns that will end up pruned away anyway. */
export function topLevelColumnNames(tree: SchemaTree): string[] {
  return tree.children.map((child) => child.element.name);
}

/**
 * Map exactly the named top-level columns to frame-arrow-typed column
 * descriptors, in the given order. Throws {@link UnsupportedParquetTypeError}
 * for any of THOSE columns that's a MAP group, a deeply-nested LIST/STRUCT
 * combination, or an otherwise-unsupported leaf type — naming it — but never
 * inspects, and therefore never throws on, a column that isn't in `names`.
 * This mirrors frame-arrow's own pruning philosophy
 * (packages/frame-arrow/src/plan.ts's module doc: "a column that's pruned
 * away is never read, never type-checked, and never throws") at the
 * schema-mapping step, before any row is even read. LIST (standard 3-level
 * convention) and flat STRUCT groups map to frame-arrow's `list<T>`/
 * `struct<...>` — see this module's doc comment.
 */
export function mapNamedColumns(tree: SchemaTree, names: readonly string[]): ParquetColumnSchema[] {
  const byName = new Map(tree.children.map((child) => [child.element.name, child] as const));
  return names.map((name) => {
    const child = byName.get(name);
    if (!child) throw new Error(`no such column "${name}"`);
    const nullable = child.element.repetition_type !== "REQUIRED";
    if (child.children.length > 0) {
      const { dtype, nestedArrowType } = mapGroupElement(child);
      return { name, dtype, nullable, nestedArrowType };
    }
    const { dtype, timezone } = mapLeafElement(child.element);
    return { name, dtype, timezone, nullable };
  });
}

/**
 * Flatten every top-level field into frame-arrow-typed column descriptors —
 * a convenience wrapper around {@link mapNamedColumns} for full-schema
 * introspection. Unlike the pruned read path in read.ts, this DOES inspect
 * (and can throw on) every column, since "all columns" is exactly what was
 * asked for.
 */
export function mapTopLevelColumns(tree: SchemaTree): ParquetColumnSchema[] {
  return mapNamedColumns(tree, topLevelColumnNames(tree));
}

/**
 * Build the apache-arrow DataType for a ParquetColumnSchema's DType tag.
 * frame-arrow's own `arrowTypeFor` (dtype.ts) isn't part of its public
 * export surface (see index.ts) — this is a small, deliberately-scoped
 * reimplementation covering exactly this package's v1 supported subset
 * (no "dictionary" case, since read.ts/schema.ts never produce that dtype —
 * "list"/"struct" ARE produced, but just return the already-computed
 * {@link ParquetColumnType.nestedArrowType} rather than rebuilding it, since
 * unlike the flat scalar dtypes a list/struct DataType can't be reconstructed
 * from the dtype tag alone).
 */
export function arrowTypeFor(col: ParquetColumnType): DataType {
  switch (col.dtype) {
    case "list":
    case "struct":
      if (!col.nestedArrowType) {
        throw new Error(`frame-parquet arrowTypeFor: dtype "${col.dtype}" is missing its precomputed nestedArrowType`);
      }
      return col.nestedArrowType;
    case "bool":
      return new Bool();
    case "int8":
      return new Int8();
    case "int16":
      return new Int16();
    case "int32":
      return new Int32();
    case "uint8":
      return new Uint8();
    case "uint16":
      return new Uint16();
    case "uint32":
      return new Uint32();
    case "int64":
      return new Int64();
    case "float32":
      return new Float32();
    case "float64":
      return new Float64();
    case "utf8":
      return new Utf8();
    case "timestamp_ms":
      return new TimestampMillisecond(col.timezone ?? null);
    case "timestamp_us":
      return new TimestampMicrosecond(col.timezone ?? null);
    default:
      throw new Error(`frame-parquet arrowTypeFor: unexpected dtype "${col.dtype}" (dictionary is not produced by the read path)`);
  }
}
