/**
 * Frame's v1 dtype surface — a coarse, JS-facing tag set that intentionally
 * does NOT try to mirror Arrow's full type system (see docs/spikes/arrow-parity.md
 * "v1 type claims"). Anything not in {@link DType} throws {@link UnsupportedTypeError}
 * the moment it is actually inspected — never silently mishandled.
 *
 * Deferred past v1 (frozen by the spike, do not silently expand or drop):
 * timestamp[ns], date32/64, time32/64, duration, decimal128, deeply nested
 * types (list<struct>, struct<list>, map), delta dictionaries, large_utf8/large_list.
 */
import {
  Bool,
  DataType,
  Dictionary,
  Field,
  Float32,
  Float64,
  Int16,
  Int32,
  Int64,
  Int8,
  List,
  Schema,
  TimestampMicrosecond,
  TimestampMillisecond,
  Uint16,
  Uint32,
  Uint8,
  Utf8,
  type Vector,
} from "apache-arrow";

export type DType =
  | "bool"
  | "int8"
  | "int16"
  | "int32"
  | "uint8"
  | "uint16"
  | "uint32"
  | "int64"
  | "float32"
  | "float64"
  | "utf8"
  | "dictionary"
  | "timestamp_ms"
  | "timestamp_us"
  | "list"
  | "struct";

/** Human-readable rendering used in error messages and Field.describe(). */
export interface FieldDescriptor {
  readonly name: string;
  readonly dtype: DType;
  readonly nullable: boolean;
  /** IANA timezone name, or null for naive; only set for timestamp_ms/timestamp_us. */
  readonly timezone?: string | null;
  /** Element dtype; only set for dtype === "list". */
  readonly itemDType?: DType;
  /** Field names of the nested struct; only set for dtype === "struct". */
  readonly structFields?: readonly FieldDescriptor[];
  /** apache-arrow's own DataType, for escape-hatch access. */
  readonly arrowType: DataType;
}

export class UnsupportedTypeError extends TypeError {
  constructor(columnName: string, arrowType: DataType) {
    super(
      `column "${columnName}" has Arrow type ${arrowType.toString()} which @johnhenry/math-plus-frame-arrow v1 does not support ` +
        `(deferred: timestamp[ns], date32/64, time32/64, duration, decimal128, deeply nested types, ` +
        `delta dictionaries, large_utf8/large_list — see docs/spikes/arrow-parity.md). ` +
        `Cast or drop this column in Arrow/pyarrow before loading it into a Frame.`,
    );
    this.name = "UnsupportedTypeError";
  }
}

/** Classify an Arrow DataType into Frame's v1 DType tag, or throw. */
export function describeField(field: Field): FieldDescriptor {
  const type = field.type;
  const base = { name: field.name, nullable: field.nullable, arrowType: type };

  if (DataType.isBool(type)) return { ...base, dtype: "bool" };
  if (DataType.isUtf8(type)) return { ...base, dtype: "utf8" };
  if (DataType.isDictionary(type)) {
    if (!DataType.isUtf8(type.dictionary)) {
      throw new UnsupportedTypeError(field.name, type);
    }
    return { ...base, dtype: "dictionary" };
  }
  if (DataType.isInt(type)) {
    switch (type.bitWidth) {
      case 8:
        return { ...base, dtype: type.isSigned ? "int8" : "uint8" };
      case 16:
        return { ...base, dtype: type.isSigned ? "int16" : "uint16" };
      case 32:
        return { ...base, dtype: type.isSigned ? "int32" : "uint32" };
      case 64:
        if (!type.isSigned) throw new UnsupportedTypeError(field.name, type);
        return { ...base, dtype: "int64" };
      default:
        throw new UnsupportedTypeError(field.name, type);
    }
  }
  if (DataType.isFloat(type)) {
    switch (type.precision) {
      case 0: // HALF
        throw new UnsupportedTypeError(field.name, type); // f16 deferred (tensor-core parity note applies to Tensor, not Frame; Frame v1 doesn't claim it)
      case 1: // SINGLE
        return { ...base, dtype: "float32" };
      case 2: // DOUBLE
        return { ...base, dtype: "float64" };
      default:
        throw new UnsupportedTypeError(field.name, type);
    }
  }
  if (DataType.isTimestamp(type)) {
    if (type.unit === 1 /* MILLISECOND */) {
      return { ...base, dtype: "timestamp_ms", timezone: type.timezone ?? null };
    }
    if (type.unit === 2 /* MICROSECOND */) {
      return { ...base, dtype: "timestamp_us", timezone: type.timezone ?? null };
    }
    throw new UnsupportedTypeError(field.name, type); // seconds, nanoseconds deferred
  }
  if (DataType.isList(type)) {
    const itemField = type.children[0] as Field;
    const item = describeField(itemField);
    if (item.dtype === "list" || item.dtype === "struct") {
      throw new UnsupportedTypeError(field.name, type); // deep nesting deferred
    }
    return { ...base, dtype: "list", itemDType: item.dtype };
  }
  if (DataType.isStruct(type)) {
    const structFields = type.children.map((child: Field) => {
      const d = describeField(child);
      if (d.dtype === "list" || d.dtype === "struct") {
        throw new UnsupportedTypeError(field.name, type); // deep nesting deferred
      }
      return d;
    });
    return { ...base, dtype: "struct", structFields };
  }
  throw new UnsupportedTypeError(field.name, type);
}

export function describeSchema(schema: Schema): FieldDescriptor[] {
  return schema.fields.map((f: Field) => describeField(f));
}

/** Build the Arrow type for a given DType tag, used when constructing new columns internally. */
export function arrowTypeFor(
  dtype: DType,
  opts: { timezone?: string | null; itemType?: DataType } = {},
): DataType {
  switch (dtype) {
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
    case "dictionary":
      return new Dictionary(new Utf8(), new Int32());
    case "timestamp_ms":
      return new TimestampMillisecond(opts.timezone ?? null);
    case "timestamp_us":
      return new TimestampMicrosecond(opts.timezone ?? null);
    case "list":
      return new List(new Field("item", opts.itemType ?? new Float64(), true));
    case "struct":
      throw new Error("arrowTypeFor(struct) requires explicit Struct field construction; not supported generically");
    default: {
      const exhaustive: never = dtype;
      throw new Error(`unknown dtype "${exhaustive as string}"`);
    }
  }
}

export function isBigIntDType(dtype: DType): boolean {
  return dtype === "int64";
}

export function isTimestampDType(dtype: DType): boolean {
  return dtype === "timestamp_ms" || dtype === "timestamp_us";
}

export function isNumericDType(dtype: DType): boolean {
  return (
    dtype === "int8" ||
    dtype === "int16" ||
    dtype === "int32" ||
    dtype === "uint8" ||
    dtype === "uint16" ||
    dtype === "uint32" ||
    dtype === "int64" ||
    dtype === "float32" ||
    dtype === "float64"
  );
}

/** Best-effort human label, e.g. "timestamp[us, America/Los_Angeles]". */
export function dtypeLabel(d: FieldDescriptor): string {
  if (d.dtype === "timestamp_ms" || d.dtype === "timestamp_us") {
    const unit = d.dtype === "timestamp_ms" ? "ms" : "us";
    return d.timezone ? `timestamp[${unit}, ${d.timezone}]` : `timestamp[${unit}]`;
  }
  if (d.dtype === "list") return `list<${d.itemDType}>`;
  if (d.dtype === "dictionary") return "dictionary<utf8>";
  return d.dtype;
}

export type { Vector };
