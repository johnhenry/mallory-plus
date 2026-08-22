/**
 * toTensor() implementation, imported ONLY via dynamic `import()` from
 * series.ts/frame.ts's async `toTensor()` methods — never statically.
 *
 * This module intentionally does NOT `import type` from "@johnhenry/math-plus-tensor-core"
 * either: frame-arrow's own typecheck (`tsc --noEmit`) must succeed even in
 * a hypothetical checkout where tensor-core's types aren't resolvable, since
 * the issue is explicit that frame-arrow has NO dependency (static OR
 * type-level) on the tensor track — only an optional peerDependency,
 * resolved lazily at call time. So the tensor-core surface used here is
 * described with a small local structural type instead, and the dynamic
 * import result is cast through `unknown`. The practical cost: `toTensor()`
 * returns `Promise<unknown>` rather than `Promise<Tensor>` — callers that
 * have tensor-core installed can cast the result themselves.
 *
 * Arrow dtype -> Tensor dtype mapping (documented, not spec'd verbatim by
 * issue #19): every frame-arrow numeric dtype has a direct 1:1 tensor-core
 * counterpart (bool/i8/i16/i32/u8/u16/u32/i64/f32/f64). utf8, dictionary,
 * timestamp, list, and struct have no Tensor equivalent and throw a clear
 * error naming the offending column and its dtype, rather than silently
 * coercing (e.g. stringifying, or truncating a timestamp to a number).
 */
import type { DType } from "./dtype.ts";
import type { Frame } from "./frame.ts";
import type { Series } from "./series.ts";

type TensorDType = "bool" | "u8" | "i8" | "u16" | "i16" | "u32" | "i32" | "u64" | "i64" | "f16" | "bf16" | "f32" | "f64";

interface TensorLike {
  readonly shape: readonly number[];
  readonly dtype: TensorDType;
}

interface TensorCoreModule {
  allocate(dtype: TensorDType, length: number): ArrayBufferView;
  isBigIntDType(dtype: TensorDType): boolean;
  Tensor: {
    fromTypedArray(data: ArrayBufferView, shape: readonly number[], options: { dtype: TensorDType }): TensorLike;
  };
}

async function loadTensorCore(): Promise<TensorCoreModule> {
  const mod = await import("@johnhenry/math-plus-tensor-core");
  return mod as unknown as TensorCoreModule;
}

function mapDType(dtype: DType, columnName: string): TensorDType {
  switch (dtype) {
    case "bool":
      return "bool";
    case "int8":
      return "i8";
    case "int16":
      return "i16";
    case "int32":
      return "i32";
    case "uint8":
      return "u8";
    case "uint16":
      return "u16";
    case "uint32":
      return "u32";
    case "int64":
      return "i64";
    case "float32":
      return "f32";
    case "float64":
      return "f64";
    default:
      throw new Error(
        `toTensor(): column "${columnName}" has dtype "${dtype}", which Tensor cannot represent ` +
          `(utf8/dictionary/timestamp/list/struct have no Tensor equivalent)`,
      );
  }
}

function fillTypedArray(
  target: ArrayBufferView,
  values: readonly unknown[],
  isBigInt: boolean,
  columnName: string,
): void {
  if (isBigInt) {
    const big = target as unknown as BigInt64Array | BigUint64Array;
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (v === null || v === undefined) {
        throw new Error(`toTensor(): column "${columnName}" contains null values; call .fillNull(...) first`);
      }
      big[i] = typeof v === "bigint" ? v : BigInt(v as number);
    }
  } else {
    const num = target as unknown as Float64Array | Float32Array | Int32Array | Uint32Array | Int16Array | Uint16Array | Int8Array | Uint8Array;
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (v === null || v === undefined) {
        throw new Error(`toTensor(): column "${columnName}" contains null values; call .fillNull(...) first`);
      }
      num[i] = typeof v === "bigint" ? Number(v) : (v as number);
    }
  }
}

export async function seriesToTensor(series: Series): Promise<unknown> {
  const { allocate, isBigIntDType, Tensor } = await loadTensorCore();
  const dtype = mapDType(series.dtype, series.name);
  const values = series.toArray();
  const data = allocate(dtype, values.length);
  fillTypedArray(data, values, isBigIntDType(dtype), series.name);
  return Tensor.fromTypedArray(data, [values.length], { dtype });
}

/**
 * Frame.toTensor(): all columns are promoted to a single dtype (float64 by
 * default) into a 2D [numRows, numCols] row-major Tensor. This is a
 * documented v1 simplification — a Frame's columns can have heterogeneous
 * dtypes, but a Tensor's backing buffer cannot, so *some* promotion policy
 * is required; float64 is the safe default (matches numeric literal
 * defaults elsewhere in this package, e.g. LiteralExpr). Pass
 * `{ dtype: "int32" }` etc. to promote to something else instead — every
 * column must actually be losslessly representable in the chosen dtype's
 * JS-number range, or you'll silently get float64-style rounding/truncation
 * exactly as `Number(bigint)`/typed-array assignment would do; this method
 * does not re-validate range beyond apache-arrow's own storage boundaries.
 */
export async function frameToTensor(frame: Frame, options: { dtype?: DType } = {}): Promise<unknown> {
  const { allocate, isBigIntDType, Tensor } = await loadTensorCore();
  const table = frame.toArrow();
  const names = table.schema.names as string[];
  const targetDType = options.dtype ?? "float64";
  const tensorDType = mapDType(targetDType, "<Frame.toTensor dtype option>");
  const numRows = table.numRows;
  const numCols = names.length;
  const data = allocate(tensorDType, numRows * numCols);
  const isBig = isBigIntDType(tensorDType);
  for (let c = 0; c < numCols; c++) {
    const columnName = names[c] as string;
    const series = frame.getSeries(columnName);
    mapDType(series.dtype, columnName); // validates this column is numeric, even if targetDType differs
    const values = series.toArray();
    for (let r = 0; r < numRows; r++) {
      const v = values[r];
      if (v === null || v === undefined) {
        throw new Error(`Frame.toTensor(): column "${columnName}" contains null values; call .fillNull(...) first`);
      }
      const flatIndex = r * numCols + c;
      if (isBig) {
        (data as unknown as BigInt64Array | BigUint64Array)[flatIndex] = typeof v === "bigint" ? v : BigInt(v as number);
      } else {
        (data as unknown as Float64Array)[flatIndex] = typeof v === "bigint" ? Number(v) : (v as number);
      }
    }
  }
  return Tensor.fromTypedArray(data, [numRows, numCols], { dtype: tensorDType });
}
