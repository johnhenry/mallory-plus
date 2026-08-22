/**
 * DType <-> ONNX Runtime Web Tensor.Type mapping (issue #18). Every mapping
 * here is a DIRECT TypedArray passthrough — @johnhenry/math-plus-tensor-core and
 * onnxruntime-web happen to back every shared dtype with the exact same
 * TypedArray class (verified against onnxruntime-common's
 * `Tensor.DataTypeMap`), so marshalling never needs to copy/convert
 * element-by-element, only reshape metadata.
 *
 * `bf16` has no ONNX Runtime Web equivalent (its `Tensor.DataTypeMap` has no
 * `bfloat16` entry, unlike `float16`, which IS supported and — like
 * tensor-core's own `bf16`/`f16` — is raw `Uint16Array` bit storage, not a
 * real numeric type either side converts). `bf16` inputs/outputs throw a
 * clear `UnsupportedDTypeError` rather than silently mis-mapping to
 * `float16` (the two aren't the same bit layout).
 */
import type { DType } from "@johnhenry/math-plus-tensor-core";
import type { Tensor as OrtTensor } from "onnxruntime-web";

export class UnsupportedDTypeError extends Error {
  constructor(dtype: string, context: string) {
    super(`@johnhenry/math-plus-adapter-onnx: dtype "${dtype}" has no ONNX Runtime Web equivalent (${context})`);
    this.name = "UnsupportedDTypeError";
  }
}

const DTYPE_TO_ORT: Partial<Record<DType, OrtTensor.Type>> = {
  bool: "bool",
  u8: "uint8",
  i8: "int8",
  u16: "uint16",
  i16: "int16",
  u32: "uint32",
  i32: "int32",
  u64: "uint64",
  i64: "int64",
  f16: "float16",
  f32: "float32",
  f64: "float64",
  // bf16: intentionally absent
};

const ORT_TO_DTYPE: Partial<Record<OrtTensor.Type, DType>> = {
  bool: "bool",
  uint8: "u8",
  int8: "i8",
  uint16: "u16",
  int16: "i16",
  uint32: "u32",
  int32: "i32",
  uint64: "u64",
  int64: "i64",
  float16: "f16",
  float32: "f32",
  float64: "f64",
  // string, uint4, int4: no tensor-core equivalent
};

export function dtypeToOrtType(dtype: DType): OrtTensor.Type {
  const ortType = DTYPE_TO_ORT[dtype];
  if (!ortType) throw new UnsupportedDTypeError(dtype, "converting a Tensor for ONNX Runtime input");
  return ortType;
}

export function ortTypeToDtype(ortType: OrtTensor.Type): DType {
  const dtype = ORT_TO_DTYPE[ortType];
  if (!dtype) {
    throw new UnsupportedDTypeError(ortType, "converting an ONNX Runtime output back into a Tensor");
  }
  return dtype;
}
