import assert from "node:assert/strict";
import { test } from "node:test";
import { BYTES_PER_ELEMENT, Tensor, isBigIntDType } from "../src/index.ts";

test("zeros allocates the right storage", () => {
  const t = Tensor.zeros([2, 3]);
  assert.equal(t.size, 6);
  assert.equal(t.ndim, 2);
  assert.equal(t.dtype, "f32");
  assert.deepEqual([...t.shape], [2, 3]);
  assert.deepEqual([...t.strides], [3, 1]);
  assert.equal(t.at(1, 2), 0);
});

test("from builds a 1-D tensor with negative indexing", () => {
  const t = Tensor.from([1, 2, 3], { dtype: "f64" });
  assert.equal(t.at(0), 1);
  assert.equal(t.at(-1), 3);
  assert.throws(() => t.at(3), RangeError);
});

test("i64 dtype is BigInt-backed (ONNX input_ids case)", () => {
  const t = Tensor.from([101, 2023, 102], { dtype: "i64" });
  assert.equal(t.at(0), 101n);
  assert.equal(typeof t.at(1), "bigint");
  assert.ok(isBigIntDType(t.dtype));
  assert.equal(BYTES_PER_ELEMENT.i64, 8);
});

test("shape is frozen — no mutation of tensor metadata", () => {
  const t = Tensor.zeros([2, 2]);
  assert.throws(() => {
    (t.shape as number[])[0] = 99;
  }, TypeError);
});
