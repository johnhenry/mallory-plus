import assert from "node:assert/strict";
import { test } from "node:test";
import {
  BYTES_PER_ELEMENT,
  Tensor,
  broadcastShapes,
  isBigIntDType,
} from "../src/index.ts";

// ---- constructors ----------------------------------------------------------

test("zeros allocates the right storage", () => {
  const t = Tensor.zeros([2, 3]);
  assert.equal(t.size, 6);
  assert.equal(t.ndim, 2);
  assert.equal(t.dtype, "f32");
  assert.deepEqual([...t.shape], [2, 3]);
  assert.deepEqual([...t.strides], [3, 1]);
  assert.equal(t.at(1, 2), 0);
});

test("ones / full / arange / fromTypedArray", () => {
  assert.equal(Tensor.ones([2, 2]).at(1, 1), 1);
  assert.equal(Tensor.full([3], 7, { dtype: "i32" }).at(2), 7);
  assert.deepEqual(Tensor.arange(0, 10, 2).toArray(), [0, 2, 4, 6, 8]);
  assert.deepEqual(Tensor.arange(3).toArray(), [0, 1, 2]);
  const backing = new Float64Array([1, 2, 3, 4]);
  const t = Tensor.fromTypedArray(backing, [2, 2], { dtype: "f64" });
  assert.equal(t.data, backing); // no copy
  assert.equal(t.at(1, 0), 3);
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

// ---- views: identity semantics ---------------------------------------------

test("permute is a view: shares storage, never copies", () => {
  const t = Tensor.from([1, 2, 3, 4, 5, 6]).reshape([2, 3]);
  const p = t.permute([1, 0]);
  assert.equal(p.data, t.data); // pointer identity — same backing buffer
  assert.deepEqual([...p.shape], [3, 2]);
  assert.deepEqual([...p.strides], [1, 3]);
  assert.equal(p.at(2, 1), t.at(1, 2));
  assert.equal(p.isContiguous, false);
});

test("transpose defaults to reversing axes", () => {
  const t = Tensor.zeros([2, 3, 4]);
  const tt = t.transpose();
  assert.deepEqual([...tt.shape], [4, 3, 2]);
  assert.equal(tt.data, t.data);
});

test("contiguous() returns `this` when already contiguous", () => {
  const t = Tensor.from([1, 2, 3, 4]).reshape([2, 2]);
  assert.equal(t.contiguous(), t); // identical object, not a copy
});

test("contiguous() packs a non-contiguous view into new storage", () => {
  const t = Tensor.from([1, 2, 3, 4, 5, 6]).reshape([2, 3]);
  const p = t.permute([1, 0]);
  const c = p.contiguous();
  assert.notEqual(c.data, t.data); // new buffer
  assert.equal(c.isContiguous, true);
  assert.deepEqual(c.toArray(), [
    [1, 4],
    [2, 5],
    [3, 6],
  ]);
});

test("reshape supports -1 inference and rejects non-contiguous input", () => {
  const t = Tensor.arange(12);
  const r = t.reshape([-1, 4]);
  assert.deepEqual([...r.shape], [3, 4]);
  assert.equal(r.data, t.data); // view
  assert.throws(() => t.reshape([-1, -1]), RangeError);
  assert.throws(() => t.reshape([5, 2]), RangeError);
  const nonContig = t.reshape([3, 4]).permute([1, 0]);
  assert.throws(() => nonContig.reshape([12]), TypeError); // no implicit copy
});

// ---- broadcasting elementwise ops -------------------------------------------

test("broadcastShapes follows NumPy trailing-axis rules", () => {
  assert.deepEqual(broadcastShapes([2, 3], [3]), [2, 3]);
  assert.deepEqual(broadcastShapes([2, 1], [1, 3]), [2, 3]);
  assert.deepEqual(broadcastShapes([], [4]), [4]);
  assert.throws(() => broadcastShapes([2, 3], [4]), RangeError);
});

test("add/sub/mul/div with broadcasting", () => {
  const a = Tensor.from([1, 2, 3, 4, 5, 6], { dtype: "f64" }).reshape([2, 3]);
  const row = Tensor.from([10, 20, 30], { dtype: "f64" });
  assert.deepEqual(a.add(row).toArray(), [
    [11, 22, 33],
    [14, 25, 36],
  ]);
  assert.deepEqual(a.sub(1).toArray(), [
    [0, 1, 2],
    [3, 4, 5],
  ]);
  assert.deepEqual(a.mul(a).toArray(), [
    [1, 4, 9],
    [16, 25, 36],
  ]);
  assert.deepEqual(a.div(2).toArray(), [
    [0.5, 1, 1.5],
    [2, 2.5, 3],
  ]);
});

test("elementwise ops on NON-CONTIGUOUS views are correct", () => {
  // Risk register #2: stride bugs produce plausible-but-wrong values.
  const a = Tensor.from([1, 2, 3, 4, 5, 6], { dtype: "f64" }).reshape([2, 3]);
  const at = a.permute([1, 0]); // [[1,4],[2,5],[3,6]], strides [1,3]
  const b = Tensor.from([10, 100], { dtype: "f64" }); // broadcasts along rows
  assert.deepEqual(at.add(b).toArray(), [
    [11, 104],
    [12, 105],
    [13, 106],
  ]);
  const col = Tensor.from([1, 2, 3], { dtype: "f64" }).reshape([3, 1]);
  assert.deepEqual(at.mul(col).toArray(), [
    [1, 4],
    [4, 10],
    [9, 18],
  ]);
});

test("dtype mismatch throws (no implicit promotion in M1)", () => {
  const a = Tensor.from([1], { dtype: "f32" });
  const b = Tensor.from([1], { dtype: "f64" });
  assert.throws(() => a.add(b), TypeError);
});

test("i64 elementwise: add/sub/mul work, div directs to cast", () => {
  const a = Tensor.from([10, 20], { dtype: "i64" });
  const b = Tensor.from([3, 4], { dtype: "i64" });
  assert.deepEqual(a.add(b).toArray(), [13n, 24n]);
  assert.deepEqual(a.mul(b).toArray(), [30n, 80n]);
  assert.throws(() => a.div(b), TypeError);
});

// ---- reductions --------------------------------------------------------------

test("sum/mean over all elements", () => {
  const t = Tensor.from([1, 2, 3, 4], { dtype: "f64" }).reshape([2, 2]);
  assert.equal(t.sum().item(), 10);
  assert.equal(t.mean().item(), 2.5);
});

test("sum/mean along an axis (incl. negative axis)", () => {
  const t = Tensor.from([1, 2, 3, 4, 5, 6], { dtype: "f64" }).reshape([2, 3]);
  assert.deepEqual(t.sum(0).toArray(), [5, 7, 9]);
  assert.deepEqual(t.sum(1).toArray(), [6, 15]);
  assert.deepEqual(t.sum(-1).toArray(), [6, 15]);
  assert.deepEqual(t.mean(0).toArray(), [2.5, 3.5, 4.5]);
});

test("reductions on non-contiguous views", () => {
  const t = Tensor.from([1, 2, 3, 4, 5, 6], { dtype: "f64" }).reshape([2, 3]);
  const p = t.permute([1, 0]); // [[1,4],[2,5],[3,6]]
  assert.deepEqual(p.sum(1).toArray(), [5, 7, 9]);
  assert.equal(p.sum().item(), 21);
});

test("mean of integer dtypes returns f64 (NumPy semantics)", () => {
  const t = Tensor.from([1, 2], { dtype: "i32" });
  const m = t.mean();
  assert.equal(m.dtype, "f64");
  assert.equal(m.item(), 1.5);
  const big = Tensor.from([3, 4], { dtype: "i64" });
  assert.equal(big.mean().dtype, "f64");
  assert.equal(big.mean().item(), 3.5);
  assert.equal(big.sum().dtype, "i64");
  assert.equal(big.sum().item(), 7n);
});

// ---- .npy I/O -----------------------------------------------------------------

test(".npy round-trip across dtypes", () => {
  for (const dtype of ["f32", "f64", "i32", "i64", "u8", "bool"] as const) {
    const t = Tensor.from([1, 0, 1, 1, 0, 1], { dtype }).reshape([2, 3]);
    const back = Tensor.fromNpy(t.toNpy());
    assert.equal(back.dtype, dtype, dtype);
    assert.deepEqual([...back.shape], [2, 3], dtype);
    assert.deepEqual(back.toArray(), t.toArray(), dtype);
  }
});

test(".npy of a non-contiguous view serializes packed values", () => {
  const t = Tensor.from([1, 2, 3, 4, 5, 6], { dtype: "f64" }).reshape([2, 3]);
  const back = Tensor.fromNpy(t.permute([1, 0]).toNpy());
  assert.deepEqual(back.toArray(), [
    [1, 4],
    [2, 5],
    [3, 6],
  ]);
});

test(".npy header is 64-byte aligned and v1.0", () => {
  const bytes = Tensor.from([1, 2, 3]).toNpy();
  assert.equal(bytes[6], 1);
  const headerLength = (bytes[8] as number) | ((bytes[9] as number) << 8);
  assert.equal((10 + headerLength) % 64, 0);
});

// ---- indexing & slicing (issue #1) --------------------------------------

test("slice is a view: shares storage, matches basic NumPy slicing", () => {
  const t = Tensor.arange(12).reshape([3, 4]); // [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
  const s = t.slice({ start: 1, end: 3 }, { start: 1, end: 3 });
  assert.equal(s.data, t.data); // pointer identity — a view, not a copy
  assert.deepEqual([...s.shape], [2, 2]);
  assert.deepEqual(s.toArray(), [
    [5, 6],
    [9, 10],
  ]);
});

test("slice: omitted trailing axes are taken whole", () => {
  const t = Tensor.arange(12).reshape([3, 4]);
  const s = t.slice({ start: 1, end: 2 }); // second axis unspecified
  assert.deepEqual([...s.shape], [1, 4]);
  assert.deepEqual(s.toArray(), [[4, 5, 6, 7]]);
});

test("slice: negative start/end count from the end (Python slice semantics)", () => {
  const t = Tensor.arange(10);
  assert.deepEqual(t.slice({ start: -3 }).toArray(), [7, 8, 9]);
  assert.deepEqual(t.slice({ end: -3 }).toArray(), [0, 1, 2, 3, 4, 5, 6]);
  assert.deepEqual(t.slice({ start: -5, end: -2 }).toArray(), [5, 6, 7]);
});

test("slice: negative step reverses via negative stride, still a view", () => {
  const t = Tensor.arange(5); // [0,1,2,3,4]
  const r = t.slice({ step: -1 });
  assert.equal(r.data, t.data);
  assert.equal(r.strides[0], -1);
  assert.deepEqual(r.toArray(), [4, 3, 2, 1, 0]);

  const partial = t.slice({ start: 3, end: 0, step: -1 });
  assert.deepEqual(partial.toArray(), [3, 2, 1]);
});

test("slice composes with permute: slice-of-a-view stays a view", () => {
  const t = Tensor.arange(12).reshape([3, 4]);
  const p = t.permute([1, 0]); // 4x3, non-contiguous
  const s = p.slice({ start: 1, end: 3 });
  assert.equal(s.data, t.data); // still the original buffer, two views deep
  assert.deepEqual([...s.shape], [2, 3]);
  assert.deepEqual(s.toArray(), p.toArray().slice(1, 3));
});

test("slice: zero-length and out-of-range specs clamp rather than throw", () => {
  const t = Tensor.arange(5);
  assert.deepEqual(t.slice({ start: 3, end: 3 }).toArray(), []);
  assert.deepEqual(t.slice({ start: 10 }).toArray(), []);
  assert.deepEqual(t.slice({ end: 100 }).toArray(), [0, 1, 2, 3, 4]);
});

test("slice: step of zero throws", () => {
  const t = Tensor.arange(5);
  assert.throws(() => t.slice({ step: 0 }), RangeError);
});

test("select drops an axis and is a view", () => {
  const t = Tensor.arange(12).reshape([3, 4]);
  const row = t.select(0, 1);
  assert.equal(row.data, t.data);
  assert.deepEqual([...row.shape], [4]);
  assert.deepEqual(row.toArray(), [4, 5, 6, 7]);
  // negative index
  assert.deepEqual(t.select(0, -1).toArray(), [8, 9, 10, 11]);
  assert.throws(() => t.select(0, 5), RangeError);
});

test("take gathers arbitrary indices along an axis (copies)", () => {
  const t = Tensor.arange(12).reshape([3, 4]);
  const picked = t.take([2, 0], { axis: 0 });
  assert.notEqual(picked.data, t.data); // copy, not a view
  assert.deepEqual(picked.toArray(), [
    [8, 9, 10, 11],
    [0, 1, 2, 3],
  ]);
  // negative indices and default axis 0
  assert.deepEqual(
    t.take([-1]).toArray(),
    [[8, 9, 10, 11]],
  );
  assert.throws(() => t.take([99]), RangeError);
});

test("gather is take with (axis, indices) argument order", () => {
  const t = Tensor.arange(12).reshape([3, 4]);
  assert.deepEqual(t.gather(1, [3, 0]).toArray(), t.take([3, 0], { axis: 1 }).toArray());
});

test("mask selects elements where the boolean tensor is true, flattening to 1-D", () => {
  const t = Tensor.from([10, 20, 30, 40], { dtype: "f64" });
  const cond = Tensor.from([1, 0, 1, 0], { dtype: "bool" });
  const selected = t.mask(cond);
  assert.deepEqual([...selected.shape], [2]);
  assert.deepEqual(selected.toArray(), [10, 30]);
});

test("mask rejects non-bool conditions and shape mismatches", () => {
  const t = Tensor.from([1, 2, 3], { dtype: "f64" });
  assert.throws(() => t.mask(Tensor.from([1, 0, 1], { dtype: "f64" })), TypeError);
  assert.throws(
    () => t.mask(Tensor.from([1, 0], { dtype: "bool" })),
    RangeError,
  );
});

test("take/select/mask on non-contiguous views read the correct (view-relative) values", () => {
  const t = Tensor.arange(6).reshape([2, 3]); // [[0,1,2],[3,4,5]]
  const p = t.permute([1, 0]); // [[0,3],[1,4],[2,5]]
  assert.deepEqual(p.select(0, 1).toArray(), [1, 4]);
  assert.deepEqual(p.take([2, 0], { axis: 0 }).toArray(), [
    [2, 5],
    [0, 3],
  ]);
});

// ---- matmul (issue #2) ---------------------------------------------------

test("matmul: 2-D x 2-D, deterministic", () => {
  const a = Tensor.from([1, 2, 3, 4, 5, 6], { dtype: "f64" }).reshape([2, 3]);
  const b = Tensor.from([7, 8, 9, 10, 11, 12], { dtype: "f64" }).reshape([3, 2]);
  const c = a.matmul(b);
  assert.deepEqual([...c.shape], [2, 2]);
  // [[1,2,3],[4,5,6]] @ [[7,8],[9,10],[11,12]]
  assert.deepEqual(c.toArray(), [
    [58, 64],
    [139, 154],
  ]);
});

test("matmul: 1-D @ 1-D collapses to a 0-d scalar", () => {
  const a = Tensor.from([1, 2, 3], { dtype: "f64" });
  const b = Tensor.from([4, 5, 6], { dtype: "f64" });
  const r = a.matmul(b);
  assert.equal(r.ndim, 0);
  assert.equal(r.item(), 32); // 1*4+2*5+3*6
});

test("matmul: 2-D @ 1-D and 1-D @ 2-D squeeze the right axis", () => {
  const mat = Tensor.from([1, 2, 3, 4, 5, 6], { dtype: "f64" }).reshape([2, 3]);
  const v3 = Tensor.from([1, 1, 1], { dtype: "f64" });
  const v2 = Tensor.from([1, 1], { dtype: "f64" });
  const mv = mat.matmul(v3); // (2,3)@(3,) -> (2,)
  assert.deepEqual([...mv.shape], [2]);
  assert.deepEqual(mv.toArray(), [6, 15]);
  const vm = v2.matmul(mat); // (2,)@(2,3) -> (3,)
  assert.deepEqual([...vm.shape], [3]);
  assert.deepEqual(vm.toArray(), [5, 7, 9]);
});

test("matmul: batched leading axes broadcast", () => {
  // batch of 2 identity-ish 2x2 matrices times a shared 2x2
  const batch = Tensor.from([1, 0, 0, 1, 2, 0, 0, 2], { dtype: "f64" }).reshape([
    2, 2, 2,
  ]);
  const shared = Tensor.from([1, 2, 3, 4], { dtype: "f64" }).reshape([2, 2]);
  const r = batch.matmul(shared);
  assert.deepEqual([...r.shape], [2, 2, 2]);
  assert.deepEqual(r.toArray(), [
    [
      [1, 2],
      [3, 4],
    ],
    [
      [2, 4],
      [6, 8],
    ],
  ]);
});

test("matmul: transposed (non-contiguous) lhs does not copy before computing", () => {
  const a = Tensor.from([1, 2, 3, 4, 5, 6], { dtype: "f64" }).reshape([2, 3]);
  const aT = a.permute([1, 0]); // (3,2), non-contiguous
  assert.equal(aT.isContiguous, false);
  const b = Tensor.from([1, 0, 0, 1], { dtype: "f64" }).reshape([2, 2]);
  const r = aT.matmul(b); // (3,2)@(2,2) -> (3,2), should equal aT itself (identity)
  assert.deepEqual(r.toArray(), aT.toArray());
});

test("matmul: inner-dimension mismatch throws", () => {
  const a = Tensor.zeros([2, 3], { dtype: "f64" });
  const b = Tensor.zeros([4, 2], { dtype: "f64" });
  assert.throws(() => a.matmul(b), RangeError);
});

test("matmul: dtype mismatch throws", () => {
  const a = Tensor.zeros([2, 2], { dtype: "f32" });
  const b = Tensor.zeros([2, 2], { dtype: "f64" });
  assert.throws(() => a.matmul(b), TypeError);
});

test("matmul: rejects 0-d (scalar) operands", () => {
  const scalar = Tensor.from([5], { dtype: "f64" }).reshape([]);
  const v = Tensor.from([1, 2], { dtype: "f64" });
  assert.throws(() => scalar.matmul(v), RangeError);
});

test("dot: rejects non-1-D operands", () => {
  const mat = Tensor.zeros([2, 2], { dtype: "f64" });
  const v = Tensor.from([1, 2], { dtype: "f64" });
  assert.throws(() => mat.dot(v), RangeError);
});

test("matmul: i64 exact accumulation, no overflow for modest values", () => {
  const a = Tensor.from([1, 2, 3, 4], { dtype: "i64" }).reshape([2, 2]);
  const b = Tensor.from([5, 6, 7, 8], { dtype: "i64" }).reshape([2, 2]);
  assert.deepEqual(a.matmul(b).toArray(), [
    [19n, 22n],
    [43n, 50n],
  ]);
});
