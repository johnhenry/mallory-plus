import assert from "node:assert/strict";
import { test } from "node:test";
import { Kernels } from "../src/index.ts";

test("add_f32 kernel adds through the wasm seam", async () => {
  const kernels = await Kernels.load();
  const out = kernels.addF32(
    new Float32Array([1, 2, 3, 4]),
    new Float32Array([10, 20, 30, 40]),
  );
  assert.deepEqual([...out], [11, 22, 33, 44]);
});

test("length mismatch throws", async () => {
  const kernels = await Kernels.load();
  assert.throws(
    () => kernels.addF32(new Float32Array(2), new Float32Array(3)),
    RangeError,
  );
});
