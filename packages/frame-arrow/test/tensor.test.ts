/**
 * toTensor() exercises the lazy dynamic import of mallory-tensor-core.
 * tensor-core IS present in this monorepo (a sibling workspace package), so
 * these tests confirm the dynamic import succeeds and produces a correct
 * Tensor — they can't prove "tensor-core absent" from inside the repo (see
 * the issue's own note on this); that half of the constraint is instead
 * verified structurally: package.json lists mallory-tensor-core ONLY under
 * peerDependencies + peerDependenciesMeta.optional, never dependencies.
 */
import assert from "node:assert/strict";
import { test } from "node:test";
import { Float64, Int32, Table, vectorFromArray } from "apache-arrow";
import { Frame } from "../src/index.ts";

test("Series.toTensor(): float64 column -> f64 Tensor with matching shape/values", async () => {
  const frame = Frame.fromArrow(new Table({ v: vectorFromArray([1.5, 2.5, 3.5], new Float64()) }));
  const tensor = (await frame.getSeries("v").toTensor()) as {
    shape: readonly number[];
    dtype: string;
    data: Float64Array;
  };
  assert.deepEqual(tensor.shape, [3]);
  assert.equal(tensor.dtype, "f64");
  assert.deepEqual(Array.from(tensor.data), [1.5, 2.5, 3.5]);
});

test("Series.toTensor(): int32 column -> i32 Tensor", async () => {
  const frame = Frame.fromArrow(new Table({ v: vectorFromArray([1, -2, 3], new Int32()) }));
  const tensor = (await frame.getSeries("v").toTensor()) as { dtype: string; data: Int32Array };
  assert.equal(tensor.dtype, "i32");
  assert.deepEqual(Array.from(tensor.data), [1, -2, 3]);
});

test("Series.toTensor() throws a clear error for a dtype Tensor can't represent (utf8)", async () => {
  const { Utf8 } = await import("apache-arrow");
  const frame = Frame.fromArrow(new Table({ name: vectorFromArray(["a", "b"], new Utf8()) }));
  await assert.rejects(() => frame.getSeries("name").toTensor(), /cannot represent/);
});

test("Series.toTensor() throws when the column contains nulls (no null representation in Tensor)", async () => {
  const frame = Frame.fromArrow(new Table({ v: vectorFromArray([1.0, null, 3.0], new Float64()) }));
  await assert.rejects(() => frame.getSeries("v").toTensor(), /null/);
});

test("Frame.toTensor(): all-numeric Frame -> 2D row-major float64 Tensor by default", async () => {
  const frame = Frame.fromArrow(
    new Table({
      a: vectorFromArray([1, 2, 3], new Float64()),
      b: vectorFromArray([10, 20, 30], new Float64()),
    }),
  );
  const tensor = (await frame.toTensor()) as { shape: readonly number[]; dtype: string; data: Float64Array };
  assert.deepEqual(tensor.shape, [3, 2]);
  assert.equal(tensor.dtype, "f64");
  assert.deepEqual(Array.from(tensor.data), [1, 10, 2, 20, 3, 30]);
});

test("mallory-frame-arrow's package.json lists mallory-tensor-core only as an optional peerDependency, pinned to that package's ACTUAL current version", async () => {
  const fs = await import("node:fs/promises");
  const url = await import("node:url");
  const pkgPath = url.fileURLToPath(new URL("../package.json", import.meta.url));
  const pkg = JSON.parse(await fs.readFile(pkgPath, "utf8"));
  // Read tensor-core's own version rather than hardcoding a literal here --
  // this test's job is to enforce the EXACT-PIN invariant, not to pin
  // itself to a version number that a release bump would silently stale.
  const tensorCorePkgPath = url.fileURLToPath(
    new URL("../../tensor-core/package.json", import.meta.url),
  );
  const tensorCorePkg = JSON.parse(await fs.readFile(tensorCorePkgPath, "utf8"));
  assert.equal(pkg.dependencies?.["mallory-tensor-core"], undefined);
  assert.equal(pkg.peerDependencies?.["mallory-tensor-core"], tensorCorePkg.version);
  assert.equal(pkg.peerDependenciesMeta?.["mallory-tensor-core"]?.optional, true);
});
