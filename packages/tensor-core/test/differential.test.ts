/**
 * Differential tests: tensor-core vs a NumPy subprocess oracle
 * (docs/PLAN.md §6.1). Exchange format is .npy in BOTH directions, so every
 * run also exercises tensor-core's .npy reader and writer against NumPy's.
 *
 * Python resolution: $MALLORY_ORACLE_PYTHON, else `python3` on PATH. If no
 * interpreter with numpy is found the suite SKIPS (it does not fail) — CI
 * environments that guarantee the oracle should also `grep -c "skipped 0"`.
 * On trycooy: nix-shell -p "python3.withPackages(ps: [ps.numpy])" provides
 * one; pip wheels don't work on NixOS.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { Tensor, isBigIntDType, type DType } from "../src/index.ts";

const ORACLE_SCRIPT = new URL("../scripts/numpy_oracle.py", import.meta.url)
  .pathname;

function findOraclePython(): string | undefined {
  const candidates = [
    process.env.MALLORY_ORACLE_PYTHON,
    "python3",
  ].filter((c): c is string => Boolean(c));
  for (const candidate of candidates) {
    try {
      execFileSync(candidate, ["-c", "import numpy"], { stdio: "ignore" });
      return candidate;
    } catch {
      // try next
    }
  }
  return undefined;
}

const PYTHON = findOraclePython();
const skip = PYTHON
  ? false
  : "no python with numpy found (set MALLORY_ORACLE_PYTHON)";

interface OracleJob {
  op: string;
  inputs?: string[];
  axis?: number;
  permutation?: number[];
  scalar?: number;
  arange?: [number, number, number];
  dtype?: string;
  specs?: Array<[number | null, number | null, number | null] | null>;
  output: string;
}

function runOracle(dir: string, job: Omit<OracleJob, "output">): Tensor {
  const output = join(dir, `out-${Math.random().toString(36).slice(2)}.npy`);
  const jobPath = join(dir, "job.json");
  writeFileSync(jobPath, JSON.stringify({ ...job, output }));
  execFileSync(PYTHON as string, [ORACLE_SCRIPT, jobPath], {
    stdio: ["ignore", "ignore", "pipe"],
  });
  return Tensor.fromNpy(new Uint8Array(readFileSync(output)));
}

function saveTensor(dir: string, name: string, t: Tensor): string {
  const path = join(dir, `${name}.npy`);
  writeFileSync(path, t.toNpy());
  return path;
}

/** Per-op absolute/relative tolerances (looser for accumulation-order-sensitive ops). */
const TOLERANCES: Record<string, { rtol: number; atol: number }> = {
  "default:f32": { rtol: 1e-5, atol: 1e-6 },
  "default:f64": { rtol: 1e-12, atol: 1e-12 },
  "sum:f32": { rtol: 1e-4, atol: 1e-5 },
  "mean:f32": { rtol: 1e-4, atol: 1e-5 },
};

function assertClose(actual: Tensor, expected: Tensor, op: string): void {
  assert.deepEqual([...actual.shape], [...expected.shape], `${op}: shape`);
  const tolerance =
    TOLERANCES[`${op}:${actual.dtype}`] ??
    TOLERANCES[`default:${actual.dtype}`] ??
    { rtol: 0, atol: 0 };
  const a = actual.contiguous();
  const e = expected.contiguous();
  for (let i = 0; i < a.size; i++) {
    const av = a.data[a.offset + i];
    const ev = e.data[e.offset + i];
    if (typeof av === "bigint" || typeof ev === "bigint") {
      assert.equal(av, ev, `${op}: element ${i}`);
    } else {
      const bound =
        tolerance.atol + tolerance.rtol * Math.abs(ev as number);
      assert.ok(
        Math.abs((av as number) - (ev as number)) <= bound,
        `${op}: element ${i}: ${av} vs ${ev} (bound ${bound})`,
      );
    }
  }
}

function randomTensor(shape: number[], dtype: DType): Tensor {
  const size = shape.reduce((x, y) => x * y, 1);
  const values = Array.from({ length: size }, () =>
    isBigIntDType(dtype)
      ? Math.floor(Math.random() * 200 - 100)
      : Math.random() * 20 - 10,
  );
  return Tensor.from(values, { dtype }).reshape(shape);
}

test("differential vs NumPy", { skip }, async (t) => {
  const dir = mkdtempSync(join(tmpdir(), "mallory-diff-"));
  t.after(() => rmSync(dir, { recursive: true, force: true }));

  await t.test("binary ops, f32 + f64, incl. broadcasting", () => {
    for (const dtype of ["f32", "f64"] as const) {
      const a = randomTensor([3, 4], dtype);
      const b = randomTensor([4], dtype); // broadcast along rows
      const aPath = saveTensor(dir, `a-${dtype}`, a);
      const bPath = saveTensor(dir, `b-${dtype}`, b);
      for (const op of ["add", "sub", "mul", "div"] as const) {
        const expected = runOracle(dir, { op, inputs: [aPath, bPath] });
        assertClose(a[op](b), expected, op);
      }
    }
  });

  await t.test("binary ops on NON-CONTIGUOUS (permuted) views", () => {
    const a = randomTensor([3, 4], "f64");
    const aT = a.permute([1, 0]);
    // Oracle sees the packed transpose; we compute on the strided view.
    const aTPath = saveTensor(dir, "aT", aT); // toNpy packs it
    const b = randomTensor([3], "f64");
    const bPath = saveTensor(dir, "b-nc", b);
    for (const op of ["add", "mul"] as const) {
      const expected = runOracle(dir, { op, inputs: [aTPath, bPath] });
      assertClose(aT[op](b), expected, op);
    }
  });

  await t.test("scalar operand", () => {
    const a = randomTensor([5], "f64");
    const aPath = saveTensor(dir, "a-scalar", a);
    const expected = runOracle(dir, { op: "mul", inputs: [aPath], scalar: 2.5 });
    assertClose(a.mul(2.5), expected, "mul");
  });

  await t.test("reductions: full, axis, negative axis, non-contiguous", () => {
    const a = randomTensor([3, 4], "f64");
    const aPath = saveTensor(dir, "a-red", a);
    assertClose(a.sum(), runOracle(dir, { op: "sum", inputs: [aPath] }), "sum");
    assertClose(a.mean(), runOracle(dir, { op: "mean", inputs: [aPath] }), "mean");
    assertClose(a.sum(0), runOracle(dir, { op: "sum", inputs: [aPath], axis: 0 }), "sum");
    assertClose(a.sum(1), runOracle(dir, { op: "sum", inputs: [aPath], axis: 1 }), "sum");
    assertClose(a.mean(1), runOracle(dir, { op: "mean", inputs: [aPath], axis: 1 }), "mean");

    const aT = a.permute([1, 0]);
    const aTPath = saveTensor(dir, "aT-red", aT);
    assertClose(
      aT.sum(0),
      runOracle(dir, { op: "sum", inputs: [aTPath], axis: 0 }),
      "sum",
    );
  });

  await t.test("i64 add matches NumPy int64 exactly", () => {
    const a = randomTensor([6], "i64");
    const b = randomTensor([6], "i64");
    const expected = runOracle(dir, {
      op: "add",
      inputs: [saveTensor(dir, "a-i64", a), saveTensor(dir, "b-i64", b)],
    });
    assert.equal(expected.dtype, "i64");
    assertClose(a.add(b), expected, "add");
  });

  await t.test("arange matches NumPy", () => {
    const expected = runOracle(dir, {
      op: "arange",
      arange: [0, 10, 3],
      dtype: "float32",
    });
    assertClose(Tensor.arange(0, 10, 3), expected, "arange");
  });

  await t.test("NumPy reads our .npy; we read NumPy's (permute round-trip)", () => {
    const a = randomTensor([2, 3, 4], "f32");
    const expected = runOracle(dir, {
      op: "permute",
      inputs: [saveTensor(dir, "a-perm", a)],
      permutation: [2, 0, 1],
    });
    assertClose(a.permute([2, 0, 1]).contiguous(), expected, "permute");
  });

  await t.test("slice matches NumPy, incl. negative step and partial specs", () => {
    const a = randomTensor([6, 5], "f64");
    const aPath = saveTensor(dir, "a-slice", a);

    const cases: Array<Array<[number | null, number | null, number | null] | null>> = [
      [[1, 5, 2], null],
      [null, [0, 4, 1]],
      [[null, null, -1], null], // reverse axis 0
      [[-4, -1, 1], [1, null, null]],
    ];
    for (const specs of cases) {
      const expected = runOracle(dir, { op: "slice", inputs: [aPath], specs });
      const jsSpecs = specs.map((s) =>
        s === null
          ? null
          : {
              start: s[0] ?? undefined,
              end: s[1] ?? undefined,
              step: s[2] ?? undefined,
            },
      );
      assertClose(a.slice(...jsSpecs).contiguous(), expected, "slice");
    }
  });

  await t.test("slice on a non-contiguous (permuted) view matches NumPy", () => {
    const a = randomTensor([4, 3], "f64");
    const aT = a.permute([1, 0]); // non-contiguous
    const aTPath = saveTensor(dir, "aT-slice", aT); // toNpy packs it
    const specs: Array<[number | null, number | null, number | null] | null> = [
      [0, 2, 1],
      [null, null, -1],
    ];
    const expected = runOracle(dir, { op: "slice", inputs: [aTPath], specs });
    const sliced = aT.slice(
      { start: 0, end: 2 },
      { step: -1 },
    );
    assertClose(sliced.contiguous(), expected, "slice");
  });
});
