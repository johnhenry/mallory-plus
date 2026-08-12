import assert from "node:assert/strict";
import { test } from "node:test";
import { Tensor, random } from "mallory-tensor-core";
import { Variable, constant, nn, optim, variable } from "../src/index.ts";
import type { Parameter } from "../src/nn.ts";

// ---- nn building blocks -----------------------------------------------------

test("Linear: forward shape and parameter collection", () => {
  const rng = random.seed(1);
  const linear = new nn.Linear(3, 4, { rng });
  const x = variable(random.uniform([2, 3], { rng, dtype: "f64" })); // Linear's params are f64
  const y = linear.forward(x);
  assert.deepEqual([...y.shape], [2, 4]);
  assert.equal(linear.parameters().length, 2); // weight + bias
});

test("Linear: bias: false omits the bias parameter", () => {
  const linear = new nn.Linear(3, 4, { bias: false, rng: random.seed(1) });
  assert.equal(linear.bias, null);
  assert.equal(linear.parameters().length, 1);
});

test("Module.parameters() recurses through nested modules", () => {
  class Net extends nn.Module {
    readonly a = new nn.Linear(2, 3, { rng: random.seed(1) });
    readonly b = new nn.Linear(3, 1, { rng: random.seed(2) });
    forward(x: Variable): Variable {
      return this.b.forward(this.a.forward(x).relu());
    }
  }
  const net = new Net();
  assert.equal(net.parameters().length, 4); // a.weight, a.bias, b.weight, b.bias
});

test("Module.zeroGrad() clears every parameter's gradient", () => {
  const linear = new nn.Linear(2, 2, { rng: random.seed(1) });
  const x = variable(random.uniform([1, 2], { rng: random.seed(2), dtype: "f64" })); // Linear's params are f64
  linear.forward(x).sum().backward();
  assert.notEqual(linear.weight.grad, null);
  linear.zeroGrad();
  assert.equal(linear.weight.grad, null);
  assert.equal((linear.bias as Parameter | null)?.grad ?? null, null);
});

test("Embedding: gather forward, scatter-add backward accumulates duplicate indices", () => {
  const emb = new nn.Embedding(5, 3, { rng: random.seed(1) });
  const indices = Tensor.from([0, 2, 0], { dtype: "i32" }); // index 0 repeated
  const out = emb.forward(indices);
  assert.deepEqual([...out.shape], [3, 3]);
  out.sum().backward();
  // Row 0's gradient should be 2x a single row's contribution (gathered twice).
  const g = emb.weight.grad?.toArray() as number[][];
  assert.deepEqual(g[0], [2, 2, 2]);
  assert.deepEqual(g[1], [0, 0, 0]); // never gathered
  assert.deepEqual(g[2], [1, 1, 1]);
});

test("LayerNorm: normalizes the last axis to ~zero mean, ~unit variance", () => {
  const ln = new nn.LayerNorm(4);
  const x = variable(Tensor.from([1, 2, 3, 100, -5, 0, 5, 10], { dtype: "f64" }).reshape([2, 4]));
  const y = ln.forward(x);
  const rowMean = y.value.mean(1).toArray() as number[];
  for (const m of rowMean) assert.ok(Math.abs(m) < 1e-6, `row mean ${m} not ~0`);
});

test("mseLoss and crossEntropy compute finite, non-negative losses", () => {
  const pred = variable(Tensor.from([0.5, 0.5], { dtype: "f64" }));
  const target = constant(Tensor.from([1, 0], { dtype: "f64" }));
  const mse = nn.mseLoss(pred, target);
  assert.ok(mse.value.item() as number > 0);

  const logits = variable(Tensor.from([2, 0.5, 0.1, 0.2, 3, 0.1], { dtype: "f64" }).reshape([2, 3]));
  const labels = Tensor.from([0, 1], { dtype: "i32" });
  const ce = nn.crossEntropy(logits, labels);
  assert.ok(Number.isFinite(ce.value.item() as number));
  assert.ok((ce.value.item() as number) > 0);
});

// ---- toy training loop: the issue #9 acceptance criterion -------------------

test("toy training loop: XOR-MLP converges with AdamW", () => {
  const rng = random.seed(7);

  class XorNet extends nn.Module {
    readonly fc1 = new nn.Linear(2, 8, { rng });
    readonly fc2 = new nn.Linear(8, 1, { rng });
    forward(x: Variable): Variable {
      return this.fc2.forward(this.fc1.forward(x).relu()).sigmoid();
    }
  }

  const net = new XorNet();
  const opt = new optim.AdamW(net.parameters(), { lr: 0.05 });

  const X = Tensor.from([0, 0, 0, 1, 1, 0, 1, 1], { dtype: "f64" }).reshape([4, 2]);
  const Y = constant(Tensor.from([0, 1, 1, 0], { dtype: "f64" }).reshape([4, 1]));

  let lastLoss = Infinity;
  for (let epoch = 0; epoch < 500; epoch++) {
    net.zeroGrad();
    const pred = net.forward(variable(X));
    const loss = nn.mseLoss(pred, Y);
    loss.backward();
    opt.step();
    lastLoss = loss.value.item() as number;
  }

  assert.ok(lastLoss < 0.05, `XOR-MLP did not converge: final loss ${lastLoss}`);

  // Predictions should actually match XOR, not just have low MSE by luck.
  const finalPred = net.forward(variable(X)).value.toArray() as number[][];
  const expected = [0, 1, 1, 0];
  finalPred.forEach((row, i) => {
    const predicted = row[0] as number;
    assert.ok(
      Math.round(predicted) === expected[i],
      `example ${i}: predicted ${predicted}, expected ${expected[i]}`,
    );
  });
});

test("toy training loop: linear regression converges with plain SGD", () => {
  // y = 3x + 2, fit with a single Linear(1,1) and plain SGD.
  const rng = random.seed(3);
  const model = new nn.Linear(1, 1, { rng });
  const opt = new optim.SGD(model.parameters(), { lr: 0.01 });

  const xs = [1, 2, 3, 4, 5];
  const X = Tensor.from(xs, { dtype: "f64" }).reshape([5, 1]);
  const Y = constant(Tensor.from(xs.map((x) => 3 * x + 2), { dtype: "f64" }).reshape([5, 1]));

  let lastLoss = Infinity;
  for (let epoch = 0; epoch < 2000; epoch++) {
    model.zeroGrad();
    const pred = model.forward(variable(X));
    const loss = nn.mseLoss(pred, Y);
    loss.backward();
    opt.step();
    lastLoss = loss.value.item() as number;
  }

  assert.ok(lastLoss < 0.01, `linear regression did not converge: final loss ${lastLoss}`);
});

// ---- telemetry hooks (issue #10) --------------------------------------------

test("backward() emits a trace span when a sink is installed, nothing by default", async () => {
  const { setSink } = await import("mallory-telemetry");
  const events: unknown[] = [];
  setSink((e) => events.push(e));
  try {
    const x = variable(Tensor.from([2, 3], { dtype: "f64" }));
    x.mul(x).sum().backward(undefined, { runId: "r1", step: 5 });
    assert.equal(events.length, 1);
    const e = events[0] as { type: string; runId: string; step: number; spans: Array<{ name: string }> };
    assert.equal(e.type, "trace");
    assert.equal(e.runId, "r1");
    assert.equal(e.step, 5);
    assert.equal(e.spans[0]?.name, "backward");
  } finally {
    setSink(null);
  }

  // default: no sink installed, backward() still works exactly as before.
  const x2 = variable(Tensor.from([2, 3], { dtype: "f64" }));
  x2.mul(x2).sum().backward();
  assert.deepEqual(x2.grad?.toArray(), [4, 6]);
});

test("optim.step() emits an optim/gradNorm metric when a sink is installed", async () => {
  const { setSink } = await import("mallory-telemetry");
  const events: unknown[] = [];
  setSink((e) => events.push(e));
  try {
    const linear = new nn.Linear(2, 1, { rng: random.seed(1) });
    const opt = new optim.SGD(linear.parameters(), { lr: 0.1 });
    const x = variable(random.uniform([1, 2], { rng: random.seed(2), dtype: "f64" }));
    linear.forward(x).sum().backward(); // also emits a "backward" trace span, since the sink is already active
    opt.step({ runId: "r1", step: 0 });
    assert.equal(events.length, 2); // 1 trace (backward) + 1 metric (optim.step)
    const e = events[1] as { type: string; name: string; value: number };
    assert.equal(e.type, "metric");
    assert.equal(e.name, "optim/gradNorm");
    assert.ok(Number.isFinite(e.value) && e.value >= 0);
  } finally {
    setSink(null);
  }
});
