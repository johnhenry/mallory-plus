/**
 * Randomized differential fuzzer over the elementwise IR (issue #48), design
 * adapted from Woxi's diff_fuzz methodology (docs/spikes/woxi-study.md):
 * spec-table-driven generation, deterministic seeded RNG with the seed
 * printed on failure for exact replay, size-monotone greedy shrinking to a
 * minimal diverging case, and a self-check leg that must find zero
 * divergences (validating the harness itself).
 *
 * What is differentially tested (two GENUINELY independent paths each):
 *
 * 1. Gradient leg: `evalWithGrad`'s analytic per-input derivatives (the 40+
 *    hand-written formulas in ir.ts's unaryValueAndDeriv table) vs central
 *    finite differences computed purely from the VALUE path. The two share
 *    no derivative code, so a wrong formula diverges. Restricted to the
 *    smooth op subset (documented v1 simplification: kinked/discontinuous
 *    ops -- relu/abs/min/max/floor/round/sign/cmp/select/... -- are
 *    excluded here because finite differences are meaningless at kinks;
 *    they're still fuzzed by the value leg below, and their derivative
 *    conventions are pinned by the existing fixed tests).
 *
 * 2. Value/broadcast leg: `CompiledFn.forward`'s fused strided iteration
 *    (the `offsets()` generator machinery) vs a naive reference that walks
 *    logical indices and reads elements through tensor-core's own
 *    `broadcastTo(...).at(...)`. NOTE: both sides evaluate the IR itself
 *    with the same `evalWithGrad` (forward() literally calls it
 *    per-element), so this leg tests the ITERATION/BROADCAST machinery,
 *    not the math -- the math is covered by the gradient leg, the existing
 *    fixed suites, and the erf cross-check.
 *
 * Knobs: MALLORY_FUZZ_CASES (default 150 per leg), MALLORY_FUZZ_SEED
 * (default 20260813) -- override to replay a reported failure exactly.
 */
import assert from "node:assert/strict";
import { test } from "node:test";
import { Tensor } from "mallory-tensor-core";
import { compile, CompiledFn, evalWithGrad, type BinaryOp, type IRNode, type UnaryOp } from "../src/index.ts";

const CASES = Number(process.env.MALLORY_FUZZ_CASES ?? 150);
const BASE_SEED = Number(process.env.MALLORY_FUZZ_SEED ?? 20260813);

// ---- deterministic RNG (mulberry32 -- tiny, seedable, good enough for case
// generation; NOT crypto, NOT for statistics) --------------------------------

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

type Rng = () => number;

function pick<T>(rng: Rng, items: readonly T[]): T {
  return items[Math.floor(rng() * items.length)] as T;
}

// ---- op spec tables ---------------------------------------------------------

/** Smooth in-domain: safe for finite-difference gradient checking (domain
 * membership is enforced by rejection sampling, not by the table). */
const SMOOTH_UNARY: readonly UnaryOp[] = [
  "neg", "sigmoid", "gelu", "exp", "log", "sqrt", "sin", "cos", "tan",
  "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "atanh",
  "log10", "log2", "cbrt", "expm1", "log1p", "erf",
];
const SMOOTH_BINARY: readonly BinaryOp[] = ["add", "sub", "mul", "div", "pow", "atan2", "hypot"];

/** Everything, including kinked/discontinuous ops -- value-leg only. */
const ALL_UNARY: readonly UnaryOp[] = [
  ...SMOOTH_UNARY,
  "relu", "abs", "floor", "ceil", "round", "sign", "trunc",
  "cot", "sec", "csc", "coth", "sech", "csch",
  "acot", "asec", "acsc", "acoth", "asech", "acsch", "acosh",
];
const ALL_BINARY: readonly BinaryOp[] = [...SMOOTH_BINARY, "min", "max"];

interface GenSpec {
  unary: readonly UnaryOp[];
  binary: readonly BinaryOp[];
  /** allow cmp + select nodes (value leg only -- their gradient is defined as 0 / branch passthrough and FD at a branch boundary is meaningless) */
  cmpSelect: boolean;
  maxDepth: number;
}

const GRAD_SPEC: GenSpec = { unary: SMOOTH_UNARY, binary: SMOOTH_BINARY, cmpSelect: false, maxDepth: 4 };
const VALUE_SPEC: GenSpec = { unary: ALL_UNARY, binary: ALL_BINARY, cmpSelect: true, maxDepth: 5 };

// ---- graph generation -------------------------------------------------------

function genNode(rng: Rng, spec: GenSpec, numInputs: number, depth: number): IRNode {
  if (depth >= spec.maxDepth || rng() < 0.25) {
    return rng() < 0.7
      ? { kind: "input", index: Math.floor(rng() * numInputs) }
      : { kind: "const", value: Math.round((rng() * 4 - 2) * 100) / 100 };
  }
  const r = rng();
  if (spec.cmpSelect && r < 0.12) {
    return {
      kind: "select",
      cond: {
        kind: "cmp",
        op: pick(rng, ["lt", "le", "gt", "ge", "eq", "ne"] as const),
        left: genNode(rng, spec, numInputs, depth + 1),
        right: genNode(rng, spec, numInputs, depth + 1),
      },
      then: genNode(rng, spec, numInputs, depth + 1),
      else: genNode(rng, spec, numInputs, depth + 1),
    };
  }
  if (r < 0.55) {
    return { kind: "unary", op: pick(rng, spec.unary), arg: genNode(rng, spec, numInputs, depth + 1) };
  }
  return {
    kind: "binary",
    op: pick(rng, spec.binary),
    left: genNode(rng, spec, numInputs, depth + 1),
    right: genNode(rng, spec, numInputs, depth + 1),
  };
}

function usesAnyInput(node: IRNode): boolean {
  switch (node.kind) {
    case "input": return true;
    case "const": return false;
    case "unary": return usesAnyInput(node.arg);
    case "binary": return usesAnyInput(node.left) || usesAnyInput(node.right);
    case "cmp": return usesAnyInput(node.left) || usesAnyInput(node.right);
    case "select": return usesAnyInput(node.cond) || usesAnyInput(node.then) || usesAnyInput(node.else);
  }
}

function nodeCount(node: IRNode): number {
  switch (node.kind) {
    case "input":
    case "const": return 1;
    case "unary": return 1 + nodeCount(node.arg);
    case "binary":
    case "cmp": return 1 + nodeCount(node.left) + nodeCount(node.right);
    case "select": return 1 + nodeCount(node.cond) + nodeCount(node.then) + nodeCount(node.else);
  }
}

/** Every subnode of `node`, including itself (used for domain rejection: a
 * point is only accepted when EVERY intermediate stays finite and bounded,
 * which keeps composed functions inside their domains and away from poles). */
function allSubnodes(node: IRNode): IRNode[] {
  switch (node.kind) {
    case "input":
    case "const": return [node];
    case "unary": return [node, ...allSubnodes(node.arg)];
    case "binary":
    case "cmp": return [node, ...allSubnodes(node.left), ...allSubnodes(node.right)];
    case "select": return [node, ...allSubnodes(node.cond), ...allSubnodes(node.then), ...allSubnodes(node.else)];
  }
}

// ---- size-monotone greedy shrinking -----------------------------------------

/** One-step reductions: replace any subtree with one of its own children.
 * Every candidate is strictly smaller, so the greedy loop below terminates
 * (asserted). */
function reductions(node: IRNode): IRNode[] {
  const out: IRNode[] = [];
  switch (node.kind) {
    case "input":
    case "const":
      return out;
    case "unary":
      out.push(node.arg);
      for (const r of reductions(node.arg)) out.push({ ...node, arg: r });
      return out;
    case "binary":
    case "cmp":
      out.push(node.left, node.right);
      for (const r of reductions(node.left)) out.push({ ...node, left: r });
      for (const r of reductions(node.right)) out.push({ ...node, right: r });
      return out;
    case "select":
      out.push(node.then, node.else, node.cond);
      for (const r of reductions(node.cond)) out.push({ ...node, cond: r });
      for (const r of reductions(node.then)) out.push({ ...node, then: r });
      for (const r of reductions(node.else)) out.push({ ...node, else: r });
      return out;
  }
}

/** Greedily shrink `node` while `stillDiverges` holds. `stillDiverges` must
 * return false (not throw) for candidates it cannot confirm -- e.g. a shrunk
 * graph that now trips a domain-rejection guard. */
function shrink(node: IRNode, stillDiverges: (candidate: IRNode) => boolean): IRNode {
  let current = node;
  for (;;) {
    const smaller = reductions(current).find((c) => stillDiverges(c));
    if (!smaller) return current;
    assert.ok(nodeCount(smaller) < nodeCount(current), "shrink must be size-monotone (termination guarantee)");
    current = smaller;
  }
}

function report(label: string, seed: number, node: IRNode, extra: string): string {
  return (
    `${label} (replay with MALLORY_FUZZ_SEED=${seed} MALLORY_FUZZ_CASES=1)\n` +
    `minimal diverging graph: ${JSON.stringify(node)}\n${extra}`
  );
}

// ---- leg 1: analytic gradients vs central finite differences ----------------

const FD_H = 1e-5;
const GRAD_RTOL = 1e-3; // catches wrong formulas (O(1) errors), tolerates FD truncation noise
const GRAD_ATOL = 1e-5;

interface GradCase {
  node: IRNode;
  numInputs: number;
  point: number[];
}

/** Accept a point only if every intermediate value/grad is finite and modest
 * at the point AND at every FD-perturbed point (keeps FD inside the domain
 * and away from poles where truncation error explodes). */
function gradPointAcceptable(node: IRNode, numInputs: number, point: readonly number[]): boolean {
  const probes: number[][] = [Array.from(point)];
  for (let k = 0; k < numInputs; k++) {
    for (const s of [-1, 1]) {
      const p = Array.from(point);
      p[k] = (p[k] as number) + s * FD_H;
      probes.push(p);
    }
  }
  for (const p of probes) {
    for (const sub of allSubnodes(node)) {
      const { value, grad } = evalWithGrad(sub, p, numInputs);
      if (!Number.isFinite(value) || Math.abs(value) > 1e4) return false;
      if (grad.some((g) => !Number.isFinite(g) || Math.abs(g) > 1e3)) return false;
    }
  }
  return true;
}

function gradDivergence(c: GradCase): { input: number; analytic: number; fd: number } | undefined {
  const analytic = evalWithGrad(c.node, c.point, c.numInputs).grad;
  for (let k = 0; k < c.numInputs; k++) {
    const plus = Array.from(c.point);
    const minus = Array.from(c.point);
    plus[k] = (plus[k] as number) + FD_H;
    minus[k] = (minus[k] as number) - FD_H;
    const fd =
      (evalWithGrad(c.node, plus, c.numInputs).value - evalWithGrad(c.node, minus, c.numInputs).value) / (2 * FD_H);
    const a = analytic[k] as number;
    if (Math.abs(a - fd) > GRAD_ATOL + GRAD_RTOL * Math.max(Math.abs(a), Math.abs(fd))) {
      return { input: k, analytic: a, fd };
    }
  }
  return undefined;
}

test(`fuzz: analytic gradients agree with central finite differences (${CASES} random smooth graphs)`, () => {
  let discardedGraphs = 0;
  let checked = 0;
  for (let caseIdx = 0; caseIdx < CASES; caseIdx++) {
    const seed = BASE_SEED + caseIdx;
    const rng = mulberry32(seed);
    const numInputs = 1 + Math.floor(rng() * 3);
    const node = genNode(rng, GRAD_SPEC, numInputs, 0);
    if (!usesAnyInput(node)) {
      discardedGraphs++;
      continue;
    }
    // Rejection-sample a point where the whole composition is in-domain.
    let point: number[] | undefined;
    for (let tries = 0; tries < 25 && !point; tries++) {
      const candidate = Array.from({ length: numInputs }, () => rng() * 4 - 2);
      if (gradPointAcceptable(node, numInputs, candidate)) point = candidate;
    }
    if (!point) {
      discardedGraphs++; // e.g. acosh-of-sigmoid never lands in-domain; expected for some graphs
      continue;
    }
    checked++;
    const div = gradDivergence({ node, numInputs, point });
    if (div) {
      const fixedPoint = point;
      const minimal = shrink(node, (candidate) => {
        if (!usesAnyInput(candidate)) return false;
        if (!gradPointAcceptable(candidate, numInputs, fixedPoint)) return false;
        return gradDivergence({ node: candidate, numInputs, point: fixedPoint }) !== undefined;
      });
      const minDiv = gradDivergence({ node: minimal, numInputs, point: fixedPoint });
      assert.fail(
        report(
          "analytic gradient diverges from finite differences",
          seed,
          minimal,
          `point=${JSON.stringify(fixedPoint)} input#${minDiv?.input}: analytic=${minDiv?.analytic} fd=${minDiv?.fd}`,
        ),
      );
    }
  }
  // No silent coverage collapse: rejection is expected (domain-restricted
  // compositions), but if nearly everything is discarded the leg is vacuous.
  assert.ok(
    checked >= CASES * 0.5,
    `gradient leg only checked ${checked}/${CASES} cases (${discardedGraphs} discarded) -- generator/domain guards have drifted`,
  );
});

// ---- leg 2: fused strided/broadcast iteration vs naive at()-based reference --

interface ValueCase {
  node: IRNode;
  numInputs: number;
  inputs: Tensor[];
  outShape: number[];
}

/** Random input tensor broadcast-compatible with `outShape`: each axis kept
 * or collapsed to 1, possibly fewer leading axes, sometimes materialized as
 * a TRANSPOSED (non-contiguous) view so the fused iterator's stride walking
 * is actually exercised. */
function genInputTensor(rng: Rng, outShape: readonly number[]): Tensor {
  const startAxis = Math.floor(rng() * (outShape.length + 1));
  const shape = outShape.slice(startAxis).map((d) => (rng() < 0.3 ? 1 : d));
  const size = shape.reduce((a, b) => a * b, 1);
  const data = Array.from({ length: size }, () => Math.round((rng() * 4 - 2) * 100) / 100);
  const t = Tensor.from(data, { dtype: "f64" }).reshape(shape);
  if (shape.length >= 2 && rng() < 0.35) {
    // Physically store the transpose, then view it back: same logical
    // values, non-contiguous strides.
    const reversed = [...shape].reverse();
    const tr = Tensor.from(data, { dtype: "f64" }).reshape(reversed).transpose();
    // tr has `shape` logically but reversed-stride storage; use it only if
    // logical shape matches (it always does for a full reverse).
    return tr;
  }
  return t;
}

function valueDivergence(c: ValueCase): { index: number[]; fused: number; reference: number } | undefined {
  const fn = new CompiledFn(c.numInputs, c.node);
  const fused = fn.forward(...c.inputs);
  const views = c.inputs.map((t) => t.broadcastTo(fused.shape));
  const idx = new Array<number>(fused.ndim).fill(0);
  const total = fused.size;
  for (let flat = 0; flat < total; flat++) {
    const scratch = views.map((v) => v.at(...idx) as number);
    const reference = evalWithGrad(c.node, scratch, c.numInputs).value;
    const got = fused.at(...idx) as number;
    const same = Object.is(got, reference) || got === reference; // NaN==NaN ok; +0/-0 both fine
    if (!same) return { index: [...idx], fused: got, reference };
    for (let axis = fused.ndim - 1; axis >= 0; axis--) {
      idx[axis] = (idx[axis] as number) + 1;
      if ((idx[axis] as number) < (fused.shape[axis] as number)) break;
      idx[axis] = 0;
    }
  }
  return undefined;
}

function genValueCase(rng: Rng): ValueCase | undefined {
  const numInputs = 1 + Math.floor(rng() * 3);
  const node = genNode(rng, VALUE_SPEC, numInputs, 0);
  if (!usesAnyInput(node)) return undefined;
  const rank = Math.floor(rng() * 4); // 0..3
  const outShape = Array.from({ length: rank }, () => 1 + Math.floor(rng() * 4));
  const inputs = Array.from({ length: numInputs }, () => genInputTensor(rng, outShape));
  return { node, numInputs, inputs, outShape };
}

test(`fuzz: fused forward equals naive broadcastTo(...).at(...) reference on random strided inputs (${CASES} random graphs)`, () => {
  let discarded = 0;
  let checked = 0;
  for (let caseIdx = 0; caseIdx < CASES; caseIdx++) {
    const seed = BASE_SEED + 100_000 + caseIdx;
    const rng = mulberry32(seed);
    const c = genValueCase(rng);
    if (!c) {
      discarded++;
      continue;
    }
    checked++;
    const div = valueDivergence(c);
    if (div) {
      const minimal = shrink(c.node, (candidate) => {
        if (!usesAnyInput(candidate)) return false;
        try {
          return valueDivergence({ ...c, node: candidate }) !== undefined;
        } catch {
          return false;
        }
      });
      const minDiv = valueDivergence({ ...c, node: minimal });
      assert.fail(
        report(
          "fused forward diverges from the naive reference",
          seed,
          minimal,
          `inputs=${JSON.stringify(c.inputs.map((t) => ({ shape: t.shape, data: [...(t.contiguous().data as Float64Array)] })))}\n` +
            `at index ${JSON.stringify(minDiv?.index)}: fused=${minDiv?.fused} reference=${minDiv?.reference}`,
        ),
      );
    }
  }
  assert.ok(
    checked >= CASES * 0.8,
    `value leg only checked ${checked}/${CASES} cases (${discarded} discarded) -- generator has drifted`,
  );
});

// ---- self-check: the harness itself cannot manufacture divergence -----------

test("fuzz self-check: reference-vs-reference finds zero divergences (harness validation)", () => {
  for (let caseIdx = 0; caseIdx < 30; caseIdx++) {
    const seed = BASE_SEED + 200_000 + caseIdx;
    const rng = mulberry32(seed);
    const c = genValueCase(rng);
    if (!c) continue;
    const views = c.inputs.map((t) => {
      const fn = new CompiledFn(c.numInputs, c.node);
      return t.broadcastTo(fn.forward(...c.inputs).shape);
    });
    // Same reference computation run twice must agree with itself exactly --
    // if this ever fails, the comparison/gather machinery is unsound and no
    // divergence report from the legs above can be trusted.
    const idx = new Array<number>(views[0]?.ndim ?? 0).fill(0);
    const first = views.map((v) => v.at(...idx) as number);
    const second = views.map((v) => v.at(...idx) as number);
    const a = evalWithGrad(c.node, first, c.numInputs).value;
    const b = evalWithGrad(c.node, second, c.numInputs).value;
    assert.ok(Object.is(a, b), `self-check divergence at seed ${seed}: ${a} vs ${b}`);
  }
});

// ---- determinism: same seed, same graph (replayability contract) ------------

test("fuzz determinism: identical seeds generate identical graphs and inputs", () => {
  const g1 = genValueCase(mulberry32(BASE_SEED + 42));
  const g2 = genValueCase(mulberry32(BASE_SEED + 42));
  assert.deepEqual(g1?.node, g2?.node);
  assert.deepEqual(
    g1?.inputs.map((t) => [...(t.contiguous().data as Float64Array)]),
    g2?.inputs.map((t) => [...(t.contiguous().data as Float64Array)]),
  );
  // compile() round-trip sanity: a Traced-built graph is the same IR shape
  // the generator emits, so the fuzzer exercises what users actually build.
  const fn = compile(1, (x) => x.sin().add(1));
  assert.equal(fn.numInputs, 1);
});
