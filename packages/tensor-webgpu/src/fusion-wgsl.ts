/**
 * IR -> WGSL lowering (issue #12, "Elementwise fusion via the #11 IR").
 *
 * `mallory-tensor-compile`'s `IRNode` (packages/tensor-compile/src/ir.ts) is a
 * small, closed, elementwise/broadcast-only IR that its own module doc calls
 * "the shared lowering target the future WebGPU kernel DSL (#12) ... reuses".
 * This module is that second backend: instead of `evalWithGrad`'s recursive
 * CPU interpreter (one JS function call per node, per element), `compileIRToWGSL`
 * walks the SAME `IRNode` tree once and emits a single WGSL expression string —
 * every op in the traced expression becomes part of ONE compute shader's body,
 * so a chain like `a.add(b).mul(c).relu()` dispatches once and touches each
 * output element exactly once, with zero intermediate GPU buffers (the
 * "elementwise fusion" this issue asks for, at v1 scope: one shader body per
 * traced expression, not arena/liveness planning across shaders).
 *
 * v1 lowers the FORWARD value only (matching `CompiledFn.forward()`), not the
 * per-input local derivative `evalWithGrad` also computes — gradient-on-GPU
 * (`forwardWithGrad`'s WGSL equivalent) is future scope. The forward value is
 * exactly what the correctness oracle in test/fusion.test.ts cross-checks:
 * running this WGSL shader on a real (or software) GPUAdapter and comparing
 * elementwise against `evalWithGrad(...).value` computed on the CPU for the
 * same IR and the same inputs — an independent-backend-same-IR cross-check,
 * the same "two independently-implemented consumers of one IR must agree"
 * pattern this repo already uses for autograd (DualNumber vs. the reverse-mode
 * tape).
 *
 * Every `UnaryOp`/`BinaryOp`/`CmpOp` in the IR is covered so no compiled
 * expression silently falls back to a partial/CPU path. WGSL 1.0 (WebGPU's
 * shading language) has no builtin `sinh`/`cosh`/`coth`/`sech`/`csch`/inverse
 * hyperbolics/`erf`/`cbrt`/`log10`, so those are expanded to the same formulas
 * `unaryValueAndDeriv` in tensor-compile/src/ir.ts uses (documented per-op
 * below), not re-derived independently — same math, different backend.
 */
import type { BinaryOp, CmpOp, IRNode, UnaryOp } from "mallory-tensor-compile";

/** WGSL source `x` is bound to the node's own value; produces an `f32` expression string (parenthesized, safe to splice into a larger expression without precedence surprises). */
function unaryExpr(op: UnaryOp, x: string): string {
  switch (op) {
    case "neg":
      return `(-(${x}))`;
    case "relu":
      return `(max(${x}, 0.0))`;
    case "sigmoid":
      return `(1.0 / (1.0 + exp(-(${x}))))`;
    case "gelu":
      // Same tanh approximation as Tensor.gelu()/unaryValueAndDeriv's "gelu" case.
      return `(0.5 * (${x}) * (1.0 + tanh(0.7978845608028654 * ((${x}) + 0.044715 * (${x}) * (${x}) * (${x})))))`;
    case "exp":
      return `(exp(${x}))`;
    case "log":
      return `(log(${x}))`;
    case "sqrt":
      return `(sqrt(${x}))`;
    case "sin":
      return `(sin(${x}))`;
    case "cos":
      return `(cos(${x}))`;
    case "tan":
      return `(tan(${x}))`;
    case "asin":
      return `(asin(${x}))`;
    case "acos":
      return `(acos(${x}))`;
    case "atan":
      return `(atan(${x}))`;
    // WGSL 1.0 has no sinh/cosh builtin; expand via exp (matches Math.sinh/cosh).
    case "sinh":
      return `((exp(${x}) - exp(-(${x}))) * 0.5)`;
    case "cosh":
      return `((exp(${x}) + exp(-(${x}))) * 0.5)`;
    case "tanh":
      return `(tanh(${x}))`;
    case "cot":
      return `(1.0 / tan(${x}))`;
    case "sec":
      return `(1.0 / cos(${x}))`;
    case "csc":
      return `(1.0 / sin(${x}))`;
    // Inverse hyperbolics: log-form definitions (same identities Math.asinh
    // etc. use internally).
    case "asinh":
      return `(log(${x} + sqrt((${x}) * (${x}) + 1.0)))`;
    case "acosh":
      return `(log(${x} + sqrt((${x}) * (${x}) - 1.0)))`;
    case "atanh":
      return `(0.5 * log((1.0 + (${x})) / (1.0 - (${x}))))`;
    case "coth":
      return `((exp(${x}) + exp(-(${x}))) / (exp(${x}) - exp(-(${x}))))`;
    case "sech":
      return `(2.0 / (exp(${x}) + exp(-(${x}))))`;
    case "csch":
      return `(2.0 / (exp(${x}) - exp(-(${x}))))`;
    case "acot":
      return `(atan(1.0 / (${x})))`;
    case "asec":
      return `(acos(1.0 / (${x})))`;
    case "acsc":
      return `(asin(1.0 / (${x})))`;
    case "acoth":
      return `(0.5 * log(((${x}) + 1.0) / ((${x}) - 1.0)))`;
    case "asech":
      return `(log((1.0 + sqrt(1.0 - (${x}) * (${x}))) / (${x})))`;
    case "acsch":
      return `(log(1.0 / (${x}) + sqrt(1.0 / ((${x}) * (${x})) + 1.0)))`;
    case "abs":
      return `(abs(${x}))`;
    // WGSL has no log10; log(x)/ln(10).
    case "log10":
      return `(log(${x}) / 2.302585092994046)`;
    case "log2":
      return `(log2(${x}))`;
    // WGSL has no cbrt; sign-preserving pow(|x|, 1/3) (pow's base must be >= 0).
    case "cbrt":
      return `(sign(${x}) * pow(abs(${x}), 0.3333333333333333))`;
    case "floor":
      return `(floor(${x}))`;
    case "ceil":
      return `(ceil(${x}))`;
    case "round":
      return `(round(${x}))`;
    case "sign":
      return `(sign(${x}))`;
    case "trunc":
      return `(trunc(${x}))`;
    case "expm1":
      return `(exp(${x}) - 1.0)`;
    case "log1p":
      return `(log(1.0 + (${x})))`;
    case "erf":
      return `(mallory_erf(${x}))`;
  }
}

function binaryExpr(op: BinaryOp, l: string, r: string): string {
  switch (op) {
    case "add":
      return `((${l}) + (${r}))`;
    case "sub":
      return `((${l}) - (${r}))`;
    case "mul":
      return `((${l}) * (${r}))`;
    case "div":
      return `((${l}) / (${r}))`;
    case "pow":
      return `(pow(${l}, ${r}))`;
    case "atan2":
      return `(atan2(${l}, ${r}))`;
    // WGSL has no hypot builtin; sqrt(a^2+b^2) (same tradeoff tensor-compile's
    // own note about this being "close enough" for v1 — no overflow guard).
    case "hypot":
      return `(sqrt((${l}) * (${l}) + (${r}) * (${r})))`;
    case "min":
      return `(min(${l}, ${r}))`;
    case "max":
      return `(max(${l}, ${r}))`;
  }
}

/** WGSL has no bool<->f32 implicit conversion; `select(falseValue, trueValue, cond)` (note the argument order — opposite of `a ? b : c`). */
function cmpExpr(op: CmpOp, l: string, r: string): string {
  const cond: Record<CmpOp, string> = {
    lt: `(${l}) < (${r})`,
    le: `(${l}) <= (${r})`,
    gt: `(${l}) > (${r})`,
    ge: `(${l}) >= (${r})`,
    eq: `(${l}) == (${r})`,
    ne: `(${l}) != (${r})`,
  };
  return `(select(0.0, 1.0, ${cond[op]}))`;
}

function lower(node: IRNode, inputVar: (index: number) => string): string {
  switch (node.kind) {
    case "input":
      return inputVar(node.index);
    case "const":
      return formatFloatLiteral(node.value);
    case "unary":
      return unaryExpr(node.op, lower(node.arg, inputVar));
    case "binary":
      return binaryExpr(node.op, lower(node.left, inputVar), lower(node.right, inputVar));
    case "cmp":
      return cmpExpr(node.op, lower(node.left, inputVar), lower(node.right, inputVar));
    case "select":
      // WGSL select(falseValue, trueValue, cond): cond must be `bool`, and our
      // "cond" sub-expression is itself an f32 (0.0/1.0, or any traced value —
      // `Traced.select`'s cond can be an arbitrary node, not just a `cmp`), so
      // compare it against 0.0 to get a real `bool`, matching evalWithGrad's
      // "cond.value !== 0" test.
      return `(select(${lower(node.else, inputVar)}, ${lower(node.then, inputVar)}, ${lower(node.cond, inputVar)} != 0.0))`;
  }
}

/** WGSL float literals need an explicit decimal point/exponent (bare integers are `i32`/`AbstractInt`); also renders non-finite values via the only portable route (0.0-division), since WGSL has no `NaN`/`Infinity` literal syntax. */
function formatFloatLiteral(value: number): string {
  if (Number.isNaN(value)) return "(0.0 / 0.0)";
  if (value === Infinity) return "(1.0 / 0.0)";
  if (value === -Infinity) return "(-1.0 / 0.0)";
  if (Number.isInteger(value)) return `${value}.0`;
  return `${value}`;
}

/**
 * `erf` via the same Abramowitz & Stegun 7.1.26 formula as tensor-compile's
 * `erf()` helper (|error| <= 1.5e-7) — kept as a real WGSL function (not
 * inlined per call site) since it's used at most once per compiled expression
 * but is long; inlining would bloat the shader source for no benefit.
 */
const ERF_WGSL_FN = `
fn mallory_erf(x: f32) -> f32 {
  let s: f32 = sign(x);
  let ax: f32 = abs(x);
  let t: f32 = 1.0 / (1.0 + 0.3275911 * ax);
  let poly: f32 = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  let y: f32 = 1.0 - poly * exp(-ax * ax);
  return s * y;
}
`;

export interface ElementwiseWGSL {
  /** Full compute shader module source — one `@compute` entry point named `main`, workgroup size 64. */
  code: string;
  /** Number of `array<f32>` storage buffers bound at group 0, bindings `[0, numInputs)`, read-only. */
  numInputs: number;
  /** Binding index of the write-only output `array<f32>` (always `numInputs`, i.e. immediately after the inputs). */
  outputBinding: number;
}

/**
 * Lower a traced elementwise expression (the same `IRNode` a `CompiledFn`
 * wraps) into a single WGSL compute shader: `numInputs` read-only storage
 * buffers in, one write-only storage buffer out, one invocation per output
 * element (flat 1-D dispatch — callers are responsible for broadcasting
 * inputs to the output shape and flattening to contiguous `Float32Array`s
 * first, mirroring `CompiledFn.forward()`'s own `#broadcastInputs` step,
 * which this module deliberately does NOT duplicate: broadcasting is a
 * layout concern, this function only compiles the math).
 */
export function compileIRToWGSL(node: IRNode, numInputs: number): ElementwiseWGSL {
  const usesErf = irUsesErf(node);
  const inputVar = (index: number): string => {
    if (index < 0 || index >= numInputs) {
      throw new RangeError(`IR references input ${index}, but numInputs is ${numInputs}`);
    }
    return `input${index}[gid.x]`;
  };
  const bindings = Array.from(
    { length: numInputs },
    (_, i) => `@group(0) @binding(${i}) var<storage, read> input${i}: array<f32>;`,
  ).join("\n");
  const outputBinding = numInputs;
  const expr = lower(node, inputVar);
  // Phony-assign every declared input (WGSL `_ = expr;`): with pipeline
  // layout "auto", a binding the shader never STATICALLY references is
  // excluded from the generated layout, so createBindGroup (which binds
  // every buffer the caller passed) fails validation -- and WebGPU
  // validation errors are asynchronous, so the dispatch silently no-ops
  // and readback returns ALL ZEROS. Found by the #58 randomized fuzzer on
  // its first run (an IR graph is free to ignore some of its declared
  // inputs; every hand-written fixed test happened to use all of them).
  const phonyUses = Array.from({ length: numInputs }, (_, i) => `  _ = input${i}[0];`).join("\n");
  const code = `${usesErf ? ERF_WGSL_FN : ""}
${bindings}
@group(0) @binding(${outputBinding}) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
${phonyUses}
  if (gid.x >= arrayLength(&output)) {
    return;
  }
  output[gid.x] = ${expr};
}
`;
  return { code, numInputs, outputBinding };
}

function irUsesErf(node: IRNode): boolean {
  switch (node.kind) {
    case "input":
    case "const":
      return false;
    case "unary":
      return node.op === "erf" || irUsesErf(node.arg);
    case "binary":
      return irUsesErf(node.left) || irUsesErf(node.right);
    case "cmp":
      return irUsesErf(node.left) || irUsesErf(node.right);
    case "select":
      return irUsesErf(node.cond) || irUsesErf(node.then) || irUsesErf(node.else);
  }
}
