/**
 * mallory-tensor-compile (issue #11) — trace once via {@link compile}, execute
 * fused: one pass over the output with zero intermediate Tensor allocations,
 * instead of one allocation per op in the chain.
 *
 * v1 scope is deliberately elementwise/broadcast-only (no reductions, no
 * matmul) — the same IR this package builds is meant to be reused as the
 * lowering target for the future WebGPU kernel DSL (#12) and the Symbolic
 * bridge (#15), so it stays small and closed rather than growing ad hoc.
 */
import { broadcastShapes, Tensor, type DType, type Shape } from "mallory-tensor-core";
import { sumToShape, Variable } from "mallory-tensor-autograd";
import { evalWithGrad, Traced, type IRNode } from "./ir.ts";

export { Traced, evalWithGrad, type IRNode, type UnaryOp, type BinaryOp } from "./ir.ts";

const FLOAT_DTYPES: readonly DType[] = ["f32", "f64"];

/** Walk storage offsets of every element of `t` in C-order — mirrors Tensor's private `elementOffsets()`, built from its public shape/strides/offset contract since compile-time fusion lives outside the class. */
function* offsets(t: Tensor): Generator<number> {
  if (t.size === 0) return;
  const ndim = t.ndim;
  if (ndim === 0) {
    yield t.offset;
    return;
  }
  const index = new Array<number>(ndim).fill(0);
  let off = t.offset;
  outer: for (;;) {
    yield off;
    for (let axis = ndim - 1; axis >= 0; axis--) {
      index[axis] = (index[axis] as number) + 1;
      off += t.strides[axis] as number;
      if ((index[axis] as number) < (t.shape[axis] as number)) continue outer;
      index[axis] = 0;
      off -= (t.shape[axis] as number) * (t.strides[axis] as number);
    }
    return;
  }
}

interface Broadcasted {
  outShape: Shape;
  dtype: DType;
  broadcasted: Tensor[];
}

/** A compiled elementwise expression, built by {@link compile}. Executing it never touches the autograd tape or the eager `Tensor`/`Variable` op methods unless you call {@link CompiledFn.asVariableOp} — plain `.forward()` is a pure fused computation (non-goal 6: strictly opt-in). */
export class CompiledFn {
  readonly numInputs: number;
  private readonly output: IRNode;

  constructor(numInputs: number, output: IRNode) {
    this.numInputs = numInputs;
    this.output = output;
  }

  #broadcastInputs(tensors: readonly Tensor[]): Broadcasted {
    if (tensors.length !== this.numInputs) {
      throw new RangeError(
        `compiled function expects ${this.numInputs} input(s), got ${tensors.length}`,
      );
    }
    const first = tensors[0];
    if (!first) throw new RangeError("compiled function requires at least one input");
    const dtype = first.dtype;
    if (!FLOAT_DTYPES.includes(dtype)) {
      throw new TypeError(
        `mallory-tensor-compile v1 supports floating dtypes only (f32/f64); got ${dtype}`,
      );
    }
    for (const t of tensors) {
      if (t.dtype !== dtype) {
        throw new TypeError(
          `mallory-tensor-compile: all inputs must share one dtype (M1 has no implicit promotion); got ${tensors.map((x) => x.dtype).join(", ")}`,
        );
      }
    }
    let outShape: Shape = first.shape;
    for (const t of tensors.slice(1)) outShape = broadcastShapes(outShape, t.shape);
    const broadcasted = tensors.map((t) => t.broadcastTo(outShape));
    return { outShape, dtype, broadcasted };
  }

  /** Fused forward value only — one pass over the output, no intermediate Tensors materialized. */
  forward(...tensors: Tensor[]): Tensor {
    const { outShape, dtype, broadcasted } = this.#broadcastInputs(tensors);
    const out = Tensor.zeros(outShape, { dtype });
    const outData = out.data as Float32Array | Float64Array;
    const iters = broadcasted.map((t) => offsets(t));
    const scratch = new Array<number>(this.numInputs);
    for (const outOff of offsets(out)) {
      for (let k = 0; k < this.numInputs; k++) {
        const step = (iters[k] as Generator<number>).next();
        scratch[k] = (broadcasted[k] as Tensor).data[step.value as number] as number;
      }
      outData[outOff] = evalWithGrad(this.output, scratch, this.numInputs).value;
    }
    return out;
  }

  /**
   * Fused forward value AND per-input local derivatives, in one pass.
   * `localGrads[k]` has the broadcast OUTPUT shape (not yet reduced to
   * input `k`'s original shape, and not yet multiplied by any upstream
   * gradient) — {@link CompiledFn.asVariableOp} does both of those, mirroring
   * how every hand-written `Variable` op backward already works
   * (`sumToShape(gradOutput.mul(localDerivative), input.shape)`).
   */
  forwardWithGrad(...tensors: Tensor[]): { value: Tensor; localGrads: Tensor[] } {
    const { outShape, dtype, broadcasted } = this.#broadcastInputs(tensors);
    const value = Tensor.zeros(outShape, { dtype });
    const valueData = value.data as Float32Array | Float64Array;
    const localGrads = tensors.map(() => Tensor.zeros(outShape, { dtype }));
    const gradData = localGrads.map((g) => g.data as Float32Array | Float64Array);
    const iters = broadcasted.map((t) => offsets(t));
    const scratch = new Array<number>(this.numInputs);
    const outIter = offsets(value);
    for (const valueOff of outIter) {
      for (let k = 0; k < this.numInputs; k++) {
        const step = (iters[k] as Generator<number>).next();
        scratch[k] = (broadcasted[k] as Tensor).data[step.value as number] as number;
      }
      const result = evalWithGrad(this.output, scratch, this.numInputs);
      valueData[valueOff] = result.value;
      for (let k = 0; k < this.numInputs; k++) {
        (gradData[k] as Float32Array | Float64Array)[valueOff] = result.grad[k] as number;
      }
    }
    return { value, localGrads };
  }

  /**
   * Bridge into the autograd tape (issue #8): the fused computation becomes
   * a single `Variable.fromOp` node, so `backward()` sees one opaque op
   * rather than the unfused chain — satisfying the "transparent to the
   * tape" acceptance criterion without the tape ever knowing fusion
   * happened.
   */
  asVariableOp(): (...vars: Variable[]) => Variable {
    return (...vars: Variable[]) => {
      if (vars.length !== this.numInputs) {
        throw new RangeError(
          `compiled function expects ${this.numInputs} input(s), got ${vars.length}`,
        );
      }
      const tensors = vars.map((v) => v.value);
      const { value, localGrads } = this.forwardWithGrad(...tensors);
      return Variable.fromOp(value, vars, (gradOutput) =>
        tensors.map((t, k) => sumToShape(gradOutput.mul(localGrads[k] as Tensor), t.shape)),
      );
    };
  }
}

/**
 * Trace `fn` once over symbolic {@link Traced} inputs to build the fused IR,
 * returning a reusable {@link CompiledFn}. Tracing itself never touches a
 * real `Tensor` — `fn` only ever sees `Traced` nodes — so `compile()` has no
 * side effect on eager `Tensor`/`Variable` usage elsewhere (non-goal 6).
 */
export function compile(numInputs: number, fn: (...args: Traced[]) => Traced): CompiledFn {
  const inputs = Array.from({ length: numInputs }, (_, i) => Traced.input(i));
  const output = fn(...inputs);
  return new CompiledFn(numInputs, output.node);
}
