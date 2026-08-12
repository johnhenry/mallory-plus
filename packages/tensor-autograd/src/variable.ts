/**
 * mallory-tensor-autograd — reverse-mode automatic differentiation over
 * mallory-tensor-core `Tensor`s (issue #8).
 *
 * Design: `Tensor` stays plain and immutable (tensor-core has no grad
 * bookkeeping fields, by design — it's the pure numeric core). `Variable`
 * wraps a `Tensor` and adds the tape: a define-by-run reverse-mode graph,
 * built as ops execute, walked backward on `.backward()`.
 *
 * Naming note: the source design sketched `Tensor.variable(x)` as the
 * grad-enabled constructor. That would require tensor-core to depend on
 * autograd, which is the wrong direction — the equivalent here is the
 * standalone `variable(x)` export (aliased as `Variable.variable(x)`).
 *
 * Non-differentiable ops (argmax, sort, comparisons, ...) simply aren't
 * `Variable` methods in v1 — call them on `.value` (a plain Tensor) instead,
 * which returns a plain Tensor with no grad tracking. That's an unambiguous
 * way to satisfy "must not silently produce wrong gradients" without a
 * runtime throw for an operation nothing calls.
 *
 * Non-goal 7 (no in-place ops on tracked tensors) is satisfied structurally:
 * `Variable` has no in-place API at all, matching tensor-core's own
 * immutable style — there's nothing to reject at runtime because the
 * mutating method doesn't exist.
 */
import { Tensor, type Axis } from "mallory-tensor-core";
import { sumToShape } from "./shape-utils.ts";

let gradEnabled = true;

export function isGradEnabled(): boolean {
  return gradEnabled;
}

/** Run `fn` with gradient tracking disabled (inference mode) — PyTorch's `torch.no_grad()`. */
export function noGrad<T>(fn: () => T): T {
  const prev = gradEnabled;
  gradEnabled = false;
  try {
    return fn();
  } finally {
    gradEnabled = prev;
  }
}

/** Run `fn` with gradient tracking forced on, even inside an enclosing `noGrad`. */
export function enableGrad<T>(fn: () => T): T {
  const prev = gradEnabled;
  gradEnabled = true;
  try {
    return fn();
  } finally {
    gradEnabled = prev;
  }
}

interface TapeNode {
  inputs: readonly Variable[];
  backwardFn: (gradOutput: Tensor) => readonly (Tensor | undefined)[];
}

let nextId = 0;

export class Variable {
  readonly value: Tensor;
  readonly requiresGrad: boolean;
  grad: Tensor | null = null;
  readonly id: number;
  readonly node: TapeNode | null;

  private constructor(value: Tensor, requiresGrad: boolean, node: TapeNode | null) {
    this.value = value;
    this.requiresGrad = requiresGrad;
    this.node = node;
    this.id = nextId++;
  }

  /** A leaf that accumulates gradients on `.backward()`. */
  static variable(value: Tensor): Variable {
    return new Variable(value, true, null);
  }

  /** A leaf that never accumulates gradients — wrap a plain Tensor to use it in graph ops. */
  static constant(value: Tensor): Variable {
    return new Variable(value, false, null);
  }

  /** Build a traced non-leaf result. Used internally by every op method below. */
  static fromOp(
    value: Tensor,
    inputs: readonly Variable[],
    backwardFn: (gradOutput: Tensor) => readonly (Tensor | undefined)[],
  ): Variable {
    const tracked = isGradEnabled() && inputs.some((v) => v.requiresGrad || v.node !== null);
    return new Variable(value, false, tracked ? { inputs, backwardFn } : null);
  }

  get shape(): Tensor["shape"] {
    return this.value.shape;
  }
  get dtype(): Tensor["dtype"] {
    return this.value.dtype;
  }
  get ndim(): number {
    return this.value.ndim;
  }

  /** Same value, no gradient history — cuts the tape at this point. */
  detach(): Variable {
    return Variable.constant(this.value);
  }

  zeroGrad(): void {
    this.grad = null;
  }

  /**
   * Walk the graph reachable from `this` in reverse topological order,
   * accumulating into every requires-grad leaf's `.grad`. `gradOutput`
   * defaults to a ones-tensor and requires `this` be scalar (size 1) —
   * matching PyTorch's `.backward()` contract.
   */
  backward(gradOutput?: Tensor): void {
    const seed =
      gradOutput ??
      (this.value.size === 1
        ? Tensor.ones(this.value.shape, { dtype: this.value.dtype })
        : (() => {
            throw new RangeError(
              "backward() with no argument requires a scalar (size-1) output; pass an explicit gradOutput for non-scalar tensors",
            );
          })());

    const order: Variable[] = [];
    const visited = new Set<number>();
    const visit = (v: Variable): void => {
      if (visited.has(v.id)) return;
      visited.add(v.id);
      if (v.node) for (const input of v.node.inputs) visit(input);
      order.push(v);
    };
    visit(this);

    const grads = new Map<number, Tensor>();
    grads.set(this.id, seed);

    for (let i = order.length - 1; i >= 0; i--) {
      const v = order[i] as Variable;
      const g = grads.get(v.id);
      if (g === undefined) continue; // not reachable via any actual gradient path
      if (v.requiresGrad) {
        v.grad = v.grad ? v.grad.add(g) : g;
      }
      if (v.node) {
        const inputGrads = v.node.backwardFn(g);
        v.node.inputs.forEach((input, idx) => {
          const contribution = inputGrads[idx];
          if (contribution === undefined) return;
          const existing = grads.get(input.id);
          grads.set(input.id, existing ? existing.add(contribution) : contribution);
        });
      }
    }
  }

  // ---- differentiable ops ----------------------------------------------------

  add(other: Variable): Variable {
    const value = this.value.add(other.value);
    return Variable.fromOp(value, [this, other], (g) => [
      sumToShape(g, this.value.shape),
      sumToShape(g, other.value.shape),
    ]);
  }

  sub(other: Variable): Variable {
    const value = this.value.sub(other.value);
    return Variable.fromOp(value, [this, other], (g) => [
      sumToShape(g, this.value.shape),
      sumToShape(g.mul(-1), other.value.shape),
    ]);
  }

  mul(other: Variable): Variable {
    const value = this.value.mul(other.value);
    return Variable.fromOp(value, [this, other], (g) => [
      sumToShape(g.mul(other.value), this.value.shape),
      sumToShape(g.mul(this.value), other.value.shape),
    ]);
  }

  div(other: Variable): Variable {
    const value = this.value.div(other.value);
    return Variable.fromOp(value, [this, other], (g) => [
      sumToShape(g.div(other.value), this.value.shape),
      // d/db (a/b) = -a/b^2
      sumToShape(g.mul(this.value).div(other.value.mul(other.value)).mul(-1), other.value.shape),
    ]);
  }

  /** 2-D only in v1 — batched/1-D matmul gradients are a follow-up, not scoped here. */
  matmul(other: Variable): Variable {
    if (this.value.ndim !== 2 || other.value.ndim !== 2) {
      throw new TypeError("Variable.matmul: only 2-D operands are supported in v1");
    }
    const value = this.value.matmul(other.value);
    return Variable.fromOp(value, [this, other], (g) => [
      g.matmul(other.value.transpose()),
      this.value.transpose().matmul(g),
    ]);
  }

  sum(axis?: Axis): Variable {
    const value = this.value.sum(axis);
    return Variable.fromOp(value, [this], (g) => {
      if (axis === undefined) return [g.broadcastTo(this.value.shape).contiguous()];
      const ax = axis < 0 ? axis + this.value.ndim : axis;
      return [g.unsqueeze(ax).broadcastTo(this.value.shape).contiguous()];
    });
  }

  mean(axis?: Axis): Variable {
    const value = this.value.mean(axis);
    const count =
      axis === undefined
        ? this.value.size
        : (this.value.shape[axis < 0 ? axis + this.value.ndim : axis] as number);
    return Variable.fromOp(value, [this], (g) => {
      const scaled = g.div(count);
      if (axis === undefined) return [scaled.broadcastTo(this.value.shape).contiguous()];
      const ax = axis < 0 ? axis + this.value.ndim : axis;
      return [scaled.unsqueeze(ax).broadcastTo(this.value.shape).contiguous()];
    });
  }

  relu(): Variable {
    const value = this.value.relu();
    return Variable.fromOp(value, [this], (g) => {
      const mask = this.value.gt(0).cast(this.value.dtype);
      return [g.mul(mask)];
    });
  }

  sigmoid(): Variable {
    const value = this.value.sigmoid();
    return Variable.fromOp(value, [this], (g) => {
      const oneMinusValue = value.mul(-1).add(1);
      return [g.mul(value).mul(oneMinusValue)];
    });
  }

  /**
   * GELU backward via the tanh-approximation's exact derivative. Uses the
   * identity tanh(x) = 2*sigmoid(2x)-1 to avoid needing a tensor-core
   * `.tanh()` method for a single niche gradient.
   */
  gelu(): Variable {
    const value = this.value.gelu();
    return Variable.fromOp(value, [this], (g) => {
      const c = Math.sqrt(2 / Math.PI);
      const x = this.value;
      const x2 = x.mul(x);
      const x3 = x2.mul(x);
      const inner = x.add(x3.mul(0.044715)).mul(c);
      const t = inner.mul(2).sigmoid().mul(2).add(-1); // tanh(inner)
      const sech2 = t.mul(t).mul(-1).add(1); // 1 - tanh^2(inner)
      const dInner = x2.mul(3 * 0.044715).add(1).mul(c); // c*(1 + 3*0.044715*x^2)
      const derivative = t.add(1).mul(0.5).add(x.mul(0.5).mul(sech2).mul(dInner));
      return [g.mul(derivative)];
    });
  }

  softmax(axis: Axis = -1): Variable {
    const value = this.value.softmax(axis);
    return Variable.fromOp(value, [this], (g) => {
      const ax = axis < 0 ? axis + this.value.ndim : axis;
      const dot = g.mul(value).sum(ax).unsqueeze(ax).broadcastTo(this.value.shape).contiguous();
      return [value.mul(g.sub(dot))];
    });
  }
}

/** `Tensor.variable(x)` from the source design — see the naming note above. */
export function variable(value: Tensor): Variable {
  return Variable.variable(value);
}

export function constant(value: Tensor): Variable {
  return Variable.constant(value);
}
