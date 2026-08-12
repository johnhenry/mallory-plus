/**
 * A small, closed, typed IR (issue #11) — elementwise/broadcast nodes only.
 * Deliberately NOT a general compiler: the node kind set is fixed (add/sub/
 * mul/div plus a curated unary set matching what tensor-autograd already
 * supports), so both the forward evaluator and its derivative are total
 * functions over a closed enum. This is the shared lowering target the
 * future WebGPU kernel DSL (#12) and the Symbolic bridge (#15) reuse.
 */

export type UnaryOp = "neg" | "relu" | "sigmoid" | "gelu" | "exp" | "log" | "sqrt";
export type BinaryOp = "add" | "sub" | "mul" | "div";

export type IRNode =
  | { kind: "input"; index: number }
  | { kind: "const"; value: number }
  | { kind: "unary"; op: UnaryOp; arg: IRNode }
  | { kind: "binary"; op: BinaryOp; left: IRNode; right: IRNode };

/**
 * Forward value AND per-input local derivative, computed together in one
 * recursive pass (forward-mode AD over the IR) — a single source of truth
 * for the math, rather than a separate forward evaluator and backward
 * evaluator that could drift apart. `grad[k]` is d(this node)/d(input k).
 */
export interface ValueAndGrad {
  value: number;
  grad: number[];
}

function zeros(n: number): number[] {
  return new Array(n).fill(0);
}

export function evalWithGrad(node: IRNode, inputs: readonly number[], numInputs: number): ValueAndGrad {
  switch (node.kind) {
    case "input": {
      const grad = zeros(numInputs);
      grad[node.index] = 1;
      return { value: inputs[node.index] as number, grad };
    }
    case "const":
      return { value: node.value, grad: zeros(numInputs) };
    case "unary": {
      const a = evalWithGrad(node.arg, inputs, numInputs);
      let value: number;
      let deriv: number; // d(this node)/d(a.value)
      switch (node.op) {
        case "neg":
          value = -a.value;
          deriv = -1;
          break;
        case "relu":
          value = a.value > 0 ? a.value : 0;
          deriv = a.value > 0 ? 1 : 0;
          break;
        case "sigmoid": {
          const s = 1 / (1 + Math.exp(-a.value));
          value = s;
          deriv = s * (1 - s);
          break;
        }
        case "gelu": {
          // Same tanh-approximation derivative as Variable.gelu()'s backward.
          const c = Math.sqrt(2 / Math.PI);
          const x = a.value;
          const inner = c * (x + 0.044715 * x ** 3);
          const t = Math.tanh(inner);
          value = 0.5 * x * (1 + t);
          const sech2 = 1 - t * t;
          const dInner = c * (1 + 3 * 0.044715 * x * x);
          deriv = 0.5 * (1 + t) + 0.5 * x * sech2 * dInner;
          break;
        }
        case "exp":
          value = Math.exp(a.value);
          deriv = value;
          break;
        case "log":
          value = Math.log(a.value);
          deriv = 1 / a.value;
          break;
        case "sqrt":
          value = Math.sqrt(a.value);
          deriv = 1 / (2 * value);
          break;
      }
      return { value, grad: a.grad.map((g) => g * deriv) };
    }
    case "binary": {
      const l = evalWithGrad(node.left, inputs, numInputs);
      const r = evalWithGrad(node.right, inputs, numInputs);
      let value: number;
      let grad: number[];
      switch (node.op) {
        case "add":
          value = l.value + r.value;
          grad = l.grad.map((g, i) => g + (r.grad[i] as number));
          break;
        case "sub":
          value = l.value - r.value;
          grad = l.grad.map((g, i) => g - (r.grad[i] as number));
          break;
        case "mul":
          value = l.value * r.value;
          grad = l.grad.map((g, i) => g * r.value + (r.grad[i] as number) * l.value);
          break;
        case "div":
          value = l.value / r.value;
          grad = l.grad.map(
            (g, i) => (g * r.value - l.value * (r.grad[i] as number)) / (r.value * r.value),
          );
          break;
      }
      return { value, grad };
    }
  }
}

/** Ergonomic IR-graph builder — mirrors Variable's method style, but builds a node tree instead of executing. */
export class Traced {
  readonly node: IRNode;

  constructor(node: IRNode) {
    this.node = node;
  }

  static input(index: number): Traced {
    return new Traced({ kind: "input", index });
  }

  #binary(op: BinaryOp, other: Traced | number): Traced {
    return new Traced({ kind: "binary", op, left: this.node, right: toNode(other) });
  }

  add(other: Traced | number): Traced {
    return this.#binary("add", other);
  }
  sub(other: Traced | number): Traced {
    return this.#binary("sub", other);
  }
  mul(other: Traced | number): Traced {
    return this.#binary("mul", other);
  }
  div(other: Traced | number): Traced {
    return this.#binary("div", other);
  }
  neg(): Traced {
    return new Traced({ kind: "unary", op: "neg", arg: this.node });
  }
  relu(): Traced {
    return new Traced({ kind: "unary", op: "relu", arg: this.node });
  }
  sigmoid(): Traced {
    return new Traced({ kind: "unary", op: "sigmoid", arg: this.node });
  }
  gelu(): Traced {
    return new Traced({ kind: "unary", op: "gelu", arg: this.node });
  }
  exp(): Traced {
    return new Traced({ kind: "unary", op: "exp", arg: this.node });
  }
  log(): Traced {
    return new Traced({ kind: "unary", op: "log", arg: this.node });
  }
  sqrt(): Traced {
    return new Traced({ kind: "unary", op: "sqrt", arg: this.node });
  }
}

function toNode(x: Traced | number): IRNode {
  return typeof x === "number" ? { kind: "const", value: x } : x.node;
}
