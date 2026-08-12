/**
 * Minimal optim slice (issue #9): AdamW (with plain SGD nearly free
 * alongside it, per the source design). Each optimizer keeps its own
 * per-parameter state keyed by `Parameter.id`, updating `param.value` in
 * place between steps (see the doc comment on `Variable.value`).
 */
import { Tensor } from "mallory-tensor-core";
import { hasSink, metric } from "mallory-telemetry";
import type { Parameter } from "./nn.ts";

/** Opt-in telemetry (issue #10) shared by every optimizer's step(). */
export interface StepOptions {
  runId?: string;
  step?: number;
}

/**
 * Global L2 grad norm across every parameter that has one — only computed
 * when a sink is installed (`hasSink()`), since it's extra tensor work
 * nothing needs on the hot path by default.
 */
function gradNorm(params: readonly Parameter[]): number {
  let sumSquares = 0;
  for (const p of params) {
    if (!p.grad) continue;
    sumSquares += (p.grad.mul(p.grad).sum().item() as number);
  }
  return Math.sqrt(sumSquares);
}

export class SGD {
  readonly params: readonly Parameter[];
  readonly lr: number;

  constructor(params: readonly Parameter[], options: { lr: number }) {
    this.params = params;
    this.lr = options.lr;
  }

  step(options: StepOptions = {}): void {
    for (const p of this.params) {
      if (!p.grad) continue;
      p.value = p.value.sub(p.grad.mul(this.lr));
    }
    if (hasSink()) {
      const runId = options.runId ?? "default";
      const step = options.step ?? 0;
      metric(runId, step, "optim/gradNorm", gradNorm(this.params));
    }
  }

  zeroGrad(): void {
    for (const p of this.params) p.zeroGrad();
  }
}

interface AdamWState {
  m: Tensor; // first moment
  v: Tensor; // second moment
  step: number;
}

export class AdamW {
  readonly params: readonly Parameter[];
  readonly lr: number;
  readonly beta1: number;
  readonly beta2: number;
  readonly eps: number;
  readonly weightDecay: number;
  readonly #state = new Map<number, AdamWState>();

  constructor(
    params: readonly Parameter[],
    options: { lr: number; beta1?: number; beta2?: number; eps?: number; weightDecay?: number },
  ) {
    this.params = params;
    this.lr = options.lr;
    this.beta1 = options.beta1 ?? 0.9;
    this.beta2 = options.beta2 ?? 0.999;
    this.eps = options.eps ?? 1e-8;
    this.weightDecay = options.weightDecay ?? 0.01;
  }

  step(options: StepOptions = {}): void {
    for (const p of this.params) {
      if (!p.grad) continue;
      let state = this.#state.get(p.id);
      if (!state) {
        state = {
          m: Tensor.zeros(p.value.shape, { dtype: p.value.dtype }),
          v: Tensor.zeros(p.value.shape, { dtype: p.value.dtype }),
          step: 0,
        };
        this.#state.set(p.id, state);
      }
      state.step += 1;

      const m = state.m.mul(this.beta1).add(p.grad.mul(1 - this.beta1));
      const v = state.v.mul(this.beta2).add(p.grad.mul(p.grad).mul(1 - this.beta2));
      state.m = m;
      state.v = v;

      const mHat = m.div(1 - this.beta1 ** state.step);
      const vHat = v.div(1 - this.beta2 ** state.step);

      // Decoupled weight decay (the "W" in AdamW): applied directly to the
      // parameter, not folded into the gradient before the moment estimates.
      const decayed = this.weightDecay !== 0 ? p.value.mul(1 - this.lr * this.weightDecay) : p.value;
      const update = mHat.div(vHat.sqrt().add(this.eps)).mul(this.lr);
      p.value = decayed.sub(update);
    }
    if (hasSink()) {
      const runId = options.runId ?? "default";
      const step = options.step ?? 0;
      metric(runId, step, "optim/gradNorm", gradNorm(this.params));
    }
  }

  zeroGrad(): void {
    for (const p of this.params) p.zeroGrad();
  }
}
