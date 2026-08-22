/**
 * trainer.configure/fit (issue #43) — a thin facade over the existing
 * forward -> loss -> backward -> step -> zeroGrad loop this package's own
 * "toy training loop" tests already hand-write (see nn.test.ts), not a new
 * training algorithm.
 *
 * `fit(dataLoader)` accepts any `AsyncIterable<{x: Tensor, y: Tensor}>` --
 * deliberately NOT a dependency on the `data` namespace (issue #22, itself
 * blocked on `@johnhenry/iteration`'s npm publish). A hand-rolled async
 * generator satisfies this interface just as well; `data`'s eventual output
 * would too, once it exists.
 *
 * `config.epochs` applies ONLY to the `fit({x, y})` full-batch overload
 * (repeating the same fixed tensor pair `epochs` times, matching the
 * existing toy-training-loop precedent exactly). For `fit(dataLoader)`,
 * `epochs` is ignored -- an arbitrary `AsyncIterable` isn't guaranteed
 * re-iterable (a hand-rolled async generator is exhausted after one pass),
 * so this facade makes exactly one pass over it per `fit()` call. A caller
 * wanting N epochs over a streaming loader should call `fit()` N times with
 * a fresh iterable each time, or use a loader that's genuinely re-iterable.
 */
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { constant, variable, type Variable } from "./variable.ts";
import type { Module } from "./nn.ts";

export interface Batch {
  readonly x: Tensor;
  readonly y: Tensor;
}

export interface Optimizer {
  step(): void;
  zeroGrad(): void;
}

export interface TrainerConfig {
  readonly model: Module;
  readonly optimizer: Optimizer;
  readonly lossFn: (prediction: Variable, target: Variable) => Variable;
  /** Only applies to `fit({x, y})` (see this module's doc comment); default 1. */
  readonly epochs?: number;
}

export interface FitResult {
  /** One loss value per training step, in order. */
  readonly lossHistory: number[];
}

/** `trainer.configure({model, optimizer, lossFn, epochs?})` -- the namespace-level entry point (see index.ts's `export * as trainer`). */
export function configure(config: TrainerConfig): Trainer {
  return new Trainer(config);
}

export class Trainer {
  private readonly model: Module;
  private readonly optimizer: Optimizer;
  private readonly lossFn: (prediction: Variable, target: Variable) => Variable;
  private readonly epochs: number;

  constructor(config: TrainerConfig) {
    this.model = config.model;
    this.optimizer = config.optimizer;
    this.lossFn = config.lossFn;
    this.epochs = config.epochs ?? 1;
  }

  async fit(data: Batch | AsyncIterable<Batch>): Promise<FitResult> {
    const lossHistory: number[] = [];
    if (isBatch(data)) {
      for (let epoch = 0; epoch < this.epochs; epoch++) {
        lossHistory.push(this.step(data.x, data.y));
      }
    } else {
      for await (const batch of data) {
        lossHistory.push(this.step(batch.x, batch.y));
      }
    }
    return { lossHistory };
  }

  private step(x: Tensor, y: Tensor): number {
    this.model.zeroGrad();
    const prediction = this.model.forward(variable(x));
    const loss = this.lossFn(prediction, constant(y));
    loss.backward();
    this.optimizer.step();
    return loss.value.item() as number;
  }
}

function isBatch(data: Batch | AsyncIterable<Batch>): data is Batch {
  return typeof data === "object" && data !== null && "x" in data && "y" in data;
}
