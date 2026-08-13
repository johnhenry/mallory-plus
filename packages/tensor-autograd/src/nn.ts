/**
 * Minimal nn slice (issue #9): Parameter, Module, Linear/Embedding/
 * LayerNorm, mseLoss/crossEntropy. Plain TypeScript class composition, no
 * decorators or metaclass-style registration — `Module.parameters()` finds
 * `Parameter`/`Module` instances via a reflection pass over own properties,
 * per the source design's explicit preference.
 */
import { Tensor, random, type Rng } from "mallory-tensor-core";
import { Variable, constant } from "./variable.ts";

/** A leaf Variable that always requires grad and is collected by `Module.parameters()`. */
export class Parameter extends Variable {
  constructor(value: Tensor) {
    super(value, true, null);
  }
}

export abstract class Module {
  // Not `...inputs: Variable[]`: Embedding legitimately takes a raw integer-
  // index Tensor (not a differentiable Variable), so the base signature
  // stays loose and each subclass narrows its own `forward`.
  abstract forward(...inputs: unknown[]): Variable;

  /** Recursively collects every `Parameter` reachable through own-property fields. */
  parameters(): Parameter[] {
    const found: Parameter[] = [];
    for (const key of Object.keys(this)) {
      const v = (this as unknown as Record<string, unknown>)[key];
      if (v instanceof Parameter) found.push(v);
      else if (v instanceof Module) found.push(...v.parameters());
    }
    return found;
  }

  /**
   * Same reflection walk as {@link parameters}, but keyed by dotted path
   * (e.g. `"layer1.weight"` for a `Parameter` nested inside a sub-`Module`
   * field named `layer1`) — the basis for {@link stateDict}/
   * {@link loadStateDict} (issue #42), where a checkpoint needs to know
   * WHICH parameter each saved tensor belongs to, not just a flat list.
   */
  namedParameters(): Record<string, Parameter> {
    const found: Record<string, Parameter> = {};
    for (const key of Object.keys(this)) {
      const v = (this as unknown as Record<string, unknown>)[key];
      if (v instanceof Parameter) {
        found[key] = v;
      } else if (v instanceof Module) {
        for (const [subKey, p] of Object.entries(v.namedParameters())) {
          found[`${key}.${subKey}`] = p;
        }
      }
    }
    return found;
  }

  /** Every named parameter's current (detached) value — a plain, serializable snapshot. See {@link loadStateDict} for the inverse. */
  stateDict(): Record<string, Tensor> {
    const out: Record<string, Tensor> = {};
    for (const [name, p] of Object.entries(this.namedParameters())) out[name] = p.value;
    return out;
  }

  /**
   * Reassigns each named `Parameter`'s mutable `.value` from `dict` (the
   * SAME "leaf reassignment between steps" mechanism `optim.*` already uses
   * — see `Variable.value`'s own doc comment: a JS object-reference repoint,
   * not an in-place Tensor mutation). Throws naming any parameter this
   * module has that's missing from `dict`, or any `dict` key that doesn't
   * match a real parameter — a checkpoint silently loading onto the wrong
   * architecture is exactly the kind of mistake that should be loud, not
   * silently partial.
   */
  loadStateDict(dict: Readonly<Record<string, Tensor>>): void {
    const named = this.namedParameters();
    const moduleKeys = new Set(Object.keys(named));
    const dictKeys = new Set(Object.keys(dict));
    for (const key of moduleKeys) {
      if (!dictKeys.has(key)) throw new Error(`loadStateDict: missing parameter "${key}" in the given state dict`);
    }
    for (const key of dictKeys) {
      if (!moduleKeys.has(key)) throw new Error(`loadStateDict: state dict has unexpected parameter "${key}" (not in this module)`);
    }
    for (const [name, p] of Object.entries(named)) {
      p.value = dict[name] as Tensor;
    }
  }

  zeroGrad(): void {
    for (const p of this.parameters()) p.zeroGrad();
  }
}

export class Linear extends Module {
  readonly weight: Parameter;
  readonly bias: Parameter | null;

  constructor(
    inFeatures: number,
    outFeatures: number,
    options: { bias?: boolean; rng?: Rng } = {},
  ) {
    super();
    const useBias = options.bias ?? true;
    // PyTorch's nn.Linear default init: uniform(-k, k), k = 1/sqrt(inFeatures).
    const k = 1 / Math.sqrt(inFeatures);
    this.weight = new Parameter(
      random.uniform([inFeatures, outFeatures], { min: -k, max: k, dtype: "f64", rng: options.rng }),
    );
    this.bias = useBias
      ? new Parameter(
          random.uniform([outFeatures], { min: -k, max: k, dtype: "f64", rng: options.rng }),
        )
      : null;
  }

  forward(x: Variable): Variable {
    const y = x.matmul(this.weight);
    return this.bias ? y.add(this.bias) : y;
  }
}

export class Embedding extends Module {
  readonly weight: Parameter;

  constructor(numEmbeddings: number, embeddingDim: number, options: { rng?: Rng } = {}) {
    super();
    this.weight = new Parameter(
      random.normal([numEmbeddings, embeddingDim], { std: 1, dtype: "f64", rng: options.rng }),
    );
  }

  /**
   * `indices`: integer tensor of row indices to gather. Backward is a
   * scatter-add of the incoming gradient's rows back into the matching
   * weight rows (accumulating duplicates) — tensor-core has no native
   * scatter-add primitive yet, so this loops over plain arrays. Fine for
   * v1/toy-scale embedding tables; a real scatter-add kernel is future work.
   */
  forward(indices: Tensor): Variable {
    const idxArray = [...(indices.toArray() as (number | bigint)[])].map(Number);
    const gathered = this.weight.value.take(idxArray, { axis: 0 });
    const [numEmbeddings, embeddingDim] = this.weight.value.shape as [number, number];

    return Variable.fromOp(gathered, [this.weight], (g) => {
      const gRows = g.contiguous().toArray() as number[][];
      const acc: number[][] = Array.from({ length: numEmbeddings }, () =>
        new Array(embeddingDim).fill(0),
      );
      idxArray.forEach((rowIdx, i) => {
        for (let d = 0; d < embeddingDim; d++) {
          (acc[rowIdx] as number[])[d] += (gRows[i] as number[])[d] as number;
        }
      });
      return [
        Tensor.from(acc.flat(), { dtype: this.weight.value.dtype }).reshape([
          numEmbeddings,
          embeddingDim,
        ]),
      ];
    });
  }
}

export class LayerNorm extends Module {
  readonly weight: Parameter;
  readonly bias: Parameter;
  readonly eps: number;

  constructor(normalizedShape: number, options: { eps?: number } = {}) {
    super();
    this.eps = options.eps ?? 1e-5;
    this.weight = new Parameter(Tensor.ones([normalizedShape], { dtype: "f64" }));
    this.bias = new Parameter(Tensor.zeros([normalizedShape], { dtype: "f64" }));
  }

  /** Normalizes over the LAST axis of `x`. */
  forward(x: Variable): Variable {
    const axis = x.ndim - 1;
    const mean = x.mean(axis).unsqueeze(axis);
    const centered = x.sub(mean);
    const variance = centered.mul(centered).mean(axis).unsqueeze(axis);
    const std = variance.add(this.eps).sqrt();
    const normalized = centered.div(std);
    // weight/bias shape [normalizedShape] broadcasts against [..., normalizedShape]
    // via ordinary trailing-axis alignment — no unsqueeze needed here.
    return normalized.mul(this.weight).add(this.bias);
  }
}

/** Mean squared error. */
export function mseLoss(prediction: Variable, target: Variable): Variable {
  const diff = prediction.sub(target);
  return diff.mul(diff).mean();
}

/**
 * Cross-entropy from raw logits (shape `[batch, numClasses]`) and integer
 * class labels (shape `[batch]`). Built on the already gradient-checked
 * `softmax`/`log` ops rather than a hand-rolled log-softmax fusion — softmax
 * is numerically stable (subtracts the row max), and its output is safely
 * > 0 for the logit magnitudes a toy training loop produces.
 */
export function crossEntropy(logits: Variable, labels: Tensor): Variable {
  const [batchSize, numClasses] = logits.shape as [number, number];
  const labelIdx = [...(labels.toArray() as (number | bigint)[])].map(Number);
  const onehotData = new Array(batchSize * numClasses).fill(0);
  labelIdx.forEach((cls, row) => {
    onehotData[row * numClasses + cls] = 1;
  });
  const onehot = constant(
    Tensor.from(onehotData, { dtype: logits.dtype }).reshape([batchSize, numClasses]),
  );

  const logProbs = logits.softmax(1).log();
  const perExampleLoss = logProbs.mul(onehot).sum(1).mul(-1);
  return perExampleLoss.mean();
}
