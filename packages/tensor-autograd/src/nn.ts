/**
 * Minimal nn slice (issue #9): Parameter, Module, Linear/Embedding/
 * LayerNorm, mseLoss/crossEntropy. Plain TypeScript class composition, no
 * decorators or metaclass-style registration — `Module.parameters()` finds
 * `Parameter`/`Module` instances via a reflection pass over own properties,
 * per the source design's explicit preference.
 */
import { Tensor, random, allocate, isBigIntDType, type AnyTypedArray, type Rng } from "mallory-tensor-core";
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
   *
   * Accumulation is sparse: contributions land in a `Map<rowIdx, Float64Array>`
   * keyed by the (few) touched rows, not a dense `numEmbeddings x embeddingDim`
   * table walked/filled on every backward call — that dense allocation used
   * to dominate cost (measured ~770ms for a batch-of-3 gradient into a
   * 50,000x256 table) regardless of how few rows the batch actually touched.
   * The only full-table-sized allocation left is the final zero-initialized
   * typed array `Tensor.fromTypedArray` needs (a native allocation, not a
   * JS-level fill loop), scattered into only at the touched rows.
   */
  forward(indices: Tensor): Variable {
    const idxArray = [...(indices.toArray() as (number | bigint)[])].map(Number);
    const gathered = this.weight.value.take(idxArray, { axis: 0 });
    const [numEmbeddings, embeddingDim] = this.weight.value.shape as [number, number];
    const dtype = this.weight.value.dtype;

    return Variable.fromOp(gathered, [this.weight], (g) => {
      const gRows = g.contiguous().toArray() as number[][];

      const acc = new Map<number, Float64Array>();
      idxArray.forEach((rowIdx, i) => {
        let row = acc.get(rowIdx);
        if (!row) {
          row = new Float64Array(embeddingDim);
          acc.set(rowIdx, row);
        }
        const gRow = gRows[i] as number[];
        for (let d = 0; d < embeddingDim; d++) {
          (row as Float64Array)[d] += gRow[d] as number;
        }
      });

      const flat = allocate(dtype, numEmbeddings * embeddingDim);
      if (isBigIntDType(dtype)) {
        const big = flat as unknown as { [i: number]: bigint };
        for (const [rowIdx, row] of acc) {
          const base = rowIdx * embeddingDim;
          for (let d = 0; d < embeddingDim; d++) {
            big[base + d] = BigInt(Math.trunc(row[d] as number));
          }
        }
      } else {
        const numeric = flat as Exclude<AnyTypedArray, BigInt64Array | BigUint64Array>;
        for (const [rowIdx, row] of acc) {
          const base = rowIdx * embeddingDim;
          numeric.set(row, base);
        }
      }

      return [Tensor.fromTypedArray(flat, [numEmbeddings, embeddingDim], { dtype })];
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

/**
 * Composes an ordered list of sub-modules, `forward` chaining them
 * (issue #71). Stores each layer as a NUMBERED own-property
 * (`this["0"]`, `this["1"]`, ...) rather than an array field — `Module`'s
 * existing `parameters()`/`namedParameters()` reflection walk only
 * recognizes `Parameter`/`Module` values on own properties (an array field
 * would be invisible to it), so this gets full `parameters()`/
 * `stateDict()`/`loadStateDict()` support with ZERO changes to the base
 * `Module` class. Dotted-path names come out as `"0.weight"`, `"1.weight"`,
 * etc. — the same convention PyTorch's own `nn.Sequential` uses.
 */
export class Sequential extends Module {
  readonly length: number;

  constructor(layers: readonly Module[]) {
    super();
    layers.forEach((layer, i) => {
      (this as unknown as Record<string, Module>)[String(i)] = layer;
    });
    this.length = layers.length;
  }

  forward(x: Variable): Variable {
    let out = x;
    for (let i = 0; i < this.length; i++) {
      out = (this as unknown as Record<string, Module>)[String(i)]!.forward(out) as Variable;
    }
    return out;
  }
}

/**
 * Inverted dropout (issue #71): zeroes each element independently with
 * probability `p`, scaling survivors by `1/(1-p)` so the expected output
 * magnitude is unchanged whether or not dropout is active — the standard
 * "inverted" convention (scale at train time, no-op at eval time, rather
 * than the reverse). `training` is an explicit `forward` parameter, not
 * module-level mode-switching state (`.train()`/`.eval()`) — this repo has
 * no such lifecycle elsewhere, and inventing one for just this module
 * would be scope beyond what's asked.
 */
export class Dropout extends Module {
  readonly p: number;

  constructor(p: number) {
    super();
    if (p < 0 || p >= 1) throw new RangeError(`Dropout: p must be in [0, 1), got ${p}`);
    this.p = p;
  }

  forward(x: Variable, training: boolean, options: { rng?: Rng } = {}): Variable {
    if (!training || this.p === 0) return x;
    // keep[i] = 1 with probability (1-p), else 0 -- P(uniform < p) = p is
    // exactly the drop event, so "keep" is the >= p side.
    const keepMask = random
      .uniform(x.shape, { min: 0, max: 1, dtype: x.dtype, rng: options.rng })
      .gte(this.p)
      .cast(x.dtype);
    const scale = 1 / (1 - this.p);
    return x.mul(constant(keepMask)).mul(scale);
  }
}

/** Mean squared error. */
export function mseLoss(prediction: Variable, target: Variable): Variable {
  const diff = prediction.sub(target);
  return diff.mul(diff).mean();
}

/**
 * Pseudo-Huber (Charbonnier) loss: `delta^2 * (sqrt(1 + ((pred-target)/delta)^2) - 1)`,
 * averaged. The smooth, fully-differentiable variant of Huber loss —
 * behaves like scaled L2 near zero and like scaled L1 far from zero
 * (Huber's whole point: quadratic near the optimum, linear/outlier-robust
 * far from it), but with a smooth transition instead of Huber's classic
 * hard piecewise switch at `delta`. Deliberate, not a shortcut: `Variable`
 * has no conditional/select op yet (see issue #64's own note on this), so
 * the piecewise form isn't buildable from existing ops without one; the
 * pseudo-Huber form needs only `sqrt`/`add`/`mul`/`div`, all already
 * gradient-checked.
 */
export function huberLoss(prediction: Variable, target: Variable, delta = 1): Variable {
  const diff = prediction.sub(target).mul(1 / delta); // Variable.div only accepts another Variable, not a scalar
  const inner = diff.mul(diff).add(1).sqrt().add(-1);
  return inner.mul(delta * delta).mean();
}

/**
 * Binary cross-entropy FROM RAW LOGITS (matches {@link crossEntropy}'s own
 * "from logits" contract — sigmoid applied internally, never fed a
 * pre-squashed probability), computed via the standard numerically-stable
 * "BCEWithLogits" formulation (issue #85):
 *
 * `L(z, y) = relu(z) - z*y + log(1 + exp(-|z|))`
 *
 * A prior version computed `p = sigmoid(z)` first, then `y*log(p) +
 * (1-y)*log(1-p)` — for `|z| >~ 37`, f64 `sigmoid` saturates to exactly
 * `1.0`/`0.0`, so the inactive side's factor becomes `0 * log(0) = 0 *
 * -Inf = NaN` (IEEE 754: `0 * Inf` is NaN regardless of the other
 * factor). This form never evaluates `log` at a saturating probability —
 * `log(1+exp(-|z|))` is rewritten as `-log(sigmoid(|z|))` (no `exp`/`abs`
 * Variable ops exist yet, so `|z|` itself is `relu(z) + relu(-z)`), and
 * `sigmoid(|z|)` is always `>= 0.5` for any finite `|z|`, so its `log`
 * never sees 0. Verified equal to the prior formula to ~1e-15 in the
 * non-saturated regime, and finite (not NaN/Infinity) at `|z|` up to at
 * least 100, before writing this.
 */
export function binaryCrossEntropy(logits: Variable, target: Variable): Variable {
  const absLogits = logits.relu().add(logits.mul(-1).relu());
  const negLogSigmoidAbs = absLogits.sigmoid().log().mul(-1);
  return logits.relu().sub(logits.mul(target)).add(negLogSigmoidAbs).mean();
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
