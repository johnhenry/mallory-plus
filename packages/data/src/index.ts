/**
 * mallory-data (issue #22): the v2 `data` namespace — async dataset
 * pipelines built on mallory-iteration, surfaced through a CURATED facade
 * rather than passing that library through raw. The curation decisions,
 * from the issue:
 *
 * - `chunk(n)` wraps upstream's `group(n)` transducer — renamed because
 *   `group` collides conceptually with `groupBy(keyFn)`, and in
 *   dataframe-adjacent code "group" strongly implies keying, not
 *   fixed-size chunking.
 * - `fold` is THE terminal reduce here (upstream `foldAsync`); upstream's
 *   `reduce*` names stay unexported (there they're the streaming
 *   transducer primitive — exporting both invites mixups).
 * - Upstream's `count*` helpers are deliberately NOT re-exported: in
 *   dataframe-land `count` means "how many rows", not an integer-sequence
 *   generator. (See the facade-contract test.)
 * - Bounded-concurrency decode/augment goes through upstream
 *   `mapConcurrentAsync` / `prefetchAsync` (never reimplemented), and
 *   cancellation is plain `AbortSignal` end to end.
 *
 * Power users can always `import ... from "mallory-iteration"` directly —
 * this facade is the supported surface, not a wall.
 */
import {
  abortable,
  foldAsync,
  mapConcurrentAsync,
  prefetchAsync,
  transducers,
  type ReducerStep,
} from "mallory-iteration";
import { Tensor } from "mallory-tensor-core";

type AnySource<T> = Iterable<T> | AsyncIterable<T>;
/** A dataset source: a (re-)iterable, or a factory producing a fresh
 * iteration — factories (and plain sync iterables, which are re-iterable by
 * nature) are what make `epochs()` possible. */
export type DatasetSource<T> = AnySource<T> | (() => AnySource<T>);

function isIterableObject<T>(x: unknown): x is AnySource<T> {
  return (
    typeof x === "object" && x !== null && (Symbol.asyncIterator in x || Symbol.iterator in x)
  );
}

// ---- seeded shuffle ----------------------------------------------------------

/** mulberry32 — same deterministic seeded RNG the family's fuzzers use. */
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

export interface ShuffleOptions {
  /** Deterministic seed. Omit for Math.random (non-reproducible). In an `epochs()` pipeline each epoch derives its own seed (`seed + epochIndex`) so epochs reshuffle differently but reproducibly. */
  seed?: number;
  /**
   * Shuffle-buffer size (the tf.data-style streaming shuffle): items are
   * held in a buffer of this size and emitted by random replacement, so
   * memory stays O(bufferSize) while mixing quality depends on it.
   * Default `Infinity` = materialize and Fisher-Yates the whole dataset
   * (perfect shuffle — fine at the dataset sizes this v1 targets).
   */
  bufferSize?: number;
}

async function* shuffled<T>(source: AnySource<T>, seed: number | undefined, bufferSize: number): AsyncGenerator<T> {
  const rng = seed === undefined ? Math.random : mulberry32(seed);
  if (bufferSize === Infinity) {
    const all: T[] = [];
    for await (const item of source) all.push(item);
    for (let i = all.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = all[i] as T;
      all[i] = all[j] as T;
      all[j] = tmp;
    }
    yield* all;
    return;
  }
  const buffer: T[] = [];
  for await (const item of source) {
    if (buffer.length < bufferSize) {
      buffer.push(item);
      continue;
    }
    const j = Math.floor(rng() * buffer.length);
    yield buffer[j] as T;
    buffer[j] = item;
  }
  // Drain: Fisher-Yates the remainder so the tail isn't emitted in arrival order.
  for (let i = buffer.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = buffer[i] as T;
    buffer[i] = buffer[j] as T;
    buffer[j] = tmp;
  }
  yield* buffer;
}

// ---- collate helpers ---------------------------------------------------------

/** `{x, y}` sample pair — `batch(n, { collate: collate.xy() })` produces exactly the `Batch` shape tensor-autograd's `trainer.fit(dataLoader)` consumes. */
export interface XYSample {
  x: readonly number[];
  y: readonly number[];
}

export const collate = {
  /** number[] samples -> one [batch, dim] Tensor (f32 by default, matching the family's ML-oriented default dtype). */
  vectors(options: { dtype?: "f32" | "f64" } = {}) {
    return (samples: ReadonlyArray<readonly number[]>): Tensor => {
      const dim = samples[0]?.length ?? 0;
      if (samples.some((s) => s.length !== dim)) {
        throw new RangeError("collate.vectors: ragged batch (samples have differing lengths)");
      }
      return Tensor.from(samples.flat() as number[], { dtype: options.dtype ?? "f32" }).reshape([
        samples.length,
        dim,
      ]);
    };
  },
  /** scalar number samples -> one [batch] Tensor. */
  scalars(options: { dtype?: "f32" | "f64" } = {}) {
    return (samples: readonly number[]): Tensor =>
      Tensor.from(samples as number[], { dtype: options.dtype ?? "f32" });
  },
  /** `{x, y}` samples -> `{ x: [batch, xDim] Tensor, y: [batch, yDim] Tensor }` — the trainer's `Batch`. */
  xy(options: { dtype?: "f32" | "f64" } = {}) {
    const vecs = collate.vectors(options);
    return (samples: readonly XYSample[]): { x: Tensor; y: Tensor } => ({
      x: vecs(samples.map((s) => s.x)),
      y: vecs(samples.map((s) => s.y)),
    });
  },
};

// ---- the Dataset facade ------------------------------------------------------

export interface MapConcurrentOptions {
  concurrency: number;
  /** Preserve source order (default true — matches upstream). */
  ordered?: boolean;
  signal?: AbortSignal;
}

/**
 * A lazy, re-iterable (when its source is) async pipeline. Every chainable
 * method returns a NEW Dataset wrapping a transformation of this one —
 * nothing runs until iteration. Iterating twice re-runs the pipeline from
 * the source, which requires a re-iterable source (a factory or a plain
 * sync iterable); a one-shot AsyncIterable source throws a clear error on
 * the second pass rather than silently yielding nothing.
 */
export class Dataset<T> implements AsyncIterable<T> {
  readonly #open: () => AnySource<T>;
  #consumedOneShot = false;
  readonly #oneShot: boolean;

  private constructor(open: () => AnySource<T>, oneShot: boolean) {
    this.#open = open;
    this.#oneShot = oneShot;
  }

  static from<T>(source: DatasetSource<T>): Dataset<T> {
    if (typeof source === "function") return new Dataset(source, false);
    if (!isIterableObject<T>(source)) {
      throw new TypeError("data.fromAsync: source must be (async) iterable or a factory returning one");
    }
    // A plain sync iterable (array, Set, generator FACTORY result excluded
    // above) is re-iterable; a bare async iterable usually is not — assume
    // one-shot unless it's sync-iterable.
    const oneShot = !(Symbol.iterator in source);
    return new Dataset(() => source, oneShot);
  }

  async *[Symbol.asyncIterator](): AsyncGenerator<T> {
    if (this.#oneShot) {
      if (this.#consumedOneShot) {
        throw new Error(
          "Dataset: this source is a one-shot AsyncIterable and was already consumed — pass a factory (() => yourAsyncIterable()) to fromAsync for re-iterable/epoch pipelines",
        );
      }
      this.#consumedOneShot = true;
    }
    yield* this.#open();
  }

  #derive<U>(transform: (source: AnySource<T>) => AnySource<U>): Dataset<U> {
    return new Dataset<U>(() => transform(this.#open()), this.#oneShot) as Dataset<U>;
  }

  /** Lazy elementwise transform (sequential; use {@link mapConcurrent} for parallel decode/augment). */
  map<U>(fn: (item: T) => U | Promise<U>): Dataset<U> {
    return this.#derive(async function* (src) {
      for await (const item of src) yield await fn(item);
    });
  }

  filter(fn: (item: T) => boolean | Promise<boolean>): Dataset<T> {
    return this.#derive(async function* (src) {
      for await (const item of src) if (await fn(item)) yield item;
    });
  }

  /** Bounded-concurrency transform via upstream `mapConcurrentAsync` — for I/O or CPU-bound decode/augment. `{signal}` cancels promptly mid-flight. */
  mapConcurrent<U>(fn: (item: T) => Promise<U> | U, options: MapConcurrentOptions): Dataset<U> {
    return this.#derive((src) => mapConcurrentAsync(fn, src, options));
  }

  /** Eagerly keep up to `n` items in flight ahead of the consumer (upstream `prefetchAsync`) — overlaps producer latency with consumer work. */
  prefetch(n: number): Dataset<T> {
    return this.#derive((src) => prefetchAsync(n, src));
  }

  /** Fixed-size chunks of `n` (trailing partial chunk included) — upstream's `group(n)` transducer, renamed per the facade contract (see module doc). */
  chunk(n: number): Dataset<T[]> {
    return this.#derive(async function* (src) {
      // Drive the transducer protocol in streaming form (transduce* is the
      // terminal fold; here each completed group is yielded as it forms).
      // The sink side-channels completed groups out via `emitted` — the
      // protocol's own buffer isn't needed for streaming.
      const emitted: T[][] = [];
      const sink: ReducerStep<T[]> = (buffer, group) => {
        emitted.push(group);
        return buffer;
      };
      const step = transducers.group<T>(n)(sink);
      for await (const item of src) {
        step([], item);
        yield* emitted.splice(0);
      }
      step.complete?.([]); // flush the trailing partial group
      yield* emitted.splice(0);
    });
  }

  /** {@link chunk} + collate: `batch(32, { collate: collate.xy() })` yields trainer-ready Tensor batches. Without a collate, yields plain `T[]` chunks. */
  batch(n: number): Dataset<T[]>;
  batch<B>(n: number, options: { collate: (samples: T[]) => B }): Dataset<B>;
  batch<B>(n: number, options?: { collate: (samples: T[]) => B }): Dataset<T[]> | Dataset<B> {
    const chunks = this.chunk(n);
    return options ? chunks.map((c) => options.collate(c)) : chunks;
  }

  /** Seeded shuffle — perfect (full-materialize Fisher-Yates) by default, streaming shuffle-buffer with `bufferSize`. See {@link ShuffleOptions}. */
  shuffle(options: ShuffleOptions = {}): Dataset<T> {
    return this.#derive((src) => shuffled(src, options.seed, options.bufferSize ?? Infinity));
  }

  take(n: number): Dataset<T> {
    return this.#derive(async function* (src) {
      if (n <= 0) return;
      let taken = 0;
      for await (const item of src) {
        yield item;
        if (++taken >= n) return;
      }
    });
  }

  drop(n: number): Dataset<T> {
    return this.#derive(async function* (src) {
      let dropped = 0;
      for await (const item of src) {
        if (dropped < n) {
          dropped++;
          continue;
        }
        yield item;
      }
    });
  }

  /** Abort-aware view: iteration rejects with the signal's reason as soon as `signal` fires (upstream `abortable`). */
  abortable(signal: AbortSignal): Dataset<T> {
    return this.#derive((src) => abortable(src, signal));
  }

  /**
   * Repeat the whole pipeline `n` times (one item stream, epoch after
   * epoch) — requires a re-iterable source. Any `shuffle({seed})` layered
   * on TOP of `epochs()` sees the concatenated stream; for the standard
   * "reshuffle each epoch" order, use the `reshuffle` option here, which
   * applies a per-epoch derived seed (`seed + epochIndex`).
   */
  epochs(n: number, options: { reshuffle?: ShuffleOptions } = {}): Dataset<T> {
    if (this.#oneShot) {
      throw new Error("Dataset.epochs: source is a one-shot AsyncIterable — pass a factory to fromAsync instead");
    }
    const open = this.#open;
    const reshuffle = options.reshuffle;
    return new Dataset<T>(async function* () {
      for (let epoch = 0; epoch < n; epoch++) {
        const src = open();
        yield* reshuffle
          ? shuffled(src, reshuffle.seed === undefined ? undefined : reshuffle.seed + epoch, reshuffle.bufferSize ?? Infinity)
          : src;
      }
    }, false);
  }

  /** Terminal reduce (upstream `foldAsync`) — THE reduce of this facade; see the module doc for why it isn't named `reduce`. */
  fold<Acc>(fn: (acc: Acc, item: T) => Acc | Promise<Acc>, init: Acc, options: { signal?: AbortSignal } = {}): Promise<Acc> {
    return foldAsync(fn, init, this.#openForTerminal(), options);
  }

  async toArray(options: { signal?: AbortSignal } = {}): Promise<T[]> {
    return this.fold<T[]>(
      (acc, item) => {
        acc.push(item);
        return acc;
      },
      [],
      options,
    );
  }

  #openForTerminal(): AnySource<T> {
    // Route through the async iterator so one-shot accounting still applies.
    return this;
  }
}

/** Build a {@link Dataset} from an (async) iterable or a factory returning one. The factory form is what enables `epochs()`/repeated iteration over async sources. */
export function fromAsync<T>(source: DatasetSource<T>): Dataset<T> {
  return Dataset.from(source);
}
