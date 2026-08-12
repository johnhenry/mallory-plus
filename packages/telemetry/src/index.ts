/**
 * mallory-telemetry (issue #10) — a stable event stream any UI can consume
 * later, designed inline alongside autograd/optim rather than bolted on
 * afterward. No UI exists yet; this is purely the emission contract.
 *
 * The differentiated panel this eventually enables isn't loss curves
 * (TensorBoard already does those well) — it's JS<->WASM memory residency
 * and copy accounting: which tensors live in linear memory and how many
 * bytes cross the boundary per step. docs/spikes/wasm-baseline.md measured
 * this exact cost causing a 4x swing that was invisible until benchmarked;
 * Python tooling has no equivalent because NumPy never crosses that
 * boundary. Hence the `metric` events tensor-wasm's allocator emits below.
 *
 * Default sink is a no-op: zero cost and zero dependencies when unused.
 * Emit SUMMARIES (shape/dtype/min/max/mean/std/finite fraction), never raw
 * tensor dumps, by default.
 */

export interface TensorSummaryStats {
  min: number;
  max: number;
  mean: number;
  std: number;
  /** Fraction of elements that are finite (not NaN/+-Infinity), in [0, 1]. */
  finite: number;
}

export interface TensorSummary {
  shape: readonly number[];
  dtype: string;
  device: string;
  stats: TensorSummaryStats;
  histogram?: { edges: readonly number[]; counts: readonly number[] };
}

export type TrainingEvent =
  | {
      type: "run.start";
      runId: string;
      time: number;
      config: Record<string, unknown>;
      environment?: Record<string, unknown>;
    }
  | {
      type: "metric";
      runId: string;
      step: number;
      time: number;
      name: string;
      value: number;
    }
  | {
      type: "tensor.summary";
      runId: string;
      step: number;
      name: string;
      tensor: TensorSummary;
    }
  | {
      type: "artifact";
      runId: string;
      step?: number;
      name: string;
      kind: "checkpoint" | "image" | "audio" | "table" | "json";
      ref: string;
    }
  | {
      type: "trace";
      runId: string;
      step: number;
      spans: ReadonlyArray<{
        name: string;
        start: number;
        duration: number;
        category: string;
      }>;
    };

export type Sink = (event: TrainingEvent) => void;

const noopSink: Sink = () => {};
let activeSink: Sink = noopSink;

/** Install a sink to receive every emitted event. Pass `null` to go back to the no-op default. */
export function setSink(sink: Sink | null): void {
  activeSink = sink ?? noopSink;
}

/** True iff a non-default sink is installed — callers can use this to skip building an event's payload entirely when telemetry is off. */
export function hasSink(): boolean {
  return activeSink !== noopSink;
}

export function emit(event: TrainingEvent): void {
  activeSink(event);
}

// ---- convenience emitters ---------------------------------------------------

export function startRun(
  runId: string,
  config: Record<string, unknown>,
  environment?: Record<string, unknown>,
): void {
  emit({ type: "run.start", runId, time: Date.now(), config, environment });
}

export function metric(runId: string, step: number, name: string, value: number): void {
  emit({ type: "metric", runId, step, time: Date.now(), name, value });
}

export function tensorSummary(
  runId: string,
  step: number,
  name: string,
  tensor: TensorSummary,
): void {
  emit({ type: "tensor.summary", runId, step, name, tensor });
}

export function artifact(
  runId: string,
  name: string,
  kind: "checkpoint" | "image" | "audio" | "table" | "json",
  ref: string,
  step?: number,
): void {
  emit({ type: "artifact", runId, name, kind, ref, step });
}

export function trace(
  runId: string,
  step: number,
  spans: ReadonlyArray<{ name: string; start: number; duration: number; category: string }>,
): void {
  emit({ type: "trace", runId, step, spans });
}

/**
 * Time a synchronous block and emit it as a one-span trace, but only if a
 * sink is actually installed — `performance.now()` calls are cheap but not
 * free, and this is meant to cost nothing on the hot path by default.
 */
export function timed<T>(
  runId: string,
  step: number,
  spanName: string,
  category: string,
  fn: () => T,
): T {
  if (!hasSink()) return fn();
  const start = performance.now();
  const result = fn();
  trace(runId, step, [{ name: spanName, start, duration: performance.now() - start, category }]);
  return result;
}
