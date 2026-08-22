/**
 * @johnhenry/math-plus-adapter-onnx (issue #18) — mode 1 only: `onnx.load(modelSource)` /
 * `model.run(inputs)`, marshalling between @johnhenry/math-plus-tensor-core's `Tensor`
 * and onnxruntime-web's `Tensor`. That marshalling is the entire job.
 *
 * Backend selection (wasm/webgl/webgpu/...) delegates entirely to ONNX
 * Runtime Web's own `env`/`SessionOptions.executionProviders` — v1 does not
 * route ORT through math-plus's own device abstraction.
 *
 * Deliberately deferred (see the issue for the reasoning): mode 2 (adapting
 * our storage to ORT's tensor storage — couples two independently evolving
 * memory models before either is stable) and mode 3 (importing/compiling
 * ONNX graphs into our own runtime — risks reimplementing ONNX's operator
 * compatibility matrix, the exact scope-creep trap the source design warns
 * about).
 */
import * as ort from "onnxruntime-web";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { dtypeToOrtType, ortTypeToDtype } from "./dtype.ts";

export { UnsupportedDTypeError } from "./dtype.ts";

/** Pass-through to ONNX Runtime Web's own session options (execution providers, graph optimization level, etc.) — v1 adds nothing on top. */
export type LoadOptions = ort.InferenceSession.SessionOptions;

export interface OnnxModel {
  readonly inputNames: readonly string[];
  readonly outputNames: readonly string[];
  /** Marshals every input Tensor to an ORT tensor, runs the graph, marshals every output ORT tensor back to a Tensor. */
  run(inputs: Record<string, Tensor>): Promise<Record<string, Tensor>>;
  /** Releases the underlying ORT session. The model is no longer usable after this resolves. */
  release(): Promise<void>;
}

/** math-plus Tensor -> ORT Tensor. Copies only if `t` isn't already contiguous (ORT tensors are always flat buffers, no stride support) — never for the common case of a freshly-built input Tensor. */
function tensorToOrt(t: Tensor): ort.Tensor {
  const packed = t.contiguous();
  const ortType = dtypeToOrtType(packed.dtype);
  // The generic `Tensor.Type` constructor overload accepts `Tensor.DataType`
  // (the union of every dtype's TypedArray); `packed.data`'s static type is
  // the narrower `AnyTypedArray`, and every dtype we support here (see
  // dtype.ts) backs both sides with the identical TypedArray class, so this
  // is a same-runtime-type passthrough, not an unsound cast.
  return new ort.Tensor(ortType, packed.data as never, packed.shape);
}

/** ORT Tensor -> math-plus Tensor. No copy — wraps the ORT tensor's own output buffer. */
function ortToTensor(t: ort.Tensor): Tensor {
  const dtype = ortTypeToDtype(t.type);
  return Tensor.fromTypedArray(t.data as never, t.dims, { dtype });
}

class OnnxModelImpl implements OnnxModel {
  readonly #session: ort.InferenceSession;

  constructor(session: ort.InferenceSession) {
    this.#session = session;
  }

  get inputNames(): readonly string[] {
    return this.#session.inputNames;
  }

  get outputNames(): readonly string[] {
    return this.#session.outputNames;
  }

  async run(inputs: Record<string, Tensor>): Promise<Record<string, Tensor>> {
    const feeds: Record<string, ort.Tensor> = {};
    for (const [name, t] of Object.entries(inputs)) {
      feeds[name] = tensorToOrt(t);
    }
    const results = await this.#session.run(feeds);
    const outputs: Record<string, Tensor> = {};
    for (const [name, ortTensor] of Object.entries(results)) {
      outputs[name] = ortToTensor(ortTensor);
    }
    return outputs;
  }

  release(): Promise<void> {
    return this.#session.release();
  }
}

/**
 * Load an ONNX model. `modelSource` is a file path/URL string, or the raw
 * model bytes (`Uint8Array`/`ArrayBuffer`) — e.g. read via `fs.readFileSync`
 * in Node or `fetch(...).arrayBuffer()` in a browser.
 */
export async function load(modelSource: string | Uint8Array | ArrayBufferLike, options?: LoadOptions): Promise<OnnxModel> {
  const session =
    typeof modelSource === "string"
      ? await ort.InferenceSession.create(modelSource, options)
      : await ort.InferenceSession.create(
          modelSource instanceof Uint8Array ? modelSource : new Uint8Array(modelSource),
          options,
        );
  return new OnnxModelImpl(session);
}

/** Namespace form matching the issue's `onnx.load(...)` usage. */
export const onnx = { load };
