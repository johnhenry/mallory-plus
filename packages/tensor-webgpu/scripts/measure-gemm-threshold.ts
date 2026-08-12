/**
 * Measures the WASM-vs-WebGPU GEMM crossover on THIS machine, the same way
 * docs/spikes/wasm-baseline.md measured the WASM-vs-pure-JS crossover:
 * real timings, not a guess. Not part of `npm test` (per the issue: "GPU
 * *performance* tests gate behind real hardware, not per-PR" — this is a
 * spike script you run manually and record the results of in
 * docs/spikes/webgpu-baseline.md, exactly like wasm-baseline.md's own
 * measurements aren't a CI-gating test either).
 *
 * Both backends are timed END-TO-END per call — allocate, copy CPU data in,
 * compute, copy result out, free — because that's the realistic cost a
 * `Tensor.matmul()` call pays today: neither backend has a "keep this
 * operand resident across multiple calls" API yet (mallory-tensor-wasm's
 * `WasmTensor` and this package's `GPUTensor` both exist, but v1 has no
 * "matmul two already-resident tensors" entry point wired up at the
 * `Tensor` level) — a resident-buffers benchmark would understate WebGPU's
 * per-call overhead (buffer creation, `mapAsync` readback) in exactly the
 * way wasm-baseline.md documented for the WASM wrapper.
 *
 * Run: `node --experimental-strip-types scripts/measure-gemm-threshold.ts`
 * (or plain `node scripts/measure-gemm-threshold.ts` on a Node version with
 * type stripping on by default, same as this package's own tests).
 */
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Kernels, WasmTensor } from "mallory-tensor-wasm";
import { bundleForBrowser, getHarness, SRC } from "../test/helpers.ts";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SIZES = [8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048];
const ITERATIONS = 5;

function randomData(size: number, seed: number): Float32Array {
  let s = seed >>> 0;
  const out = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = (s / 0xffffffff) * 2 - 1;
  }
  return out;
}

async function measureWasm(kernels: Kernels, n: number): Promise<number> {
  const a = randomData(n * n, n + 1);
  const b = randomData(n * n, n + 2);
  const times: number[] = [];
  for (let i = 0; i < ITERATIONS; i++) {
    const t0 = performance.now();
    const ta = WasmTensor.fromArray(kernels, a, [n, n]);
    const tb = WasmTensor.fromArray(kernels, b, [n, n]);
    const out = kernels.zeros([n, n]);
    kernels.matmulInto(out, ta, tb);
    out.toFloat32Array();
    const t1 = performance.now();
    ta.free();
    tb.free();
    out.free();
    times.push(t1 - t0);
  }
  times.sort((x, y) => x - y);
  return times[Math.floor(times.length / 2)] as number; // median
}

async function main(): Promise<void> {
  const kernels = await Kernels.load();

  const harness = await getHarness();
  if ("unavailable" in harness) {
    console.error(`Cannot measure WebGPU side: headless WebGPU unavailable (${harness.reason})`);
    console.error("WASM-only timings follow (no crossover can be determined):");
    for (const n of SIZES) {
      const wasmMs = await measureWasm(kernels, n);
      console.log(`n=${n}\twasm=${wasmMs.toFixed(3)}ms`);
    }
    process.exit(1);
  }

  const bundle = bundleForBrowser([path.join(SRC, "gemm.ts")]);

  console.log("n\twasm_ms\twebgpu_ms\tfaster");
  const rows: Array<{ n: number; wasmMs: number; webgpuMs: number }> = [];
  for (const n of SIZES) {
    const wasmMs = await measureWasm(kernels, n);
    const a = randomData(n * n, n + 1);
    const b = randomData(n * n, n + 2);
    const webgpuMs = await harness.run<number>(
      `
      const adapter = await navigator.gpu.requestAdapter();
      const device = await adapter.requestDevice();
      const a = new Float32Array(${JSON.stringify(Array.from(a))});
      const b = new Float32Array(${JSON.stringify(Array.from(b))});
      const times = [];
      for (let i = 0; i < ${ITERATIONS}; i++) {
        const t0 = performance.now();
        await runGemmWGSL(device, a, b, ${n}, ${n}, ${n});
        const t1 = performance.now();
        times.push(t1 - t0);
      }
      times.sort((x, y) => x - y);
      return times[Math.floor(times.length / 2)];
      `,
      bundle,
    );
    rows.push({ n, wasmMs, webgpuMs });
    console.log(`${n}\t${wasmMs.toFixed(3)}\t${webgpuMs.toFixed(3)}\t${webgpuMs < wasmMs ? "webgpu" : "wasm"}`);
  }

  const crossover = rows.find((r) => r.webgpuMs < r.wasmMs);
  if (crossover) {
    console.log(
      `\nCrossover: webgpu becomes faster at n=${crossover.n} (${crossover.n * crossover.n} elements)`,
    );
  } else {
    console.log(`\nNo crossover found within tested sizes (up to n=${SIZES[SIZES.length - 1]}) — wasm always faster here`);
  }

  harness.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
