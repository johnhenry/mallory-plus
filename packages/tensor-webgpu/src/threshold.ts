/**
 * GEMM backend selection (issue #12, v1 scope item 1: "GEMM above a measured
 * size threshold — small matmuls stay on WASM since GPU dispatch overhead
 * dominates at small sizes"). The crossover was MEASURED, not guessed — see
 * docs/spikes/webgpu-baseline.md for the full methodology and raw numbers —
 * and the honest result on THIS machine is: no crossover exists within a
 * practical size range. `mallory-tensor-wasm`'s `matmulInto` was faster than
 * this package's `runGemmWGSL` at every measured size from 8x8 up to
 * 768x768 (589,824 output elements), by a factor that stayed in the 5-10x
 * range rather than narrowing toward 1x as size grew — i.e. this isn't
 * "the crossover is just past what got measured," it's "no crossover is
 * visible in the trend at all" on this hardware+kernel combination.
 *
 * Two real, named reasons, not hand-waving:
 *  1. This machine has no discrete GPU — WebGPU here runs through ANGLE's GL
 *     backend against an Intel integrated GPU under Xvfb (confirmed via
 *     `chrome://gpu`, not SwiftShader as originally expected — see
 *     docs/spikes/webgpu-baseline.md), which is a real hardware path but a
 *     much weaker one than a discrete GPU's.
 *  2. `runGemmWGSL`'s shader (gemm.ts) is intentionally naive for v1 — one
 *     thread per output element, no shared-memory tiling — exactly the kind
 *     of kernel that's most exposed to memory-bandwidth-bound performance on
 *     weak hardware. A tiled kernel is documented future work in gemm.ts's
 *     own module doc; it would very plausibly change this number.
 *
 * `GEMM_ELEMENT_THRESHOLD` is set to `Infinity` — `chooseGemmBackend` always
 * returns `"wasm"` — as the only honest default given the measurement above:
 * a specific finite number here would imply a crossover was found and
 * extrapolated beyond the tested range, which didn't happen. Re-measure with
 * `scripts/measure-gemm-threshold.ts` on real discrete-GPU hardware (or
 * after a tiled kernel lands) and replace this constant with whatever that
 * run finds — the mechanism (`chooseGemmBackend`, this named constant) is
 * built for exactly that recalibration, it's just not been triggered yet.
 */

/** See this module's doc comment: no measured crossover exists on this machine's hardware+kernel combination, so `chooseGemmBackend` conservatively never returns `"webgpu"` until re-measured. */
export const GEMM_ELEMENT_THRESHOLD = Number.POSITIVE_INFINITY;

/** `"wasm"` below the measured crossover, `"webgpu"` at or above it. Pure size-based heuristic — v1 doesn't factor in `k` separately or GPU queue occupancy. */
export function chooseGemmBackend(m: number, n: number): "wasm" | "webgpu" {
  return m * n >= GEMM_ELEMENT_THRESHOLD ? "webgpu" : "wasm";
}
