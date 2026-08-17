/**
 * findPeaks (issue #44) — local-maxima detection with `height`/`distance`/
 * `prominence` filtering, matching `scipy.signal.find_peaks`'s basic
 * behavior and algorithms (plateau-aware local-maxima detection, the exact
 * greedy tallest-first `distance` selection algorithm, and the standard
 * "walk outward to the nearest higher point on each side" prominence
 * definition).
 */
import { Tensor } from "mallory-tensor-core";

export interface FindPeaksOptions {
  /** Minimum peak height. A peak below this is discarded. */
  readonly height?: number;
  /** Minimum required horizontal distance (in samples) between neighboring peaks. When two peaks are closer, the shorter one is discarded (ties: the earlier-index one is discarded), matching scipy's exact greedy algorithm. */
  readonly distance?: number;
  /** Minimum required topographic prominence (see `prominences`'s doc comment). */
  readonly prominence?: number;
}

export interface FindPeaksResult {
  readonly indices: number[];
  readonly heights: number[];
  readonly prominences: number[];
}

/** Every plateau-aware local maximum's representative index (the midpoint of a flat plateau, floor-rounded, matching scipy's convention) and value. */
function localMaxima(x: Float64Array): { indices: number[]; heights: number[] } {
  const indices: number[] = [];
  const heights: number[] = [];
  let i = 1;
  const n = x.length;
  while (i < n - 1) {
    if ((x[i] as number) > (x[i - 1] as number)) {
      let iAhead = i + 1;
      while (iAhead < n - 1 && (x[iAhead] as number) === (x[i] as number)) iAhead++;
      if ((x[iAhead] as number) < (x[i] as number)) {
        const mid = Math.floor((i + iAhead - 1) / 2);
        indices.push(mid);
        heights.push(x[i] as number);
        i = iAhead;
      } else {
        i = iAhead;
      }
    } else {
      i++;
    }
  }
  return { indices, heights };
}

/**
 * Topographic prominence of each peak: from a peak, walk left until either
 * the signal boundary or a point higher than the peak; take the minimum
 * value seen along the way (the "left base"). Do the same walking right.
 * Prominence = peak height - max(leftBase, rightBase) (the higher, i.e.
 * more restrictive, of the two bases -- the standard definition: how far
 * you'd have to descend from the peak before you could walk to a higher
 * peak without re-ascending past the starting peak's own height).
 */
function prominences(x: Float64Array, peakIndices: readonly number[]): number[] {
  const n = x.length;
  return peakIndices.map((peakIdx) => {
    const peakHeight = x[peakIdx] as number;

    let leftBase = peakHeight;
    for (let i = peakIdx - 1; i >= 0; i--) {
      const v = x[i] as number;
      if (v > peakHeight) break;
      if (v < leftBase) leftBase = v;
    }

    let rightBase = peakHeight;
    for (let i = peakIdx + 1; i < n; i++) {
      const v = x[i] as number;
      if (v > peakHeight) break;
      if (v < rightBase) rightBase = v;
    }

    return peakHeight - Math.max(leftBase, rightBase);
  });
}

/**
 * Greedy tallest-first selection: repeatedly take the tallest remaining
 * peak, discard every other remaining peak within `distance` samples of it.
 * Matches scipy's exact `_select_by_peak_distance` algorithm.
 *
 * `indices` arrives already sorted ascending by position -- `localMaxima`
 * scans left-to-right, and any height/prominence filtering upstream uses
 * `Array.prototype.filter`, which preserves relative order. That sortedness
 * means the set of candidates within `distance` of an accepted peak is
 * always a *contiguous* run around it in position-rank space, so instead of
 * rescanning every other candidate for every accepted peak (O(n^2)), walk
 * outward left/right from each accepted peak only as far as `distance`
 * reaches. A doubly linked list over "still-alive" ranks (`left`/`right`)
 * lets each walk skip already-suppressed candidates in O(1) per hop and
 * relinks past whatever it just suppressed, so no rank is ever revisited
 * once dead -- O(n log n) total (the height sort dominates; the suppression
 * walk itself is O(n) amortized, standard union-find-style path
 * compression). Produces bit-identical output to the O(n^2) version: both
 * compute the same "first sufficiently tall, earlier-processed peak within
 * `distance` wins" union over accepted peaks' windows, just at different
 * speeds.
 */
function filterByDistance(indices: number[], heights: number[], distance: number): boolean[] {
  const n = indices.length;
  const keep = new Array<boolean>(n).fill(true);
  if (n === 0) return keep;

  const left = new Array<number>(n);
  const right = new Array<number>(n);
  for (let k = 0; k < n; k++) {
    left[k] = k - 1;
    right[k] = k + 1;
  }

  const order = indices.map((_, i) => i).sort((a, b) => (heights[b] as number) - (heights[a] as number));

  for (const i of order) {
    if (!keep[i]) continue;
    const pos = indices[i] as number;

    let k = left[i] as number;
    while (k >= 0 && pos - (indices[k] as number) < distance) {
      keep[k] = false;
      k = left[k] as number;
    }
    left[i] = k;
    if (k >= 0) right[k] = i;

    let m = right[i] as number;
    while (m < n && (indices[m] as number) - pos < distance) {
      keep[m] = false;
      m = right[m] as number;
    }
    right[i] = m;
    if (m < n) left[m] = i;
  }

  return keep;
}

export function findPeaks(signal: Tensor, options: FindPeaksOptions = {}): FindPeaksResult {
  if (signal.shape.length !== 1) throw new RangeError("findPeaks: v1 supports 1-D Tensor only");
  // Read-only below, so no defensive copy is needed even when `.data`
  // aliases `signal`'s own storage.
  const x = signal.contiguous().data as Float64Array;

  let { indices, heights } = localMaxima(x);

  if (options.height !== undefined) {
    const h = options.height;
    const kept = indices.map((_, i) => (heights[i] as number) >= h);
    indices = indices.filter((_, i) => kept[i]);
    heights = heights.filter((_, i) => kept[i]);
  }

  let peakProminences = prominences(x, indices);

  if (options.prominence !== undefined) {
    const p = options.prominence;
    const kept = peakProminences.map((v) => v >= p);
    indices = indices.filter((_, i) => kept[i]);
    heights = heights.filter((_, i) => kept[i]);
    peakProminences = peakProminences.filter((_, i) => kept[i]);
  }

  if (options.distance !== undefined) {
    if (options.distance < 1) throw new RangeError(`findPeaks: distance must be >= 1, got ${options.distance}`);
    const keep = filterByDistance(indices, heights, options.distance);
    indices = indices.filter((_, i) => keep[i]);
    heights = heights.filter((_, i) => keep[i]);
    peakProminences = peakProminences.filter((_, i) => keep[i]);
  }

  return { indices, heights, prominences: peakProminences };
}
