/**
 * Graph -> CSR bridge (issue #16). mallory-math's `Graph<T>` stores
 * `Map<T, Map<T, number>>` adjacency privately (no public accessor for the
 * raw nested Map) and already exposes `vertices()`, `edges()`, and the
 * public `readonly directed` flag — `toCSR` reconstructs the exact same
 * per-vertex row structure `Graph`'s own `addEdge` builds, from that public
 * surface, rather than reaching into a private field.
 *
 * `edges()` de-duplicates an undirected graph's symmetric pairs (returns
 * each undirected edge once, not once per direction) — `addEdge`'s own
 * mirroring means the adjacency Map itself has BOTH directions with the
 * same weight, so `toCSR` mirrors non-self-loop edges into both rows for
 * undirected graphs to reconstruct that faithfully. A self-loop
 * (`from === to`) is never doubled: `addEdge(u, u, w)` sets the same Map
 * entry twice (a no-op the second time), not two distinct entries.
 *
 * Implemented directly, without a separate `sparse` package dependency —
 * the CSR bundle here is a plain typed-array shape with no class wrapper,
 * so there's nothing to actually depend ON yet. A future `sparse` package
 * (PLAN.md's minimal CSR/COO package) can consume this shape directly once
 * it exists.
 *
 * ## The semantics clash this bridge exists to get right
 *
 * `Graph.toAdjacencyMatrix()` uses `Infinity` for "no edge" and hardcodes
 * `0` on the diagonal (a shortest-path/Floyd-Warshall convention: "distance
 * from a vertex to itself is 0"). That is the EXACT INVERSE of CSR, where
 * an absent (row, col) pair is an implicit zero, not stored at all. `toCSR`
 * never touches `toAdjacencyMatrix()` — it stores only the graph's REAL
 * (finite) edges, including any weight-0 edge that was actually added
 * (never conflating "explicitly zero" with "absent"). `toDense`'s inverse
 * direction makes that choice explicit via `missing: "zero" | "infinity"`:
 * `"zero"` is the standard sparse-matrix convention (absent = 0, no special
 * diagonal case); `"infinity"` replicates `toAdjacencyMatrix()`'s own
 * convention (absent = Infinity, diagonal defaults to 0 unless a real
 * self-loop edge overrides it) for feeding shortest-path algorithms.
 */
import type { Graph } from "mallory-math";

export interface CSRGraph<T> {
  /** Length `order.length + 1`. Row `i`'s entries are `columnIndices`/`values` in `[rowPointers[i], rowPointers[i+1])`. */
  rowPointers: Uint32Array;
  /** Length = number of stored (real) edges. Column-sorted within each row. */
  columnIndices: Uint32Array;
  /** Parallel to `columnIndices` — the edge weight at that (row, col). */
  values: Float64Array;
  /** Row/column label vector: `order[i]` is the vertex CSR row/column `i` refers to. */
  order: T[];
}

export function toCSR<T>(g: Graph<T>): CSRGraph<T> {
  const order = g.vertices();
  const index = new Map(order.map((v, i) => [v, i]));
  const n = order.length;

  const rows: Array<Array<{ col: number; value: number }>> = Array.from({ length: n }, () => []);
  for (const { from, to, weight } of g.edges()) {
    const fi = index.get(from) as number;
    const ti = index.get(to) as number;
    rows[fi].push({ col: ti, value: weight });
    if (!g.directed && from !== to) {
      rows[ti].push({ col: fi, value: weight });
    }
  }

  const rowPointers = new Uint32Array(n + 1);
  let nnz = 0;
  for (let i = 0; i < n; i++) {
    rowPointers[i] = nnz;
    nnz += (rows[i] as { col: number; value: number }[]).length;
  }
  rowPointers[n] = nnz;

  const columnIndices = new Uint32Array(nnz);
  const values = new Float64Array(nnz);
  let k = 0;
  for (let i = 0; i < n; i++) {
    const sorted = [...(rows[i] as { col: number; value: number }[])].sort((a, b) => a.col - b.col);
    for (const { col, value } of sorted) {
      columnIndices[k] = col;
      values[k] = value;
      k++;
    }
  }

  return { rowPointers, columnIndices, values, order };
}

export interface ToDenseOptions {
  /** What an absent (row, col) pair means in the output. `"zero"` (standard sparse-matrix convention) or `"infinity"` (matches `Graph.toAdjacencyMatrix()`, for shortest-path consumers). */
  missing: "zero" | "infinity";
}

export function toDense<T>(csr: CSRGraph<T>, options: ToDenseOptions): number[][] {
  const n = csr.order.length;
  const fill = options.missing === "infinity" ? Number.POSITIVE_INFINITY : 0;
  const matrix = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (options.missing === "infinity" && i === j ? 0 : fill)),
  );
  for (let row = 0; row < n; row++) {
    const start = csr.rowPointers[row] as number;
    const end = csr.rowPointers[row + 1] as number;
    for (let k = start; k < end; k++) {
      const col = csr.columnIndices[k] as number;
      (matrix[row] as number[])[col] = csr.values[k] as number;
    }
  }
  return matrix;
}
