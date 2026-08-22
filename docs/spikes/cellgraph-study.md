# Spike: CellGraph (mallory, formerly mallory-graph) as prior art for frame-arrow's lazy planner

Studied: `~/Projects/mallory-graph/src/lib/cell-graph.ts` (331 lines) + call sites
(`use-cell.ts`, `GraphCanvas.tsx:72-150`, `ExpressionRow.tsx:42-140`). Target:
PLAN.md §6.2 — frame-arrow's immutable expression-oriented `Frame`, `.collect()`
boundary, column pruning, predicate pushdown. Read-only spike; no code changes.

## 1. Mechanism inventory

- **Auto-recorded dependency edges via an evaluation stack.** `stack: string[]`
  (L43); `get()` records an edge from `stack.at(-1)` (the cell currently
  computing) to the cell being read (L134-139). Edges are *rebuilt from scratch*
  on every recompute — `recomputeAndEmit` detaches the old dependency set before
  pushing onto the stack and running `compute()` (L150-162) — so conditional
  reads keep the edge set exact per-evaluation. No manual tag declarations.
- **Eager dirty-marking, lazy recompute.** `set()` writes a free value, bumps
  version, emits, then `propagateDirty()` walks transitive dependents marking
  `dirty = true` and emitting — *no recompute* (L60-75, L261-280). Recompute
  happens only in `get()` when `cell.dirty && cell.compute` (L141-145): a pull
  model where subscribers re-read via `get()` and pay for recompute on demand.
- **structuralEqual-based change suppression.** After a recompute, if the new
  value deep-equals the cached one, the old *reference* is kept, version is not
  bumped, and no emit happens (L166-176; same check on `set`, L65-67). Deep
  equality is `structuralEqual` (L316-331). This is what lets React's
  `Object.is` snapshot check bail (`use-cell.ts:9-16` wires `get()` +
  `getVersion` into `useSyncExternalStore`).
- **Version counters.** `version` per cell (L22), bumped only on real change
  (L72, L92, L174); `getVersion()` (L180-182) is "the value
  useSyncExternalStore observes." Dirty-marking deliberately does **not** bump
  (comment L273-276) — unaffected downstream branches skip redraws.
- **Free / dependent / auxiliary roles.** Role is structural: `set` (no compute
  fn) ⇒ free, `define` (compute fn) ⇒ dependent, never-written ⇒ unknown
  (L103-107). `auxiliary` (L27, applied L64/L89) marks hidden-by-default cells
  for a GeoGebra-style Algebra listing via `list()` (L121-128). Consumed by
  `AlgebraView.tsx`.
- **CircularDependencyError.** Thrown from `get()` when the requested id is
  already on the evaluation stack (L142, class L32-37). Detection is lazy and
  per-evaluation — a cycle through a currently-clean cell isn't seen.
- **Reentrancy hardening** (battle scars worth knowing about): `emit` guards
  per-id against the useSyncExternalStore synchronous-getSnapshot storm
  (L282-312); `propagateDirty` snapshots the dependents set before iterating
  (L264-269); `delete()` dirty-marks former dependents *after* removal so
  reentrant recomputes see the post-delete world (L214-242).

Representative usage: computes are opaque closures over string ids —
`GraphCanvas.tsx:90-137` chains expr → freeVars → params → sampled path;
`ExpressionRow.tsx:82-99` exploits `get()`-on-nonexistent-id to pre-register an
edge on a cell that may be created later.

## 2. Mapping onto frame-arrow's lazy planner

| CellGraph mechanism | Applicable? | Why / where the analogy breaks |
|---|---|---|
| Pull-based laziness (`get()` recomputes on demand) | **Yes, conceptually** | `.collect()` *is* frame-arrow's `get()`: nothing executes until demanded. This is the strongest carry-over, and it's an idea, not code. |
| Auto-recorded edges via evaluation stack | **No** | CellGraph needs runtime tracking because computes are opaque closures over dynamic string ids. A Frame plan DAG's edges are explicit by construction — each node holds its child node(s). Stack tracking solves a problem frame-arrow doesn't have. |
| Eager dirty-marking + invalidation | **Mostly no; narrow yes later** | Frame plans are immutable expression DAGs — a node never "changes," so nothing dirties. The only invalidation frame-arrow could ever need is for *mutable external sources* (`scanParquet` on a file that changed between collects), and the right shape there is cache keys `(plan hash, source snapshot version/mtime)`, not push-dirty over a mutable id graph. v1 (§6.2) doesn't cache across `.collect()` calls at all, so this is N/A until a reactive/cached layer exists — which in mallory is the *app* layer, not the graph. |
| structuralEqual suppression | **No — cost model inverts** | Deep-equality over an Arrow table is an O(n·cols) scan; doing it to *save* downstream work is exactly backwards at dataframe scale. The analog, if ever needed, is cheap fingerprints (schema + row count + column stats/hashes). The *reference-preservation* trick (keep old identity on no-op recompute) is worth remembering for any future reactive binding. |
| Version counters for useSyncExternalStore | **Not for the planner** | Belongs to a hypothetical future React-binding package (a notebook UI re-collecting frames), which would sit *above* frame-arrow — same layering as `use-cell.ts` sitting above CellGraph. |
| Free/dependent/auxiliary roles | **Loose echo only** | Leaf scan nodes ≈ free, derived nodes ≈ dependent, optimizer-inserted nodes ≈ auxiliary. Cosmetic; falls out of node types for free. |
| CircularDependencyError | **No** | An immutable DAG built by construction cannot be cyclic — a node can only reference nodes that already exist. Cycle detection is only needed when identity is a mutable name, as with CellGraph's string ids. |
| Shared-subcomputation caching | **Yes — the one code-adjacent idea** | Two dependents reading one cell share its cached value; the planner analog is memoizing shared subplans (by node identity or plan hash) within one `.collect()` execution, i.e. cheap common-subexpression sharing. |

Root cause of most "No"s: CellGraph is *dynamic-by-ID over mutable cells with
opaque computes*; frame-arrow is *static-by-construction over immutable nodes
with reified operations*. Each design is good precisely where the other's core
machinery is unnecessary.

## 3. What frame-arrow needs that CellGraph lacks

- **Reified plan nodes.** CellGraph computes are closures — you can execute
  them but never inspect, rewrite, split, or reorder them. Everything below
  requires operations as data (`{op: "filter", predicate, input}`).
- **Column pruning.** Requires schema propagation through nodes plus a
  top-down required-columns analysis. CellGraph's `get()` is all-or-nothing per
  cell; there is no notion of demanding a *projection* of a value.
- **Predicate pushdown.** Requires an expression algebra: split conjunctions,
  compute per-predicate column references, and rewrite the tree to move
  filters below joins/into scans. No CellGraph counterpart at all.
- **`.collect()` as a batch boundary.** CellGraph materializes one cell per
  `get()`; a planner optimizes the *whole* DAG at the boundary before
  executing — pruning/pushdown are boundary-time rewrites, then execution.
- **Optimizer passes generally** (rewrite rules, later: worker/WASM execution
  strategies, §6.2 deferred list). CellGraph has zero rewriting machinery.

## 4. Extract vs. reference: recommendation

**Confirm the standing decision — design reference only, do not extract.**
Honestly evaluated, the case is stronger than expected: roughly 60% of
cell-graph.ts by line count (stack tracking, dirty propagation, emit/propagate
reentrancy guards, delete semantics, version plumbing) exists *because*
CellGraph is a mutable, dynamic-ID store driving a synchronous React UI. The
planner needs none of that code; what carries over — pull-until-demanded,
memoize shared subresults, preserve identity on no-op recompute — is three
ideas totaling zero reusable lines. Extraction would freeze a private app's
internal API into a shared package for negative benefit. **Revisit trigger:**
if math-plus later grows a reactive notebook layer ("auto re-collect when
the Parquet file changes, don't re-render unchanged panes"), *that layer* is
shaped exactly like CellGraph and extraction becomes worth re-asking — but it
would depend on frame-arrow, not live inside it.

## 5. Sharp edges in cell-graph.ts worth reporting upstream

1. **`set()` on a formerly-dependent cell leaks stale edges** (L60-75): it
   clears `compute` but never detaches `cell.dependencies` (unlike
   `recomputeAndEmit` L153-154 and `delete` L231). Old upstream writes then
   spuriously dirty/emit the now-free cell and its subtree (recomputes are
   suppressed by structuralEqual, so it's correctness-safe but wasteful), and
   the free cell's `dirty` flag sticks `true` until its next `set()`.
2. **`structuralEqual` treats any two zero-own-enumerable-key objects as
   equal** (L323-325): two distinct `Date`s, `Map`s, `Set`s, or real DOM
   `Path2D`s compare equal, silently suppressing updates. Safe today only
   because cells hold plain data (the app's "Path2D" is its own
   `{stroke, commands}` object — sample-function.ts:130-131); one future cell
   holding a `Map` breaks invisibly. Also: no cycle guard — a cyclic value
   causes infinite recursion.
3. **A throwing compute never caches its failure** (L156-164): `dirty` stays
   `true`, so every subsequent `get()` re-runs the failing (possibly
   expensive) compute and rethrows — including from inside React's
   `getSnapshot`. Call sites defend with try/catch inside computes
   (GraphCanvas.tsx:120-134), but the class contract doesn't say they must.
4. **`define()` unconditionally bumps version + emits** (L91-93) even when the
   new compute yields a structurally identical value — a redefine-per-render
   pattern would cause render churn; call sites all guard with
   `has()`/`hasValue()` first, which is convention, not enforcement.
5. **Unsubscribe never cleans up** (L185-191): empty listener Sets linger, and
   `subscribe`'s `ensure(id)` creates cell records that only an explicit
   `delete()` removes — unbounded `Map` growth under long-lived sessions with
   dynamic row ids (ExpressionRow's per-row cells are the live example).
