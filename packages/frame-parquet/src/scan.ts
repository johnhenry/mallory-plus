/**
 * `scanParquet("/events/*.parquet")` — glob/partitioned Parquet scans.
 *
 * DESIGN DECISION (issue #20 leaves this open, "your call, document which
 * you chose and why"): this is an EAGER v1, not a lazy one. `scanParquet`
 * resolves the glob, calls `readParquet` on every match (each pushing its
 * own `columns`/`filter` down to hyparquet, so per-file I/O reduction is
 * still real), and concatenates the results via `Frame.concat` immediately
 * — nothing is deferred to `.collect()`.
 *
 * True laziness — registering multiple files as plan-level sources and only
 * touching the ones a later `.collect()` actually needs — would require a
 * new `PlanNode` variant (e.g. a "parquetSource" kind alongside plan.ts's
 * existing `"source"`) that `execute.ts` and `plan.ts`'s `planArrowSchema`
 * both know how to handle. That's a change to mallory-frame-arrow itself
 * (an already-merged, closed package from issue #19), not something
 * frame-parquet can bolt on from outside — Frame's plan is a closed
 * discriminated union with no source-level extension point, and its
 * constructor is private. Reopening frame-arrow's plan machinery for this
 * is a well-defined but separable piece of work (tracked as a follow-up
 * issue); the eager fallback here is honestly scoped, matches the
 * concat-based pattern issue #20 itself allows, and is not a lot of code to
 * later replace once frame-arrow grows a source-level extension point.
 *
 * Uses Node's built-in `fs.promises.glob` (stable-enough since Node 22.0,
 * comfortably covered by this repo's `engines.node: ">=22.12.0"` floor;
 * verified working, no experimental-warning noise, on Node v26.5.0 in this
 * package's own dev environment) rather than adding a new glob-matching
 * dependency, per issue #20's own preference.
 *
 * A real bug found while building this (not hypothetical): apache-arrow
 * 21.2.0's `Table.concat()` produces a Table whose `getChild()` throws
 * ("Vector constructor expects an Array of Data instances") once ANY of the
 * concatenated tables has zero rows — reproducible directly against
 * apache-arrow with no frame-arrow/frame-parquet code involved at all. A
 * `filter` that matches rows in some partition files but not others (a
 * completely ordinary `scanParquet` use case — that's the whole point of
 * partitioned data) routinely produces exactly that mix once each file is
 * read+filtered individually. Filtering out zero-row frames before handing
 * them to `Frame.concat` (frame-arrow's own concat, which delegates to
 * `Table.concat` — see packages/frame-arrow/src/execute.ts's "concat" case)
 * sidesteps it entirely; if EVERY matched file's frame is empty after
 * filtering, one empty frame is returned as-is (still the right schema, no
 * concat needed). Tracked as a follow-up issue against frame-arrow itself,
 * since any other caller of `Frame.concat` could hit the same apache-arrow
 * bug — not scoped to this package.
 */
import { glob } from "node:fs/promises";
import { Frame } from "mallory-frame-arrow";
import { readParquet, type ReadParquetOptions } from "./read.ts";

export type ScanParquetOptions = ReadParquetOptions;

export async function scanParquet(pattern: string, options: ScanParquetOptions = {}): Promise<Frame> {
  const paths: string[] = [];
  for await (const p of glob(pattern)) paths.push(p);
  paths.sort(); // deterministic order regardless of filesystem readdir order
  if (paths.length === 0) {
    throw new Error(`scanParquet: no files matched pattern "${pattern}"`);
  }
  const frames = await Promise.all(paths.map((p) => readParquet(p, options)));
  // See module doc: apache-arrow 21.2.0's Table.concat() breaks getChild()
  // when any input has zero rows, so those are excluded from the concat set.
  const nonEmpty = frames.filter((f) => f.length > 0);
  if (nonEmpty.length === 0) return frames[0] as Frame;
  return Frame.concat(nonEmpty);
}
