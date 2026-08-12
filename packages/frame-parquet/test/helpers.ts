/**
 * pyarrow oracle resolution — same convention as
 * packages/tensor-core/test/differential.test.ts / docs/TESTING.md's
 * MALLORY_ORACLE_PYTHON: `$MALLORY_ORACLE_PYTHON`, else `python3` on PATH.
 * Tests that need pyarrow SKIP (never fail) when no interpreter with
 * pyarrow is found.
 *
 * On trycooy: nix-shell -p "python3.withPackages(ps: [ps.pyarrow ps.pandas])"
 */
import { execFileSync } from "node:child_process";

function resolvePython(): string | undefined {
  const candidates = [process.env.MALLORY_ORACLE_PYTHON, "python3"].filter((c): c is string => Boolean(c));
  for (const candidate of candidates) {
    try {
      execFileSync(candidate, ["-c", "import pyarrow"], { stdio: "ignore" });
      return candidate;
    } catch {
      // try next
    }
  }
  return undefined;
}

export const PYTHON = resolvePython();
export const PYARROW_SKIP_REASON = "no python with pyarrow found (set MALLORY_ORACLE_PYTHON)";

/** A URL base (not a plain path) — use `new URL(name, FIXTURES_DIR)` to resolve a fixture file. */
export const FIXTURES_DIR = new URL("./fixtures/", import.meta.url);

/** Run a small inline python snippet against `pyarrow`, returning parsed stdout JSON. */
export function runPyarrowJson(script: string): unknown {
  if (!PYTHON) throw new Error(PYARROW_SKIP_REASON);
  const out = execFileSync(PYTHON, ["-c", script], { encoding: "utf8" });
  return JSON.parse(out);
}
