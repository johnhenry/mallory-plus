# Testing

## Unit tests

```bash
npm test            # all workspaces
npm test -w mallory-tensor-core
```

Tests are TypeScript run directly by `node --test` (native type stripping, Node ≥22.12).

## Differential tests (NumPy oracle)

`packages/tensor-core/test/differential.test.ts` compares tensor-core ops against a NumPy
subprocess (`packages/tensor-core/scripts/numpy_oracle.py`), exchanging data as `.npy` in both
directions — so every run also validates `.npy` I/O against NumPy's implementation.

Python resolution: `$MALLORY_ORACLE_PYTHON`, else `python3` on PATH. **The suite skips (does not
fail) when no interpreter with numpy is found** — environments that guarantee the oracle should
assert `skipped 0`.

### On NixOS (trycooy)

pip wheels don't work (libstdc++/libz linkage). Use a nix-provided Python:

```bash
ORACLE_PY=$(nix-shell -p "python3.withPackages(ps: with ps; [numpy pyarrow pandas])" --run "which python3")
MALLORY_ORACLE_PYTHON=$ORACLE_PY npm test -w mallory-tensor-core
```

The resolved store path stays valid until garbage-collected; re-run the `nix-shell` line to refresh.
The same `$MALLORY_ORACLE_PYTHON` (or a bare `python3` on PATH) is what `mallory-frame-arrow`'s and
`mallory-frame-parquet`'s pyarrow-round-trip tests look for too (see below) — one env var covers
every Python oracle in the repo.

### Tolerances

Per-op tolerance registry in `differential.test.ts` (`TOLERANCES`): f64 default `rtol 1e-12`,
f32 default `rtol 1e-5`, looser for accumulation-order-sensitive ops (`sum`/`mean` on f32).
Integer dtypes (incl. `i64`) compare exactly.

## pyarrow/pandas round-trip tests (frame-arrow, frame-parquet)

Some `mallory-frame-arrow`/`mallory-frame-parquet` tests verify a JS-written Arrow IPC/Parquet file
by reading it back with `pyarrow`/`pandas` directly (not just round-tripping through the package's
own reader — self-consistency isn't proof of a valid file). Same resolution and skip-don't-fail
behavior as the NumPy oracle above (`$MALLORY_ORACLE_PYTHON`, else `python3` on PATH,
`packages/frame-parquet/test/helpers.ts`) — the NixOS nix-shell line above already includes
`pyarrow`/`pandas` for exactly this reason.

Most fixtures in these two packages (and in `packages/interop-python/tests/fixtures/`, see below)
are **pre-generated and committed**, not regenerated live on every test run — only the tests that
verify THIS run's own freshly-written output (e.g. `writeParquet`'s pyarrow-round-trip tests) spawn
a live Python subprocess.

## Gradient oracles (autograd)

`adapter-math`'s `mallory-adapter-math/test-utils` subpath (`dualGrad`/`dualGradN`) wraps
mallory-math's `DualNumber` forward-mode autodiff — a third gradient oracle, algorithmically
independent of both `tensor-autograd`'s reverse-mode tape and finite differences. Its own tests
(`adapters/adapter-math/test/test-utils.test.ts`) validate it against finite differences AND
against `tensor-autograd`'s `Variable`/`grad.of` on real scalar/multivariate functions.

## Python-side interop tests (`packages/interop-python`)

`mallory-interop` is a PyPI package outside the npm/Cargo workspaces (see docs/RELEASING.md) — its
`pytest` suite runs separately from `npm test`. Bidirectional conformance (JS writes/Python reads,
and the inverse) is proven with committed fixtures on both sides — see
`packages/interop-python/README.md`'s "Bidirectional conformance fixtures" section for exactly
which test proves which direction, including the two JS-side tests
(`packages/frame-arrow/test/interop-python.test.ts`,
`packages/frame-parquet/test/interop-python.test.ts`) that verify Python-written fixtures read back
correctly — real cross-language verification, not two suites independently trusting their own
output.

```bash
nix-shell -p "python3.withPackages(ps: [ps.pyarrow ps.pandas ps.numpy ps.pytest])" \
  --run "cd packages/interop-python && PYTHONPATH=src python3 -m pytest tests/ -v"
```
