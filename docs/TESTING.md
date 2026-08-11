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

### Tolerances

Per-op tolerance registry in `differential.test.ts` (`TOLERANCES`): f64 default `rtol 1e-12`,
f32 default `rtol 1e-5`, looser for accumulation-order-sensitive ops (`sum`/`mean` on f32).
Integer dtypes (incl. `i64`) compare exactly.

## Gradient oracles (pre-autograd)

`packages/scalar-types/test/dualnumber-oracle.test.ts` cross-checks mallory-math's `DualNumber`
forward-mode autodiff against central finite differences — the pure-JS third gradient oracle from
docs/PLAN.md §6.1, validated ahead of tensor-autograd. These helpers migrate to `adapter-math`
when that package lands.
