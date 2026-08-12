#!/usr/bin/env python3
"""Generates the "Python writes, JS reads" half of interop-python's
bidirectional conformance suite (issue #21). Output is committed (small
binary fixtures) rather than regenerated at test time, matching this
repo's established convention -- see this directory's own README section
in ../README.md for how to regenerate.

Run via the nix-provisioned Python env (see ../README.md):
  nix-shell -p "python3.withPackages(ps: [ps.pyarrow ps.pandas ps.numpy])" \
    --run "python3 packages/interop-python/tests/fixtures/generate_py_fixtures.py"
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd

from mallory_interop import save_npy, save_npz, write_ipc, write_parquet

HERE = Path(__file__).parent

df = pd.DataFrame(
    {
        "id": pd.array([1, 2, 3, 4, 5], dtype="Int32"),
        "value": pd.array([1.5, None, 3.25, -0.5, 100], dtype="Float64"),
        "label": pd.array(["alpha", "beta", None, "delta", ""], dtype="string"),
        "active": pd.array([True, False, None, True, False], dtype="boolean"),
    }
)

write_ipc(df, str(HERE / "py_written.arrow"))
write_parquet(df, str(HERE / "py_written.parquet"), compression="snappy")

# JSON-safe expected values (NaN/NA -> null), read by both the Python
# conformance test (self-check) and a JS-side test (frame-arrow/frame-parquet
# reading these Python-written fixtures back).
expected = {
    "columns": list(df.columns),
    "rows": json.loads(df.where(pd.notnull(df), None).to_json(orient="records")),
}
(HERE / "py_written_expected.json").write_text(json.dumps(expected, indent=2))

# .npy / .npz fixtures
arr = np.array([1.0, 2.5, -3.0, 4.25, 0.0], dtype=np.float64)
save_npy(str(HERE / "py_written.npy"), arr)
save_npz(
    str(HERE / "py_written.npz"),
    {"a": arr, "b": np.array([1, 2, 3], dtype=np.int32)},
)

print("wrote py_written.arrow, py_written.parquet, py_written_expected.json, py_written.npy, py_written.npz")
