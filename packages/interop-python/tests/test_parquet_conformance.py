"""Bidirectional Parquet conformance (issue #21) -- same structure as
test_ipc_conformance.py, see its module doc for the "JS writes/Python
writes" fixture-generation split.
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from math_plus_interop import read_parquet, write_parquet

FIXTURES = Path(__file__).parent / "fixtures"


def _rows_as_plain(df: pd.DataFrame) -> list[dict]:
    records = []
    for _, row in df.iterrows():
        record = {}
        for col in df.columns:
            v = row[col]
            record[col] = None if pd.isna(v) else (bool(v) if df[col].dtype.kind == "b" else v)
        records.append(record)
    return records


def test_reads_js_written_parquet_matching_expected():
    expected = json.loads((FIXTURES / "js_written_expected.json").read_text())
    df = read_parquet(str(FIXTURES / "js_written.parquet"))
    assert list(df.columns) == expected["columns"]
    assert _rows_as_plain(df) == expected["rows"]


def test_reads_own_py_written_parquet_matching_expected():
    expected = json.loads((FIXTURES / "py_written_expected.json").read_text())
    df = read_parquet(str(FIXTURES / "py_written.parquet"))
    assert list(df.columns) == expected["columns"]
    assert _rows_as_plain(df) == expected["rows"]


def test_column_projection():
    df = read_parquet(str(FIXTURES / "js_written.parquet"), columns=["id", "label"])
    assert list(df.columns) == ["id", "label"]
    assert len(df) == 5


def test_pyarrow_native_filter_pushdown():
    df = read_parquet(str(FIXTURES / "js_written.parquet"), filters=[("id", ">", 3)])
    assert sorted(df["id"].tolist()) == [4, 5]


def test_write_then_read_back_round_trips_exactly_snappy():
    df = pd.DataFrame(
        {
            "x": pd.array([10, 20, 30], dtype="Int64"),
            "y": pd.array([1.1, None, 3.3], dtype="Float64"),
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "roundtrip.parquet")
        write_parquet(df, path, compression="snappy")
        back = read_parquet(path)
        assert back["x"].tolist() == [10, 20, 30]
        assert back["y"].tolist()[0] == 1.1
        assert pd.isna(back["y"].tolist()[1])


def test_write_then_read_back_preserves_nan_as_nan_not_null():
    """issue #103: same NaN-vs-null distinction as
    test_ipc_conformance.py's version of this test -- a genuine NaN in a
    plain (non-nullable-extension) float column must round-trip as NaN, not
    get silently coerced to an Arrow/Parquet null."""
    df = pd.DataFrame({"y": [1.1, float("nan"), 3.3]})
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "nan_roundtrip.parquet")
        write_parquet(df, path)

        # The written file itself must hold a real NaN, not a Parquet null,
        # for the affected value.
        table = pq.read_table(path)
        assert table.column("y").null_count == 0

        back = read_parquet(path)
        values = back["y"].tolist()
        assert values[0] == 1.1
        assert values[2] == 3.3
        nan_value = values[1]
        assert nan_value is not pd.NA
        assert nan_value != nan_value  # NaN is the only value unequal to itself


def test_write_zstd_produces_a_file_pyarrow_reads_directly():
    # A sanity check that write_parquet's zstd path is a REAL codec, not
    # frame-parquet's own JS-side footgun (pyarrow ships a native zstd
    # encoder, so there's no analogous "silently corrupt" risk here -- see
    # parquet.py's module doc -- but verify end to end anyway).
    df = pd.DataFrame({"x": pd.array([1, 2, 3], dtype="Int64")})
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "zstd.parquet")
        write_parquet(df, path, compression="zstd")
        table = pq.read_table(path)
        assert table.column("x").to_pylist() == [1, 2, 3]
        assert pq.ParquetFile(path).metadata.row_group(0).column(0).compression == "ZSTD"
