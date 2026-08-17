"""Parquet convenience wrappers over pyarrow.parquet (issue #21).

`columns`/`filters` are pyarrow's OWN option shapes (a plain column-name
list; pyarrow's tuple/list filter expressions -- see pyarrow.parquet's own
docs), not a translation of mallory-frame-parquet's mongo-style `filter`
option ($gt/$lt/etc.) -- the two ecosystems' pushdown APIs are different by
design (hyparquet vs. Arrow's own C++ Parquet reader) and translating
between them isn't part of this package's v1 scope. Both independently
achieve real row-group-statistics-based I/O pruning; see this package's
conformance suite for cross-checking the same query against both.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ._arrow_nan import dataframe_to_table


def read_parquet(
    path: str,
    columns: list[str] | None = None,
    filters: Any | None = None,
) -> pd.DataFrame:
    """Read a Parquet file (optionally with column projection / pyarrow-native filters) into a pandas DataFrame."""
    table = pq.read_table(path, columns=columns, filters=filters)
    return table.to_pandas(types_mapper=pd.ArrowDtype)


def write_parquet(df: pd.DataFrame, path: str, compression: str = "snappy") -> None:
    """Write a pandas DataFrame to a Parquet file.

    `compression` accepts anything pyarrow.parquet.write_table does
    ("snappy", "zstd", "gzip", "brotli", "lz4", "none", ...) -- pyarrow ships
    a real zstd ENCODER natively (unlike the JS side's hyparquet-writer,
    which needed a bring-your-own WASM compressor -- see
    mallory-frame-parquet's src/zstd.ts), so there's no analogous footgun
    to guard against here.

    Uses ``dataframe_to_table`` (not a bare ``pa.Table.from_pandas``) so
    genuine NaN values in plain float columns round-trip as NaN rather than
    being silently coerced to Arrow null (see ``_arrow_nan.py``).
    """
    table = dataframe_to_table(df, preserve_index=False)
    pq.write_table(table, path, compression=compression)
