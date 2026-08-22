"""Internal helper (issue #103): ``pa.Table.from_pandas`` silently coerces
genuine NaN float values to Arrow nulls, collapsing "not-a-number" and
"missing" into the same representation on write. That's the right default
for pandas' own nullable extension dtypes (``Float64``, etc.), which already
fold NaN into ``pd.NA`` before pyarrow ever sees the column -- there's no
distinct NaN to lose there. But for plain NumPy-backed float columns (the
common case: a DataFrame built from ``float`` literals/``np.nan``, no
extension dtype requested), a real NaN and a real missing value are
different things, and pyarrow's pandas-integration path erases that
distinction unless told not to via ``from_pandas=False`` on the low-level
``pa.array()`` constructor (see ``pa.array``'s own docs: "from_pandas: bool,
default None ... if True, ... NaN and None will both be converted to null
values, irrespective of type").

Not part of the package's public API -- ``ipc.py``/``parquet.py`` use this in
place of a bare ``pa.Table.from_pandas(df, preserve_index=False)`` call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa


def dataframe_to_table(df: pd.DataFrame, *, preserve_index: bool = False) -> pa.Table:
    """Like ``pa.Table.from_pandas(df, preserve_index=preserve_index)``, but
    preserves genuine NaN values in plain NumPy-backed float columns as
    Arrow NaN rather than coercing them to Arrow null.
    """
    table = pa.Table.from_pandas(df, preserve_index=preserve_index)
    for name in df.columns:
        dtype = df[name].dtype
        # Only plain NumPy float dtypes (float16/32/64) are affected -- pandas
        # nullable extension dtypes (e.g. Float64Dtype) are not `np.dtype`
        # instances and already disambiguate NaN vs. missing via `pd.NA`
        # before this ever runs, so leave those to the default conversion.
        if isinstance(dtype, np.dtype) and dtype.kind == "f":
            idx = table.column_names.index(name)
            field = table.schema.field(idx)
            fixed = pa.array(df[name].to_numpy(), type=field.type, from_pandas=False)
            table = table.set_column(idx, field, fixed)
    return table
