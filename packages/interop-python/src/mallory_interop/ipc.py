"""Arrow IPC convenience wrappers (issue #21).

Thin wrappers around pyarrow's IPC reader/writer, matching the file-format
side of what mallory-frame-arrow's ``Frame.fromIPC``/``.toIPC()`` produce
and consume on the JS side (Arrow IPC **file** format, not stream format --
matches docs/spikes/arrow-parity.md's methodology, which verified both
formats round-trip but frame-arrow's own toIPC()/fromIPC() use file format).
"""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from ._arrow_nan import dataframe_to_table


def read_ipc(path: str) -> pd.DataFrame:
    """Read an Arrow IPC file into a pandas DataFrame.

    Uses ``types_mapper=pd.ArrowDtype`` so nullable extension dtypes (e.g. a
    nullable ``int32``/``bool`` column written by mallory-frame-arrow) round-trip
    exactly instead of pandas' classic NumPy-backed dtypes silently coercing
    nullable ints to float64 (docs/spikes/arrow-parity.md's own recorded gotcha).
    """
    with pa.memory_map(path, "rb") as source:
        table = ipc.open_file(source).read_all()
    return table.to_pandas(types_mapper=pd.ArrowDtype)


def write_ipc(df: pd.DataFrame, path: str) -> None:
    """Write a pandas DataFrame to an Arrow IPC file.

    Uses ``dataframe_to_table`` (not a bare ``pa.Table.from_pandas``) so
    genuine NaN values in plain float columns round-trip as NaN rather than
    being silently coerced to Arrow null (see ``_arrow_nan.py``).
    """
    table = dataframe_to_table(df, preserve_index=False)
    with pa.OSFile(path, "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
