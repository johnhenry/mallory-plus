"""mallory-interop -- the Python-side half of mallory-plus's interop story
(issue #21). Ships to PyPI as `mallory-interop`, installed alongside
pyarrow/pandas.

Explicitly NOT the Arrow C Data Interface / PyCapsule zero-copy bridge --
deferred, since it only pays off with a native Node addon in the same
process, which is downstream of mallory-plus's WASM work. Explicitly NOT
built on Python's `__dataframe__()` DataFrame Interchange Protocol either --
pandas deprecates it and drops the fallback entirely in pandas 4.0; Arrow
IPC is the portable interop format for browser/Node/Deno/Python alike.
"""

from .ipc import read_ipc, write_ipc
from .npy import load_npy, load_npz, save_npy, save_npz
from .parquet import read_parquet, write_parquet

__all__ = [
    "read_ipc",
    "write_ipc",
    "read_parquet",
    "write_parquet",
    "load_npy",
    "save_npy",
    "load_npz",
    "save_npz",
]

__version__ = "0.0.1"
