"""`.npy`/`.npz` helpers (issue #21) -- largely thin, since these formats are
NumPy-native on the Python side already (``numpy.load``/``numpy.save`` ARE
the interop point; there is no pandas/Arrow layer to bridge here the way
``ipc.py``/``parquet.py`` bridge pyarrow<->pandas). These wrappers exist for
API-surface discoverability and symmetry with @johnhenry/math-plus-tensor-core's own
``.npy`` read/write (packages/tensor-core/src/npy.ts) -- not because
``numpy.load``/``numpy.save`` need wrapping for correctness.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_npy(path: str) -> np.ndarray:
    """Load a single array from a `.npy` file -- an alias for `numpy.load`."""
    return np.load(path)


def save_npy(path: str, array: np.ndarray) -> None:
    """Save a single array to a `.npy` file -- an alias for `numpy.save`."""
    np.save(path, array)


def load_npz(path: str) -> dict[str, np.ndarray]:
    """Load every array from a `.npz` archive into a plain dict (eagerly, unlike
    `numpy.load`'s lazy `NpzFile`, since the ORIGINAL AS3-era Mallory /
    @johnhenry/math-plus-tensor-core naming convention this bridges to expects named
    tensors as a plain mapping, not a file handle)."""
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def save_npz(path: str, arrays: dict[str, np.ndarray], compressed: bool = False) -> None:
    """Save a dict of named arrays to a `.npz` archive."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


__all__ = ["load_npy", "save_npy", "load_npz", "save_npz"]
