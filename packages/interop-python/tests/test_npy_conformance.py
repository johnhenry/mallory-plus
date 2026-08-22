"""`.npy`/`.npz` helper tests (issue #21) -- thin wrapper sanity checks, not
a conformance suite in the bidirectional sense (see module doc on npy.py):
`.npy`/`.npz` are NumPy-native on both sides already (@johnhenry/math-plus-tensor-core's
own `.npy` reader/writer, packages/tensor-core/src/npy.ts, targets the same
format numpy.save/numpy.load do -- that conformance is already covered by
tensor-core's own NumPy-oracle differential tests, see docs/TESTING.md).
"""

import tempfile
from pathlib import Path

import numpy as np

from math_plus_interop import load_npy, load_npz, save_npy, save_npz

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_npy_reads_committed_fixture():
    arr = load_npy(str(FIXTURES / "py_written.npy"))
    np.testing.assert_array_equal(arr, np.array([1.0, 2.5, -3.0, 4.25, 0.0]))


def test_load_npz_reads_committed_fixture_as_a_dict():
    arrays = load_npz(str(FIXTURES / "py_written.npz"))
    assert set(arrays.keys()) == {"a", "b"}
    np.testing.assert_array_equal(arrays["a"], np.array([1.0, 2.5, -3.0, 4.25, 0.0]))
    np.testing.assert_array_equal(arrays["b"], np.array([1, 2, 3], dtype=np.int32))


def test_save_npy_then_load_npy_round_trips():
    arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "roundtrip.npy")
        save_npy(path, arr)
        back = load_npy(path)
        np.testing.assert_array_equal(back, arr)


def test_save_npz_then_load_npz_round_trips_compressed_and_uncompressed():
    arrays = {"m": np.eye(3), "v": np.array([1, 2, 3])}
    with tempfile.TemporaryDirectory() as tmp:
        for compressed in (False, True):
            path = str(Path(tmp) / f"roundtrip_{compressed}.npz")
            save_npz(path, arrays, compressed=compressed)
            back = load_npz(path)
            np.testing.assert_array_equal(back["m"], arrays["m"])
            np.testing.assert_array_equal(back["v"], arrays["v"])
