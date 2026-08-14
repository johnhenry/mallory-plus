#!/usr/bin/env python3
"""NumPy eigenvalue oracle for adapter-math's eigGeneral (issue #68).

Reads a JSON job {"matrix": [[...], ...]} on stdin, writes
{"eigenvalues": [[re, im], ...]} (numpy.linalg.eigvals, unordered) to
stdout. Plain JSON both ways -- eigGeneral returns ComplexNumber[], not a
Tensor, so there's no .npy round-trip to dogfood here unlike the other
oracles in this repo.
"""
import json
import sys

import numpy as np


def main() -> None:
    job = json.load(sys.stdin)
    a = np.array(job["matrix"], dtype="float64")
    values = np.linalg.eigvals(a)
    json.dump({"eigenvalues": [[complex(v).real, complex(v).imag] for v in values]}, sys.stdout)


if __name__ == "__main__":
    main()
