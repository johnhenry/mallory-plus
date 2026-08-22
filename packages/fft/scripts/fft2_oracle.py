#!/usr/bin/env python3
"""NumPy oracle for @johnhenry/math-plus-fft's fft2/ifft2/fftshift/ifftshift (issue #69)
and fftn/ifftn (issue #84).

Reads {"op": "fft2"|"ifft2"|"fftshift"|"ifftshift"|"fftn"|"ifftn", "real":
[[...]], "imag": [[...]], "axes": [...]?} on stdin, writes {"real": [[...]],
"imag": [[...]]} to stdout. Plain JSON both ways (matching eig_oracle.py's
precedent) -- ComplexTensor doesn't map cleanly to a single .npy file.
"""
import json
import sys

import numpy as np


def main() -> None:
    job = json.load(sys.stdin)
    a = np.array(job["real"], dtype="float64") + 1j * np.array(job["imag"], dtype="float64")
    op = job["op"]
    axes = job.get("axes")
    if op == "fft2":
        result = np.fft.fft2(a)
    elif op == "ifft2":
        result = np.fft.ifft2(a)
    elif op == "fftshift":
        result = np.fft.fftshift(a, axes=axes)
    elif op == "ifftshift":
        result = np.fft.ifftshift(a, axes=axes)
    elif op == "fftn":
        result = np.fft.fftn(a, axes=axes)
    elif op == "ifftn":
        result = np.fft.ifftn(a, axes=axes)
    else:
        raise SystemExit(f"unknown op {op!r}")
    json.dump({"real": result.real.tolist(), "imag": result.imag.tolist()}, sys.stdout)


if __name__ == "__main__":
    main()
