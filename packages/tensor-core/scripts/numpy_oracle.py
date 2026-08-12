#!/usr/bin/env python3
"""NumPy oracle for mallory-tensor-core differential tests.

Reads a JSON job file:
  {
    "op": "add" | "sub" | "mul" | "div" | "sum" | "mean" | "arange" | "permute" | "slice"
        | "matmul" | "dot" | "cast" | "eq" | "ne" | "lt" | "lte" | "gt" | "gte"
        | "min" | "max" | "argmin" | "argmax",
    "inputs": ["/path/a.npy", ...],       # .npy files written by tensor-core
    "axis": 1,                            # optional (reductions)
    "permutation": [1, 0],                # optional (permute op)
    "scalar": 2.5,                        # optional (binary ops vs scalar)
    "arange": [start, stop, step],        # optional (arange op)
    "dtype": "float32",                   # optional (arange dtype)
    "specs": [[1, 10, 2], null, [null, null, -1]],  # optional (slice op) -- per-axis
                                           # [start, end, step] triples or null for the
                                           # whole axis, matching tensor-core's SliceSpec
    "output": "/path/out.npy"
  }
Computes the equivalent NumPy result and writes it to `output` as .npy.
The exchange format is .npy in both directions, dogfooding tensor-core's I/O.
"""

import json
import sys

import numpy as np

# tensor-core dtype names -> NumPy dtype names. Names already spelled the
# NumPy way (e.g. the "float32" the arange tests pass) fall through unchanged.
DTYPE_MAP = {
    "bool": "bool",
    "u8": "uint8", "i8": "int8",
    "u16": "uint16", "i16": "int16",
    "u32": "uint32", "i32": "int32",
    "u64": "uint64", "i64": "int64",
    "f32": "float32", "f64": "float64",
}


def to_numpy_dtype(name: str) -> str:
    return DTYPE_MAP.get(name, name)


def main() -> None:
    with open(sys.argv[1]) as f:
        job = json.load(f)

    op = job["op"]
    inputs = [np.load(path) for path in job.get("inputs", [])]

    if op in ("add", "sub", "mul", "div"):
        a = inputs[0]
        b = job["scalar"] if "scalar" in job else inputs[1]
        result = {
            "add": lambda: a + b,
            "sub": lambda: a - b,
            "mul": lambda: a * b,
            "div": lambda: a / b,
        }[op]()
    elif op == "sum":
        result = inputs[0].sum(axis=job.get("axis"))
    elif op == "mean":
        result = inputs[0].mean(axis=job.get("axis"))
    elif op == "arange":
        start, stop, step = job["arange"]
        result = np.arange(start, stop, step, dtype=job.get("dtype", "float32"))
    elif op == "permute":
        result = np.ascontiguousarray(inputs[0].transpose(job["permutation"]))
    elif op == "slice":
        index = tuple(
            slice(*spec) if spec is not None else slice(None)
            for spec in job["specs"]
        )
        result = np.ascontiguousarray(inputs[0][index])
    elif op == "matmul":
        result = np.ascontiguousarray(np.matmul(inputs[0], inputs[1]))
    elif op == "dot":
        result = np.asarray(np.dot(inputs[0], inputs[1]))
    elif op == "cast":
        result = inputs[0].astype(to_numpy_dtype(job["dtype"]))
    elif op in ("eq", "ne", "lt", "lte", "gt", "gte"):
        a = inputs[0]
        b = job["scalar"] if "scalar" in job else inputs[1]
        result = {
            "eq": lambda: a == b,
            "ne": lambda: a != b,
            "lt": lambda: a < b,
            "lte": lambda: a <= b,
            "gt": lambda: a > b,
            "gte": lambda: a >= b,
        }[op]()
    elif op == "min":
        result = inputs[0].min(axis=job.get("axis"))
    elif op == "max":
        result = inputs[0].max(axis=job.get("axis"))
    elif op == "argmin":
        result = inputs[0].argmin(axis=job.get("axis")).astype("int32")
    elif op == "argmax":
        result = inputs[0].argmax(axis=job.get("axis")).astype("int32")
    else:
        raise SystemExit(f"unknown op {op!r}")

    np.save(job["output"], np.asarray(result))


if __name__ == "__main__":
    main()
