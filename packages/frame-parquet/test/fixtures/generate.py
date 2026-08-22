#!/usr/bin/env python3
"""
Generate the committed .parquet fixtures under test/fixtures/ using real
pyarrow (not hand-rolled buffers — see docs/TESTING.md's MATH_PLUS_ORACLE_PYTHON
convention and docs/spikes/parquet-bakeoff.md's own methodology, which this
mirrors).

Run via nix-shell on trycooy (pip wheels don't work on NixOS):

    nix-shell -p "python3.withPackages(ps: [ps.pyarrow ps.pandas])" \
      --run "python3 test/fixtures/generate.py"

The .parquet bytes this script produces are committed to the repo so tests
never need a Python interpreter at run time — only this generator does.
"""
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))


def write(table, name, **kwargs):
    path = os.path.join(HERE, name)
    pq.write_table(table, path, **kwargs)
    print(f"wrote {name}: {pq.ParquetFile(path).num_row_groups} row groups, {table.num_rows} rows")


# ---------------------------------------------------------------------------
# basic.parquet / basic_zstd.parquet — one of every v1-supported scalar
# dtype, with nulls at different strides, several row groups. Mirrors the
# spike's own fixture shape (f64/i64/utf8/bool with nulls, multi-row-group).
# ---------------------------------------------------------------------------
N = 100
f64 = [float(i) + 0.5 if i % 7 != 0 else None for i in range(N)]
i64 = [i * 1000 if i % 11 != 0 else None for i in range(N)]
i32 = [i - 50 if i % 6 != 0 else None for i in range(N)]
u32 = [i * 3 if i % 9 != 0 else None for i in range(N)]
f32 = [float(i) * 1.5 if i % 8 != 0 else None for i in range(N)]
utf8 = [f"row-{i}" if i % 13 != 0 else None for i in range(N)]
boolean = [(i % 2 == 0) if i % 17 != 0 else None for i in range(N)]
ts_ms = [pa.scalar(1_700_000_000_000 + i * 1000, type=pa.timestamp("ms")) if i % 5 != 0 else None for i in range(N)]
ts_us = [pa.scalar(1_700_000_000_000_000 + i * 137, type=pa.timestamp("us")) if i % 5 != 0 else None for i in range(N)]

basic_table = pa.table({
    "f64": pa.array(f64, type=pa.float64()),
    "i64": pa.array(i64, type=pa.int64()),
    "i32": pa.array(i32, type=pa.int32()),
    "u32": pa.array(u32, type=pa.uint32()),
    "f32": pa.array(f32, type=pa.float32()),
    "utf8": pa.array(utf8, type=pa.utf8()),
    "bool": pa.array(boolean, type=pa.bool_()),
    "ts_ms": pa.array(ts_ms, type=pa.timestamp("ms")),
    "ts_us": pa.array(ts_us, type=pa.timestamp("us")),
})

write(basic_table, "basic.parquet", compression="snappy", row_group_size=25)
write(basic_table, "basic_zstd.parquet", compression="zstd", row_group_size=25)

expected = {
    "f64": f64, "i64": i64, "i32": i32, "u32": u32, "f32": f32,
    "utf8": utf8, "bool": boolean,
    "ts_ms": [None if v is None else v.value for v in ts_ms],
    "ts_us": [None if v is None else v.value for v in ts_us],
}
with open(os.path.join(HERE, "basic_expected.json"), "w") as f:
    json.dump(expected, f)

# ---------------------------------------------------------------------------
# pushdown.parquet — a bigger, decisively-monotonic file for the
# I/O-skipping proof (test/pushdown.test.ts): several row groups, a
# monotonic `value` column so a $gt filter provably skips whole groups via
# statistics, matching the spike's own row-group-skip methodology.
# ---------------------------------------------------------------------------
PN = 6000
ROW_GROUP = 500  # -> 12 row groups
pd_id = list(range(PN))
pd_value = [float(i) for i in range(PN)]  # strictly monotonic -> decisive min/max stats
pd_label = [f"item-{i % 37}" for i in range(PN)]

pushdown_table = pa.table({
    "id": pa.array(pd_id, type=pa.int64()),
    "value": pa.array(pd_value, type=pa.float64()),
    "label": pa.array(pd_label, type=pa.utf8()),
})
write(pushdown_table, "pushdown.parquet", compression="snappy", row_group_size=ROW_GROUP)

# ---------------------------------------------------------------------------
# quirks.parquet — a dictionary-encoded low-cardinality utf8 column, to
# prove frame-parquet's read path decodes it as a plain "utf8" Frame column
# (dictionary encoding not preserved — documented v1 simplification, see
# schema.ts's module doc).
# ---------------------------------------------------------------------------
cats = ["alpha", "beta", "gamma"]
dict_values = [cats[i % 3] for i in range(N)]
dict_col = pa.array(dict_values).dictionary_encode()
quirks_table = pa.table({
    "id": pa.array(list(range(N)), type=pa.int64()),
    "cat": dict_col,
})
write(quirks_table, "quirks.parquet", compression="snappy", row_group_size=40)
with open(os.path.join(HERE, "quirks_expected.json"), "w") as f:
    json.dump({"cat": dict_values}, f)

# ---------------------------------------------------------------------------
# nested_map.parquet — a MAP column, to prove readParquet throws a clear,
# named UnsupportedParquetTypeError rather than silently mishandling or
# crashing on a Parquet type this package's v1 doesn't support. (LIST/STRUCT
# used to serve this role before issue #30 added real support for them —
# MAP is the one nested type that's still a genuine gap, since frame-arrow
# has no map DType to map it to.)
# ---------------------------------------------------------------------------
nested_map_table = pa.table({
    "id": pa.array(list(range(10)), type=pa.int64()),
    "attrs": pa.array([{f"k{i}": i * 2} for i in range(10)], type=pa.map_(pa.string(), pa.int64())),
})
write(nested_map_table, "nested_map.parquet", compression="snappy")

# ---------------------------------------------------------------------------
# nested_list.parquet — list<double>, issue #30: proves readParquet maps a
# standard 3-level-convention Parquet LIST column to frame-arrow's list<T>,
# with nulls at every level distinguished: a null list (None), an empty list
# ([]), and null elements inside a non-null list. Several row groups, like
# this package's other fixtures.
# ---------------------------------------------------------------------------
LN = 60


def make_list(i):
    if i % 11 == 0:
        return None  # null list
    if i % 7 == 0:
        return []  # empty, non-null list
    length = (i % 4) + 1
    return [float(i) + 0.25 * j if (i + j) % 5 != 0 else None for j in range(length)]


list_values = [make_list(i) for i in range(LN)]
nested_list_table = pa.table({
    "id": pa.array(list(range(LN)), type=pa.int64()),
    "values": pa.array(list_values, type=pa.list_(pa.float64())),
})
write(nested_list_table, "nested_list.parquet", compression="snappy", row_group_size=20)
with open(os.path.join(HERE, "nested_list_expected.json"), "w") as f:
    json.dump({"values": list_values}, f)

# ---------------------------------------------------------------------------
# nested_struct.parquet — struct<a: double, b: string>, issue #30: proves
# readParquet maps a flat Parquet STRUCT group to frame-arrow's struct<...>,
# with nulls at every level distinguished: a null struct (None) vs. a
# non-null struct with a null field. Several row groups.
# ---------------------------------------------------------------------------
SN = 60


def make_struct(i):
    if i % 9 == 0:
        return None  # null struct
    a = None if i % 5 == 0 else float(i) + 0.5
    b = None if i % 6 == 0 else f"s-{i}"
    return {"a": a, "b": b}


struct_values = [make_struct(i) for i in range(SN)]
nested_struct_table = pa.table({
    "id": pa.array(list(range(SN)), type=pa.int64()),
    "point": pa.array(struct_values, type=pa.struct([("a", pa.float64()), ("b", pa.string())])),
})
write(nested_struct_table, "nested_struct.parquet", compression="snappy", row_group_size=20)
with open(os.path.join(HERE, "nested_struct_expected.json"), "w") as f:
    json.dump({"point": struct_values}, f)

# ---------------------------------------------------------------------------
# scan/part-*.parquet — three same-schema partitions for the scanParquet
# glob test.
# ---------------------------------------------------------------------------
scan_dir = os.path.join(HERE, "scan")
os.makedirs(scan_dir, exist_ok=True)
for part in range(3):
    part_ids = list(range(part * 10, part * 10 + 10))
    part_table = pa.table({
        "id": pa.array(part_ids, type=pa.int64()),
        "part": pa.array([part] * 10, type=pa.int32()),
        "name": pa.array([f"p{part}-{i}" for i in part_ids], type=pa.utf8()),
    })
    write(part_table, os.path.join("scan", f"part-{part}.parquet"), compression="snappy")

print("done")
