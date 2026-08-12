import assert from "node:assert/strict";
import { test } from "node:test";
import { Int32, Table, Utf8, vectorFromArray } from "apache-arrow";
import { Frame } from "../src/index.ts";

function usersFrame(): Frame {
  return Frame.fromArrow(
    new Table({
      id: vectorFromArray([1, 2, 3], new Int32()),
      name: vectorFromArray(["alice", "bob", "carol"], new Utf8()),
    }),
  );
}

function ordersFrame(): Frame {
  return Frame.fromArrow(
    new Table({
      id: vectorFromArray([1, 1, 4], new Int32()),
      item: vectorFromArray(["widget", "gadget", "orphan"], new Utf8()),
    }),
  );
}

test("inner join keeps only matching keys, and duplicates left rows for multiple right matches", () => {
  const joined = usersFrame().join(ordersFrame(), { on: "id", how: "inner" }).sortBy("item");
  const rows = joined.toRows();
  assert.deepEqual(
    rows.map((r) => [r.id, r.name, r.item]),
    [
      [1, "alice", "gadget"],
      [1, "alice", "widget"],
    ],
  );
});

test("left join keeps every left row, nulling right-side columns when unmatched", () => {
  const joined = usersFrame().join(ordersFrame(), { on: "id", how: "left" }).sortBy("name", "item");
  const rows = joined.toRows();
  assert.deepEqual(
    rows.map((r) => [r.id, r.name, r.item]),
    [
      [1, "alice", "gadget"],
      [1, "alice", "widget"],
      [2, "bob", null],
      [3, "carol", null],
    ],
  );
});

test("right join keeps every right row, nulling left-side columns when unmatched", () => {
  const joined = usersFrame().join(ordersFrame(), { on: "id", how: "right" }).sortBy("item");
  const rows = joined.toRows();
  assert.deepEqual(
    rows.map((r) => [r.id, r.name, r.item]),
    [
      [1, "alice", "gadget"],
      [4, null, "orphan"],
      [1, "alice", "widget"],
    ],
  );
});

test("outer join keeps everything from both sides", () => {
  const joined = usersFrame().join(ordersFrame(), { on: "id", how: "outer" });
  assert.equal(joined.length, 5); // alice x2, bob, carol, orphan(id=4)
});

test("overlapping non-key column names get the configured suffix", () => {
  const left = Frame.fromArrow(
    new Table({ id: vectorFromArray([1], new Int32()), label: vectorFromArray(["L"], new Utf8()) }),
  );
  const right = Frame.fromArrow(
    new Table({ id: vectorFromArray([1], new Int32()), label: vectorFromArray(["R"], new Utf8()) }),
  );
  const joined = left.join(right, { on: "id" });
  assert.deepEqual(joined.columns, ["id", "label", "label_right"]);
  assert.deepEqual(joined.toRows(), [{ id: 1, label: "L", label_right: "R" }]);
});
