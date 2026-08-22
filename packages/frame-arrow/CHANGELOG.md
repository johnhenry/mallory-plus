# @johnhenry/math-plus-frame-arrow

## 1.0.0

### Patch Changes

- Updated dependencies [262a154]
  - @johnhenry/math-plus-tensor-core@0.2.0

## 0.1.0

### Minor Changes

- 7b0ced4: Fixes johnhenry/math-plus#86: `Frame.fromCSV()`, the reader counterpart to the existing `.toCSV()` writer -- RFC-4180 parsing (quoted fields, doubled-quote escapes, commas/newlines inside quotes, CRLF or LF) plus per-column dtype inference (bool/int64/float64/utf8, widening to the narrowest type every non-empty cell agrees on; large integers stay exact via `BigInt`, unlike a `Number()`-based parser). Ragged rows and unterminated quotes throw a clear error rather than silently mishandling the input.

  Also widens the `@johnhenry/math-plus-tensor-core` peerDependency from an exact pin (`0.1.0`) to a caret range (`^0.1.0`), a backward-compatible relaxation: the exact pin made `npm install` `ERESOLVE` for any consumer (e.g. `mallory-graph`) already depending on tensor-core via its own `^0.1.0` range, even though frame-arrow's actual coupling to tensor-core is a small, stable, dynamically-imported surface with no static or type-level dependency at all.
