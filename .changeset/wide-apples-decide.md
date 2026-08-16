---
"mallory-adapter-math": patch
---

Widen internal `mallory-math`/`mallory-tensor-core`/`mallory-tensor-compile` dependency ranges from exact/narrow-caret pins to `>=X <1.0.0`. Under npm's 0.x caret semantics, `^0.8.0` excludes `0.9.0+`, so any consumer already depending on `mallory-math@^0.9.0` (e.g. mallory-graph) got a second, nested, older `mallory-math` copy — silently correct at the JS level, but a `ComplexNumber` returned by `eigGeneral` (typed against the nested class) was rejected by TypeScript as unassignable to the app's own (structurally different) `ComplexNumber` type. Fixes #83.
