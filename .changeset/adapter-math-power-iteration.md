---
"mallory-adapter-math": minor
---

Add `linalg.powerIteration`: matrix-free dominant (Perron) eigenvalue via a `matvec` closure, so a height-h strip transfer matrix (`|tiles|^h x |tiles|^h` for the generalized Wang tile laboratory, johnhenry/mallory-graph#92) never needs materializing. `eigGeneral` remains the differential-test oracle for small, materialized cases. Fixes #84 (item 4 of 4, closes the issue).
