---
"mallory-signal": minor
---

Add `correlate2D`: true 2-D cross-correlation via FFT (existing `correlate` is 1-D with row/column batching, not genuine 2-D). Same `correlate(a,b) === convolve(a, flip(b))` convention as `correlate1D`. Upstream for the generalized Wang tile laboratory's autocorrelation-surface analysis (johnhenry/mallory-graph#92). Fixes #84 (item 3 of 4).
