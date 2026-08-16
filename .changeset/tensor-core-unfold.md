---
"mallory-tensor-core": minor
---

Add `Tensor.prototype.unfold(windowShape, axes?)`: sliding-window ("patch") view via NumPy's `sliding_window_view` stride trick, never copies. Upstream for the generalized Wang tile laboratory's patch-census machinery (johnhenry/mallory-graph#92). Fixes #84 (item 1 of 4).
