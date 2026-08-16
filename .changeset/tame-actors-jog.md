---
"mallory-tensor-autograd": patch
---

Fix `nn.binaryCrossEntropy` returning NaN once a classifier converges well (saturated logits, `|z| >~ 37`). Reformulated using the standard numerically-stable BCEWithLogits formula, `relu(z) - z*y + log(1+exp(-|z|))`, built from existing `relu`/`sigmoid`/`log` ops (no `exp`/`abs` ops needed — `log(1+exp(-|z|))` rewritten as `-log(sigmoid(|z|))`, and `|z|` as `relu(z) + relu(-z)`). Byte-equivalent to the prior formula in the non-saturated regime (verified to ~1e-15); now finite everywhere. Fixes #85.
