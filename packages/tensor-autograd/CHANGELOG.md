# mallory-tensor-autograd

## 0.2.2

### Patch Changes

- Updated dependencies [262a154]
  - mallory-tensor-core@0.2.0

## 0.2.1

### Patch Changes

- 3f3824a: Fix `nn.binaryCrossEntropy` returning NaN once a classifier converges well (saturated logits, `|z| >~ 37`). Reformulated using the standard numerically-stable BCEWithLogits formula, `relu(z) - z*y + log(1+exp(-|z|))`, built from existing `relu`/`sigmoid`/`log` ops (no `exp`/`abs` ops needed — `log(1+exp(-|z|))` rewritten as `-log(sigmoid(|z|))`, and `|z|` as `relu(z) + relu(-z)`). Byte-equivalent to the prior formula in the non-saturated regime (verified to ~1e-15); now finite everywhere. Fixes #85.

## 0.2.0

### Minor Changes

- 21981eb: Fixes johnhenry/mallory-plus#89: `optim.SGD` gains an optional `momentum`/`nesterov` option, following PyTorch's own update convention (`buf = momentum*buf + grad`, Nesterov's lookahead `d_p = grad + momentum*buf` applied after the buffer update). Both default to off (`0`/`false`), so `new SGD(params, { lr })` is byte-identical to the pre-#89 plain-SGD update -- no existing caller's behavior changes. Constructing with `nesterov: true` and no (or zero) `momentum` throws a `RangeError`, since Nesterov's lookahead is meaningless without a momentum term to look ahead with.

## 0.1.0

### Minor Changes

- aeeeb35: Gap-analysis backlog (issues #64-#72): additive new API surface across six packages, all backward-compatible.

  - **mallory-tensor-core**: eager `Tensor` unary op-table parity with the compiled IR (`exp`/`pow`/`abs`/`neg`/full trig+hyperbolic families/`cbrt`/`log10`/`log2`/`expm1`/`log1p`/`floor`/`ceil`/`round`/`trunc`), plus structural ops `clip`/`prod`/`pad`/`split`/`repeat`/`flip`/`roll`/`nonzero`.
  - **mallory-tensor-wasm**: `subInto`/`divInto` WASM kernels, parity with the existing `addInto`/`mulInto`.
  - **mallory-adapter-math**: `det`/`inv` (derived from the existing `lu`/`solve`), and `eigGeneral` — eigenvalues of a general non-symmetric real matrix via Hessenberg reduction + shifted QR, returned as `ComplexNumber[]` to support genuine complex-conjugate pairs.
  - **mallory-fft**: `fft2`/`ifft2` and `fftshift`/`ifftshift`.
  - **mallory-signal**: `correlate`/`correlate1D` (convolution's cross-correlation dual), `freqz` (SOS filter frequency response), `welch` (power spectral density).
  - **mallory-tensor-autograd**: `nn.Sequential`, `nn.Dropout`, `nn.huberLoss`, `nn.binaryCrossEntropy`; `optim.Adam`, `optim.RMSprop`, `optim.StepLR` (a learning-rate scheduler — `optim.SGD`/`optim.AdamW`'s `lr` field is now mutable rather than `readonly` to support this, a backward-compatible widening).

### Patch Changes

- Updated dependencies [aeeeb35]
  - mallory-tensor-core@0.1.0
