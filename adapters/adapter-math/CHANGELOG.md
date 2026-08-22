# @johnhenry/math-plus-adapter-math

## 0.2.0

### Minor Changes

- 262a154: Add `linalg.powerIteration`: matrix-free dominant (Perron) eigenvalue via a `matvec` closure, so a height-h strip transfer matrix (`|tiles|^h x |tiles|^h` for the generalized Wang tile laboratory, johnhenry/mallory#92) never needs materializing. `eigGeneral` remains the differential-test oracle for small, materialized cases. Fixes #84 (item 4 of 4, closes the issue).

### Patch Changes

- Updated dependencies [262a154]
  - @johnhenry/math-plus-tensor-core@0.2.0
  - @johnhenry/math-plus-tensor-compile@0.0.5

## 0.1.2

### Patch Changes

- 76ad8a7: Widen internal `@johnhenry/math`/`@johnhenry/math-plus-tensor-core`/`@johnhenry/math-plus-tensor-compile` dependency ranges from exact/narrow-caret pins to `>=X <1.0.0`. Under npm's 0.x caret semantics, `^0.8.0` excludes `0.9.0+`, so any consumer already depending on `@johnhenry/math@^0.9.0` (e.g. mallory-graph) got a second, nested, older `@johnhenry/math` copy — silently correct at the JS level, but a `ComplexNumber` returned by `eigGeneral` (typed against the nested class) was rejected by TypeScript as unassignable to the app's own (structurally different) `ComplexNumber` type. Fixes #83.
  - @johnhenry/math-plus-tensor-compile@0.0.4

## 0.1.1

### Patch Changes

- @johnhenry/math-plus-tensor-compile@0.0.3

## 0.1.0

### Minor Changes

- aeeeb35: Gap-analysis backlog (issues #64-#72): additive new API surface across six packages, all backward-compatible.

  - **@johnhenry/math-plus-tensor-core**: eager `Tensor` unary op-table parity with the compiled IR (`exp`/`pow`/`abs`/`neg`/full trig+hyperbolic families/`cbrt`/`log10`/`log2`/`expm1`/`log1p`/`floor`/`ceil`/`round`/`trunc`), plus structural ops `clip`/`prod`/`pad`/`split`/`repeat`/`flip`/`roll`/`nonzero`.
  - **@johnhenry/math-plus-tensor-wasm**: `subInto`/`divInto` WASM kernels, parity with the existing `addInto`/`mulInto`.
  - **@johnhenry/math-plus-adapter-math**: `det`/`inv` (derived from the existing `lu`/`solve`), and `eigGeneral` — eigenvalues of a general non-symmetric real matrix via Hessenberg reduction + shifted QR, returned as `ComplexNumber[]` to support genuine complex-conjugate pairs.
  - **@johnhenry/math-plus-fft**: `fft2`/`ifft2` and `fftshift`/`ifftshift`.
  - **@johnhenry/math-plus-signal**: `correlate`/`correlate1D` (convolution's cross-correlation dual), `freqz` (SOS filter frequency response), `welch` (power spectral density).
  - **@johnhenry/math-plus-tensor-autograd**: `nn.Sequential`, `nn.Dropout`, `nn.huberLoss`, `nn.binaryCrossEntropy`; `optim.Adam`, `optim.RMSprop`, `optim.StepLR` (a learning-rate scheduler — `optim.SGD`/`optim.AdamW`'s `lr` field is now mutable rather than `readonly` to support this, a backward-compatible widening).

### Patch Changes

- Updated dependencies [aeeeb35]
  - @johnhenry/math-plus-tensor-core@0.1.0
  - @johnhenry/math-plus-tensor-compile@0.0.2
