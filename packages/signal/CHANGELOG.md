# @johnhenry/math-plus-signal

## 0.3.0

### Minor Changes

- 262a154: Add `correlate2D`: true 2-D cross-correlation via FFT (existing `correlate` is 1-D with row/column batching, not genuine 2-D). Same `correlate(a,b) === convolve(a, flip(b))` convention as `correlate1D`. Upstream for the generalized Wang tile laboratory's autocorrelation-surface analysis (johnhenry/mallory#92). Fixes #84 (item 3 of 4).

### Patch Changes

- Updated dependencies [262a154]
- Updated dependencies [262a154]
  - @johnhenry/math-plus-fft@0.2.0
  - @johnhenry/math-plus-tensor-core@0.2.0

## 0.2.0

### Minor Changes

- 59162f6: Fixes johnhenry/math-plus#90: `butter()` gains `"bandpass"`/`"bandstop"` support, alongside the existing `"lowpass"`/`"highpass"`. `wn` takes a `[low, high]` pair for the two new types (a `number` still works for lowpass/highpass, unchanged and non-breaking) -- expressed via function overloads so the compiler enforces the right shape per `btype` at the call site.

  Implements scipy's `lp2bp_zpk`/`lp2bs_zpk` analog frequency transforms, and replaces the old lowpass/highpass-only `zpk2sos` shortcut (which assumed every digital zero was real and identical, a shape bandpass/bandstop's zeros don't have) with a general real-coefficient pairing: complex-conjugate pairs and leftover real values are grouped independently on the zero side and the pole side, which is provably always the same group count on both sides for any real-coefficient system with equal zero/pole counts -- see the module's own doc comment for the parity argument. Verified byte-identical output to the pre-#90 specialized `zpk2sos` for lowpass/highpass's own shape, and differentially tested end-to-end (via `sosFilter` vs. scipy's `sosfilt`) against real `scipy.signal.butter` for bandpass and bandstop across multiple orders, including the order-dependent edge case where the prototype's single real pole (odd order only) becomes either a complex-conjugate pair or two real poles depending on bandwidth/center-frequency.

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
  - @johnhenry/math-plus-fft@0.1.0
