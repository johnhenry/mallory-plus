# mallory-signal

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
  - mallory-fft@0.1.0
