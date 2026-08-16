---
"mallory-fft": minor
---

Add `fftn`/`ifftn`: n-D FFT, the general case `fft2`/`ifft2` already cover in 2-D. Separable (one `fft` pass per axis), optional `axes` subset. Upstream for the generalized Wang tile laboratory's diffraction-spectrum machinery on Wang cubes (johnhenry/mallory-graph#92). Fixes #84 (item 2 of 4).
