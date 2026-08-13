/** Hann window (issue #44's stft/istft dependency). `periodic=true` (the default) matches `scipy.signal.get_window`'s "periodic" convention (denominator `n`, not `n-1`) -- what `scipy.signal.stft` uses internally by default for its own Hann window, needed to differential-test against it exactly. */
export function hannWindow(n: number, periodic = true): Float64Array {
  const w = new Float64Array(n);
  const denom = periodic ? n : n - 1;
  if (denom === 0) {
    w[0] = 1;
    return w;
  }
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / denom);
  return w;
}

/** Hamming window (symmetric, `sym=True` convention -- denominator `n-1`), `resamplePoly`'s anti-aliasing FIR filter dependency. */
export function hammingWindow(n: number): Float64Array {
  const w = new Float64Array(n);
  if (n === 1) {
    w[0] = 1;
    return w;
  }
  for (let i = 0; i < n; i++) w[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (n - 1));
  return w;
}
