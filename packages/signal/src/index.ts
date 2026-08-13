/**
 * mallory-signal — the signal-processing slice of the v2/v3 SciPy-
 * equivalent bundle (issue #44): `convolve`, `stft`/`istft`, `findPeaks`,
 * `sosFilter`, `butter`, `resamplePoly`. See each module's own doc comment
 * for algorithm sources and v1 scope decisions.
 */
export { convolve, convolve1D, type ConvolveMode, type ConvolveOptions } from "./convolve.ts";
export { butter, type ButterOptions, type FilterType } from "./filter-design.ts";
export { findPeaks, type FindPeaksOptions, type FindPeaksResult } from "./find-peaks.ts";
export { resamplePoly } from "./resample.ts";
export { sosFilter, type Sos, type SosSection } from "./sos-filter.ts";
export { istft, stft, type StftOptions } from "./stft.ts";
export { hammingWindow, hannWindow } from "./window.ts";
