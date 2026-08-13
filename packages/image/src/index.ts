/**
 * mallory-image — resize/normalize tensor ops (issue #41), the v2
 * "practical ML/media compute" bundle's image slice. Scoped tightly: not a
 * general image-processing library.
 */
export { normalize, type NormalizeOptions } from "./normalize.ts";
export { resize, type ResizeMethod, type ResizeOptions, type ResizeSize } from "./resize.ts";
