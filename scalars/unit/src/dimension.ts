/**
 * The 7 SI base dimensions, represented as an exponent vector. `Dimension`
 * arithmetic (multiply/divide/pow) is just per-axis addition/subtraction/
 * scaling of exponents — the standard way dimensional analysis is modeled.
 */
export const BASE_DIMENSIONS = [
  "length",
  "mass",
  "time",
  "current",
  "temperature",
  "amount",
  "luminosity",
] as const;

export type BaseDimension = (typeof BASE_DIMENSIONS)[number];
export type Dimension = Readonly<Record<BaseDimension, number>>;

export const DIMENSIONLESS: Dimension = Object.freeze({
  length: 0,
  mass: 0,
  time: 0,
  current: 0,
  temperature: 0,
  amount: 0,
  luminosity: 0,
});

/** Build a `Dimension` from a partial exponent map; unmentioned axes default to 0. */
export function dim(partial: Partial<Record<BaseDimension, number>>): Dimension {
  return Object.freeze({ ...DIMENSIONLESS, ...partial });
}

export function multiplyDimensions(a: Dimension, b: Dimension): Dimension {
  const out = {} as Record<BaseDimension, number>;
  for (const k of BASE_DIMENSIONS) out[k] = a[k] + b[k];
  return Object.freeze(out);
}

export function divideDimensions(a: Dimension, b: Dimension): Dimension {
  const out = {} as Record<BaseDimension, number>;
  for (const k of BASE_DIMENSIONS) out[k] = a[k] - b[k];
  return Object.freeze(out);
}

export function powDimension(a: Dimension, n: number): Dimension {
  const out = {} as Record<BaseDimension, number>;
  for (const k of BASE_DIMENSIONS) out[k] = a[k] * n;
  return Object.freeze(out);
}

export function dimensionsEqual(a: Dimension, b: Dimension): boolean {
  return BASE_DIMENSIONS.every((k) => a[k] === b[k]);
}

export function isDimensionless(d: Dimension): boolean {
  return BASE_DIMENSIONS.every((k) => d[k] === 0);
}

const SYMBOLS: Record<BaseDimension, string> = {
  length: "L",
  mass: "M",
  time: "T",
  current: "I",
  temperature: "Θ", // Theta
  amount: "N",
  luminosity: "J",
};

/** Human-readable exponent-vector form for error messages, e.g. "L*T^-2". */
export function dimensionToString(d: Dimension): string {
  const parts = BASE_DIMENSIONS.filter((k) => d[k] !== 0).map((k) =>
    d[k] === 1 ? SYMBOLS[k] : `${SYMBOLS[k]}^${d[k]}`,
  );
  return parts.length > 0 ? parts.join("*") : "dimensionless";
}
