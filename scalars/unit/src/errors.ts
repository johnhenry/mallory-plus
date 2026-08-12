import { type Dimension, dimensionToString } from "./dimension.ts";

/** Any malformed unit expression — an unknown symbol, a bad exponent, an unsupported composition. */
export class UnitParseError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UnitParseError";
  }
}

/** A specific, common `UnitParseError`: a symbol (e.g. after stripping any prefix) that isn't in the unit table. */
export class UnknownUnitError extends UnitParseError {
  readonly symbol: string;
  constructor(symbol: string) {
    super(`unknown unit "${symbol}"`);
    this.name = "UnknownUnitError";
    this.symbol = symbol;
  }
}

/** Thrown by `.to()`/`.add()`/`.sub()` when the two sides' dimensions don't match — the dimensional-analysis check the issue asks for. */
export class DimensionMismatchError extends Error {
  readonly left: Dimension;
  readonly right: Dimension;
  constructor(left: Dimension, right: Dimension, message?: string) {
    super(message ?? `dimension mismatch: [${dimensionToString(left)}] vs [${dimensionToString(right)}]`);
    this.name = "DimensionMismatchError";
    this.left = left;
    this.right = right;
  }
}
