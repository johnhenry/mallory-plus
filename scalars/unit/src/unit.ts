import {
  DIMENSIONLESS,
  dimensionsEqual,
  divideDimensions,
  multiplyDimensions,
  powDimension,
  type Dimension,
} from "./dimension.ts";
import { DimensionMismatchError, UnitParseError } from "./errors.ts";
import { parseUnitExpression } from "./parser.ts";

function composeSymbols(a: string, b: string, op: "*" | "/"): string {
  // v1 does not algebraically simplify composite symbols — multiplying "m"
  // by "m" yields the symbol "m*m", not "m^2". Both parse back to the same
  // dimension and scale (see parser.ts), so correctness isn't affected, only
  // display. See Unit.pow()'s doc comment for why this matters more there.
  return `${a}${op}${b}`;
}

/**
 * A magnitude plus dimension metadata: `Unit.of(55, "cm").to("m")`.
 * Immutable — every method returns a new `Unit`.
 *
 * Units never enter tensor storage (non-goal 3, docs/PLAN.md): a tensor of
 * quantities is `Tensor<"f64">` plus a separate unit annotation, never
 * `Tensor<Unit>`. This package is pure TypeScript with zero dependencies —
 * conversion/parsing is metadata transformation, not bulk numeric work, per
 * the acceleration test (docs/PLAN.md §3).
 */
export class Unit {
  /** The magnitude, expressed in `this.symbol`. */
  readonly value: number;
  /** The unit expression this magnitude is expressed in (`""` for dimensionless). */
  readonly symbol: string;
  readonly dimension: Dimension;

  readonly #scale: number;
  readonly #offset: number;

  private constructor(value: number, symbol: string, dimension: Dimension, scale: number, offset: number) {
    this.value = value;
    this.symbol = symbol;
    this.dimension = dimension;
    this.#scale = scale;
    this.#offset = offset;
  }

  static of(value: number, unitExpr: string): Unit {
    const p = parseUnitExpression(unitExpr);
    return new Unit(value, p.canonical, p.dimension, p.scale, p.offset);
  }

  static dimensionless(value: number): Unit {
    return new Unit(value, "", DIMENSIONLESS, 1, 0);
  }

  get isDimensionless(): boolean {
    return this.symbol === "";
  }

  /** Magnitude in a fixed SI-coherent reference for this dimension — internal to conversion/arithmetic, not part of the public surface. */
  #baseValue(): number {
    return this.value * this.#scale + this.#offset;
  }

  #assertSameDimension(other: Unit, verb: string): void {
    if (!dimensionsEqual(this.dimension, other.dimension)) {
      throw new DimensionMismatchError(
        this.dimension,
        other.dimension,
        `cannot ${verb} "${this.symbol || "(dimensionless)"}" and "${other.symbol || "(dimensionless)"}"`,
      );
    }
  }

  /** Convert to another unit of the SAME dimension. Throws `DimensionMismatchError` otherwise — the dimensional-analysis check. */
  to(unitExpr: string): Unit {
    const p = parseUnitExpression(unitExpr);
    if (!dimensionsEqual(this.dimension, p.dimension)) {
      throw new DimensionMismatchError(this.dimension, p.dimension, `cannot convert "${this.symbol}" to "${unitExpr}"`);
    }
    const value = (this.#baseValue() - p.offset) / p.scale;
    return new Unit(value, p.canonical, p.dimension, p.scale, p.offset);
  }

  add(other: Unit): Unit {
    this.#assertSameDimension(other, "add");
    if (this.isDimensionless) return Unit.dimensionless(this.value + other.value);
    const otherHere = other.to(this.symbol);
    return new Unit(this.value + otherHere.value, this.symbol, this.dimension, this.#scale, this.#offset);
  }

  sub(other: Unit): Unit {
    this.#assertSameDimension(other, "subtract");
    if (this.isDimensionless) return Unit.dimensionless(this.value - other.value);
    const otherHere = other.to(this.symbol);
    return new Unit(this.value - otherHere.value, this.symbol, this.dimension, this.#scale, this.#offset);
  }

  mul(other: Unit | number): Unit {
    if (typeof other === "number") {
      return new Unit(this.value * other, this.symbol, this.dimension, this.#scale, this.#offset);
    }
    if (this.#offset !== 0 || other.#offset !== 0) {
      throw new UnitParseError(
        `cannot multiply affine units ("${this.symbol}"/"${other.symbol}") — convert to a non-affine unit (e.g. "K" instead of "degC") first`,
      );
    }
    const dimension = multiplyDimensions(this.dimension, other.dimension);
    const symbol = this.isDimensionless ? other.symbol : other.isDimensionless ? this.symbol : composeSymbols(this.symbol, other.symbol, "*");
    return new Unit(this.value * other.value, symbol, dimension, this.#scale * other.#scale, 0);
  }

  div(other: Unit | number): Unit {
    if (typeof other === "number") {
      return new Unit(this.value / other, this.symbol, this.dimension, this.#scale, this.#offset);
    }
    if (this.#offset !== 0 || other.#offset !== 0) {
      throw new UnitParseError(
        `cannot divide affine units ("${this.symbol}"/"${other.symbol}") — convert to a non-affine unit (e.g. "K" instead of "degC") first`,
      );
    }
    const dimension = divideDimensions(this.dimension, other.dimension);
    const symbol = other.isDimensionless ? this.symbol : composeSymbols(this.symbol || "1", other.symbol, "/");
    return new Unit(this.value / other.value, symbol, dimension, this.#scale / other.#scale, 0);
  }

  /**
   * Integer/real power. v1 only supports SIMPLE symbols (no `*`/`/`/`^`
   * already in `this.symbol`) — `(m/s)^2` can't be spelled as `"m/s^2"`
   * (that means `m/(s^2)`, not `(m/s)^2`) without parentheses, which this
   * package's expression grammar deliberately doesn't support. Call `.mul()`
   * repeatedly for composite units instead (`speed.mul(speed)`).
   */
  pow(n: number): Unit {
    if (this.#offset !== 0) {
      throw new UnitParseError(`cannot exponentiate an affine unit ("${this.symbol}")`);
    }
    if (/[*/^]/.test(this.symbol)) {
      throw new UnitParseError(
        `Unit.pow() only supports a simple (non-composite) unit symbol; "${this.symbol}" is composite — use repeated .mul() instead`,
      );
    }
    const dimension = powDimension(this.dimension, n);
    const symbol = this.isDimensionless ? "" : n === 1 ? this.symbol : `${this.symbol}^${n}`;
    return new Unit(this.value ** n, symbol, dimension, this.#scale ** n, 0);
  }

  toString(digits?: number): string {
    const v = digits === undefined ? String(this.value) : this.value.toPrecision(digits);
    return this.symbol ? `${v} ${this.symbol}` : v;
  }
}
