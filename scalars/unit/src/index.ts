/**
 * mallory-unit — the only net-new scalar package in mallory-plus
 * (mallory-math supplies ComplexNumber/Rational/Decimal, re-exported via
 * mallory-scalar-types, but has no Unit type). See docs/PLAN.md's
 * "scalars/unit" entry and issue #23.
 */
export { Unit } from "./unit.ts";
export {
  BASE_DIMENSIONS,
  DIMENSIONLESS,
  dim,
  dimensionsEqual,
  dimensionToString,
  divideDimensions,
  isDimensionless,
  multiplyDimensions,
  powDimension,
  type BaseDimension,
  type Dimension,
} from "./dimension.ts";
export { BASE_UNITS, PREFIXES, lookupUnitSymbol, type UnitDef } from "./units-table.ts";
export { DimensionMismatchError, UnitParseError, UnknownUnitError } from "./errors.ts";
