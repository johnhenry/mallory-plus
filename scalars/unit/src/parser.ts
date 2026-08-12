import { DIMENSIONLESS, multiplyDimensions, powDimension, type Dimension } from "./dimension.ts";
import { UnitParseError, UnknownUnitError } from "./errors.ts";
import { lookupUnitSymbol } from "./units-table.ts";

export interface ParsedUnit {
  dimension: Dimension;
  /** `baseValue = value * scale + offset`. */
  scale: number;
  /** Nonzero only for a bare single affine unit (`degC`/`degF`); always 0 inside a composite expression. */
  offset: number;
  /** The (trimmed) expression this was parsed from — used as the resulting `Unit.symbol`. */
  canonical: string;
}

/**
 * Parses a unit expression: a bare symbol (`"m"`, `"degC"`, `"km"`) or a
 * `*`/`/`-separated chain of symbols each with an optional `^exponent`
 * (`"kg*m/s^2"`, `"m/s^2"`, `"mL/min"`). Left-to-right, no operator
 * precedence ambiguity to resolve (`^` always binds to the single term
 * immediately preceding it) and no parentheses — v1 deliberately doesn't
 * support them; see `Unit.pow()`'s doc comment for the one place that
 * matters.
 */
export function parseUnitExpression(expr: string): ParsedUnit {
  const trimmed = expr.trim();
  if (trimmed.length === 0) {
    throw new UnitParseError("empty unit expression");
  }

  // A bare single symbol is looked up directly so affine units (degC/degF)
  // can carry their offset — offset only makes sense for a standalone unit.
  if (!/[*/^]/.test(trimmed)) {
    const def = lookupUnitSymbol(trimmed);
    if (!def) throw new UnknownUnitError(trimmed);
    return { dimension: def.dimension, scale: def.scale, offset: def.offset ?? 0, canonical: trimmed };
  }

  const tokens = trimmed.match(/[*/]|[^*/]+/g);
  if (!tokens) throw new UnitParseError(`could not parse unit expression "${expr}"`);

  let dimension: Dimension = DIMENSIONLESS;
  let scale = 1;
  let op: "*" | "/" = "*";
  for (const raw of tokens) {
    if (raw === "*" || raw === "/") {
      op = raw;
      continue;
    }
    const termStr = raw.trim();
    if (termStr.length === 0) continue;

    const [symbolPart, expPart, ...rest] = termStr.split("^");
    if (rest.length > 0) {
      throw new UnitParseError(`too many "^" in term "${termStr}" (in "${expr}")`);
    }
    const exponent = expPart === undefined ? 1 : Number(expPart);
    if (!Number.isFinite(exponent)) {
      throw new UnitParseError(`invalid exponent in "${termStr}" (in "${expr}")`);
    }

    const def = lookupUnitSymbol(symbolPart as string);
    if (!def) throw new UnknownUnitError(symbolPart as string);
    if (def.offset) {
      throw new UnitParseError(
        `affine unit "${symbolPart}" cannot be used inside a composite unit expression ("${expr}") — use it standalone only`,
      );
    }

    const signedExponent = op === "/" ? -exponent : exponent;
    dimension = multiplyDimensions(dimension, powDimension(def.dimension, signedExponent));
    scale *= def.scale ** signedExponent;
    op = "*";
  }

  return { dimension, scale, offset: 0, canonical: trimmed };
}
