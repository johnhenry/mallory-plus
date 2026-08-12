import { dim, type Dimension } from "./dimension.ts";

/**
 * A registered unit symbol's conversion contract: `baseValue = value * scale
 * + offset`, where `baseValue` is the magnitude in a fixed SI-coherent
 * reference for that dimension (metre, kilogram, second, ampere, kelvin,
 * mole, candela — the usual SI base/coherent-derived choice). `offset` is
 * nonzero only for affine units (`degC`, `degF`) and is never combined into
 * a composite expression (see `parser.ts`).
 */
export interface UnitDef {
  dimension: Dimension;
  scale: number;
  offset?: number;
  /** Whether SI prefixes (`k`, `m`, `c`, ...) may combine with this symbol — e.g. `g` (gram) is prefixable to `kg`/`mg`, but the already-prefixed `kg` itself is not (no `"kkg"`). */
  prefixable?: boolean;
}

export const BASE_UNITS: Record<string, UnitDef> = {
  // length
  m: { dimension: dim({ length: 1 }), scale: 1, prefixable: true },
  in: { dimension: dim({ length: 1 }), scale: 0.0254 },
  ft: { dimension: dim({ length: 1 }), scale: 0.3048 },
  yd: { dimension: dim({ length: 1 }), scale: 0.9144 },
  mi: { dimension: dim({ length: 1 }), scale: 1609.344 },

  // mass — the prefixable root is g (gram); kg is the SI-coherent base value
  // and is listed directly so prefixes never stack onto it ("kkg" etc.).
  g: { dimension: dim({ mass: 1 }), scale: 0.001, prefixable: true },
  kg: { dimension: dim({ mass: 1 }), scale: 1 },
  lb: { dimension: dim({ mass: 1 }), scale: 0.45359237 },
  oz: { dimension: dim({ mass: 1 }), scale: 0.028349523125 },

  // time
  s: { dimension: dim({ time: 1 }), scale: 1, prefixable: true },
  min: { dimension: dim({ time: 1 }), scale: 60 },
  h: { dimension: dim({ time: 1 }), scale: 3600 },
  day: { dimension: dim({ time: 1 }), scale: 86400 },

  // electric current
  A: { dimension: dim({ current: 1 }), scale: 1, prefixable: true },

  // temperature — degC/degF are affine (see offset)
  K: { dimension: dim({ temperature: 1 }), scale: 1 },
  degC: { dimension: dim({ temperature: 1 }), scale: 1, offset: 273.15 },
  degF: { dimension: dim({ temperature: 1 }), scale: 5 / 9, offset: 273.15 - 32 * (5 / 9) },

  // amount of substance
  mol: { dimension: dim({ amount: 1 }), scale: 1, prefixable: true },

  // luminous intensity
  cd: { dimension: dim({ luminosity: 1 }), scale: 1, prefixable: true },

  // named derived units
  Hz: { dimension: dim({ time: -1 }), scale: 1, prefixable: true },
  N: { dimension: dim({ mass: 1, length: 1, time: -2 }), scale: 1, prefixable: true },
  J: { dimension: dim({ mass: 1, length: 2, time: -2 }), scale: 1, prefixable: true },
  W: { dimension: dim({ mass: 1, length: 2, time: -3 }), scale: 1, prefixable: true },
  Pa: { dimension: dim({ mass: 1, length: -1, time: -2 }), scale: 1, prefixable: true },
  L: { dimension: dim({ length: 3 }), scale: 0.001, prefixable: true }, // litre = dm^3
};

/** SI prefix -> multiplier. Sorted longest-first at lookup time so "da" (deca) is tried before single-letter prefixes. */
export const PREFIXES: Record<string, number> = {
  y: 1e-24,
  z: 1e-21,
  a: 1e-18,
  f: 1e-15,
  p: 1e-12,
  n: 1e-9,
  u: 1e-6,
  "µ": 1e-6, // µ (micro sign)
  m: 1e-3,
  c: 1e-2,
  d: 1e-1,
  da: 1e1,
  h: 1e2,
  k: 1e3,
  M: 1e6,
  G: 1e9,
  T: 1e12,
  P: 1e15,
  E: 1e18,
  Z: 1e21,
  Y: 1e24,
};

const PREFIX_KEYS_LONGEST_FIRST = Object.keys(PREFIXES).sort((a, b) => b.length - a.length);

/** Resolve a single symbol (no `*`/`/`/`^`) to its `UnitDef`, trying a direct match first, then prefix decomposition. `undefined` if nothing matches. */
export function lookupUnitSymbol(symbol: string): UnitDef | undefined {
  const direct = BASE_UNITS[symbol];
  if (direct) return direct;

  for (const prefix of PREFIX_KEYS_LONGEST_FIRST) {
    if (symbol.length > prefix.length && symbol.startsWith(prefix)) {
      const rest = symbol.slice(prefix.length);
      const base = BASE_UNITS[rest];
      if (base?.prefixable) {
        return { dimension: base.dimension, scale: base.scale * (PREFIXES[prefix] as number) };
      }
    }
  }
  return undefined;
}
